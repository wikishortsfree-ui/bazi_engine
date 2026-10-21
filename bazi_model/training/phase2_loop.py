"""
Shared Phase 2 training-loop helpers (MLX).

Moved from ``scripts/train_phase2.py`` so that multiple Phase 2 training
entry points (e.g. ``scripts/train_phase2.py`` for prose curation and
``scripts/train_projection_qa.py`` for Stage 1 QA) can share the optimizer,
grad clipping, gradient-tree manipulation, checkpoint I/O, batch conversion,
and validation logic without duplication.

Nothing here changes runtime behavior compared with the previous inline
implementations in ``train_phase2.py``. See that script's module docstring
for checkpoint format and resume semantics.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def make_lr_schedule(
    total_steps: int,
    peak_lr: float,
    warmup_fraction: float = 0.1,
    cosine_final_fraction: float = 0.1,
):
    """Linear warmup then cosine decay to ``cosine_final_fraction * peak_lr``."""
    import mlx.optimizers as optim

    warmup_steps = max(1, int(total_steps * warmup_fraction))
    decay_steps = total_steps - warmup_steps
    final_lr = peak_lr * cosine_final_fraction

    warmup_schedule = optim.linear_schedule(0.0, peak_lr, warmup_steps)
    cosine_schedule = optim.cosine_decay(peak_lr, decay_steps, end=final_lr)
    return optim.join_schedules([warmup_schedule, cosine_schedule], [warmup_steps])


def create_optimizer(
    projection_lr: float = 5e-5,
    total_steps: int = 1000,
    weight_decay: float = 0.01,
    warmup_fraction: float = 0.1,
    cosine_final_fraction: float = 0.1,
):
    """AdamW with warmup+cosine schedule for projection training."""
    import mlx.optimizers as optim

    lr_schedule = make_lr_schedule(
        total_steps,
        peak_lr=projection_lr,
        warmup_fraction=warmup_fraction,
        cosine_final_fraction=cosine_final_fraction,
    )
    return optim.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)


def clip_grads(grads: dict, max_norm: float = 1.0) -> tuple[dict, float]:
    """Clip gradients by global norm. Returns ``(clipped_grads, original_norm)``."""
    import mlx.optimizers as optim

    clipped, norm = optim.clip_grad_norm(grads, max_norm)
    return clipped, float(norm)


def _add_grad_trees(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _add_grad_trees(a[k], b[k]) for k in a if k in b}
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a)(_add_grad_trees(ai, bi) for ai, bi in zip(a, b))
    return a + b


def _scale_grad_tree(tree, scale: float):
    if isinstance(tree, dict):
        return {k: _scale_grad_tree(v, scale) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_scale_grad_tree(v, scale) for v in tree)
    return tree * scale


def _zero_tree(tree):
    import mlx.core as mx
    if isinstance(tree, dict):
        return {k: _zero_tree(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_zero_tree(v) for v in tree)
    return mx.zeros_like(tree)


def _zero_grads_except_trainable(grads: dict) -> None:
    """In place: zero every grad branch except projection_layer / cross_interaction_encoder."""
    trainable_keys = {"projection_layer", "cross_interaction_encoder"}
    for key in list(grads.keys()):
        if key not in trainable_keys:
            grads[key] = _zero_tree(grads[key])


def _batch_to_mx(batch: dict, n_virtual: int):
    """Convert collated numpy batch to MLX arrays (chart inputs + ids + mask)."""
    import mlx.core as mx
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if k in ("full_texts", "task_types"):
            out[k] = v
            continue
        if k == "chart_natal_inputs":
            out[k] = {kk: mx.array(vv) for kk, vv in v.items()}
            continue
        if k == "chart_dynamic_inputs":
            if v is None:
                out[k] = None
            else:
                out[k] = {kk: mx.array(vv) for kk, vv in v.items()}
            continue
        if isinstance(v, (list, tuple)):
            continue
        try:
            out[k] = mx.array(v)
        except Exception:
            out[k] = v
    return out


def _flatten_params(params, prefix: str = ""):
    """Flatten ``Module.parameters()`` to dotted-key -> array dict."""
    out: dict[str, Any] = {}
    if isinstance(params, dict):
        for k, v in params.items():
            out.update(_flatten_params(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(params, (list, tuple)):
        for i, v in enumerate(params):
            out.update(_flatten_params(v, f"{prefix}.{i}" if prefix else str(i)))
    else:
        out[prefix] = params
    return out


def _load_projection_from_checkpoint(projection, path: Path, cross_encoder=None) -> None:
    """Load projection + optional cross_interaction_encoder weights from a Phase 2 .npz."""
    import numpy as np
    from bazi_model.encoders.natal_encoder_mlx import _flat_to_nested

    data = dict(np.load(path, allow_pickle=True))

    proj_flat = {
        k.replace("projection_layer.", ""): np.asarray(v)
        for k, v in data.items()
        if k.startswith("projection_layer.")
    }
    if proj_flat:
        nested = _flat_to_nested(proj_flat)
        projection.update(nested, strict=False)

    if cross_encoder is not None:
        cross_flat = {
            k.replace("cross_interaction_encoder.", ""): np.asarray(v)
            for k, v in data.items()
            if k.startswith("cross_interaction_encoder.")
        }
        if cross_flat:
            nested = _flat_to_nested(cross_flat)
            cross_encoder.update(nested, strict=False)


def _load_optimizer_state(optimizer, opt_path: Path) -> bool:
    """Restore Adam moments + per-param step counters; returns True on success."""
    import numpy as np
    from bazi_model.encoders.natal_encoder_mlx import _flat_to_nested

    if not opt_path.exists():
        return False
    opt_data = {k: v for k, v in np.load(opt_path, allow_pickle=True).items()}
    nested = _flat_to_nested(opt_data)
    optimizer.state = nested
    return True


def _save_checkpoint(
    model,
    optimizer,
    step: int,
    best_val_loss: float,
    checkpoint_dir: Path,
    best: bool,
) -> None:
    """Save ``checkpoint_{best,latest}.npz`` + optimizer state in ``checkpoint_dir``."""
    import numpy as np

    prefix = "checkpoint_best" if best else "checkpoint_latest"
    state = {
        "step": np.array(step, dtype=np.int64),
        "best_val_loss": np.array(best_val_loss, dtype=np.float32),
    }
    flat = _flatten_params(model.projection_layer.parameters())
    for k, v in flat.items():
        state[f"projection_layer.{k}"] = np.asarray(v)
    if model.cross_interaction_encoder is not None:
        flat_cross = _flatten_params(model.cross_interaction_encoder.parameters())
        for k, v in flat_cross.items():
            state[f"cross_interaction_encoder.{k}"] = np.asarray(v)
    np.savez(str(checkpoint_dir / f"{prefix}.npz"), **state)
    try:
        opt_state = dict(optimizer.state)
        flat_opt = _flatten_params(opt_state)
        np.savez(
            str(checkpoint_dir / f"{prefix}_optimizer.npz"),
            **{k: np.asarray(v) for k, v in flat_opt.items()},
        )
    except Exception as e:
        print(f"  Note: could not save optimizer state: {e}", file=sys.stderr)
    print(f"  Saved {prefix} (step={step}, best_val_loss={best_val_loss:.4f})")


def _run_validation(model, val_ds, tokenizer, args, n_virtual, loss_fn, collate_fn=None):
    """Run validation over ``val_ds``; return mean val loss.

    ``collate_fn`` defaults to ``phase2_collate`` (prose dataset) but can be
    swapped for QA dataset consumers. The loop calls
    ``collate_fn(batch, tokenizer, max_length=args.max_length)``.
    """
    if len(val_ds) == 0:
        return 0.0

    if collate_fn is None:
        from bazi_model.data.phase2_dataset import phase2_collate as collate_fn  # type: ignore

    batch_size = args.batch_size
    losses = []
    for start in range(0, len(val_ds), batch_size):
        indices = list(range(start, min(start + batch_size, len(val_ds))))
        batch = collate_fn([val_ds[i] for i in indices], tokenizer, max_length=args.max_length)
        if not batch:
            continue
        batch_mx = _batch_to_mx(batch, n_virtual)
        loss = loss_fn(model, batch_mx)
        losses.append(float(loss))
    return sum(losses) / len(losses) if losses else 0.0


def _log_sample_generations(model, val_ds, tokenizer, args, n: int = 2) -> None:
    """Placeholder hook for sample decoding during validation."""
    if not val_ds or n <= 0:
        return
    print("  [sample_generations] enabled (decode stub; extend for full generation)")
