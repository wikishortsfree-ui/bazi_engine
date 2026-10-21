#!/usr/bin/env python3
"""
Projection re-training on Stage 1 factual QA (short-answer) data.

Rationale
---------
The 64-token projection was originally trained on long-form prose narratives,
which smeared the per-fact gradient signal and left the probe at ~30% overall
accuracy (0% on pillar identity, ~75% on interactions). This script retrains
the projection (and the cross-interaction encoder) on single-fact QA so that
loss gradients land where the probe actually evaluates: short factual answers.

Everything except the dataset is shared with ``scripts/train_phase2.py``:
- frozen Qwen + trainable projection + cross-interaction encoder
- AdamW with warmup+cosine schedule, grad clipping, grad accumulation
- ``phase2_collate`` produces the same ``input_ids / labels / loss_mask`` shape
  by setting ``input_sequence = question`` and ``target_sequence = answer``;
  the mask therefore covers only answer tokens. See ``bazi_model/data/phase2_dataset.py``.
- checkpoint format is drop-in compatible with ``scripts/probe_factual_64token.py``.

Usage (from repo root, Phase 2 env):
  python scripts/train_projection_qa.py --subset_size 5000 --epochs 1 --init_mode fresh
  python scripts/train_projection_qa.py --subset_size 20000 --epochs 3 --init_mode warm \\
      --projection_checkpoint checkpoints/phase3_prep/checkpoint_best.npz

See ``plans/projection_qa_retrain_*.plan.md`` for the full execution flow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from bazi_model.training.phase2_loop import (  # noqa: E402
    _add_grad_trees,
    _batch_to_mx,
    _load_optimizer_state,
    _load_projection_from_checkpoint,
    _run_validation,
    _save_checkpoint,
    _scale_grad_tree,
    clip_grads,
    create_optimizer,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrain 64-token projection on Stage 1 factual QA.",
    )
    # Data
    p.add_argument(
        "--qa_data",
        type=Path,
        default=_root / "data" / "stage1_qa.jsonl",
        help="Stage 1 QA JSONL (chart_id/question/answer/question_category/template_id).",
    )
    p.add_argument(
        "--qa_charts",
        type=Path,
        default=_root / "data" / "qa_charts.jsonl",
        help="Chart lookup JSONL (chart_id -> chart); joined in memory.",
    )
    p.add_argument(
        "--subset_size",
        type=int,
        default=20000,
        help="Cap on QA records loaded from --qa_data. Pilot: 5000; full: 20000.",
    )
    p.add_argument(
        "--qa_val_per_category",
        type=int,
        default=500,
        help="Per-bucket val hold-out (capped at available count).",
    )
    # Init
    p.add_argument(
        "--init_mode",
        choices=("fresh", "warm"),
        default="fresh",
        help="'fresh' = random init for projection; 'warm' = load --projection_checkpoint.",
    )
    p.add_argument(
        "--projection_checkpoint",
        type=Path,
        default=None,
        help="Required when --init_mode=warm.",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Resume from checkpoint .npz: restores model weights, optimizer state, "
            "step counter, and LR-schedule position. Mirrors --resume in train_phase2.py."
        ),
    )
    # Batch and accumulation
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accumulation_steps", type=int, default=2)
    # Optimizer / schedule
    p.add_argument("--projection_lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.1)
    p.add_argument("--cosine_final_fraction", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # Training length
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None)
    # Model / env
    p.add_argument(
        "--n_virtual_tokens",
        type=int,
        default=63,
        help="Total virtual tokens = 7 slots * 9 virtual tokens per slot.",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=320,
        help="Short QA fits in ~320 tokens (much faster than prose's 960).",
    )
    # Logging / checkpointing
    p.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=_root / "checkpoints" / "phase3_prep" / "qa",
        help="Where to write checkpoint_{best,latest}.npz (and _optimizer.npz).",
    )
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--val_every_steps", type=int, default=200)
    p.add_argument("--log_every_steps", type=int, default=50)
    p.add_argument("--save_every_steps", type=int, default=500)
    # Model paths
    p.add_argument(
        "--llm_path",
        type=Path,
        default=_root / "models" / "qwen2.5-7b-instruct-mlx",
    )
    p.add_argument(
        "--natal_mode",
        choices=("encoder", "embedding"),
        default="encoder",
        help=(
            "'encoder' = frozen pretrained NatalEncoderMLX (default); "
            "'embedding' = EmbeddingNatalMLX trained jointly with projection."
        ),
    )
    p.add_argument(
        "--natal_weights",
        type=Path,
        default=_root / "checkpoints" / "natal_encoder_mlx.npz",
        help="Required for --natal_mode=encoder; ignored for embedding.",
    )
    p.add_argument(
        "--dynamic_weights",
        type=Path,
        default=_root / "checkpoints" / "dynamic_encoder_mlx.npz",
    )
    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--balance_categories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Weight samples so sampler matches TARGET_CATEGORY_DISTRIBUTION.",
    )
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--hybrid_chart_text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set, prepend a structured Chinese chart text block to each "
            "QA prompt so the LLM sees virtual tokens + text + question. "
            "See plans/hybrid-prompt-v1-and-lora_*.plan.md §4."
        ),
    )
    p.add_argument(
        "--skip_grad_sanity",
        action="store_true",
        help="Skip the first-batch grad-sanity assertions (not recommended).",
    )
    return p.parse_args()


def _grad_is_all_none_or_zero(tree) -> bool:
    """Return True iff every leaf in ``tree`` is None or a zero-norm array.

    MLX ``value_and_grad`` on a module with frozen params returns ``None``
    leaves for the frozen sub-tree.
    """
    import mlx.core as mx

    if tree is None:
        return True
    if isinstance(tree, dict):
        return all(_grad_is_all_none_or_zero(v) for v in tree.values())
    if isinstance(tree, (list, tuple)):
        return all(_grad_is_all_none_or_zero(v) for v in tree)
    if isinstance(tree, mx.array):
        return float(mx.abs(tree).sum()) == 0.0
    return False


def _grad_tree_nonzero_and_finite(tree) -> tuple[bool, bool]:
    """Return ``(any_nonzero, all_finite)`` across every array leaf in ``tree``."""
    import mlx.core as mx

    any_nonzero = False
    all_finite = True

    def _visit(node):
        nonlocal any_nonzero, all_finite
        if node is None:
            return
        if isinstance(node, dict):
            for v in node.values():
                _visit(v)
            return
        if isinstance(node, (list, tuple)):
            for v in node:
                _visit(v)
            return
        if isinstance(node, mx.array):
            if not bool(mx.all(mx.isfinite(node))):
                all_finite = False
            if float(mx.abs(node).sum()) > 0.0:
                any_nonzero = True

    _visit(tree)
    return any_nonzero, all_finite


def _run_grad_sanity(model, grads: dict, *, trainable_keys: tuple[str, ...]) -> None:
    """Assert frozen-LLM grads are None/zero and trainable grads are finite + nonzero.

    ``trainable_keys`` lists submodules expected to receive non-zero gradients
    (always includes ``projection_layer``; includes ``natal_encoder`` when
    ``--natal_mode=embedding``).
    """
    llm_grad = grads.get("llm_model")
    if not _grad_is_all_none_or_zero(llm_grad):
        raise RuntimeError(
            "Grad sanity failed: llm_model gradients are not None/zero. "
            "freeze_except_projection() was not applied or module structure changed."
        )

    optional_keys = {"cross_interaction_encoder"}
    for key in trainable_keys:
        sub = grads.get(key)
        if sub is None:
            if key in optional_keys:
                continue
            raise RuntimeError(f"Grad sanity failed: no gradients for {key}.")
        any_nz, all_finite = _grad_tree_nonzero_and_finite(sub)
        if not all_finite:
            raise RuntimeError(f"Grad sanity failed: non-finite gradients in {key}.")
        if not any_nz:
            raise RuntimeError(f"Grad sanity failed: {key} gradients are all zero.")

    print("  [grad_sanity] OK: LLM grads None/zero; trainable grads finite and nonzero.")


def main() -> int:
    args = _parse_args()

    if args.init_mode == "warm":
        if args.projection_checkpoint is None or not args.projection_checkpoint.exists():
            print(
                "Error: --init_mode=warm requires --projection_checkpoint to an existing .npz.",
                file=sys.stderr,
            )
            return 1

    if args.resume is not None and not args.resume.exists():
        print(f"Error: --resume checkpoint not found: {args.resume}", file=sys.stderr)
        return 1

    try:
        import numpy as np
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load as mlx_lm_load
        from mlx_lm.utils import does_model_support_input_embeddings

        from bazi_model.data.phase2_dataset import phase2_collate
        from bazi_model.data.stage1_qa_dataset import (
            Stage1QADataset,
            get_category_weights_for_sampler,
            stratified_holdout_indices,
        )
        from bazi_model.encoders.dynamic_encoder_mlx import DynamicEncoderMLX
        from bazi_model.encoders.embedding_natal_mlx import EmbeddingNatalMLX
        from bazi_model.encoders.natal_encoder_mlx import NatalEncoderMLX
        from bazi_model.encoders.cross_interaction_encoder_mlx import CrossInteractionEncoderMLX
        from bazi_model.model.bazi_llm import BaziLLM
        from bazi_model.projection import ProjectionLayer
        from bazi_model.training.phase2_loss import compute_lm_loss
    except ImportError as e:
        print(
            "Error: MLX / bazi_model not available. Install Phase 2 deps: "
            "pip install -r requirements_phase2.txt",
            file=sys.stderr,
        )
        print(f"ImportError: {e}", file=sys.stderr)
        return 1

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # --- Data -------------------------------------------------------------
    if not args.qa_data.exists():
        print(f"Error: QA data not found: {args.qa_data}", file=sys.stderr)
        return 1
    if not args.qa_charts.exists():
        print(f"Error: QA charts not found: {args.qa_charts}", file=sys.stderr)
        return 1

    print(f"Loading QA dataset (limit={args.subset_size}) ...")
    full_ds = Stage1QADataset(
        args.qa_data,
        args.qa_charts,
        limit=args.subset_size,
        hybrid_chart_text=args.hybrid_chart_text,
    )
    if args.hybrid_chart_text:
        print("  hybrid_chart_text: ON (chart text prepended to QA prompts)")
    if len(full_ds) == 0:
        print("Error: QA dataset is empty.", file=sys.stderr)
        return 1
    print(f"  {len(full_ds)} QA pairs loaded")

    train_idx, val_idx = stratified_holdout_indices(
        full_ds, per_category=args.qa_val_per_category, seed=args.seed
    )
    print(f"  split: train={len(train_idx)}  val={len(val_idx)}")

    class _Subset:
        """Index-view over Stage1QADataset that still exposes ``_records``
        so ``get_category_weights_for_sampler`` and ``phase2_collate`` work."""

        def __init__(self, parent: Stage1QADataset, indices: np.ndarray) -> None:
            self._parent = parent
            self._indices = indices
            self._records = [parent._records[int(i)] for i in indices]

        def __len__(self) -> int:
            return len(self._indices)

        def __getitem__(self, i: int) -> dict:
            return self._parent[int(self._indices[i])]

    train_ds = _Subset(full_ds, train_idx)
    val_ds = _Subset(full_ds, val_idx)

    if args.balance_categories:
        weights = get_category_weights_for_sampler(train_ds)  # type: ignore[arg-type]
        w = np.asarray(weights, dtype=np.float64)
        total = w.sum()
        if total <= 0:
            print("Warning: category weights sum to 0; falling back to uniform.", file=sys.stderr)
            w = None
        else:
            w = w / total
    else:
        w = None

    required_paths = [args.llm_path, args.dynamic_weights]
    if args.natal_mode == "encoder":
        required_paths.append(args.natal_weights)
    for path in required_paths:
        if not path.exists():
            print(f"Error: required model path not found: {path}", file=sys.stderr)
            return 1

    print("Loading encoders and projection ...")
    if args.natal_mode == "encoder":
        natal_encoder = NatalEncoderMLX(d_emb=32, d_model=256, num_layers=3)
        natal_encoder.load_weights(str(args.natal_weights))
        print("  natal: NatalEncoderMLX (frozen, pretrained)")
    else:
        natal_encoder = EmbeddingNatalMLX(d_emb=64, d_model=256)
        print("  natal: EmbeddingNatalMLX (trained jointly with projection)")
    dynamic_encoder = DynamicEncoderMLX(d_emb=32, d_model=256, num_layers=2)
    dynamic_encoder.load_weights(str(args.dynamic_weights))
    cross_encoder = CrossInteractionEncoderMLX(d_input=27, d_hidden=128, d_model=256)

    # Per-slot projection: 5 natal slots + 1 dynamic + 1 cross = 7 slots.
    n_slots = 7
    if args.n_virtual_tokens % n_slots != 0:
        print(
            f"Error: --n_virtual_tokens ({args.n_virtual_tokens}) must be divisible by "
            f"{n_slots} (per-slot projection); use 63 (= 7 * 9).",
            file=sys.stderr,
        )
        return 1
    n_virtual_per_slot = args.n_virtual_tokens // n_slots
    projection = ProjectionLayer(
        d_in_per_slot=256,
        d_llm=3584,
        n_slots=n_slots,
        n_virtual_per_slot=n_virtual_per_slot,
    )

    if args.resume is not None:
        print(f"Resuming projection + cross encoder weights from {args.resume} ...")
        _load_projection_from_checkpoint(projection, args.resume, cross_encoder)
    elif args.init_mode == "warm":
        print(f"Warm-starting projection + cross encoder from {args.projection_checkpoint} ...")
        _load_projection_from_checkpoint(projection, args.projection_checkpoint, cross_encoder)
    else:
        print("Fresh projection init (no warm start).")

    print(f"Loading Qwen from {args.llm_path} ...")
    llm_model, tokenizer_wrapper = mlx_lm_load(str(args.llm_path))
    if not does_model_support_input_embeddings(llm_model):
        print("Error: this model does not support input_embeddings.", file=sys.stderr)
        return 1
    tokenizer = getattr(tokenizer_wrapper, "_tokenizer", tokenizer_wrapper)
    if not callable(tokenizer):
        print(
            f"Error: tokenizer ({type(tokenizer)}) is not callable; expected HF tokenizer.",
            file=sys.stderr,
        )
        return 1

    model = BaziLLM(
        natal_encoder,
        dynamic_encoder,
        projection,
        llm_model,
        tokenizer,
        cross_interaction_encoder=cross_encoder,
        max_length=args.max_length,
        n_virtual_tokens=args.n_virtual_tokens,
    )
    model.freeze_except_projection()
    if args.natal_mode == "embedding":
        # Train the embedding natal head jointly with projection + cross encoder.
        model.natal_encoder.unfreeze()
    if args.gradient_checkpointing:
        model.gradient_checkpointing = True
        print("Gradient checkpointing: ON")

    trainable_keys: tuple[str, ...] = ("projection_layer", "cross_interaction_encoder")
    if args.natal_mode == "embedding":
        trainable_keys = trainable_keys + ("natal_encoder",)

    from mlx.utils import tree_flatten
    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(
        f"Trainable params: {n_trainable:,}  "
        f"(trainable submodules: {', '.join(trainable_keys)})"
    )

    n_virtual = args.n_virtual_tokens
    accum_steps = args.accumulation_steps
    max_norm = args.max_grad_norm
    batch_size = args.batch_size

    num_micro = max(1, (len(train_ds) + batch_size - 1) // batch_size)
    steps_per_epoch = max(1, num_micro // accum_steps)
    if args.max_steps is not None:
        total_steps = args.max_steps
        epochs = max(1, (total_steps + steps_per_epoch - 1) // steps_per_epoch)
    else:
        epochs = args.epochs
        total_steps = steps_per_epoch * epochs

    optimizer = create_optimizer(
        projection_lr=args.projection_lr,
        total_steps=total_steps,
        weight_decay=args.weight_decay,
        warmup_fraction=args.warmup_fraction,
        cosine_final_fraction=args.cosine_final_fraction,
    )

    def loss_fn(m, batch_mx):
        logits = m(
            batch_mx["chart_natal_inputs"],
            batch_mx["chart_dynamic_inputs"],
            batch_mx["full_texts"],
            cross_features=batch_mx.get("cross_features"),
        )
        L = batch_mx["labels"].shape[1]
        logits_text = logits[:, n_virtual : n_virtual + L, :]
        return compute_lm_loss(logits_text, batch_mx["labels"], batch_mx["loss_mask"])

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # --- Grad sanity on first batch --------------------------------------
    if not args.skip_grad_sanity:
        sanity_indices = np.arange(min(batch_size, len(train_ds)))
        sanity_batch = phase2_collate(
            [train_ds[int(i)] for i in sanity_indices],
            tokenizer,
            max_length=args.max_length,
        )
        sanity_mx = _batch_to_mx(sanity_batch, n_virtual)
        print("Running grad sanity on first batch ...")
        loss, grads = loss_and_grad_fn(model, sanity_mx)
        mx.eval(loss, grads)
        print(f"  sanity_loss={float(loss):.4f}")
        _run_grad_sanity(model, grads, trainable_keys=trainable_keys)

    # --- Training loop ----------------------------------------------------
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    no_improve_count = 0
    global_step = 0
    start_epoch = 0
    skip_micros = 0

    if args.resume is not None:
        ckpt_data = dict(np.load(args.resume, allow_pickle=True))
        global_step = int(ckpt_data["step"])
        best_val_loss = float(ckpt_data["best_val_loss"])
        start_epoch = global_step // steps_per_epoch
        steps_into_epoch = global_step - start_epoch * steps_per_epoch
        skip_micros = steps_into_epoch * accum_steps
        opt_path = args.resume.parent / args.resume.name.replace(".npz", "_optimizer.npz")
        if _load_optimizer_state(optimizer, opt_path):
            print(
                f"Resumed: step={global_step}, best_val_loss={best_val_loss:.4f}, "
                "optimizer+LR restored"
            )
        else:
            print(
                f"Resumed: step={global_step}, best_val_loss={best_val_loss:.4f} "
                "(optimizer state not found; LR restarts)",
                file=sys.stderr,
            )

    for epoch in range(start_epoch, epochs):
        if w is not None:
            epoch_indices = np.random.choice(len(train_ds), size=len(train_ds), replace=True, p=w)
        else:
            epoch_indices = np.random.permutation(len(train_ds))

        epoch_loss_sum = 0.0
        epoch_batches = 0
        accum_loss = 0.0
        accumulated_grads = None
        batch_count_in_accum = 0

        for micro_idx, start in enumerate(range(0, len(epoch_indices), batch_size)):
            if epoch == start_epoch and micro_idx < skip_micros:
                continue
            indices = epoch_indices[start : start + batch_size]
            if len(indices) == 0:
                continue
            batch = phase2_collate(
                [train_ds[int(i)] for i in indices],
                tokenizer,
                max_length=args.max_length,
            )
            if not batch:
                continue
            batch_mx = _batch_to_mx(batch, n_virtual)
            loss, grads = loss_and_grad_fn(model, batch_mx)
            mx.eval(loss, grads)
            accum_loss += float(loss)
            batch_count_in_accum += 1
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = _add_grad_trees(accumulated_grads, grads)

            if batch_count_in_accum >= accum_steps:
                accumulated_grads = _scale_grad_tree(accumulated_grads, 1.0 / accum_steps)
                mx.eval(accumulated_grads)
                clipped, gnorm = clip_grads(accumulated_grads, max_norm=max_norm)
                optimizer.update(model, clipped)
                mx.eval(model.parameters(), optimizer.state)

                step_loss = accum_loss / accum_steps
                epoch_loss_sum += step_loss
                epoch_batches += 1
                global_step += 1

                if global_step % args.log_every_steps == 0:
                    lr = optimizer.learning_rate
                    if isinstance(lr, mx.array):
                        lr_val = float(lr)
                    elif callable(lr):
                        lr_val = float(lr(global_step))
                    else:
                        lr_val = lr
                    print(
                        f"  step {global_step}  train_loss={step_loss:.4f}  "
                        f"grad_norm={gnorm:.4f}  lr={lr_val:.2e}"
                    )

                if global_step % args.val_every_steps == 0:
                    val_loss = _run_validation(
                        model, val_ds, tokenizer, args, n_virtual, loss_fn,
                        collate_fn=phase2_collate,
                    )
                    print(f"  step {global_step}  val_loss={val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        _save_checkpoint(
                            model, optimizer, global_step, best_val_loss,
                            args.checkpoint_dir, best=True,
                        )
                    else:
                        no_improve_count += 1
                    if no_improve_count >= args.early_stopping_patience:
                        print(
                            f"Early stopping: no val improvement for "
                            f"{args.early_stopping_patience} checks."
                        )
                        break

                if global_step % args.save_every_steps == 0:
                    _save_checkpoint(
                        model, optimizer, global_step, best_val_loss,
                        args.checkpoint_dir, best=False,
                    )

                if args.max_steps is not None and global_step >= args.max_steps:
                    break

                accum_loss = 0.0
                accumulated_grads = None
                batch_count_in_accum = 0

        if args.max_steps is not None and global_step >= args.max_steps:
            print(f"Reached max_steps={args.max_steps}.")
            break
        if no_improve_count >= args.early_stopping_patience:
            break

        val_loss = _run_validation(
            model, val_ds, tokenizer, args, n_virtual, loss_fn,
            collate_fn=phase2_collate,
        )
        avg_train = epoch_loss_sum / epoch_batches if epoch_batches else 0.0
        print(f"epoch {epoch + 1}/{epochs}  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            _save_checkpoint(
                model, optimizer, global_step, best_val_loss,
                args.checkpoint_dir, best=True,
            )
        else:
            no_improve_count += 1
        if no_improve_count >= args.early_stopping_patience:
            print(f"Early stopping after epoch {epoch + 1}.")
            break

    print(f"Done. Best val_loss={best_val_loss:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
