#!/usr/bin/env python3
"""
Phase 2 training: projection-layer (and optionally encoder) fine-tuning on MLX.

Uses BaziLLM with frozen LLM; trainable: projection (default), optionally encoders at low LR.
Data: phase2_train.jsonl / phase2_val.jsonl (Task 2.3). Loss: masked causal LM (reasoning + output only).

Optimizer: AdamW with 10% warmup + cosine decay to 10% of peak LR.
Gradient clipping: max_norm=1.0. Gradient accumulation: default 2 (effective batch = batch_size * accumulation_steps).

Usage (from repo root, Phase 2 env):
  python scripts/train_phase2.py [options]
  python -m scripts.train_phase2 --help

Ref: docs/phase2_task_2_4_plan.md; docs/design_decisions.md §17–19.

---
BaziLLM training convention (full sequence, labels, loss_mask)
-------------------------------------------------------------
- **Model input:** BaziLLM.forward receives the FULL sequence as query_text: that is,
  task_token + "\\n" + query + "\\n" + reasoning + "\\n" + output (teacher-forcing).
  So the text sequence = input_sequence + "\\n" + target_sequence from the prepared JSONL.
- **Tokenization:** The full string is tokenized once; we get input_ids (batch, seq_len)
  after padding/truncation to max_length.
- **Labels:** Causal LM uses shifted labels: labels[i] = token at position i+1.
  So labels has the same length as input_ids; labels[seq_len-1] is set to ignore_index
  (or pad) since there is no next token to predict.
- **Loss mask:** We compute loss only on positions where the NEXT token is part of
  reasoning + output (not on task+query). So: prefix_len = number of tokens in
  (task_token + "\\n" + query). Then loss_mask[i] = 1 for prefix_len-1 <= i <= seq_len-2,
  and 0 elsewhere (and 0 where labels are pad/ignore). So the model is trained to
  predict reasoning and output tokens; virtual tokens + task + query are context only.
- **Data pipeline:** Phase2ChartDataset loads phase2_*.jsonl; phase2_collate(tokenizer, ...)
  produces input_ids, labels, loss_mask, full_texts, chart_natal_inputs, chart_dynamic_inputs.
  The training loop passes the full sequence (as list of strings) to BaziLLM.forward;
  for loss we use the precomputed labels and loss_mask from the collate (same
  tokenization ensures alignment). See bazi_model/data/phase2_dataset.py.

Checkpoint format (checkpoints/phase2_baseline/): checkpoint_latest.npz and checkpoint_best.npz
contain step (int64), best_val_loss (float32), and projection_layer.* (e.g. projection_layer.linear.weight,
projection_layer.linear.bias). See docs/design_decisions.md §19.2.

Resume: --resume <checkpoint.npz> restores model weights, optimizer state (Adam moments + per-param
step counters), global step, and best_val_loss. The LR schedule resumes at the correct position
because MLX's schedule is driven by the per-parameter step counters in the optimizer state.
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
    _log_sample_generations,
    _run_validation,
    _save_checkpoint,
    _scale_grad_tree,
    _zero_grads_except_trainable,
    _zero_tree,
    clip_grads,
    create_optimizer,
    make_lr_schedule,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: train projection (and optionally encoders) with BaziLLM on MLX.",
    )
    # Data (paths; actual loading in data-loading todo)
    parser.add_argument(
        "--train_data",
        type=Path,
        default=_root / "data" / "phase2_train.jsonl",
        help="Path to phase2 train JSONL.",
    )
    parser.add_argument(
        "--val_data",
        type=Path,
        default=_root / "data" / "phase2_val.jsonl",
        help="Path to phase2 validation JSONL.",
    )
    # Batch and accumulation
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size.")
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (effective batch = batch_size * accumulation_steps).",
    )
    # Learning rates
    parser.add_argument(
        "--projection_lr",
        type=float,
        default=5e-5,
        help="Learning rate for projection layer.",
    )
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=0.0,
        help="Learning rate for encoders (0 = frozen).",
    )
    # Optimizer
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )
    # Schedule
    parser.add_argument(
        "--warmup_fraction",
        type=float,
        default=0.1,
        help="Fraction of total steps for linear warmup (e.g. 0.1 = 10%%).",
    )
    parser.add_argument(
        "--cosine_final_fraction",
        type=float,
        default=0.1,
        help="Final LR as fraction of peak after cosine decay (e.g. 0.1 = 10%%).",
    )
    # Gradient clipping
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max global norm for gradient clipping.",
    )
    # Training length
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max training steps (overrides epochs if set).",
    )
    # Model / env
    parser.add_argument("--n_virtual_tokens", type=int, default=64, help="Number of virtual tokens.")
    parser.add_argument(
        "--max_length",
        type=int,
        default=960,
        help="Max text token length (virtual + text <= 1024 for Qwen context).",
    )
    # Checkpoint and validation
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=_root / "checkpoints" / "phase3_prep",
        help="Directory for checkpoints.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Stop after N consecutive validation checks without improvement.",
    )
    parser.add_argument(
        "--val_every_steps",
        type=int,
        default=200,
        help="Run validation every N steps (and at end of each epoch).",
    )
    parser.add_argument(
        "--log_every_steps",
        type=int,
        default=50,
        help="Log train loss every N steps.",
    )
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=500,
        help="Save latest checkpoint every N steps (best is saved on val improvement).",
    )
    parser.add_argument(
        "--sample_generations",
        action="store_true",
        help="Log a few sample generations at validation time (optional).",
    )
    # Model paths
    parser.add_argument(
        "--llm_path",
        type=Path,
        default=_root / "models" / "qwen2.5-7b-instruct-mlx",
        help="Path to MLX-converted Qwen model directory.",
    )
    parser.add_argument(
        "--natal_weights",
        type=Path,
        default=_root / "checkpoints" / "natal_encoder_mlx.npz",
        help="Path to natal encoder MLX weights (.npz).",
    )
    parser.add_argument(
        "--dynamic_weights",
        type=Path,
        default=_root / "checkpoints" / "dynamic_encoder_mlx.npz",
        help="Path to dynamic encoder MLX weights (.npz).",
    )
    parser.add_argument(
        "--projection_checkpoint",
        type=Path,
        default=None,
        help="Optional path to load projection weights from (e.g. previous Phase 2 run).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint .npz: restores model weights, optimizer state, step counter, and LR schedule position.",
    )
    parser.add_argument(
        "--balance_tasks",
        action="store_true",
        default=True,
        help="Use task-balanced sampling (default: True).",
    )
    parser.add_argument(
        "--no_balance_tasks",
        action="store_false",
        dest="balance_tasks",
        help="Disable task-balanced sampling.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Use gradient checkpointing on LLM forward (trades ~2x compute for much less memory). Default: on.",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
        help="Disable gradient checkpointing (needs more RAM).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    try:
        import numpy as np
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx_lm import load as mlx_lm_load
        from mlx_lm.utils import does_model_support_input_embeddings

        from bazi_model.data.phase2_dataset import (
            Phase2ChartDataset,
            phase2_collate,
            get_task_weights_for_sampler,
        )
        from bazi_model.encoders.dynamic_encoder_mlx import DynamicEncoderMLX
        from bazi_model.encoders.natal_encoder_mlx import NatalEncoderMLX
        from bazi_model.encoders.cross_interaction_encoder_mlx import CrossInteractionEncoderMLX
        from bazi_model.model.bazi_llm import BaziLLM
        from bazi_model.projection import ProjectionLayer
        from bazi_model.training.phase2_loss import compute_lm_loss
    except ImportError as e:
        print(
            "Error: MLX / bazi_model not available. Install Phase 2 deps: pip install -r requirements_phase2.txt",
            file=sys.stderr,
        )
        print(f"ImportError: {e}", file=sys.stderr)
        return 1

    # --- Data ---
    if not args.train_data.exists():
        print(f"Error: Train data not found: {args.train_data}", file=sys.stderr)
        return 1
    if not args.val_data.exists():
        print(f"Error: Val data not found: {args.val_data}", file=sys.stderr)
        return 1

    train_ds = Phase2ChartDataset(args.train_data)
    val_ds = Phase2ChartDataset(args.val_data)
    if len(train_ds) == 0:
        print("Error: Train dataset is empty.", file=sys.stderr)
        return 1

    # --- Model ---
    if not args.llm_path.is_dir():
        print(f"Error: LLM path not found: {args.llm_path}", file=sys.stderr)
        return 1
    if not args.natal_weights.exists():
        print(f"Error: Natal encoder weights not found: {args.natal_weights}", file=sys.stderr)
        return 1
    if not args.dynamic_weights.exists():
        print(f"Error: Dynamic encoder weights not found: {args.dynamic_weights}", file=sys.stderr)
        return 1

    print("Loading encoders and projection ...")
    natal_encoder = NatalEncoderMLX(d_emb=32, d_model=256, num_layers=3)
    natal_encoder.load_weights(str(args.natal_weights))
    dynamic_encoder = DynamicEncoderMLX(d_emb=32, d_model=256, num_layers=2)
    dynamic_encoder.load_weights(str(args.dynamic_weights))
    cross_encoder = CrossInteractionEncoderMLX(d_input=27, d_hidden=128, d_model=256)
    d_in = 768  # natal(256) + dynamic(256) + cross(256)
    projection = ProjectionLayer(d_in=d_in, d_llm=3584, n_virtual_tokens=args.n_virtual_tokens)
    if args.resume is not None:
        if not args.resume.exists():
            print(f"Error: Resume checkpoint not found: {args.resume}", file=sys.stderr)
            return 1
        _load_projection_from_checkpoint(projection, args.resume, cross_encoder)
    elif args.projection_checkpoint is not None and args.projection_checkpoint.exists():
        _load_projection_from_checkpoint(projection, args.projection_checkpoint, cross_encoder)

    print("Loading Qwen from", args.llm_path, "...")
    llm_model, tokenizer_wrapper = mlx_lm_load(str(args.llm_path))
    if not does_model_support_input_embeddings(llm_model):
        print("Error: This model does not support input_embeddings.", file=sys.stderr)
        return 1

    # mlx_lm returns a TokenizerWrapper; BaziLLM and phase2_collate need the
    # underlying HuggingFace tokenizer (callable with return_tensors="np").
    tokenizer = getattr(tokenizer_wrapper, "_tokenizer", tokenizer_wrapper)
    if not callable(tokenizer):
        print(
            f"Error: tokenizer ({type(tokenizer)}) is not callable; "
            "expected HuggingFace PreTrainedTokenizer.",
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
    if args.gradient_checkpointing:
        model.gradient_checkpointing = True
        print("Gradient checkpointing: ON (saves memory, ~2x slower forward/backward)")
    from mlx.utils import tree_flatten
    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"Trainable params: {n_trainable:,}  (projection + cross_interaction_encoder)")

    n_virtual = args.n_virtual_tokens
    accum_steps = args.accumulation_steps
    max_norm = args.max_grad_norm
    batch_size = args.batch_size

    # Steps: num_micro = micro-batches per epoch, steps_per_epoch = optimizer updates per epoch
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
        # Logits: (batch, n_virtual + L, vocab_size); labels/loss_mask are (batch, L)
        L = batch_mx["labels"].shape[1]
        logits_text = logits[:, n_virtual : n_virtual + L, :]
        return compute_lm_loss(logits_text, batch_mx["labels"], batch_mx["loss_mask"])

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Task-balanced epoch indices
    if args.balance_tasks:
        weights = get_task_weights_for_sampler(train_ds)
        w = np.array(weights, dtype=np.float64)
        w = w / w.sum()
    else:
        w = None

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
            print(f"Resumed: step={global_step}, best_val_loss={best_val_loss:.4f}, optimizer+LR restored")
        else:
            print(f"Resumed: step={global_step}, best_val_loss={best_val_loss:.4f} (optimizer state not found; LR restarts)", file=sys.stderr)

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
            if micro_idx < 3:
                import time as _t; _t0 = _t.time()
                print(f"  [dbg] micro={micro_idx} collating ...", flush=True)
            batch = phase2_collate(
                [train_ds[i] for i in indices],
                tokenizer,
                max_length=args.max_length,
            )
            if not batch:
                print("  Warning: collate returned empty batch; skipping.", file=sys.stderr)
                continue
            batch_mx = _batch_to_mx(batch, n_virtual)
            if micro_idx < 3:
                print(f"  [dbg] micro={micro_idx} collated in {_t.time()-_t0:.1f}s, running fwd+bwd ...", flush=True)

            loss, grads = loss_and_grad_fn(model, batch_mx)
            mx.eval(loss, grads)
            if micro_idx < 3:
                print(f"  [dbg] micro={micro_idx} done in {_t.time()-_t0:.1f}s  loss={float(loss):.4f}", flush=True)
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
                    print(f"  step {global_step}  train_loss={step_loss:.4f}  grad_norm={gnorm:.4f}  lr={lr_val:.2e}")

                # Validation
                if global_step % args.val_every_steps == 0:
                    val_loss = _run_validation(model, val_ds, tokenizer, args, n_virtual, loss_fn)
                    print(f"  step {global_step}  val_loss={val_loss:.4f}")
                    if args.sample_generations and val_ds:
                        _log_sample_generations(model, val_ds, tokenizer, args, n=2)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        _save_checkpoint(
                            model, optimizer, global_step, best_val_loss, args.checkpoint_dir, best=True
                        )
                    else:
                        no_improve_count += 1

                    if no_improve_count >= args.early_stopping_patience:
                        print(f"Early stopping: no val improvement for {args.early_stopping_patience} checks.")
                        break

                # Save latest periodically
                if global_step % args.save_every_steps == 0:
                    _save_checkpoint(model, optimizer, global_step, best_val_loss, args.checkpoint_dir, best=False)

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

        # End of epoch: validation
        val_loss = _run_validation(model, val_ds, tokenizer, args, n_virtual, loss_fn)
        avg_train = epoch_loss_sum / epoch_batches if epoch_batches else 0.0
        print(f"epoch {epoch + 1}/{epochs}  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            _save_checkpoint(model, optimizer, global_step, best_val_loss, args.checkpoint_dir, best=True)
        else:
            no_improve_count += 1
        if no_improve_count >= args.early_stopping_patience:
            print(f"Early stopping after epoch {epoch + 1}.")
            break

    print(f"Done. Best val_loss={best_val_loss:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
