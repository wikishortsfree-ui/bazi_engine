#!/usr/bin/env python3
"""
Train the interaction-detection head: NatalEncoder + InteractionHead.

Predicts 64-dim binary interaction vector from natal chart embedding.
Uses ChartDataset, train/val split, AdamW, and the task-agnostic Trainer.

Usage:
  python scripts/train_interaction.py --data data/charts.jsonl [options]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from debug_interaction_training import debug_interaction_model
import torch
from torch.utils.data import DataLoader, random_split

from bazi_model.data import ChartDataset, chart_collate_fn_inputs_targets
from bazi_model.encoders.natal_encoder import NatalEncoder
from bazi_model.tasks.interaction_detector import InteractionHead
from bazi_model.training import Trainer

_NATAL_INPUT_KEYS = (
    "stem_ids",
    "branch_ids",
    "stem_elements",
    "stem_polarities",
    "pillar_positions",
)


def _cast_inputs_to_float32(inputs: dict[str, torch.Tensor]) -> None:
    """Cast float tensors in inputs to float32 in place."""
    for k in list(inputs.keys()):
        if inputs[k].is_floating_point():
            inputs[k] = inputs[k].to(torch.float32)


def compute_pos_weight(dataset: ChartDataset) -> torch.Tensor:
    """(1-p)/p per bit from targets for BCEWithLogitsLoss pos_weight."""
    all_vectors = []
    for i in range(len(dataset)):
        _, targets = dataset[i]
        all_vectors.append(targets["interaction_vector"])
    all_vectors = torch.stack(all_vectors)  # (N, 64)
    positive_rate = all_vectors.mean(dim=0)  # (64,)
    return (1 - positive_rate) / (positive_rate + 1e-8)


def print_interaction_target_stats(dataset: ChartDataset) -> None:
    """Print positive rate and theoretical baseline loss for the 64-dim interaction target."""
    all_vectors = []
    for i in range(len(dataset)):
        _, targets = dataset[i]
        all_vectors.append(targets["interaction_vector"])
    all_vectors = torch.stack(all_vectors)  # (N, 64)
    global_pos = all_vectors.float().mean().item()
    # BCE with "predict all zeros": for target 0 loss=0, for target 1 loss=-log(eps). Mean = p * (-log(eps)) with eps~0 -> large. Use eps=1e-7.
    # Actually: always predict p=0 gives loss = (1-p)*0 + p*(-log(1e-7)) ≈ p*16.1. So mean loss ≈ 16.1 * global_pos.
    # "Random" (p=0.5): every bit gives 0.693, so mean loss = 0.693.
    # If model predicts sigmoid=0.3 on average: loss per 0 = -log(0.7)≈0.36, per 1 = -log(0.3)≈1.2; mean = (1-global_pos)*0.36 + global_pos*1.2.
    baseline_random = 0.693
    if global_pos > 0 and global_pos < 1:
        baseline_bias_zero = (1 - global_pos) * 0.36 + global_pos * 1.2  # approx for sigmoid~0.3
        print(
            f"[interaction] target positive rate: {global_pos:.4f} (sparse). "
            f"Theoretical loss: random init ~{baseline_random:.3f}; biased-to-zero init ~{baseline_bias_zero:.3f}"
        )
    else:
        print(f"[interaction] target positive rate: {global_pos:.4f}. Theoretical random loss ~{baseline_random:.3f}")


class InteractionModel(torch.nn.Module):
    """Wrapper: NatalEncoder + InteractionHead for the interaction task."""

    def __init__(self, encoder: NatalEncoder, head: InteractionHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, **input_kwargs) -> torch.Tensor:
        natal_emb = self.encoder(**input_kwargs)
        return self.head(natal_emb)


def make_step_fn(optimizer: torch.optim.Optimizer):
    """step_fn((inputs, targets), model, device). Encoder sees only inputs; loss uses targets."""

    def step_fn(
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        model: torch.nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        inputs, targets = batch
        _cast_inputs_to_float32(inputs)
        encoder_kw = {k: inputs[k] for k in _NATAL_INPUT_KEYS if k in inputs}
        natal_emb = model.encoder(**encoder_kw)
        loss, logits = model.head.forward_loss(targets, natal_emb)
        if model.training:
            model.zero_grad()
            loss.backward()
            optimizer.step()
            return loss, None
        pred = (logits.sigmoid() >= 0.5).float()
        target = targets["interaction_vector"]
        if target.dtype != pred.dtype:
            target = target.to(pred.dtype)
        binary_accuracy = (pred == target).float().mean().item()
        return loss, {"binary_accuracy": binary_accuracy}

    return step_fn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train interaction head (NatalEncoder + InteractionHead)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/charts.jsonl"),
        help="Path to charts.jsonl",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation (e.g. 0.1 for 80/10/10)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test fraction for held-out set (0 = no test split; use same as evaluate_phase1 for test metrics)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for saving best checkpoint",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=None,
        help="TensorBoard log dir (optional)",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and split
    full_ds = ChartDataset(args.data, task_name="interaction", temporal_only=False)

    # Check one sample
    inputs, targets = full_ds[0]

    print("=== INPUTS (should be MINIMAL) ===")
    for k, v in inputs.items():
        print(f"  {k}: shape={v.shape}")

    print("\n=== TARGETS (the answers) ===")
    for k, v in targets.items():
        print(f"  {k}: shape={v.shape}")

    # THE KEY CHECK:
    if "interaction_vector" in inputs:
        print("\n❌ ERROR: interaction_vector is in INPUTS! This is the answer!")
        print("The model is just copying the answer from input to output.")
        sys.exit(1)

    print("\n✅ No leakage detected (interaction_vector only in targets)")

    print_interaction_target_stats(full_ds)

    n = len(full_ds)
    n_val = max(1, int(n * args.val_ratio))
    if args.test_ratio > 0:
        n_test = int(n * args.test_ratio)
        n_train = n - n_val - n_test
        if n_train < 1 or n_test < 1:
            raise ValueError("val_ratio + test_ratio too large for dataset size")
        train_ds, val_ds, _ = random_split(
            full_ds,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        n_train = n - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=chart_collate_fn_inputs_targets,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=chart_collate_fn_inputs_targets,
        num_workers=0,
    )

    # Model: NatalEncoder + InteractionHead (pos_weight from dataset for rare bits)
    encoder = NatalEncoder()
    # pos_weight = compute_pos_weight(full_ds)
    head = InteractionHead(d_in=256, d_out=64, pos_weight=None)
    model = InteractionModel(encoder=encoder, head=head)
    # Debug before training
    # debug_interaction_model(model, train_loader, device)
    # sys.exit(0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    warmup_epochs = max(1, args.epochs // 20)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )
    step_fn = make_step_fn(optimizer)

    # Log true initial loss (first batch, no steps). Epoch 1 train_loss is an average over the whole epoch (many steps).
    model.to(device)
    model.eval()
    with torch.no_grad():
        init_batch = next(iter(train_loader))
        init_inputs = {k: v.to(device) for k, v in init_batch[0].items()}
        init_targets = {k: v.to(device) for k, v in init_batch[1].items()}
        _cast_inputs_to_float32(init_inputs)
        enc_kw = {k: init_inputs[k] for k in _NATAL_INPUT_KEYS if k in init_inputs}
        emb = encoder(**enc_kw)
        init_loss, _ = head.forward_loss(init_targets, emb)
    print(f"[interaction] True initial loss (first batch, no training): {init_loss.item():.6f}")
    model.train()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        step_fn=step_fn,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping_patience=15,
        log_dir=str(args.log_dir) if args.log_dir else None,
        checkpoint_dir=str(args.checkpoint_dir) if args.checkpoint_dir else None,
    )

    best_val_loss = trainer.run(num_epochs=args.epochs)
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Load best model from Trainer's checkpoint so task-specific file has best weights, not final
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_pt = args.checkpoint_dir / "best.pt"
    if best_pt.exists():
        ckpt = torch.load(best_pt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    ckpt_path = args.checkpoint_dir / "interaction_best.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        ckpt_path,
    )
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
