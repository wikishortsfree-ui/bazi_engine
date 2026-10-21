#!/usr/bin/env python3
"""
Train cross-temporal head: NatalEncoder + DynamicEncoder + CrossTemporalHead.

Uses ChartDataset(..., temporal_only=True). Step: natal_emb = natal_encoder(batch);
dynamic_emb = dynamic_encoder(batch); loss, logits = head.forward_loss(batch, natal_emb, dynamic_emb).

Usage:
  python scripts/train_cross_temporal.py --data data/charts.jsonl [options]
  python -m scripts.train_cross_temporal --data data/charts.jsonl [options]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
from torch.utils.data import DataLoader, random_split

from bazi_model.data import ChartDataset, chart_collate_fn_inputs_targets
from bazi_model.encoders.dynamic_encoder import DynamicEncoder
from bazi_model.encoders.natal_encoder import NatalEncoder
from bazi_model.tasks.cross_temporal_detector import (
    CROSS_TEMPORAL_SEGMENT_RANGES,
    CrossTemporalHead,
    BINARY_INDICES,
    CONTINUOUS_INDICES,
)
from bazi_model.training import Trainer

N_BINARY_DIMS = len(BINARY_INDICES)

_NATAL_INPUT_KEYS = (
    "stem_ids",
    "branch_ids",
    "stem_elements",
    "stem_polarities",
    "pillar_positions",
)
_DYNAMIC_INPUT_KEYS = (
    "dayun_stem_id",
    "dayun_branch_id",
    "liunian_stem_id",
    "liunian_branch_id",
    "liuyue_stem_id",
    "liuyue_branch_id",
    "pillar_presence_mask",
)


def _cast_inputs_to_float32(inputs: dict[str, torch.Tensor]) -> None:
    """Cast float tensors in inputs to float32 in place."""
    for k in list(inputs.keys()):
        if inputs[k].is_floating_point():
            inputs[k] = inputs[k].to(torch.float32)


def _compute_pos_weight_from_loader(
    loader: DataLoader,
    binary_indices: list[int],
    min_positive_rate: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute per-dimension pos_weight = (1 - p) / p from training set positive rates."""
    n_dims = len(binary_indices)
    counts = torch.zeros(n_dims, dtype=torch.float64, device=device)
    total = 0
    for _, targets in loader:
        vec = targets["dynamic_interaction_vector"].to(device)
        # Extract binary dims: vec has shape (batch, 66); we need columns at binary_indices.
        bin_vec = vec[:, binary_indices]
        counts += bin_vec.sum(dim=0).double()
        total += bin_vec.shape[0]
    if total == 0:
        return torch.ones(n_dims, dtype=torch.float32)
    p = (counts / total).clamp(min=min_positive_rate, max=1.0 - 1e-6)
    pos_weight = (1.0 - p) / p
    return pos_weight.float()


class CrossTemporalModel(torch.nn.Module):
    """Wrapper: NatalEncoder + DynamicEncoder + CrossTemporalHead. Encoders get inputs only."""

    def __init__(
        self,
        natal_encoder: NatalEncoder,
        dynamic_encoder: DynamicEncoder,
        head: CrossTemporalHead,
    ):
        super().__init__()
        self.natal_encoder = natal_encoder
        self.dynamic_encoder = dynamic_encoder
        self.head = head

    def forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (natal_emb, dynamic_emb) from inputs only (no ground truth)."""
        natal_kw = {k: inputs[k] for k in _NATAL_INPUT_KEYS if k in inputs}
        dynamic_kw = {k: inputs[k] for k in _DYNAMIC_INPUT_KEYS if k in inputs}
        natal_emb = self.natal_encoder(**natal_kw)
        dynamic_emb = self.dynamic_encoder(**dynamic_kw)
        return natal_emb, dynamic_emb


def make_step_fn(optimizer: torch.optim.Optimizer):
    """step_fn((inputs, targets), model, device). Encoders see only inputs; loss uses targets."""

    def step_fn(
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        model: torch.nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        inputs, targets = batch
        _cast_inputs_to_float32(inputs)
        wrapper = model
        natal_emb, dynamic_emb = wrapper(inputs)
        loss, logits = wrapper.head.forward_loss(targets, natal_emb, dynamic_emb)
        if model.training:
            model.zero_grad()
            loss.backward()
            optimizer.step()
            return loss, None
        target = targets["dynamic_interaction_vector"]
        if target.dtype != logits.dtype:
            target = target.to(logits.dtype)
        bi = wrapper.head._binary_idx
        pred_bin = (logits[:, bi].sigmoid() >= 0.5).float()
        binary_accuracy = (pred_bin == target[:, bi]).float().mean().item()
        ci = wrapper.head._continuous_idx
        delta_mse = (logits[:, ci] - target[:, ci]).pow(2).mean().item()
        return loss, {"binary_accuracy": binary_accuracy, "delta_mse": delta_mse}

    return step_fn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train cross-temporal head (NatalEncoder + DynamicEncoder + CrossTemporalHead)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/charts.jsonl"),
        help="Path to charts.jsonl",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
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
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument(
        "--segment_weights",
        type=str,
        default=None,
        help="Optional comma-separated list of 7 floats to upweight BCE per segment "
        "(natal×大运, natal×流年, natal×流月, 大运×流年, 大运×流月, 流年×流月, useful_god_active). "
        'E.g. "2,1,1,1,1,1,4" to upweight natal×大运 and useful_god_active.',
    )
    parser.add_argument(
        "--pos_weight_from_data",
        action="store_true",
        help="Compute pos_weight for the 61 binary dims from training set positive rates (1-p)/p; cap p at 0.01.",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ChartDataset(args.data, task_name="cross_temporal", temporal_only=True)
    n = len(dataset)
    if n == 0:
        raise ValueError("No samples with temporal context in dataset (temporal_only=True).")

    n_val = max(1, int(n * args.val_ratio))
    if args.test_ratio > 0:
        n_test = int(n * args.test_ratio)
        n_train = n - n_val - n_test
        if n_train < 1 or n_test < 1:
            raise ValueError("val_ratio + test_ratio too large for dataset size")
        train_ds, val_ds, _ = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        n_train = n - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=chart_collate_fn_inputs_targets,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=chart_collate_fn_inputs_targets,
        num_workers=args.num_workers,
    )

    segment_weights_tensor = None
    if args.segment_weights is not None:
        parts = [float(x.strip()) for x in args.segment_weights.split(",")]
        if len(parts) != len(CROSS_TEMPORAL_SEGMENT_RANGES):
            raise ValueError(
                f"--segment_weights must have {len(CROSS_TEMPORAL_SEGMENT_RANGES)} comma-separated floats, got {len(parts)}"
            )
        segment_weights_tensor = torch.tensor(parts, dtype=torch.float32)

    pos_weight_tensor = None
    if args.pos_weight_from_data:
        pos_weight_tensor = _compute_pos_weight_from_loader(
            train_loader, BINARY_INDICES, min_positive_rate=0.01, device=device
        )

    natal_encoder = NatalEncoder()
    dynamic_encoder = DynamicEncoder()
    head = CrossTemporalHead(
        segment_weights=segment_weights_tensor,
        pos_weight=pos_weight_tensor,
    )
    model = CrossTemporalModel(natal_encoder, dynamic_encoder, head)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
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

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

    best_pt = args.checkpoint_dir / "best.pt"
    if best_pt.exists():
        ckpt = torch.load(best_pt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    ckpt_path = args.checkpoint_dir / "cross_temporal_best.pt"
    torch.save(
        {
            "natal_encoder": natal_encoder.state_dict(),
            "dynamic_encoder": dynamic_encoder.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        ckpt_path,
    )
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
