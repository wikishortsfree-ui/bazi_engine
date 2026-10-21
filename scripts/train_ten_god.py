#!/usr/bin/env python3
"""
Train ten-god classification head: NatalEncoder + TenGodHead on charts.jsonl.

Uses train/val split, AdamW, optional scheduler, and the task-agnostic Trainer.
Saves the best model state to checkpoint_dir (final state after run; Trainer may
save during run in a future version).

Usage:
  python scripts/train_ten_god.py --dataset data/charts.jsonl --epochs 10
  python scripts/train_ten_god.py --dataset data/charts.jsonl --batch-size 32 --lr 1e-3 --checkpoint-dir checkpoints --log-dir runs/ten_god
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

# Add project root for imports when run as script
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from bazi_model.data import ChartDataset, chart_collate_fn_inputs_targets
from bazi_model.encoders.natal_encoder import NatalEncoder
from bazi_model.tasks.ten_god_head import NUM_CLASSES, TenGodHead
from bazi_model.training import Trainer

# NatalEncoder accepts only minimal inputs (no ground truth).
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
            inputs[k] = inputs[k].float()


def compute_class_weights_ten_god(train_dataset: torch.utils.data.Dataset, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    Sqrt-inverse-frequency class weights from the ten-god labels in the train set.

    Each chart has 4 positions; we count all label occurrences, then compute
    weight[c] = 1 / sqrt(freq[c]), normalized by mean (same rationale as useful_god:
    avoids saddle at uniform predictions while up-weighting rarer classes).
    """
    counts = torch.zeros(num_classes)
    for i in range(len(train_dataset)):
        _, targets = train_dataset[i]
        ids = targets["ten_god_ids"]  # (4,)
        for j in range(ids.numel()):
            c = ids.view(-1)[j].item()
            if 0 <= c < num_classes:
                counts[c] += 1
    counts = counts.clamp(min=1)
    freq = counts / counts.sum()
    weights = 1.0 / freq.sqrt()
    weights = weights / weights.mean()
    return weights


class TenGodModel(torch.nn.Module):
    """Wrapper combining NatalEncoder and TenGodHead for the ten-god task."""

    def __init__(self, encoder: NatalEncoder, head: TenGodHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, **input_kwargs) -> torch.Tensor:
        """Forward through encoder then head; returns logits. For loss use head.forward_loss."""
        natal_emb = self.encoder(**input_kwargs)
        return self.head(natal_emb)


def _make_step_fn(optimizer: torch.optim.Optimizer):
    """Build step_fn((inputs, targets), model, device). Encoder sees only inputs; loss uses targets."""

    def step_fn(
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        model: torch.nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        inputs, targets = batch
        _cast_inputs_to_float32(inputs)
        encoder_kwargs = {k: inputs[k] for k in _NATAL_INPUT_KEYS if k in inputs}
        natal_emb = model.encoder(**encoder_kwargs)
        ten_god_ids = targets["ten_god_ids"]
        loss, logits = model.head.forward_loss(natal_emb, ten_god_ids)
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.detach(), None
        pred = logits.view(-1, logits.size(-1)).argmax(dim=-1)
        target = ten_god_ids.reshape(-1)
        accuracy = (pred == target).float().mean().item()
        return loss.detach(), {"accuracy": accuracy}

    return step_fn


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ten-god head (NatalEncoder + TenGodHead)")
    p.add_argument("--dataset", type=str, default="data/charts.jsonl", help="Path to charts.jsonl")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoint")
    p.add_argument("--log-dir", type=str, default="runs/ten_god", help="TensorBoard log directory")
    p.add_argument("--split", type=float, default=0.9, help="Train ratio when test_ratio=0; else train = 1 - val_ratio - test_ratio")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Validation fraction (e.g. 0.1 for 80/10/10)")
    p.add_argument("--test_ratio", type=float, default=0.1, help="Test fraction for held-out set (e.g. 0.1 for 80/10/10); 0 = no test split")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--scheduler", type=str, default="none", choices=("none", "cosine", "plateau"), help="LR scheduler")
    p.add_argument("--early-stopping-patience", type=int, default=5, help="Early stopping patience (epochs)")
    p.add_argument("--no-class-weights", action="store_true", help="Disable sqrt-inverse class weights for CE/focal")
    p.add_argument("--no-focal-loss", action="store_false", dest="focal_loss", default=True, help="Disable Focal loss (default: Focal loss on with gamma=2)")
    p.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma when --no-focal-loss not set (default: 2.0)")
    p.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing for CE/focal (default: 0.1)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    dataset = ChartDataset(str(dataset_path), task_name="ten_god", temporal_only=False)
    n_total = len(dataset)
    if args.test_ratio > 0:
        n_test = int(n_total * args.test_ratio)
        n_val = max(1, int(n_total * args.val_ratio))
        n_train = n_total - n_val - n_test
        if n_train < 1 or n_test < 1:
            print("Dataset too small for train/val/test split.", file=sys.stderr)
            return 1
        train_ds, val_ds, _ = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(args.seed)
        )
    else:
        n_train = int(n_total * args.split)
        n_val = n_total - n_train
        if n_train == 0 or n_val == 0:
            print("Dataset too small for train/val split.", file=sys.stderr)
            return 1
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
        )

    # Sanity check: inputs and targets must not overlap (no data leakage)
    inputs, targets = dataset[0]
    assert set(inputs.keys()).isdisjoint(set(targets.keys())), "Input and target keys must not overlap"
    print("=== INPUT FEATURES (no ground truth) ===")
    for k, v in inputs.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    print("=== GROUND TRUTH TARGETS ===")
    for k, v in targets.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    print("  No overlap between inputs and targets")

    class_weight = None
    if not args.no_class_weights:
        class_weight = compute_class_weights_ten_god(train_ds, num_classes=NUM_CLASSES)
        print(f"Ten-god class weights (sqrt-inverse): {class_weight.tolist()}")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=chart_collate_fn_inputs_targets,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=chart_collate_fn_inputs_targets,
        num_workers=0,
    )

    encoder = NatalEncoder()
    head = TenGodHead(
        d_model=encoder.d_model,
        class_weight=class_weight,
        label_smoothing=args.label_smoothing,
        use_focal=args.focal_loss,
        focal_gamma=args.focal_gamma,
    )
    model = TenGodModel(encoder=encoder, head=head)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

    step_fn = _make_step_fn(optimizer)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        step_fn=step_fn,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping_patience=args.early_stopping_patience,
        log_dir=args.log_dir or None,
        checkpoint_dir=str(checkpoint_dir) or None,
    )

    best_val_loss = trainer.run(num_epochs=args.epochs)
    print(f"Best validation loss: {best_val_loss:.6f}")

    best_pt = checkpoint_dir / "best.pt"
    if best_pt.exists():
        ckpt = torch.load(best_pt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    ckpt_path = checkpoint_dir / "ten_god_best.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
