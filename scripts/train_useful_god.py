#!/usr/bin/env python3
"""
Train Useful God (用神) classifier: NatalEncoder + UsefulGodHead.

Usage:
    python scripts/train_useful_god.py --data data/charts.jsonl [options]

Step function: natal_emb = encoder(batch); loss, logits = head.forward_loss(natal_emb, batch["useful_god_id"]).
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
from torch.utils.data import DataLoader, random_split, Subset

from bazi_model.data import ChartDataset, chart_collate_fn_inputs_targets
from bazi_model.encoders.natal_encoder import NatalEncoder
from bazi_model.tasks.useful_god_classifier import UsefulGodHead
from bazi_model.training import Trainer

_NATAL_INPUT_KEYS = (
    "stem_ids",
    "branch_ids",
    "stem_elements",
    "stem_polarities",
    "pillar_positions",
)


def _cast_inputs_float32(inputs: dict[str, torch.Tensor]) -> None:
    """In-place cast float tensors in inputs to float32."""
    for k in list(inputs.keys()):
        if inputs[k].is_floating_point():
            inputs[k] = inputs[k].to(torch.float32)


def compute_class_weights(dataset, n_classes: int = 6) -> torch.Tensor:
    """Sqrt-inverse-frequency class weights for balanced CE loss.

    Plain inverse-frequency weights make P(c)*w(c) identical for all classes,
    which creates a saddle point at uniform predictions (expected gradient = 0).
    Sqrt-inverse-frequency breaks this symmetry while still up-weighting rare
    classes, giving the optimizer a non-trivial gradient to escape the uniform
    solution.
    """
    counts = torch.zeros(n_classes)
    for i in range(len(dataset)):
        _, targets = dataset[i]
        label = targets["useful_god_id"].item()
        if label == -1:
            label = 5
        counts[label] += 1
    counts = counts.clamp(min=1)
    freq = counts / counts.sum()
    weights = 1.0 / freq.sqrt()
    weights = weights / weights.mean()
    return weights


class UsefulGodModel(torch.nn.Module):
    """Wrapper: NatalEncoder + UsefulGodHead for training and checkpointing."""

    def __init__(self, encoder: NatalEncoder, head: UsefulGodHead):
        super().__init__()
        self.natal_encoder = encoder
        self.head = head

    def forward(self, **input_kwargs) -> torch.Tensor:
        natal_emb = self.natal_encoder(**input_kwargs)
        return self.head(natal_emb)


def make_step_fn(optimizer: torch.optim.Optimizer):
    """step_fn((inputs, targets), model, device). Encoder sees only inputs; loss uses targets."""

    def step_fn(
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        model: UsefulGodModel,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        inputs, targets = batch
        _cast_inputs_float32(inputs)
        encoder_kw = {k: inputs[k] for k in _NATAL_INPUT_KEYS if k in inputs}
        natal_emb = model.natal_encoder(**encoder_kw)
        useful_god_id = targets["useful_god_id"]  # (batch, 1), 0-4 or 5 for ambiguous
        loss, logits = model.head.forward_loss(natal_emb, useful_god_id)
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss, None
        target = useful_god_id.squeeze(-1)
        pred = logits.argmax(dim=-1)
        accuracy = (pred == target).float().mean().item()
        return loss, {"accuracy": accuracy}

    return step_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Useful God classifier (NatalEncoder + UsefulGodHead).")
    parser.add_argument("--data", type=str, default="data/charts.jsonl", help="Path to charts.jsonl")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio (e.g. 0.1 for 80/10/10)")
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test fraction for held-out set (0 = no test split; use same as evaluate_phase1 for test metrics)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard log directory (optional)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument(
        "--encoder_checkpoint",
        type=str,
        default=None,
        help="Path to element_strength checkpoint (.pt) to load encoder from; train only UsefulGodHead (optionally freeze encoder with --freeze_encoder).",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder when using --encoder_checkpoint; only train UsefulGodHead.",
    )
    parser.add_argument(
        "--unambiguous_only",
        action="store_true",
        help="Train only on samples with unambiguous useful_god (target in 0–4); exclude ambiguous (target -1/5).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    path = Path(args.data)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataset = ChartDataset(path, task_name="useful_god", temporal_only=False)
    if args.unambiguous_only:
        unambiguous_indices = [
            i
            for i in range(len(dataset))
            if dataset[i][1]["useful_god_id"].item() in (0, 1, 2, 3, 4)
        ]
        dataset = Subset(dataset, unambiguous_indices)
        print(f"unambiguous_only: training on {len(dataset)} samples (excluded ambiguous).")
    n = len(dataset)
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
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=chart_collate_fn_inputs_targets,
        num_workers=0,
    )

    class_weight = compute_class_weights(dataset, n_classes=6)
    print(f"Class weights: {class_weight.tolist()}")

    encoder = NatalEncoder()
    head = UsefulGodHead(
        input_dim=256,
        n_classes=6,
        class_weight=class_weight,
        focal_gamma=2.0,
    )
    model = UsefulGodModel(encoder, head)

    if args.encoder_checkpoint:
        ckpt_path = Path(args.encoder_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Encoder checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Element-strength checkpoint: state_dict keys are "encoder.*" and "head.*"
        encoder_state = {k[8:]: v for k, v in ckpt.items() if k.startswith("encoder.")}
        if not encoder_state:
            raise ValueError(
                f"No 'encoder.*' keys in {ckpt_path}; is this an element_strength checkpoint?"
            )
        natal_encoder_state = {f"natal_encoder.{k}": v for k, v in encoder_state.items()}
        model.load_state_dict(natal_encoder_state, strict=False)
        print(f"Loaded encoder from {ckpt_path} ({len(natal_encoder_state)} keys).")
        if args.freeze_encoder:
            for p in model.natal_encoder.parameters():
                p.requires_grad = False
            print("Encoder frozen; training only UsefulGodHead.")

    params_for_optimizer = (
        model.head.parameters() if (args.encoder_checkpoint and args.freeze_encoder) else model.parameters()
    )
    optimizer = torch.optim.AdamW(
        params_for_optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    step_fn = make_step_fn(optimizer)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        step_fn=step_fn,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping_patience=args.patience,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    best_val = trainer.run(num_epochs=args.epochs)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_pt = checkpoint_dir / "best.pt"
    if best_pt.exists():
        ckpt = torch.load(best_pt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    ckpt_path = checkpoint_dir / "useful_god_best.pt"
    torch.save({"model_state_dict": model.state_dict(), "best_val_loss": best_val}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path} (best_val_loss={best_val:.6f})")


if __name__ == "__main__":
    main()
