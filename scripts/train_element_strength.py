"""
Train element strength head: NatalEncoder + ElementStrengthHead.

Target: elemental_strengths (batch, 5), MSE + optional ordinal loss. Full dataset (no temporal_only).

Loss often decreases slowly or plateaus early if (1) LR is too low, (2) there is no scheduler,
or (3) the encoder gets weak gradients. This script uses a higher default LR (1e-3), optional
encoder-vs-head LR ratio, and CosineAnnealingLR so loss can keep improving.

Usage:
  python scripts/train_element_strength.py --data data/charts.jsonl [options]
  python scripts/train_element_strength.py --data data/charts.jsonl --epochs 20 --lr 0.001 --checkpoint_dir checkpoints
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
from bazi_model.encoders import NatalEncoder
from bazi_model.tasks import ElementStrengthHead
from bazi_model.training import Trainer

_NATAL_INPUT_KEYS = (
    "stem_ids",
    "branch_ids",
    "stem_elements",
    "stem_polarities",
    "pillar_positions",
)


def _cast_inputs_to_f32(inputs: dict[str, torch.Tensor]) -> None:
    """Cast float tensors in inputs to float32 in place."""
    for k in list(inputs.keys()):
        if inputs[k].dtype in (torch.float64, torch.float32):
            inputs[k] = inputs[k].float()


class ElementStrengthModel(torch.nn.Module):
    """Wrapper: NatalEncoder + ElementStrengthHead for single checkpoint save."""

    def __init__(self, encoder: NatalEncoder, head: ElementStrengthHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, **input_kwargs) -> torch.Tensor:
        natal_emb = self.encoder(**input_kwargs)
        return self.head(natal_emb)


def make_step_fn(optimizer: torch.optim.Optimizer):
    """Build step_fn((inputs, targets), model, device). Encoder sees only inputs; loss uses targets."""

    def step_fn(
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        model: torch.nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        inputs, targets = batch
        _cast_inputs_to_f32(inputs)
        encoder_kw = {k: inputs[k] for k in _NATAL_INPUT_KEYS if k in inputs}
        natal_emb = model.encoder(**encoder_kw)
        elemental_strengths = targets["elemental_strengths"]
        loss, probs = model.head.forward_loss(natal_emb, elemental_strengths)
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss, None
        mae = (probs - elemental_strengths).abs().mean().item()
        # Ordinal accuracy: fraction of samples where argsort(pred) == argsort(true)
        pred_np = probs.detach().cpu().float().numpy()
        true_np = elemental_strengths.cpu().float().numpy()
        pred_rank = pred_np.argsort(axis=1).argsort(axis=1)
        true_rank = true_np.argsort(axis=1).argsort(axis=1)
        ordinal_ok = (pred_rank == true_rank).all(axis=1).mean()
        return loss, {"mae": mae, "ordinal_accuracy": float(ordinal_ok)}

    return step_fn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train element strength head (NatalEncoder + ElementStrengthHead)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/charts.jsonl"),
        help="Path to charts.jsonl",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for head (and encoder if --encoder_lr not set). Use 1e-3 if loss plateaus at ~0.09.",
    )
    parser.add_argument(
        "--encoder_lr_ratio",
        type=float,
        default=2.0,
        help="Encoder LR = lr * this (default 2.0). Higher ratio gives encoder stronger signal when loss is flat.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help="Directory to save best checkpoint (e.g. checkpoints/)",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=None,
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (e.g. 0.1 for 80/10/10)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test fraction for held-out set (0 = no test split; use same as evaluate_phase1 for test metrics)",
    )
    parser.add_argument(
        "--ordinal_loss_weight",
        type=float,
        default=0.15,
        help="Element head: weight for pairwise ranking loss (default 0.15).",
    )
    parser.add_argument(
        "--ordinal_margin_min",
        type=float,
        default=0.0,
        help="Only penalize pairs with true strength diff >= this (default 0). Use 0.02 to skip borderline pairs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = ChartDataset(args.data, task_name="element_strength", temporal_only=False)
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
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
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

    encoder = NatalEncoder()
    head = ElementStrengthHead(
        d_natal=256,
        n_elements=5,
        ordinal_loss_weight=args.ordinal_loss_weight,
        ordinal_margin_min=args.ordinal_margin_min,
    )
    model = ElementStrengthModel(encoder=encoder, head=head)

    encoder_lr = args.lr * args.encoder_lr_ratio
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": encoder_lr},
            {"params": head.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    step_fn = make_step_fn(optimizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = str(args.checkpoint_dir) if args.checkpoint_dir else None
    log_dir = str(args.log_dir) if args.log_dir else None

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        step_fn=step_fn,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping_patience=5,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
    )

    best_val_loss = trainer.run(num_epochs=args.epochs)

    if args.checkpoint_dir:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_pt = args.checkpoint_dir / "best.pt"
        if best_pt.exists():
            ckpt = torch.load(best_pt, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
        ckpt_path = args.checkpoint_dir / "element_strength_best.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved model state to {ckpt_path}")


if __name__ == "__main__":
    main()
