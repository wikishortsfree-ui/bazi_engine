#!/usr/bin/env python3
"""
Evaluate an element-strength checkpoint (encoder + element head from train_element_strength.py).

Reports MSE, MAE, ordinal accuracy, and prints a few example rows comparing ground truth
to model predictions (element strengths 木/火/土/金/水).

Usage:
  python scripts/evaluate_element_strength.py --checkpoint checkpoints/element_strength_best.pt --data data/charts.jsonl
  python scripts/evaluate_element_strength.py --checkpoint checkpoints/element_strength_best.pt --data data/charts.jsonl --n_examples 5 --output reports/element_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch
from torch.utils.data import DataLoader

from bazi_model.data import ChartDataset, chart_collate_fn_inputs_targets
from bazi_model.encoders.natal_encoder import NatalEncoder
from bazi_model.tasks.element_regressor import ElementStrengthHead

ELEMENT_NAMES = ("木", "火", "土", "金", "水")  # wood, fire, earth, metal, water

_NATAL_INPUT_KEYS = (
    "stem_ids",
    "branch_ids",
    "stem_elements",
    "stem_polarities",
    "pillar_positions",
)


def _cast_inputs_f32(inputs: dict[str, torch.Tensor]) -> None:
    for k in list(inputs.keys()):
        if inputs[k].is_floating_point():
            inputs[k] = inputs[k].float()


def _ordinal_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Fraction of samples where argsort(pred) == argsort(true). pred/true (N, 5)."""
    if pred.size == 0:
        return 0.0
    pred_rank = np.argsort(np.argsort(pred, axis=1), axis=1)
    true_rank = np.argsort(np.argsort(true, axis=1), axis=1)
    return float(np.all(pred_rank == true_rank, axis=1).mean())


def _ordinal_accuracy_relaxed(
    pred: np.ndarray, true: np.ndarray, margin_min: float = 0.02
) -> float:
    """Ordinal correct when only considering pairs with true strength diff >= margin_min."""
    if pred.size == 0:
        return 0.0
    n, k = pred.shape
    correct = np.ones(n, dtype=bool)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            margin = true[:, i] - true[:, j]
            clear = margin >= margin_min
            wrong_order = (pred[:, j] >= pred[:, i]) & clear
            correct &= ~wrong_order
    return float(correct.mean())


def _rank_string(values: np.ndarray) -> str:
    """Return element names in order of strength (strongest first). values shape (5,)."""
    order = np.argsort(values)[::-1]
    return " > ".join(ELEMENT_NAMES[i] for i in order)


class _ElementModel(torch.nn.Module):
    """Thin wrapper so we can load state_dict from ElementStrengthModel."""

    def __init__(self, encoder: NatalEncoder, head: ElementStrengthHead):
        super().__init__()
        self.encoder = encoder
        self.head = head


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate element-strength checkpoint: metrics + example comparisons."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/element_strength_best.pt"),
        help="Path to element_strength_best.pt",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/charts.jsonl"),
        help="Path to charts.jsonl",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=10,
        help="Number of example rows to print (ground truth vs predicted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional: write metrics JSON here",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: evaluate only on first N samples (for quick runs)",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.data.exists():
        print(f"Data not found: {args.data}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = NatalEncoder()
    head = ElementStrengthHead(d_natal=256, n_elements=5)
    model = _ElementModel(encoder=encoder, head=head)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    dataset = ChartDataset(args.data, task_name="element_strength", temporal_only=False)
    if args.limit is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=chart_collate_fn_inputs_targets,
        num_workers=0,
    )

    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, targets in loader:
            _cast_inputs_f32(inputs)
            encoder_kw = {k: inputs[k].to(device) for k in _NATAL_INPUT_KEYS if k in inputs}
            probs = model.head(model.encoder(**encoder_kw))
            true = targets["elemental_strengths"]
            all_pred.append(probs.cpu().numpy())
            all_true.append(true.numpy())

    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)
    n = pred.shape[0]

    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    ordinal_acc = _ordinal_accuracy(pred, true)
    ordinal_relaxed = _ordinal_accuracy_relaxed(pred, true, margin_min=0.02)

    metrics = {
        "n_samples": n,
        "mse": mse,
        "mae": mae,
        "ordinal_accuracy": ordinal_acc,
        "ordinal_accuracy_relaxed": ordinal_relaxed,
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
    }

    # ----- Print report -----
    print("=== Element strength evaluation ===\n")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data:       {args.data}")
    print(f"N samples:  {n}\n")
    print("--- Metrics ---")
    print(f"  MSE:                    {mse:.6f}")
    print(f"  MAE:                    {mae:.4f}")
    print(f"  Ordinal accuracy:       {ordinal_acc:.2%}")
    print(f"  Ordinal (relaxed 0.02): {ordinal_relaxed:.2%}  (ignores pairs with true diff < 0.02)\n")

    # ----- Example rows: ground truth vs predicted -----
    n_show = min(args.n_examples, n)
    print(f"--- Example rows (ground truth vs predicted), first {n_show} samples ---\n")
    print("Element order: 木 火 土 金 水 (wood, fire, earth, metal, water)\n")

    for i in range(n_show):
        t = true[i]
        p = pred[i]
        pred_rank = np.argsort(p)[::-1]
        true_rank = np.argsort(t)[::-1]
        ok = np.array_equal(np.argsort(p), np.argsort(t))
        status = "✓ ordinal OK" if ok else "✗ ordinal wrong"

        print(f"  Sample {i + 1}  {status}")
        print(f"    Ground truth:  " + "  ".join(f"{ELEMENT_NAMES[j]}: {t[j]:.3f}" for j in range(5)))
        print(f"    Predicted:     " + "  ".join(f"{ELEMENT_NAMES[j]}: {p[j]:.3f}" for j in range(5)))
        print(f"    True rank:     {_rank_string(t)}")
        print(f"    Pred rank:     {_rank_string(p)}")
        print()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2))
        print(f"Metrics written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
