#!/usr/bin/env python3
"""
Analyze element-strength ordinal errors: find samples where argsort(pred) != argsort(true)
and classify as borderline (true strengths very close) vs clear errors.

Helps determine if the 90% ordinal target is achievable or capped by label noise.
Usage:
  python scripts/analyze_ordinal_errors.py --data data/charts.jsonl
  python scripts/analyze_ordinal_errors.py --data data/charts.jsonl --checkpoint checkpoints/multi_task_best.pt --output reports/ordinal_error_analysis.json
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
from torch.utils.data import DataLoader, random_split

from bazi_model.data import ChartDataset, chart_collate_fn_inputs_targets
from bazi_model.encoders.natal_encoder import NatalEncoder
from bazi_model.tasks.element_regressor import ElementStrengthHead

_NATAL_INPUT_KEYS = (
    "stem_ids",
    "branch_ids",
    "stem_elements",
    "stem_polarities",
    "pillar_positions",
)


def _cast_inputs_float32(inputs: dict) -> None:
    for k in list(inputs.keys()):
        if inputs[k].is_floating_point():
            inputs[k] = inputs[k].float()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze ordinal errors for element strength (borderline vs clear)."
    )
    parser.add_argument("--data", type=Path, default=Path("data/charts.jsonl"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to multi-task .pt (default: checkpoints/multi_task_best.pt)",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--borderline_threshold",
        type=float,
        default=0.02,
        help="True strength diff below this counts as borderline (default 0.02)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/ordinal_error_analysis.json"),
        help="Write JSON report here",
    )
    args = parser.parse_args()

    ckpt_path = args.checkpoint or (Path("checkpoints") / "multi_task_best.pt")
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1
    if not args.data.exists():
        print(f"Data not found: {args.data}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    encoder = NatalEncoder().to(device)
    encoder.load_state_dict(ckpt["encoder"], strict=True)
    head = ElementStrengthHead(d_natal=256, n_elements=5).to(device)
    head.load_state_dict(ckpt["element_head"], strict=True)
    encoder.eval()
    head.eval()

    dataset = ChartDataset(args.data, task_name="multi", temporal_only=False)
    n = len(dataset)
    n_val = max(1, int(n * args.val_ratio))
    n_test = max(1, int(n * args.test_ratio))
    n_train = n - n_val - n_test
    if n_train < 1 or n_test < 1:
        print("Invalid split: need test_ratio > 0 and enough data.", file=sys.stderr)
        return 1
    _, _, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=chart_collate_fn_inputs_targets,
        num_workers=0,
    )

    all_pred, all_true = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            _cast_inputs_float32(inputs)
            encoder_kw = {k: inputs[k].to(device) for k in _NATAL_INPUT_KEYS if k in inputs}
            natal_emb = encoder(**encoder_kw)
            probs = head(natal_emb)
            all_pred.append(probs.cpu().numpy())
            all_true.append(targets["elemental_strengths"].numpy())
    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)
    N, K = pred.shape
    assert K == 5

    # Ordinal correct: argsort(pred) == argsort(true) per sample
    pred_rank = np.argsort(np.argsort(pred, axis=1), axis=1)
    true_rank = np.argsort(np.argsort(true, axis=1), axis=1)
    correct = np.all(pred_rank == true_rank, axis=1)
    n_correct = int(correct.sum())
    n_wrong = N - n_correct
    ordinal_accuracy = n_correct / N if N else 0.0

    # Balanced chart: all 5 element strengths within 0.05 (spread = max - min <= 0.05).
    # If wrong-ordinal cases are mostly balanced, the error is unavoidable noise, not model failure.
    balanced_threshold = 0.05
    true_spread = true.max(axis=1) - true.min(axis=1)
    wrong_indices = np.where(~correct)[0]
    wrong_spreads = true_spread[wrong_indices] if len(wrong_indices) else np.array([])
    n_balanced_wrong = int((wrong_spreads <= balanced_threshold).sum())
    n_unbalanced_wrong = len(wrong_indices) - n_balanced_wrong

    # For each wrong sample: min true-strength difference over "swapped" pairs
    # Pair (i,j) is swapped if true says i > j but pred says j > i (or vice versa).
    # Margin = |true[i] - true[j]|. If margin < threshold, borderline.
    borderline_threshold = args.borderline_threshold
    wrong_borderline = 0
    wrong_clear = 0
    min_margins = []

    for idx in wrong_indices:
        t = true[idx]
        p = pred[idx]
        tr = true_rank[idx]
        pr = pred_rank[idx]
        margins = []
        for i in range(K):
            for j in range(i + 1, K):
                if (tr[i] > tr[j]) != (pr[i] > pr[j]):
                    margins.append(abs(t[i] - t[j]))
        if margins:
            min_margin = float(min(margins))
            min_margins.append(min_margin)
            if min_margin < borderline_threshold:
                wrong_borderline += 1
            else:
                wrong_clear += 1

    # Histogram of min_margin for wrong samples
    bins = [0, 0.02, 0.05, 0.1, 0.2, 1.0]
    hist = np.histogram(min_margins, bins=bins)[0].tolist()
    bin_labels = [f"[{bins[i]:.2f},{bins[i+1]:.2f})" for i in range(len(bins) - 1)]

    report = {
        "n_total": N,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "ordinal_accuracy": round(ordinal_accuracy, 4),
        "n_wrong_balanced": n_balanced_wrong,
        "n_wrong_unbalanced": n_unbalanced_wrong,
        "pct_balanced_among_wrong": round(100.0 * n_balanced_wrong / n_wrong, 2) if n_wrong else 0.0,
        "balanced_threshold": balanced_threshold,
        "n_wrong_borderline": wrong_borderline,
        "n_wrong_clear": wrong_clear,
        "borderline_threshold": borderline_threshold,
        "min_margin_histogram_bins": bin_labels,
        "min_margin_histogram_counts": hist,
        "checkpoint": str(ckpt_path),
        "eval_split": "test",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Report written to {args.output}")

    print("\n--- Ordinal error analysis ---")
    print(f"  Ordinal accuracy: {ordinal_accuracy:.2%} ({n_correct}/{N})")
    print(f"  Wrong samples: {n_wrong}")
    if n_wrong:
        print(f"  Of wrong: balanced (true spread <= {balanced_threshold}): {n_balanced_wrong} ({100*n_balanced_wrong/n_wrong:.0f}%)")
    else:
        print("  Of wrong: (none)")
    print(f"  Of wrong: unbalanced (spread > {balanced_threshold}): {n_unbalanced_wrong}")
    print(f"  Of wrong: borderline (min pair diff < {borderline_threshold}): {wrong_borderline}")
    print(f"  Of wrong: clear (min pair diff >= {borderline_threshold}): {wrong_clear}")
    if n_wrong:
        pct_borderline = 100 * wrong_borderline / n_wrong
        print(f"  → {pct_borderline:.0f}% of errors are borderline (targets very close)")
    print("  Min-margin histogram (wrong samples):", dict(zip(bin_labels, hist)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
