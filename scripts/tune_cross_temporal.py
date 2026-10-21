#!/usr/bin/env python3
"""
Hyperparameter tuning for cross-temporal training: short-horizon trials + optional full run.

Runs multiple short training runs (e.g. 5–10 epochs) with different configs (lr,
segment_weights, pos_weight_from_data), records best validation loss per trial,
then optionally runs one full training with the best config.

Usage:
  python scripts/tune_cross_temporal.py --data data/charts.jsonl --epochs_per_trial 5 --n_trials 8
  python scripts/tune_cross_temporal.py --data data/charts.jsonl --epochs_per_trial 5 --n_trials 8 --run_full_at_end
  python scripts/tune_cross_temporal.py --data data/charts.jsonl --search random --n_trials 12 --output reports/tune_cross_temporal.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent


def _parse_best_val_loss(stdout: str) -> float | None:
    """Parse 'Best validation loss: X' from train_cross_temporal.py output."""
    m = re.search(r"Best validation loss:\s+([\d.]+)", stdout)
    return float(m.group(1)) if m else None


def _run_trial(
    data: Path,
    epochs: int,
    lr: float,
    segment_weights: str | None,
    pos_weight_from_data: bool,
    batch_size: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    checkpoint_dir: Path,
    log_dir: Path | None,
    weight_decay: float = 0.01,
) -> tuple[float | None, str]:
    """Run one train_cross_temporal.py trial. Returns (best_val_loss or None, stderr snippet)."""
    cmd = [
        sys.executable,
        str(_root / "scripts" / "train_cross_temporal.py"),
        "--data",
        str(data),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--batch_size",
        str(batch_size),
        "--val_ratio",
        str(val_ratio),
        "--test_ratio",
        str(test_ratio),
        "--seed",
        str(seed),
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--weight_decay",
        str(weight_decay),
    ]
    if segment_weights is not None:
        cmd.extend(["--segment_weights", segment_weights])
    if pos_weight_from_data:
        cmd.append("--pos_weight_from_data")
    if log_dir is not None:
        cmd.extend(["--log_dir", str(log_dir)])

    result = subprocess.run(
        cmd,
        cwd=str(_root),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    best = _parse_best_val_loss(result.stdout) if result.returncode == 0 else None
    err = result.stderr[-500:] if result.stderr else ""
    if result.returncode != 0:
        err = f"exit {result.returncode}\n" + err
    return best, err


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-temporal tuning: short trials then optional full run."
    )
    parser.add_argument("--data", type=Path, default=Path("data/charts.jsonl"), help="Path to charts.jsonl")
    parser.add_argument(
        "--epochs_per_trial",
        type=int,
        default=5,
        help="Epochs per trial (short horizon; 5–10 typical)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=8,
        help="Number of trials (grid: lr × segment_weights × pos_weight; random: n_trials samples)",
    )
    parser.add_argument(
        "--search",
        choices=["grid", "random"],
        default="grid",
        help="grid: lr × segment_weights × pos_weight_from_data; random: sample n_trials",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tune_dir",
        type=Path,
        default=Path("checkpoints/tune_cross_temporal"),
        help="Directory for trial checkpoints (subdirs trial_0, trial_1, ...)",
    )
    parser.add_argument(
        "--run_full_at_end",
        action="store_true",
        help="After tuning, run one full training (60 epochs) with best config",
    )
    parser.add_argument(
        "--full_epochs",
        type=int,
        default=60,
        help="Epochs for full run when --run_full_at_end",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="AdamW weight decay passed to train_cross_temporal",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write results JSON here (configs and best_val_loss per trial)",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Dataset not found: {args.data}", file=sys.stderr)
        return 1

    # Search space
    lr_options = [5e-4, 1e-3, 2e-3]
    segment_weight_presets = [
        None,  # no segment weighting
        "1,1,1,1,1,1,1",
        "2,1,1,1,1,1,4",
        "3,1,1,1,1,1,5",
    ]
    pos_weight_options = [False, True]

    if args.search == "grid":
        configs = []
        for lr in lr_options:
            for sw in segment_weight_presets:
                for pw in pos_weight_options:
                    configs.append({"lr": lr, "segment_weights": sw, "pos_weight_from_data": pw})
        configs = configs[: args.n_trials]
    else:
        random.seed(args.seed)
        configs = []
        for _ in range(args.n_trials):
            configs.append({
                "lr": random.choice(lr_options),
                "segment_weights": random.choice(segment_weight_presets),
                "pos_weight_from_data": random.choice(pos_weight_options),
            })

    args.tune_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, cfg in enumerate(configs):
        ckpt_dir = args.tune_dir / f"trial_{i}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir = args.tune_dir / "logs" / f"trial_{i}"
        sw_str = cfg["segment_weights"] or "none"
        pw_str = "yes" if cfg["pos_weight_from_data"] else "no"
        print(f"Trial {i + 1}/{len(configs)}: lr={cfg['lr']}, segment_weights={sw_str}, pos_weight_from_data={pw_str}")

        best_metric, err = _run_trial(
            data=args.data,
            epochs=args.epochs_per_trial,
            lr=cfg["lr"],
            segment_weights=cfg["segment_weights"],
            pos_weight_from_data=cfg["pos_weight_from_data"],
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            checkpoint_dir=ckpt_dir,
            log_dir=log_dir,
            weight_decay=args.weight_decay,
        )

        results.append({
            "trial": i,
            "lr": cfg["lr"],
            "segment_weights": cfg["segment_weights"],
            "pos_weight_from_data": cfg["pos_weight_from_data"],
            "best_val_loss": best_metric,
        })
        if best_metric is not None:
            print(f"  best_val_loss={best_metric:.6f}")
        else:
            print(f"  failed: {err.strip()[:200]}")

    valid = [r for r in results if r["best_val_loss"] is not None]
    if not valid:
        print("No successful trials.", file=sys.stderr)
        if args.output:
            args.output.write_text(json.dumps({"trials": results}, indent=2))
        return 1

    best_trial = min(valid, key=lambda r: r["best_val_loss"])
    print(f"\nBest config: lr={best_trial['lr']}, segment_weights={best_trial['segment_weights']}, pos_weight_from_data={best_trial['pos_weight_from_data']} (val_loss={best_trial['best_val_loss']:.6f})")
    print("Full training command:")
    sw_arg = f' --segment_weights "{best_trial["segment_weights"]}"' if best_trial["segment_weights"] else ""
    pw_arg = " --pos_weight_from_data" if best_trial["pos_weight_from_data"] else ""
    print(
        f"  python scripts/train_cross_temporal.py --data {args.data} --epochs {args.full_epochs} --batch_size {args.batch_size}"
        f" --lr {best_trial['lr']} --val_ratio {args.val_ratio} --test_ratio {args.test_ratio} --seed {args.seed}"
        f" --checkpoint_dir checkpoints/sequential/cross_temporal --log_dir runs/cross_temporal{sw_arg}{pw_arg}"
    )

    if args.output:
        out_data = {"trials": results, "best": best_trial}
        args.output.write_text(json.dumps(out_data, indent=2))
        print(f"\nResults written to {args.output}")

    if args.run_full_at_end:
        print(f"\nRunning full {args.full_epochs}-epoch training with best config...")
        full_ckpt = Path("checkpoints/sequential/cross_temporal")
        full_ckpt.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(_root / "scripts" / "train_cross_temporal.py"),
            "--data",
            str(args.data),
            "--epochs",
            str(args.full_epochs),
            "--lr",
            str(best_trial["lr"]),
            "--batch_size",
            str(args.batch_size),
            "--val_ratio",
            str(args.val_ratio),
            "--test_ratio",
            str(args.test_ratio),
            "--seed",
            str(args.seed),
            "--checkpoint_dir",
            str(full_ckpt),
            "--log_dir",
            "runs/cross_temporal",
            "--weight_decay",
            str(args.weight_decay),
        ]
        if best_trial["segment_weights"]:
            cmd.extend(["--segment_weights", best_trial["segment_weights"]])
        if best_trial["pos_weight_from_data"]:
            cmd.append("--pos_weight_from_data")

        ret = subprocess.run(cmd, cwd=str(_root))
        if ret.returncode != 0:
            print("Full run failed.", file=sys.stderr)
            return 1
        print(f"Full run completed. Checkpoint: {full_ckpt}/cross_temporal_best.pt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
