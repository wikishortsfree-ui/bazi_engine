#!/usr/bin/env python3
"""
Efficient Phase 1 hyperparameter tuning: short-horizon trials + optional full run.

Runs multiple short training runs (e.g. 5 epochs each) with different configs
(lr, task_weights), records best validation loss per trial, then optionally
runs one full 30-epoch training with the best config. This minimizes total
compute vs running full 30 epochs for every candidate.

Usage:
  python scripts/tune_phase1.py --data data/charts.jsonl --epochs_per_trial 5 --n_trials 9
  python scripts/tune_phase1.py --data data/charts.jsonl --epochs_per_trial 5 --n_trials 9 --run_full_at_end
  python scripts/tune_phase1.py --data data/charts.jsonl --epochs_per_trial 5 --search random --n_trials 12

See docs/phase1_training_strategy_clarification.md § "Most efficient manner".
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
    """Parse 'Best validation loss (sum): X' or 'Best validation composite: X' from train_multi_task.py output."""
    m = re.search(r"Best validation (?:loss \(sum\)|composite):\s+([\d.]+)", stdout)
    return float(m.group(1)) if m else None


def _run_trial(
    data: Path,
    epochs: int,
    lr: float,
    task_weights: str,
    batch_size: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    checkpoint_dir: Path,
    log_dir: Path | None,
    save_by: str = "loss",
    ordinal_loss_weight: float = 0.15,
) -> tuple[float | None, str]:
    """Run one train_multi_task.py trial. Returns (best_val_metric or None, stderr snippet)."""
    cmd = [
        sys.executable,
        str(_root / "scripts" / "train_multi_task.py"),
        "--data",
        str(data),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--task_weights",
        task_weights,
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
        "--patience",
        str(max(epochs, 5)),  # avoid early stop during short trials
        "--save_by",
        save_by,
        "--ordinal_loss_weight",
        str(ordinal_loss_weight),
    ]
    if log_dir:
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
        description="Efficient Phase 1 tuning: short trials then optional full run."
    )
    parser.add_argument("--data", type=Path, default=Path("data/charts.jsonl"), help="Path to charts.jsonl")
    parser.add_argument(
        "--epochs_per_trial",
        type=int,
        default=5,
        help="Epochs per trial (short horizon; 5–10 is typical)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=15,
        help="Number of trials (for grid: 15 = 3 lr × 5 element weights)",
    )
    parser.add_argument(
        "--search",
        choices=["grid", "random"],
        default="grid",
        help="grid: lr × element_weight grid; random: sample n_trials from space",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tune_dir",
        type=Path,
        default=Path("checkpoints/tune_phase1"),
        help="Directory for trial checkpoints (subdirs trial_0, trial_1, ...)",
    )
    parser.add_argument(
        "--run_full_at_end",
        action="store_true",
        help="After tuning, run one full 30-epoch training with best config",
    )
    parser.add_argument(
        "--save_by",
        choices=["loss", "composite"],
        default="loss",
        help="Same as train_multi_task: save best checkpoint by 'loss' (min) or 'composite' (max). Affects trial selection and full run.",
    )
    parser.add_argument(
        "--ordinal_loss_weight",
        type=float,
        default=0.15,
        help="Element head pairwise ranking weight passed to train_multi_task (0.15 = better MSE; try 0.25 for more ordinal focus).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write results JSON here (configs and best_val_metric per trial)",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Dataset not found: {args.data}", file=sys.stderr)
        return 1

    # Search space: lr × element_weight (other task weights fixed at 1)
    lr_options = [5e-4, 1e-3, 2e-3]
    element_weights = [1.0, 2.0, 3.0, 4.0, 5.0]
    if args.search == "grid":
        configs = [
            {"lr": lr, "task_weights": f"1,{ew},1,1"}
            for lr in lr_options
            for ew in element_weights
        ]
        configs = configs[: args.n_trials]
    else:
        random.seed(args.seed)
        configs = []
        for _ in range(args.n_trials):
            lr = random.choice(lr_options)
            ew = random.choice(element_weights)
            configs.append({"lr": lr, "task_weights": f"1,{ew},1,1"})

    args.tune_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, cfg in enumerate(configs):
        ckpt_dir = args.tune_dir / f"trial_{i}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir = args.tune_dir / "logs" / f"trial_{i}"
        print(f"Trial {i + 1}/{len(configs)}: lr={cfg['lr']}, task_weights={cfg['task_weights']}")
        best_metric, err = _run_trial(
            data=args.data,
            epochs=args.epochs_per_trial,
            lr=cfg["lr"],
            task_weights=cfg["task_weights"],
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            checkpoint_dir=ckpt_dir,
            log_dir=log_dir,
            save_by=args.save_by,
            ordinal_loss_weight=args.ordinal_loss_weight,
        )
        results.append(
            {
                "trial": i,
                "lr": cfg["lr"],
                "task_weights": cfg["task_weights"],
                "best_val_metric": best_metric,
            }
        )
        if best_metric is not None:
            print(f"  best_val_metric={best_metric:.6f}")
        else:
            print(f"  failed: {err.strip()[:200]}")

    valid = [r for r in results if r["best_val_metric"] is not None]
    if not valid:
        print("No successful trials.", file=sys.stderr)
        if args.output:
            args.output.write_text(json.dumps({"trials": results}, indent=2))
        return 1

    # Best trial: min metric when save_by=loss, max when save_by=composite
    if args.save_by == "composite":
        best_trial = max(valid, key=lambda r: r["best_val_metric"])
    else:
        best_trial = min(valid, key=lambda r: r["best_val_metric"])
    metric_label = "composite" if args.save_by == "composite" else "val_loss"
    print(f"\nBest config: lr={best_trial['lr']}, task_weights={best_trial['task_weights']} ({metric_label}={best_trial['best_val_metric']:.6f})")
    print("Full training command:")
    save_by_arg = f" --save_by {args.save_by}" if args.save_by != "loss" else ""
    ord_arg = f" --ordinal_loss_weight {args.ordinal_loss_weight}"
    print(
        f"  python scripts/train_multi_task.py --data {args.data} --epochs 30 --batch_size {args.batch_size}"
        f" --lr {best_trial['lr']} --task_weights {best_trial['task_weights']}"
        f" --val_ratio {args.val_ratio} --test_ratio {args.test_ratio} --seed {args.seed}"
        f" --checkpoint_dir checkpoints --log_dir runs/multi_task{ord_arg}{save_by_arg}"
    )

    if args.output:
        out_data = {"trials": results, "best": best_trial}
        args.output.write_text(json.dumps(out_data, indent=2))
        print(f"\nResults written to {args.output}")

    if args.run_full_at_end:
        print("\nRunning full 30-epoch training with best config...")
        cmd = [
            sys.executable,
            str(_root / "scripts" / "train_multi_task.py"),
            "--data",
            str(args.data),
            "--epochs",
            "30",
            "--lr",
            str(best_trial["lr"]),
            "--task_weights",
            best_trial["task_weights"],
            "--batch_size",
            str(args.batch_size),
            "--val_ratio",
            str(args.val_ratio),
            "--test_ratio",
            str(args.test_ratio),
            "--seed",
            str(args.seed),
            "--checkpoint_dir",
            "checkpoints",
            "--log_dir",
            "runs/multi_task",
            "--save_by",
            args.save_by,
            "--ordinal_loss_weight",
            str(args.ordinal_loss_weight),
        ]
        ret = subprocess.run(cmd, cwd=str(_root))
        if ret.returncode != 0:
            print("Full run failed.", file=sys.stderr)
            return 1
        print("Full run completed. Checkpoint: checkpoints/multi_task_best.pt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
