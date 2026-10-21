"""
Prepare Phase 2 dataset: curated JSONL → train/val split with task tokens.

Reads a curated JSONL with (chart, task_type, query, reasoning, output), validates
each record, adds task_token for each task_type, and writes train/val JSONL (and
optionally test). For training, loss is computed only on reasoning + output;
the query is in the model input only (see docs/phase2_implementation.md).

Usage:
  python scripts/prepare_phase2_dataset.py --input data/curated_synthetic.jsonl
  python scripts/prepare_phase2_dataset.py -i data/curated_synthetic.jsonl --val_ratio 0.1 --seed 42
  python scripts/prepare_phase2_dataset.py -i data/curated_synthetic.jsonl --val_ratio 0.1 --test_ratio 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bazi_engine.models import BaziChart

# Task types and target distribution (single source of truth; reuse in train_phase2 for balancing).
TASK_TYPES = ("本命分析", "流年", "流月", "八字问答")
TARGET_DISTRIBUTION = {
    "本命分析": 0.40,
    "流年": 0.30,
    "流月": 0.10,
    "八字问答": 0.20,
}

TASK_TOKEN_MAP = {
    "本命分析": "[TASK:本命分析]",
    "流年": "[TASK:流年]",
    "流月": "[TASK:流月]",
    "八字问答": "[TASK:八字问答]",
}


def _chart_from_record(record: dict) -> BaziChart | None:
    """Parse chart from record; return None if invalid."""
    chart_val = record.get("chart")
    if chart_val is None:
        return None
    try:
        if isinstance(chart_val, dict):
            chart_str = json.dumps(chart_val, ensure_ascii=False)
        else:
            chart_str = str(chart_val)
        return BaziChart.from_json(chart_str)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _validate_record(record: dict, *, validate_chart_features: bool = True) -> tuple[bool, str]:
    """
    Validate one curated record. Returns (ok, message).
    For 流年/流月, chart must have temporal_context and cross_interactions.
    """
    for key in ("chart", "task_type", "query", "reasoning", "output"):
        if key not in record:
            return False, f"missing field: {key}"
    task_type = record["task_type"]
    if task_type not in TASK_TYPES:
        return False, f"invalid task_type: {task_type!r}"

    chart = _chart_from_record(record)
    if chart is None:
        return False, "chart parse failed"

    if task_type in ("流年", "流月"):
        if chart.temporal_context is None:
            return False, "流年/流月 requires chart.temporal_context"
        if chart.cross_interactions is None:
            return False, "流年/流月 requires chart.cross_interactions"

    if validate_chart_features:
        try:
            if task_type in ("流年", "流月"):
                chart.to_input_features("cross_temporal")
            else:
                chart.to_input_features("multi")
        except (ValueError, KeyError, TypeError) as e:
            return False, f"chart features failed: {e}"

    return True, ""


def _stratified_split(
    records: list[dict],
    val_ratio: float,
    seed: int,
    *,
    test_ratio: float = 0.0,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split records into train / val / test, stratified by task_type.

    Counts per stratum use floor: n_test = int(n * test_ratio), n_val = int(n * val_ratio),
    n_train = n - n_test - n_val. Requires val_ratio + test_ratio < 1.
    """
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r["task_type"]].append(r)

    train, val, test = [], [], []
    rng = random.Random(seed)
    for _task_type, items in by_task.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(0, int(n * test_ratio)) if test_ratio > 0 else 0
        n_val = max(0, int(n * val_ratio))
        n_train = n - n_test - n_val
        if n_train < 0:
            n_val = max(0, n_val + n_train)
            n_train = n - n_test - n_val
        if n_train < 0:
            n_test = max(0, n_test + n_train)
            n_train = n - n_test - n_val
        train.extend(items[n_test + n_val :])
        val.extend(items[n_test : n_test + n_val])
        test.extend(items[:n_test])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _add_task_token(record: dict, *, add_input_target_sequences: bool = True) -> dict:
    """Add task_token (and optionally input_sequence, target_sequence) to record."""
    out = dict(record)
    out["task_token"] = TASK_TOKEN_MAP[out["task_type"]]
    if add_input_target_sequences:
        out["input_sequence"] = out["task_token"] + "\n" + out["query"]
        out["target_sequence"] = out["reasoning"] + "\n" + out["output"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Phase 2 train/val JSONL from curated synthetic data."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("data/curated_synthetic.jsonl"),
        help="Path to curated JSONL (default: data/curated_synthetic.jsonl)",
    )
    parser.add_argument(
        "--train_out",
        type=Path,
        default=Path("data/phase2_train.jsonl"),
        help="Output train JSONL (default: data/phase2_train.jsonl)",
    )
    parser.add_argument(
        "--val_out",
        type=Path,
        default=Path("data/phase2_val.jsonl"),
        help="Output val JSONL (default: data/phase2_val.jsonl)",
    )
    parser.add_argument(
        "--test_out",
        type=Path,
        default=Path("data/phase2_test.jsonl"),
        help="Output test JSONL when --test_ratio > 0 (default: data/phase2_test.jsonl)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation fraction per task_type stratum (default: 0.1)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.0,
        help="Optional test fraction per task_type stratum (default: 0 → no test file)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split (default: 42)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max sequence length in tokens (not used in this script; training truncates/pads to this)",
    )
    parser.add_argument(
        "--add_task_tokens",
        action="store_true",
        default=True,
        help="Add task_token to each record (default: True)",
    )
    parser.add_argument(
        "--no_add_task_tokens",
        action="store_false",
        dest="add_task_tokens",
        help="Do not add task_token",
    )
    parser.add_argument(
        "--no_validate_features",
        action="store_true",
        help="Skip chart to_input_features() validation",
    )
    parser.add_argument(
        "--no_input_target_sequences",
        action="store_true",
        help="Do not add input_sequence and target_sequence to output",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1
    if args.val_ratio < 0 or args.val_ratio >= 1:
        print("Error: --val_ratio must be in [0, 1)", file=sys.stderr)
        return 1
    if args.test_ratio < 0 or args.test_ratio >= 1:
        print("Error: --test_ratio must be in [0, 1)", file=sys.stderr)
        return 1
    if args.val_ratio + args.test_ratio >= 1:
        print(
            "Error: --val_ratio + --test_ratio must be < 1 (leave room for train)",
            file=sys.stderr,
        )
        return 1
    if args.train_out.resolve() == args.val_out.resolve():
        print("Error: train_out and val_out must differ", file=sys.stderr)
        return 1
    if args.test_ratio > 0:
        tr, va, te = args.train_out.resolve(), args.val_out.resolve(), args.test_out.resolve()
        if te == tr or te == va:
            print("Error: test_out must differ from train_out and val_out", file=sys.stderr)
            return 1

    records: list[dict] = []
    skipped = 0
    skip_reasons: defaultdict[str, int] = defaultdict(int)

    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skip_reasons["json_decode"] += 1
                skipped += 1
                continue
            if not isinstance(record, dict):
                skip_reasons["not_a_dict"] += 1
                skipped += 1
                continue
            ok, msg = _validate_record(
                record,
                validate_chart_features=not args.no_validate_features,
            )
            if not ok:
                skip_reasons[msg or "unknown"] += 1
                skipped += 1
                continue
            if args.add_task_tokens:
                record = _add_task_token(
                    record,
                    add_input_target_sequences=not args.no_input_target_sequences,
                )
            records.append(record)

    if not records:
        print("Error: no valid records after validation", file=sys.stderr)
        return 1

    train, val, test = _stratified_split(
        records, args.val_ratio, args.seed, test_ratio=args.test_ratio
    )

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.val_out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.train_out, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.val_out, "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.test_ratio > 0:
        args.test_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.test_out, "w", encoding="utf-8") as f:
            for r in test:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    test_counts = defaultdict(int)
    for r in train:
        train_counts[r["task_type"]] += 1
    for r in val:
        val_counts[r["task_type"]] += 1
    for r in test:
        test_counts[r["task_type"]] += 1

    print(f"Total valid: {len(records)}, skipped: {skipped}")
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"  skip reason: {reason!r} -> {count}")
    print(f"Train: {len(train)}, val: {len(val)}", end="")
    if args.test_ratio > 0:
        print(f", test: {len(test)}")
    else:
        print()
    print("Train task_type counts:", dict(train_counts))
    print("Val task_type counts:", dict(val_counts))
    if args.test_ratio > 0:
        print("Test task_type counts:", dict(test_counts))
        print(f"Wrote {args.train_out}, {args.val_out}, {args.test_out}")
    else:
        print(f"Wrote {args.train_out}, {args.val_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
