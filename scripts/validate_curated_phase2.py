"""
Validate curated Phase 2 JSONL against the schema expected by prepare_phase2_dataset.py.

Runs the same validation as prepare_phase2_dataset._validate_record (field presence,
task_type, chart parse, temporal requirements, to_input_features), plus distribution
and length checks.  Optionally validates against the JSON Schema in
schemas/training_example.json.

Usage:
    python scripts/validate_curated_phase2.py data/curated_synthetic.jsonl
    python scripts/validate_curated_phase2.py data/curated_synthetic.jsonl --json-schema
    python scripts/validate_curated_phase2.py data/curated_synthetic.jsonl --skip-features
    python scripts/validate_curated_phase2.py data/curated_synthetic.jsonl --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bazi_engine.models import BaziChart

TASK_TYPES = ("本命分析", "流年", "流月", "八字问答")
REQUIRED_FIELDS = ("chart", "task_type", "query", "reasoning", "output")

TARGET_DISTRIBUTION = {
    "本命分析": 0.40,
    "流年": 0.30,
    "流月": 0.10,
    "八字问答": 0.20,
}


def _chart_from_record(record: dict) -> BaziChart | None:
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


def _validate_record(
    record: dict,
    *,
    validate_features: bool = True,
) -> list[str]:
    """Return list of error strings (empty = valid)."""
    errors: list[str] = []

    for key in REQUIRED_FIELDS:
        if key not in record:
            errors.append(f"missing field: {key}")
    if errors:
        return errors

    task_type = record["task_type"]
    if task_type not in TASK_TYPES:
        errors.append(f"invalid task_type: {task_type!r}")

    for field in ("query", "reasoning", "output"):
        val = record[field]
        if not isinstance(val, str) or not val.strip():
            errors.append(f"empty or non-string {field}")

    chart = _chart_from_record(record)
    if chart is None:
        errors.append("chart parse failed (BaziChart.from_json)")
        return errors

    if task_type in ("流年", "流月"):
        if chart.temporal_context is None:
            errors.append("流年/流月 requires chart.temporal_context (dynamic != null)")
        if chart.cross_interactions is None:
            errors.append("流年/流月 requires chart.cross_interactions (dynamic != null)")

    if validate_features and not errors:
        try:
            if task_type in ("流年", "流月"):
                chart.to_input_features("cross_temporal")
            else:
                chart.to_input_features("multi")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"to_input_features failed: {exc}")

    return errors


def _validate_json_schema(record: dict, schema: dict) -> list[str]:
    """Validate record against JSON Schema. Requires jsonschema package."""
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema package not installed; run: pip install jsonschema"]
    try:
        jsonschema.validate(record, schema)
        return []
    except jsonschema.ValidationError as exc:
        return [f"JSON Schema: {exc.message} (at {'/'.join(str(p) for p in exc.absolute_path)})"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate curated Phase 2 JSONL for prepare_phase2_dataset.py compatibility."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to curated JSONL (e.g. data/curated_synthetic.jsonl)",
    )
    parser.add_argument(
        "--json-schema",
        action="store_true",
        help="Also validate each record against schemas/training_example.json (requires jsonschema)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip chart.to_input_features() validation (faster)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print every error per record (default: summary only)",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=50,
        help="Max per-record errors to print in verbose mode (default: 50)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        return 1

    schema: dict | None = None
    if args.json_schema:
        schema_path = ROOT / "schemas" / "training_example.json"
        if not schema_path.exists():
            print(f"Error: JSON Schema not found: {schema_path}", file=sys.stderr)
            return 1
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

    total = 0
    valid = 0
    invalid = 0
    parse_errors = 0
    error_counts: defaultdict[str, int] = defaultdict(int)
    task_type_counts: Counter[str] = Counter()
    errors_printed = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors += 1
                invalid += 1
                if args.verbose and errors_printed < args.max_errors:
                    print(f"  line {lineno}: JSON parse error: {exc}")
                    errors_printed += 1
                continue

            if not isinstance(record, dict):
                parse_errors += 1
                invalid += 1
                continue

            errs = _validate_record(record, validate_features=not args.skip_features)
            if schema is not None:
                errs.extend(_validate_json_schema(record, schema))

            if errs:
                invalid += 1
                for e in errs:
                    error_counts[e] += 1
                if args.verbose and errors_printed < args.max_errors:
                    print(f"  line {lineno}: {'; '.join(errs)}")
                    errors_printed += 1
            else:
                valid += 1
                tt = record.get("task_type", "?")
                task_type_counts[tt] += 1

    print(f"File: {args.input}")
    print(f"Total records: {total}")
    print(f"  Valid:   {valid}")
    print(f"  Invalid: {invalid}")
    if parse_errors:
        print(f"  JSON parse errors: {parse_errors}")

    if error_counts:
        print("\nError breakdown:")
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  [{count:>4}] {err}")

    if task_type_counts:
        print(f"\nTask type distribution (valid records only, n={valid}):")
        for tt in TASK_TYPES:
            count = task_type_counts.get(tt, 0)
            pct = 100 * count / valid if valid else 0
            target = TARGET_DISTRIBUTION.get(tt, 0) * 100
            delta = pct - target
            flag = " !" if abs(delta) > 10 else ""
            print(f"  {tt}: {count:>5} ({pct:5.1f}%, target {target:.0f}%){flag}")

    if invalid > 0:
        pct_invalid = 100 * invalid / total if total else 0
        print(f"\nResult: FAIL — {invalid}/{total} records invalid ({pct_invalid:.1f}%)")
        if not args.verbose:
            print("Re-run with --verbose to see per-record errors.")
        return 1

    print(f"\nResult: PASS — all {valid} records valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
