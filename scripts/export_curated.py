"""
Export accepted (and optionally flagged) rows from the curation app to curated JSONL.

Reads the curation state sidecar (produced by scripts/curation_app/app.py) alongside
the source JSONL, and writes only accepted rows in the schema expected by
scripts/prepare_phase2_dataset.py:

    chart, task_type, query, reasoning, output

Optionally writes flagged rows to a separate file for follow-up revision.

Usage:
    python scripts/export_curated.py
    python scripts/export_curated.py -i data/synthetic_final.jsonl
    python scripts/export_curated.py -i data/synthetic_final.jsonl --state data/synthetic_final.curation_state.json
    python scripts/export_curated.py --include-flagged
    python scripts/export_curated.py --flagged-out data/flagged_for_revision.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_OUTPUT_FIELDS = ("chart", "task_type", "query", "reasoning", "output")
OPTIONAL_METADATA_FIELDS = ("name", "selection_source")


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_state(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_output_record(rec: dict, review: dict) -> dict:
    """Build an output record with the canonical schema fields."""
    out: dict = {}
    for key in REQUIRED_OUTPUT_FIELDS:
        out[key] = rec[key]
    out["human_review"] = {
        "action": review.get("action", ""),
        "notes": review.get("notes", ""),
        "timestamp": review.get("timestamp", ""),
    }
    for key in OPTIONAL_METADATA_FIELDS:
        if key in rec:
            out[key] = rec[key]
    return out


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export accepted curation rows to curated_synthetic.jsonl."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=ROOT / "data" / "synthetic_final.jsonl",
        help="Source JSONL reviewed in the curation app (default: data/synthetic_final.jsonl)",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help="Curation state JSON. Default: <input>.curation_state.json",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=ROOT / "data" / "curated_synthetic.jsonl",
        help="Output path for accepted rows (default: data/curated_synthetic.jsonl)",
    )
    parser.add_argument(
        "--include-flagged",
        action="store_true",
        help="Include flagged records in the main output alongside accepted ones",
    )
    parser.add_argument(
        "--flagged-out",
        type=Path,
        default=None,
        help="Separate output for flagged rows (e.g. data/flagged_for_revision.jsonl). "
             "Written regardless of --include-flagged.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing files",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    state_path: Path = args.state or args.input.with_suffix(".curation_state.json")
    if not state_path.exists():
        print(f"Error: curation state not found: {state_path}", file=sys.stderr)
        print("Run the curation app first, or supply --state explicitly.", file=sys.stderr)
        return 1

    records = _load_jsonl(args.input)
    state = _load_state(state_path)

    if not records:
        print("Error: no records in input file", file=sys.stderr)
        return 1

    accepted: list[dict] = []
    flagged: list[dict] = []
    rejected = 0
    unreviewed = 0
    missing_fields = 0
    task_counts: defaultdict[str, int] = defaultdict(int)
    tier_counts: defaultdict[int, int] = defaultdict(int)

    for i, rec in enumerate(records):
        review = state.get(str(i), {})
        action = review.get("action", "")

        has_required = all(k in rec for k in REQUIRED_OUTPUT_FIELDS)
        if not has_required:
            missing_fields += 1
            continue

        if action == "accept":
            out = _build_output_record(rec, review)
            accepted.append(out)
            task_counts[rec.get("task_type", "?")] += 1
            tier = rec.get("quality", {}).get("tier")
            if tier is not None:
                tier_counts[tier] += 1
        elif action == "flag":
            out = _build_output_record(rec, review)
            flagged.append(out)
            if args.include_flagged:
                accepted.append(out)
                task_counts[rec.get("task_type", "?")] += 1
        elif action == "reject":
            rejected += 1
        else:
            unreviewed += 1

    print(f"Source: {args.input} ({len(records)} records)")
    print(f"State:  {state_path} ({len(state)} reviewed)")
    print()
    print(f"  Accepted:   {len(accepted) - (len(flagged) if args.include_flagged else 0)}")
    print(f"  Flagged:    {len(flagged)}" + (" (included in output)" if args.include_flagged else ""))
    print(f"  Rejected:   {rejected}")
    print(f"  Unreviewed: {unreviewed}")
    if missing_fields:
        print(f"  Missing fields (skipped): {missing_fields}")

    total_export = len(accepted)
    print(f"\nTotal to export: {total_export}")

    if task_counts:
        print("\nTask type breakdown (exported):")
        for tt, count in sorted(task_counts.items()):
            pct = 100 * count / total_export if total_export else 0
            print(f"  {tt}: {count} ({pct:.1f}%)")

    if tier_counts:
        print("\nTier breakdown (accepted only):")
        for tier in sorted(tier_counts):
            print(f"  Tier {tier}: {tier_counts[tier]}")

    if args.dry_run:
        print("\n(dry run — no files written)")
        return 0

    if total_export == 0:
        print("\nWarning: no records to export", file=sys.stderr)
        return 1

    _write_jsonl(accepted, args.output)
    print(f"\nWrote {total_export} records to {args.output}")

    if args.flagged_out and flagged:
        _write_jsonl(flagged, args.flagged_out)
        print(f"Wrote {len(flagged)} flagged records to {args.flagged_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
