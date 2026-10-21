"""
Export curated Phase 2 JSONL from scored records (no Claude, no UI).

Reads output of phase2_quality_filter.py (with quality.tier), keeps rows whose
tier is <= --max-tier (default 2), strips QC-only fields, and writes a file
suitable for prepare_phase2_dataset.py (chart, task_type, query, reasoning,
output, plus optional name / selection_source when present).

Usage:
  python scripts/export_curated_from_scored.py \\
    -i data/synthetic_scored.jsonl -o data/curated_synthetic.jsonl
  python scripts/export_curated_from_scored.py -i data/synthetic_scored.jsonl \\
    -o data/curated_synthetic.jsonl --max-tier 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CORE_FIELDS = ("chart", "task_type", "query", "reasoning", "output")
OPTIONAL_FIELDS = ("name", "selection_source")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build curated_synthetic.jsonl from synthetic_scored.jsonl (tier filter + strip QC fields)."
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Scored JSONL (e.g. data/synthetic_scored.jsonl)")
    p.add_argument("--output", "-o", type=Path, required=True, help="Output curated JSONL path")
    p.add_argument(
        "--max-tier",
        type=int,
        default=2,
        help="Keep records with quality.tier <= this value (default: 2 = Tier 1 and 2)",
    )
    p.add_argument("--print-summary", action="store_true", help="Print counts to stderr")
    args = p.parse_args()

    if not args.input.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        return 1

    kept = 0
    skipped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, encoding="utf-8") as f_in, open(args.output, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = rec.get("quality") or {}
            tier = q.get("tier", 99)
            if tier > args.max_tier:
                skipped += 1
                continue
            out: dict = {k: rec[k] for k in CORE_FIELDS if k in rec}
            for k in OPTIONAL_FIELDS:
                if k in rec and rec[k] is not None:
                    out[k] = rec[k]
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1

    if args.print_summary:
        print(
            f"export_curated_from_scored: kept={kept} skipped_tier={skipped} max_tier={args.max_tier}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
