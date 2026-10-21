"""
Combined scoring + tier pipeline for Phase 2 curated records.

Runs factual_check and language_quality_check on each record, computes a
composite score (starts at 100, deducts per issue), assigns a tier
(1: >=80, 2: 60-79, 3: <60), and writes scored JSONL.

Usage:
  python scripts/phase2_quality_filter.py -i data/qwen_raw.jsonl -o data/synthetic_scored.jsonl
  python scripts/phase2_quality_filter.py -i data/qwen_raw.jsonl --print-summary
  python scripts/phase2_quality_filter.py -i data/qwen_raw.jsonl -o data/synthetic_scored.jsonl --tier2-list data/tier2_list.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.factual_check import check_factual_consistency
from scripts.language_quality_check import check_language_quality

DEDUCTIONS: dict[str, int] = {
    "chart_parse": 100,
    "phantom_ten_god": 8,
    "composite_ten_god": 5,
    "strength_mismatch": 15,
    "phantom_interaction": 8,
    "useful_god_mismatch": 15,
    "short_reasoning": 10,
    "short_output": 10,
    "generic_phrases": 5,
    "thin_terminology": 5,
    "repetition": 10,
}

DEFAULT_DEDUCTION = 5

TIER_THRESHOLDS = (80, 60)


def compute_score(issues: list[dict[str, str]]) -> int:
    score = 100
    for issue in issues:
        score -= DEDUCTIONS.get(issue["type"], DEFAULT_DEDUCTION)
    return max(score, 0)


def assign_tier(score: int) -> int:
    if score >= TIER_THRESHOLDS[0]:
        return 1
    if score >= TIER_THRESHOLDS[1]:
        return 2
    return 3


def score_record(
    record: dict[str, Any],
    *,
    check_interactions: bool = True,
    min_reasoning_chars: int = 100,
    min_output_chars: int = 200,
    max_generic_hits: int = 8,
    min_classical_terms: int = 2,
) -> dict[str, Any]:
    """Run both QC checks, compute score and tier, return annotated record."""
    # 流年/流月 text naturally references cross-temporal interactions (natal↔temporal)
    # that don't exist in the natal chart; skip natal interaction checks for these.
    task_type = record.get("task_type", "")
    effective_check_interactions = check_interactions and task_type not in ("流年", "流月")
    factual = check_factual_consistency(record, check_interactions=effective_check_interactions)
    language = check_language_quality(
        record,
        min_reasoning_chars=min_reasoning_chars,
        min_output_chars=min_output_chars,
        max_generic_hits=max_generic_hits,
        min_classical_terms=min_classical_terms,
    )
    all_issues = factual + language
    score = compute_score(all_issues)
    tier = assign_tier(score)

    out = dict(record)
    out["factual_issues"] = factual
    out["language_issues"] = language
    out["factual_ok"] = len(factual) == 0
    out["language_ok"] = len(language) == 0
    out["quality"] = {
        "score": score,
        "tier": tier,
        "factual_issue_count": len(factual),
        "language_issue_count": len(language),
    }
    return out


def _print_summary(
    total: int,
    tier_counts: dict[int, int],
    issue_counts: dict[str, int],
    score_sum: int,
    score_min: int,
    score_max: int,
) -> None:
    avg = score_sum / total if total else 0
    lines = [
        f"phase2_quality_filter: total={total}",
        f"  tiers: T1={tier_counts.get(1, 0)} T2={tier_counts.get(2, 0)} T3={tier_counts.get(3, 0)}",
        f"  score: min={score_min} max={score_max} avg={avg:.1f}",
        f"  issues: {dict(sorted(issue_counts.items()))}",
    ]
    print("\n".join(lines), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Combined QC scoring + tier pipeline for Phase 2 records.")
    p.add_argument("--input", "-i", type=Path, required=True, help="Input JSONL path")
    p.add_argument("--output", "-o", type=Path, default=None, help="Scored output JSONL (default: stdout)")
    p.add_argument(
        "--tier2-list",
        type=Path,
        default=None,
        help="Write Tier 2 records separately for Claude regeneration",
    )
    p.add_argument("--print-summary", action="store_true", help="Print summary stats to stderr")
    p.add_argument("--no-interactions", action="store_true", help="Skip natal interaction phantom checks")
    p.add_argument("--min-reasoning", type=int, default=100)
    p.add_argument("--min-output", type=int, default=200)
    p.add_argument("--max-generic", type=int, default=8)
    p.add_argument("--min-classical", type=int, default=2)
    args = p.parse_args(argv)

    total = 0
    tier_counts: dict[int, int] = {}
    issue_counts: dict[str, int] = {}
    score_sum = 0
    score_min = 101
    score_max = -1

    out_f = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    tier2_f = open(args.tier2_list, "w", encoding="utf-8") if args.tier2_list else None
    try:
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                scored = score_record(
                    rec,
                    check_interactions=not args.no_interactions,
                    min_reasoning_chars=args.min_reasoning,
                    min_output_chars=args.min_output,
                    max_generic_hits=args.max_generic,
                    min_classical_terms=args.min_classical,
                )
                total += 1
                tier = scored["quality"]["tier"]
                score = scored["quality"]["score"]
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
                score_sum += score
                score_min = min(score_min, score)
                score_max = max(score_max, score)
                for issue in scored["factual_issues"] + scored["language_issues"]:
                    issue_counts[issue["type"]] = issue_counts.get(issue["type"], 0) + 1

                out_f.write(json.dumps(scored, ensure_ascii=False) + "\n")
                if tier2_f and tier == 2:
                    tier2_f.write(json.dumps(scored, ensure_ascii=False) + "\n")
    finally:
        if args.output:
            out_f.close()
        if tier2_f:
            tier2_f.close()

    if total == 0:
        score_min = 0
        score_max = 0

    if args.print_summary:
        _print_summary(total, tier_counts, issue_counts, score_sum, score_min, score_max)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
