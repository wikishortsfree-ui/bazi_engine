#!/usr/bin/env python3
"""
Chart selection for Phase 2 data curation (tasks 2.2.1, 2.2.1b, 2.2.2).

Selects charts from astrodatabank_bazi.jsonl with stratified priority sampling,
generates synthetic strong day-master charts for gap-fill, merges them,
and prints a distribution report on the final selected_charts.jsonl.

Phases:
  1. Load astrodatabank_bazi.jsonl, exclude no-time records.
  2. Include ALL strong day-master charts (strength >= threshold), fill remaining
     quota from weaker charts prioritizing AA Rodden rating + longer biography.
  3. Generate synthetic strong day-master charts via acceptance sampling.
  4. Merge, shuffle, write selected_charts.jsonl.
  5. Print distribution report.

Usage:
  python scripts/select_charts.py \\
    --input data/astrodatabank_bazi.jsonl \\
    --output data/selected_charts.jsonl \\
    --target-astro 9000 --synthetic-target 2000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from bazi_engine import BaziCalculator
from bazi_engine.strength import compute_day_master_strength

RODDEN_PRIORITY = {"AA": 0, "A": 1}
NO_TIME_INDICATORS = frozenset({"", "unknown", "?", "-", "none", "n/a"})

STRENGTH_BINS = [
    ("very_strong", 0.80, 1.01),
    ("strong", 0.65, 0.80),
    ("mid_strong", 0.50, 0.65),
    ("moderate", 0.30, 0.50),
    ("weak", 0.20, 0.30),
    ("very_weak", 0.00, 0.20),
]

BRANCHES_12 = "子丑寅卯辰巳午未申酉戌亥"
DAY_MASTERS_10 = "甲乙丙丁戊己庚辛壬癸"
INTERACTION_TYPES = [
    "stem_combinations",
    "stem_clashes",
    "branch_six_combinations",
    "branch_clashes",
    "three_harmony",
    "harms",
    "punishments",
    "destructions",
]
USEFUL_GOD_CATEGORIES = ["印", "比劫", "财", "食伤", "官杀"]

DATE_MIN, DATE_MAX = 1900, 2025
LAT_MIN, LAT_MAX = -60.0, 60.0
LNG_MIN, LNG_MAX = -180.0, 180.0
TZ_MIN, TZ_MAX = -12, 14


def _has_valid_time(record: dict) -> bool:
    time_str = (record.get("time") or "").strip().lower()
    return time_str not in NO_TIME_INDICATORS


def _get_strength(record: dict) -> float:
    chart = record.get("chart") or {}
    dm = chart.get("day_master") or {}
    return float(dm.get("strength_percentile", 0.0))


def _get_bio_length(record: dict) -> int:
    return len(record.get("biography") or "")


def _get_rodden_priority(record: dict) -> int:
    rating = (record.get("rodden_rating") or "").strip().upper()
    return RODDEN_PRIORITY.get(rating, 99)


def _strength_bin(strength: float) -> str:
    for label, lo, hi in STRENGTH_BINS:
        if lo <= strength < hi:
            return label
    return "very_weak"


# ---------------------------------------------------------------------------
# Phase 1+2: Astrodatabank stratified selection
# ---------------------------------------------------------------------------


def select_astrodatabank(
    records: list[dict],
    target: int,
    strong_threshold: float,
) -> list[dict]:
    """
    Include ALL charts with strength >= strong_threshold.
    Fill remaining slots from weaker charts: AA > A Rodden, then longer biography.
    """
    valid = [r for r in records if _has_valid_time(r)]
    excluded_no_time = len(records) - len(valid)
    print(f"  Excluded {excluded_no_time} records with no birth time")

    strong = [r for r in valid if _get_strength(r) >= strong_threshold]
    rest = [r for r in valid if _get_strength(r) < strong_threshold]

    print(f"  Strong day masters (>= {strong_threshold}): {len(strong)}")
    print(f"  Remaining pool: {len(rest)}")

    rest.sort(key=lambda r: (_get_rodden_priority(r), -_get_bio_length(r)))

    remaining_slots = max(0, target - len(strong))
    selected_rest = rest[:remaining_slots]

    selected = strong + selected_rest
    random.shuffle(selected)

    print(
        f"  Selected {len(strong)} strong + {len(selected_rest)} priority-ranked"
        f" = {len(selected)} total"
    )
    return selected


# ---------------------------------------------------------------------------
# Phase 3: Synthetic gap-fill via acceptance sampling
# ---------------------------------------------------------------------------


def _random_datetime(rng: random.Random) -> datetime:
    year = rng.randint(DATE_MIN, DATE_MAX)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    tz = timezone(timedelta(hours=rng.randint(TZ_MIN, TZ_MAX)))
    return datetime(year, month, day, hour, minute, second, tzinfo=tz)


def generate_synthetic_strong(
    target: int,
    strong_threshold: float,
    rng: random.Random,
    max_attempts: int,
) -> list[dict]:
    results: list[dict] = []
    attempts = 0

    while len(results) < target and attempts < max_attempts:
        attempts += 1
        dt = _random_datetime(rng)
        lat = round(rng.uniform(LAT_MIN, LAT_MAX), 6)
        lng = round(rng.uniform(LNG_MIN, LNG_MAX), 6)
        try:
            chart = BaziCalculator.from_datetime(dt, lat, lng)
        except Exception:
            continue

        pillars = [
            chart.year_pillar,
            chart.month_pillar,
            chart.day_pillar,
            chart.hour_pillar,
        ]
        stems = [p.stem.value for p in pillars]
        branches = [p.branch.value for p in pillars]
        strength = compute_day_master_strength(stems, branches)

        if strength >= strong_threshold:
            record = {
                "chart": chart._to_json_dict(),
                "selection_source": "synthetic",
                "latitude": lat,
                "longitude": lng,
            }
            results.append(record)
            if len(results) % 200 == 0:
                rate = len(results) / attempts * 100
                print(
                    f"  Generated {len(results)}/{target} strong charts"
                    f" ({attempts} attempts, {rate:.1f}% acceptance)",
                    end="\r",
                )

    rate = len(results) / max(1, attempts) * 100
    print(
        f"\n  Synthetic: {len(results)} charts from {attempts} attempts"
        f" ({rate:.1f}% acceptance rate)"
    )
    return results


# ---------------------------------------------------------------------------
# Distribution report
# ---------------------------------------------------------------------------


def print_distribution_report(records: list[dict]) -> str:
    """Build and print distribution report; returns report text."""
    n = len(records)
    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)

    p(f"\n{'='*60}")
    p(f"DISTRIBUTION REPORT — {n} charts")
    p(f"{'='*60}")

    sources = Counter(r.get("selection_source", "astrodatabank") for r in records)
    p("\n--- Source ---")
    for src, count in sources.most_common():
        p(f"  {src}: {count} ({100*count/n:.1f}%)")

    ratings = Counter(
        (r.get("rodden_rating") or "N/A").strip() or "N/A" for r in records
    )
    p("\n--- Rodden rating ---")
    for rating, count in ratings.most_common():
        p(f"  {rating}: {count} ({100*count/n:.1f}%)")

    dm_counts: Counter[str] = Counter()
    for r in records:
        dm_counts[((r.get("chart") or {}).get("day_master") or {}).get("stem", "?")] += 1
    p("\n--- Day master (天干) ---")
    for stem in DAY_MASTERS_10:
        c = dm_counts.get(stem, 0)
        p(f"  {stem}: {c} ({100*c/n:.1f}%)")

    mb_counts: Counter[str] = Counter()
    for r in records:
        np_ = (r.get("chart") or {}).get("natal_pillars") or {}
        mb_counts[(np_.get("month") or {}).get("branch", "?")] += 1
    p("\n--- Month branch (地支) ---")
    for branch in BRANCHES_12:
        c = mb_counts.get(branch, 0)
        p(f"  {branch}: {c} ({100*c/n:.1f}%)")

    strength_counts: Counter[str] = Counter()
    for r in records:
        strength_counts[_strength_bin(_get_strength(r))] += 1
    p("\n--- Day master strength ---")
    for label, lo, hi in STRENGTH_BINS:
        c = strength_counts.get(label, 0)
        p(f"  {label} ({lo:.2f}–{hi:.2f}): {c} ({100*c/n:.1f}%)")

    ug_counts: Counter[str] = Counter()
    for r in records:
        ug = (r.get("chart") or {}).get("useful_god")
        if ug and not ug.get("is_ambiguous", True):
            ug_counts[ug.get("primary", "unknown")] += 1
        else:
            ug_counts["ambiguous/none"] += 1
    p("\n--- Useful god (用神) ---")
    for cat in USEFUL_GOD_CATEGORIES + ["ambiguous/none"]:
        c = ug_counts.get(cat, 0)
        p(f"  {cat}: {c} ({100*c/n:.1f}%)")

    p("\n--- Interaction type presence ---")
    for itype in INTERACTION_TYPES:
        count = sum(
            1
            for r in records
            if ((r.get("chart") or {}).get("interactions") or {}).get(itype)
        )
        p(f"  {itype}: {count} ({100*count/n:.1f}%)")

    astro = [
        r for r in records if r.get("selection_source", "astrodatabank") == "astrodatabank"
    ]
    if astro:
        bio_lens = sorted(_get_bio_length(r) for r in astro)
        p(f"\n--- Biography length (astrodatabank, n={len(astro)}) ---")
        p(f"  min={bio_lens[0]}, median={bio_lens[len(bio_lens)//2]}, max={bio_lens[-1]}")
        for threshold in [200, 400, 800]:
            c = sum(1 for bl in bio_lens if bl >= threshold)
            p(f"  >= {threshold} chars: {c} ({100*c/len(astro):.1f}%)")

    p(f"\n{'='*60}")
    report = "\n".join(lines)
    print(report)
    return report


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select charts for Phase 2 data curation (tasks 2.2.1, 2.2.1b, 2.2.2).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/astrodatabank_bazi.jsonl"),
        help="Input astrodatabank JSONL (default: data/astrodatabank_bazi.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/selected_charts.jsonl"),
        help="Output selected charts JSONL (default: data/selected_charts.jsonl)",
    )
    parser.add_argument(
        "--target-astro",
        type=int,
        default=9000,
        help="Target count from astrodatabank (default: 9000)",
    )
    parser.add_argument(
        "--synthetic-target",
        type=int,
        default=2000,
        help="Target count of synthetic strong charts (default: 2000)",
    )
    parser.add_argument(
        "--strong-threshold",
        type=float,
        default=0.50,
        help="Day master strength threshold for 'strong' (default: 0.50)",
    )
    parser.add_argument(
        "--max-synthetic-attempts",
        type=int,
        default=100_000,
        help="Max random chart attempts for synthetic generation (default: 100000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Skip synthetic gap-fill (astrodatabank selection only)",
    )
    parser.add_argument(
        "--report-only",
        type=Path,
        default=None,
        metavar="PATH",
        help="Print distribution report for an existing JSONL and exit.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write distribution report to file (in addition to stdout).",
    )
    args = parser.parse_args()

    if args.report_only is not None:
        if not args.report_only.exists():
            print(f"Error: file not found: {args.report_only}", file=sys.stderr)
            return 1
        records = _load_jsonl(args.report_only)
        report = print_distribution_report(records)
        if args.report_output:
            args.report_output.parent.mkdir(parents=True, exist_ok=True)
            args.report_output.write_text(report, encoding="utf-8")
            print(f"\nReport written to {args.report_output}")
        return 0

    rng = random.Random(args.seed)
    random.seed(args.seed)

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    # Phase 1: Load
    print(f"Phase 1: Loading {args.input}...")
    all_records = _load_jsonl(args.input)
    print(f"  Loaded {len(all_records)} records")

    # Phase 2: Stratified selection
    print(f"\nPhase 2: Selecting up to {args.target_astro} astrodatabank charts...")
    selected_astro = select_astrodatabank(
        all_records, args.target_astro, args.strong_threshold
    )
    for r in selected_astro:
        r["selection_source"] = "astrodatabank"

    # Phase 3: Synthetic gap-fill
    synthetic: list[dict] = []
    if not args.no_synthetic and args.synthetic_target > 0:
        print(f"\nPhase 3: Generating {args.synthetic_target} synthetic strong charts...")
        synthetic = generate_synthetic_strong(
            args.synthetic_target,
            args.strong_threshold,
            rng,
            args.max_synthetic_attempts,
        )
    else:
        print("\nPhase 3: Skipped (--no-synthetic or target=0)")

    # Phase 4: Merge and write
    merged = selected_astro + synthetic
    rng.shuffle(merged)
    print(
        f"\nPhase 4: Merged {len(selected_astro)} astrodatabank"
        f" + {len(synthetic)} synthetic = {len(merged)} total"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in merged:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Written to {args.output}")

    # Phase 5: Distribution report
    report = print_distribution_report(merged)
    if args.report_output:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(report, encoding="utf-8")
        print(f"\nReport written to {args.report_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
