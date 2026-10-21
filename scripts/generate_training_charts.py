#!/usr/bin/env python3
"""
Generate a CSV of random birth date/time and location rows for training chart computation.

Output columns: birth_datetime, timezone_offset, lat, lng [, gender, flowing_year, flowing_month]
Compatible with scripts/batch_generate_charts.py and scripts/compute_charts.py.

- birth_datetime: YYYY-MM-DD HH:MM:SS (no timezone; timezone_offset is separate).
- timezone_offset: integer hours (e.g. -12 to 14).
- lat, lng: latitude/longitude (lat ±60 to avoid polar edge cases).
- gender: optional; when present with flowing_year, used for 大运 (male/female).
- flowing_year: optional; year for temporal context (大运/流年/流月).
- flowing_month: optional; month 1–12 for 流月 when flowing_year is set; improves 流月 diversity.

Use --with-dynamic-fraction to add gender, flowing_year, and flowing_month for a fraction of rows
so that charts.jsonl will have a temporal subset for the cross-temporal head.

Use --stratified to ensure equal representation across (month, hour) bins (288 bins),
then shuffle; improves Phase 1 data quality for encoder training.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root for imports when run as script (e.g. for --validate)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Default ranges (documented for reproducibility)
DATE_MIN = 1900
DATE_MAX = 2025
TZ_OFFSET_MIN = -12
TZ_OFFSET_MAX = 14
LAT_MIN = -60
LAT_MAX = 60
LNG_MIN = -180
LNG_MAX = 180

GENDERS = ("male", "female")

# Day master labels for validation (天干 cycle order)
DAY_MASTER_LABELS = "甲乙丙丁戊己庚辛壬癸"


def _row_to_datetime(row: dict[str, str | int | float]) -> datetime:
    """Parse birth_datetime + timezone_offset from a CSV row (same format we write)."""
    s = str(row["birth_datetime"]).strip()
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    offset = int(row.get("timezone_offset", 8))
    return dt.replace(tzinfo=timezone(timedelta(hours=offset)))


def validate_day_master_distribution(dates_csv: Path) -> None:
    """Check that all 10 day masters (甲-癸) appear with ~equal frequency."""
    from bazi_engine import BaziCalculator

    day_masters: list[str] = []
    with dates_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            dt = _row_to_datetime(row)
            lat = float(row["lat"])
            lng = float(row["lng"])
            chart = BaziCalculator.from_datetime(dt, lat, lng)
            day_masters.append(chart.day_pillar.stem.value)

    if not day_masters:
        print("No rows in CSV; cannot validate day master distribution.")
        return

    counts = Counter(day_masters)
    total = len(day_masters)
    print("Day Master Distribution:")
    for stem in DAY_MASTER_LABELS:
        n = counts.get(stem, 0)
        pct = 100 * n / total
        print(f"  {stem}: {n:,} ({pct:.1f}%)")

    min_pct = min(counts.get(s, 0) / total for s in DAY_MASTER_LABELS)
    max_pct = max(counts.get(s, 0) / total for s in DAY_MASTER_LABELS)
    if min_pct < 0.08 or max_pct > 0.12:
        print(
            f"⚠️  Warning: Day master distribution skewed (min {min_pct:.1%}, max {max_pct:.1%})"
        )
    else:
        print("✅ Day master distribution is balanced")


def _random_datetime(rng: random.Random) -> str:
    """Sample a datetime in DATE_MIN--DATE_MAX, 00:00:00--23:59:59."""
    year = rng.randint(DATE_MIN, DATE_MAX)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)  # avoid month-end edge cases
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"


def _datetime_in_bin(rng: random.Random, month: int, hour: int) -> str:
    """Sample a datetime with fixed month and hour; year, day, minute, second random."""
    year = rng.randint(DATE_MIN, DATE_MAX)
    day = rng.randint(1, 28)  # avoid month-end edge cases
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"


def _random_timezone_offset(rng: random.Random) -> int:
    return rng.randint(TZ_OFFSET_MIN, TZ_OFFSET_MAX)


def _random_lat(rng: random.Random) -> float:
    return round(rng.uniform(LAT_MIN, LAT_MAX), 6)


def _random_lng(rng: random.Random) -> float:
    return round(rng.uniform(LNG_MIN, LNG_MAX), 6)


def _random_flowing_year(rng: random.Random) -> int:
    """Year for temporal context; keep within same era as date range."""
    return rng.randint(DATE_MIN, DATE_MAX)


def _stratified_month_hour_bins(num_samples: int, rng: random.Random) -> list[tuple[int, int]]:
    """Return num_samples (month, hour) pairs with one per bin (1–12, 0–23), then shuffled."""
    bins = [(m, h) for m in range(1, 13) for h in range(24)]  # 288 bins
    per_bin, remainder = divmod(num_samples, len(bins))
    # Assign counts: first `remainder` bins get per_bin+1, rest get per_bin
    pairs: list[tuple[int, int]] = []
    for i, (m, h) in enumerate(bins):
        n = per_bin + (1 if i < remainder else 0)
        pairs.extend([(m, h)] * n)
    rng.shuffle(pairs)
    return pairs


def generate_rows(
    num_samples: int,
    *,
    rng: random.Random,
    with_dynamic_fraction: float = 0.0,
    stratified: bool = False,
) -> list[dict[str, str | int | float]]:
    """
    Generate num_samples random rows.

    If with_dynamic_fraction > 0, that fraction of rows get gender, flowing_year, and flowing_month set;
    the rest have empty gender, flowing_year, and flowing_month.

    If stratified is True, datetimes are sampled so each (month, hour) bin gets
    equal representation (12 months × 24 hours = 288 bins), then rows are shuffled.
    """
    rows: list[dict[str, str | int | float]] = []
    num_dynamic = int(round(num_samples * with_dynamic_fraction)) if with_dynamic_fraction > 0 else 0
    dynamic_indices = set(rng.sample(range(num_samples), min(num_dynamic, num_samples)))

    if stratified:
        month_hour_pairs = _stratified_month_hour_bins(num_samples, rng)
        for i, (month, hour) in enumerate(month_hour_pairs):
            row: dict[str, str | int | float] = {
                "birth_datetime": _datetime_in_bin(rng, month, hour),
                "timezone_offset": _random_timezone_offset(rng),
                "lat": _random_lat(rng),
                "lng": _random_lng(rng),
            }
            if i in dynamic_indices:
                row["gender"] = rng.choice(GENDERS)
                row["flowing_year"] = _random_flowing_year(rng)
                row["flowing_month"] = rng.randint(1, 12)
            else:
                row["gender"] = ""
                row["flowing_year"] = ""
                row["flowing_month"] = ""
            rows.append(row)
        rng.shuffle(rows)
    else:
        for i in range(num_samples):
            row = {
                "birth_datetime": _random_datetime(rng),
                "timezone_offset": _random_timezone_offset(rng),
                "lat": _random_lat(rng),
                "lng": _random_lng(rng),
            }
            if i in dynamic_indices:
                row["gender"] = rng.choice(GENDERS)
                row["flowing_year"] = _random_flowing_year(rng)
                row["flowing_month"] = rng.randint(1, 12)
            else:
                row["gender"] = ""
                row["flowing_year"] = ""
                row["flowing_month"] = ""
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a CSV of random birth date/time and location for training chart computation.",
        epilog="Output is compatible with batch_generate_charts.py and compute_charts.py.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        metavar="N",
        help="Number of rows to generate (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dates.csv"),
        metavar="PATH",
        help="Output CSV path (default: dates.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--with-dynamic-fraction",
        type=float,
        default=0.0,
        metavar="F",
        help="Fraction of rows (0.0--1.0) to add gender, flowing_year, and flowing_month for temporal subset (e.g. 0.2)",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Stratify by (month, hour): 288 bins get equal counts, then shuffle. Use for Phase 1 data quality.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="After writing CSV, validate day master (天干) distribution; expect ~10%% each ±2%%.",
    )
    parser.add_argument(
        "--validate-only",
        type=Path,
        metavar="PATH",
        default=None,
        help="Only run day master distribution validation on existing CSV; then exit.",
    )
    args = parser.parse_args()

    if args.validate_only is not None:
        if not args.validate_only.exists():
            print(f"Error: File not found: {args.validate_only}", file=sys.stderr)
            return 1
        validate_day_master_distribution(args.validate_only)
        return 0

    if args.num_samples < 1:
        print("Error: --num-samples must be >= 1", file=sys.stderr)
        return 1
    if not 0 <= args.with_dynamic_fraction <= 1:
        print("Error: --with-dynamic-fraction must be between 0 and 1", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rows = generate_rows(
        args.num_samples,
        rng=rng,
        with_dynamic_fraction=args.with_dynamic_fraction,
        stratified=args.stratified,
    )

    fieldnames = ["birth_datetime", "timezone_offset", "lat", "lng", "gender", "flowing_year", "flowing_month"]
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"Wrote {len(rows)} rows to {out_path}", file=sys.stderr)
    if args.validate:
        validate_day_master_distribution(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
