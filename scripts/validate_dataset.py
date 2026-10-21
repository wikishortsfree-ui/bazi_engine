#!/usr/bin/env python3
"""
Validate dataset diversity and quality.

Reads a JSONL chart file (each line = BaziChart.to_json()) and reports:
- Total charts and share with temporal (dynamic) data
- Day master (天干) distribution
- Month branch (地支) distribution
- Balance checks for day master and month branch

Usage:
  python scripts/validate_dataset.py path/to/charts.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

STEMS = "甲乙丙丁戊己庚辛壬癸"
BRANCHES = "子丑寅卯辰巳午未申酉戌亥"


def validate_dataset(path: Path) -> bool:
    """Check day master, month, and temporal distributions. Returns False if file is empty."""
    day_masters: list[str] = []
    months: list[str] = []
    has_temporal = 0
    total = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            chart = json.loads(line)
            # BaziChart JSON: day_master.stem, natal_pillars.month.branch
            day_masters.append(chart["day_master"]["stem"])
            months.append(chart["natal_pillars"]["month"]["branch"])
            if chart.get("dynamic"):
                has_temporal += 1
            total += 1

    if total == 0:
        print("No charts in file.")
        return False

    print(f"Total charts: {total:,}")
    print(f"Temporal charts: {has_temporal:,} ({100 * has_temporal / total:.1f}%)")

    print("\nDay Master Distribution:")
    dm_counts = Counter(day_masters)
    for stem in STEMS:
        count = dm_counts.get(stem, 0)
        pct = 100 * count / total
        print(f"  {stem}: {count:,} ({pct:.1f}%)")

    print("\nMonth Branch Distribution:")
    month_counts = Counter(months)
    for branch in BRANCHES:
        count = month_counts.get(branch, 0)
        pct = 100 * count / total
        print(f"  {branch}: {count:,} ({pct:.1f}%)")

    # Balance: expect ~10% per stem, ~8.33% per branch
    dm_min = min(dm_counts.values()) / total
    dm_max = max(dm_counts.values()) / total
    month_min = min(month_counts.values()) / total
    month_max = max(month_counts.values()) / total

    print(f"\nDay master range: {dm_min:.1%} - {dm_max:.1%}")
    print(f"Month branch range: {month_min:.1%} - {month_max:.1%}")

    if dm_min > 0.08 and dm_max < 0.12:
        print("✅ Day master distribution is balanced")
    else:
        print("⚠️ Day master distribution may be skewed")

    if month_min > 0.06 and month_max < 0.11:
        print("✅ Month branch distribution is balanced")
    else:
        print("⚠️ Month branch distribution may be skewed")

    return True


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_dataset.py <path/to/charts.jsonl>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1
    if not validate_dataset(path):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
