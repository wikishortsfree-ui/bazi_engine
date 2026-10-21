#!/usr/bin/env python3
"""
Batch generate BaziChart JSONL from a CSV of birth datetimes.

CSV columns: birth_datetime, timezone_offset, gender [, lat, lng]
Output: one JSON line per chart (compact, no indentation).

Usage:
  python scripts/batch_generate_charts.py --input birth_dates.csv --output charts.jsonl
  python scripts/batch_generate_charts.py --input birth_dates.csv --output charts.jsonl --include-dynamic --current-date 2024-03-15
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tqdm import tqdm

# Add project root for imports when run as script
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from bazi_engine import BaziCalculator
from bazi_engine.models import BaziChart
from bazi_engine.temporal import (
    TemporalContext,
    compute_liunian,
    compute_liuyue,
    detect_cross_interactions,
)

DEFAULT_LAT, DEFAULT_LNG = 39.9, 116.4


def _parse_datetime(s: str, offset_hours: int) -> datetime:
    """Parse 'YYYY-MM-DD HH:MM:SS' (or ISO) and attach timezone from offset."""
    s = s.strip()
    if "T" in s:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    else:
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            dt = datetime.strptime(s, "%Y-%m-%d")
    tz = timezone(timedelta(hours=offset_hours))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt


def _gender_to_int(g: str) -> int:
    """'male' -> 1, 'female' -> 0."""
    g = (g or "").strip().lower()
    if g in ("male", "m", "1"):
        return 1
    if g in ("female", "f", "0"):
        return 0
    raise ValueError(f"Invalid gender: {g!r}")


def _process_row(
    row: dict,
    current_date: datetime.date | None,
    include_dynamic: bool,
) -> str | None:
    """
    Build BaziChart for one row; return compact JSON string or None on error.
    """
    try:
        birth_str = row.get("birth_datetime") or row.get("birth_datetime_utc")
        if not birth_str:
            raise ValueError("Missing birth_datetime")
        offset = int(row.get("timezone_offset", 8))
        lat = float(row.get("lat", DEFAULT_LAT))
        lng = float(row.get("lng", DEFAULT_LNG))
        dt = _parse_datetime(birth_str, offset)
        chart = BaziCalculator.from_datetime(dt, lat, lng)
        if include_dynamic and current_date is not None:
            gender = _gender_to_int(row.get("gender", "male"))
            dayun_seq = BaziCalculator().compute_dayun(chart, gender, max_periods=10)
            dayun = dayun_seq.get_period_for_year(current_date.year)
            liunian = compute_liunian(
                current_date.year, day_master_stem=chart.day_pillar.stem.value
            )
            liuyue = compute_liuyue(
                current_date.year,
                current_date.month,
                day_master_stem=chart.day_pillar.stem.value,
            )
            ctx = TemporalContext(dayun=dayun, liunian=liunian, liuyue=liuyue, liuri=None)
            stems = [
                chart.year_pillar.stem.value,
                chart.month_pillar.stem.value,
                chart.day_pillar.stem.value,
                chart.hour_pillar.stem.value,
            ]
            branches = [
                chart.year_pillar.branch.value,
                chart.month_pillar.branch.value,
                chart.day_pillar.branch.value,
                chart.hour_pillar.branch.value,
            ]
            cross = detect_cross_interactions(stems, branches, ctx, useful_god=None)
            chart = BaziChart(
                birth_dt=chart.birth_dt,
                solar_dt=chart.solar_dt,
                lat=chart.lat,
                lng=chart.lng,
                year_pillar=chart.year_pillar,
                month_pillar=chart.month_pillar,
                day_pillar=chart.day_pillar,
                hour_pillar=chart.hour_pillar,
                sect=chart.sect,
                temporal_context=ctx,
                cross_interactions=cross,
            )
        return chart.to_json(indent=None)
    except Exception as e:
        raise RuntimeError(f"Row error: {e}") from e


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate BaziChart JSONL from CSV of birth datetimes."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument(
        "--include-dynamic",
        action="store_true",
        help="Compute 大运/流年/流月 for --current-date",
    )
    parser.add_argument(
        "--current-date",
        default=None,
        help="Date for dynamic calculation (YYYY-MM-DD); default: today",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip rows that fail; default is fail-fast",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    current_date = None
    if args.include_dynamic:
        if args.current_date:
            try:
                current_date = datetime.strptime(args.current_date, "%Y-%m-%d").date()
            except ValueError:
                print(f"Error: invalid --current-date {args.current_date!r}", file=sys.stderr)
                return 1
        else:
            current_date = datetime.now(timezone.utc).date()

    rows = list(csv.DictReader(input_path.open(newline="", encoding="utf-8")))
    required = {"birth_datetime"}
    if not rows:
        print("Error: CSV has no data rows", file=sys.stderr)
        return 1
    for col in required:
        if col not in rows[0] and "birth_datetime_utc" not in rows[0]:
            print(f"Error: CSV must have column {col}", file=sys.stderr)
            return 1

    iter_rows = tqdm(rows, desc="Charts", unit="chart", file=sys.stderr)
    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(iter_rows):
            try:
                line = _process_row(row, current_date, args.include_dynamic)
                if line is not None:
                    f.write(line + "\n")
                    written += 1
            except Exception as e:
                if args.skip_errors:
                    print(f"Row {i + 2}: {e}", file=sys.stderr)
                else:
                    print(f"Row {i + 2}: {e}", file=sys.stderr)
                    return 1

    print(f"Wrote {written} charts to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
