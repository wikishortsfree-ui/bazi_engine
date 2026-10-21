#!/usr/bin/env python3
"""
Batch compute BaziChart from a CSV of birth dates/locations using multiprocessing.

Reads dates.csv (from generate_training_charts.py or compatible CSV), computes one
BaziChart per row via BaziCalculator, optionally adds temporal context (大运/流年/流月)
when flowing_year and gender are present, and writes one JSON line per chart to
data/charts.jsonl.

CSV columns:
  - birth_datetime (YYYY-MM-DD HH:MM:SS) OR birth_date + birth_time
  - timezone_offset (int hours)
  - lat, lng (floats; optional, defaults used if missing)
  - gender (optional; for dynamic: male/female or 1/0)
  - flowing_year (optional; when set with gender, temporal context is computed for this year)
  - flowing_month (optional; 1–12 for 流月; when missing, defaults to 6 for backward compatibility)

Usage:
  python scripts/compute_charts.py --input dates.csv --output data/charts.jsonl
  python scripts/compute_charts.py --input dates.csv --output data/charts.jsonl --workers 8 --skip-errors
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Add project root for imports when run as script
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from bazi_engine import BaziCalculator
from bazi_engine.models import BaziChart, _compute_useful_god
from bazi_engine.strength import compute_day_master_strength
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
            try:
                dt = datetime.strptime(s, "%Y-%m-%d %H:%M")
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


def _row_to_datetime(row: dict[str, Any]) -> datetime:
    """Build timezone-aware datetime from row (birth_datetime or birth_date + birth_time)."""
    birth_str = row.get("birth_datetime") or row.get("birth_datetime_utc")
    if birth_str:
        offset = int(row.get("timezone_offset", 8))
        return _parse_datetime(birth_str.strip(), offset)
    birth_date = row.get("birth_date", "").strip()
    birth_time = row.get("birth_time", "00:00:00").strip()
    if not birth_date:
        raise ValueError("Missing birth_datetime or birth_date")
    offset = int(row.get("timezone_offset", 8))
    if len(birth_time) <= 5:  # HH:MM
        birth_time = f"{birth_time}:00"
    combined = f"{birth_date} {birth_time}"
    return _parse_datetime(combined, offset)


def _process_one_row(row: dict[str, Any], row_index: int) -> tuple[int, str | None, str | None]:
    """
    Build BaziChart for one row. Returns (row_index, json_line, error_message).
    For multiprocessing: must be a top-level function for pickling.
    """
    try:
        dt = _row_to_datetime(row)
        lat = float(row.get("lat", DEFAULT_LAT))
        lng = float(row.get("lng", DEFAULT_LNG))
        chart = BaziCalculator.from_datetime(dt, lat, lng)

        flowing_year_str = row.get("flowing_year", "").strip()
        gender_str = (row.get("gender") or "").strip()
        if flowing_year_str and gender_str:
            try:
                flowing_year = int(flowing_year_str)
                gender = _gender_to_int(gender_str)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid flowing_year or gender: {e}") from e
            flowing_month_str = (row.get("flowing_month") or "").strip()
            if flowing_month_str:
                try:
                    m = int(flowing_month_str)
                    flowing_month = m if 1 <= m <= 12 else 6
                except (ValueError, TypeError):
                    flowing_month = 6
            else:
                flowing_month = 6  # backward compat when only flowing_year is given
            dayun_seq = BaziCalculator().compute_dayun(chart, gender, max_periods=10)
            dayun = dayun_seq.get_period_for_year(flowing_year)
            liunian = compute_liunian(
                flowing_year, day_master_stem=chart.day_pillar.stem.value
            )
            liuyue = compute_liuyue(
                flowing_year,
                flowing_month,
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
            dm_pct = compute_day_master_strength(stems, branches)
            ug_dict = _compute_useful_god(dm_pct, chart.day_pillar.stem.element)
            ug_element = ug_dict.get("primary_element") if ug_dict else None
            cross = detect_cross_interactions(stems, branches, ctx, useful_god=ug_element)
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
        json_line = chart.to_json(indent=None)
        return (row_index, json_line, None)
    except Exception as e:
        return (row_index, None, str(e))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute BaziChart JSONL from CSV using multiprocessing."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV (e.g. dates.csv from generate_training_charts.py)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL (e.g. data/charts.jsonl)",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip rows that fail and continue; report count at end. Default is fail-fast.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of worker processes (default: CPU count - 1, or 1 if 1 CPU)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(input_path.open(newline="", encoding="utf-8")))
    if not rows:
        print("Error: CSV has no data rows", file=sys.stderr)
        return 1

    has_datetime = "birth_datetime" in rows[0] or "birth_datetime_utc" in rows[0]
    has_date_time = "birth_date" in rows[0]
    if not has_datetime and not has_date_time:
        print(
            "Error: CSV must have birth_datetime (or birth_datetime_utc) or birth_date",
            file=sys.stderr,
        )
        return 1

    workers = args.workers
    if workers is None:
        import multiprocessing
        workers = max(1, multiprocessing.cpu_count() - 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    from multiprocessing import Pool

    task_args = [(row, i) for i, row in enumerate(rows)]
    written = 0
    errors: list[tuple[int, str]] = []

    with Pool(processes=workers) as pool:
        results = pool.starmap(_process_one_row, task_args)

    for row_index, json_line, err in results:
        if err is not None:
            errors.append((row_index + 2, err))  # +2 for 1-based + header
            if not args.skip_errors:
                print(f"Row {row_index + 2}: {err}", file=sys.stderr)
                return 1

    # Write output in original row order (results are in order by row_index)
    with output_path.open("w", encoding="utf-8") as f:
        for row_index, json_line, err in tqdm(
            results, desc="Writing charts", unit="chart", file=sys.stderr
        ):
            if err is None and json_line is not None:
                f.write(json_line + "\n")
                written += 1

    if errors:
        for row_num, msg in errors:
            print(f"Row {row_num}: {msg}", file=sys.stderr)
        print(f"Skipped {len(errors)} rows due to errors.", file=sys.stderr)
    print(f"Wrote {written} charts to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
