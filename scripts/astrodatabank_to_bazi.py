"""
Convert Astrodatabank raw JSONL to BaziChart JSONL with true solar time.

Reads data/astrodatabank_raw.jsonl (name, date, time, longitude, latitude, ...),
resolves timezone from (lat, lon) via timezonefinder, localizes birth datetime,
builds BaziChart via BaziCalculator.from_datetime (which uses true solar time),
and writes data/astrodatabank_bazi.jsonl with chart + metadata per line.

Usage:
  python scripts/astrodatabank_to_bazi.py --input data/astrodatabank_raw.jsonl --output data/astrodatabank_bazi.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dateutil import parser as dateutil_parser

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo

from bazi_engine.calculator import BaziCalculator


def _parse_date(s: str) -> datetime | None:
    """Parse date string; return naive datetime at 00:00:00 or None."""
    if not s or not s.strip():
        return None
    s = s.strip()
    try:
        dt = dateutil_parser.parse(s, default=datetime(2000, 1, 1))
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    except (ValueError, TypeError):
        return None


def _parse_time(s: str) -> tuple[int, int, int] | None:
    """Parse time string to (hour, minute, second); return None if missing/invalid."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip().lower()
    if s in ("unknown", "?", "-", ""):
        return None
    try:
        dt = dateutil_parser.parse(s, default=datetime(2000, 1, 1))
        return (dt.hour, dt.minute, dt.second)
    except (ValueError, TypeError):
        return None


def _build_birth_datetime(
    date_str: str,
    time_str: str,
    tz: ZoneInfo,
    *,
    default_noon_if_no_time: bool = True,
) -> datetime | None:
    """
    Build timezone-aware datetime from raw date/time strings and timezone.

    If time is missing and default_noon_if_no_time is True, use 12:00:00.
    """
    dt_date = _parse_date(date_str)
    if dt_date is None:
        return None
    time_parts = _parse_time(time_str)
    if time_parts is not None:
        hour, minute, second = time_parts
    elif default_noon_if_no_time:
        hour, minute, second = 12, 0, 0
    else:
        return None
    naive = dt_date.replace(hour=hour, minute=minute, second=second, microsecond=0)
    return naive.replace(tzinfo=tz)


def convert_row(raw: dict, tz_finder: TimezoneFinder) -> dict | None:
    """
    Convert one raw Astrodatabank record to BaziChart + metadata, or None if skipped.

    Resolves timezone from (latitude, longitude), localizes birth datetime,
    builds BaziChart via BaziCalculator.from_datetime, returns dict with
    chart (BaziChart JSON dict), name, biography, and other preserved fields.
    """
    lat = raw.get("latitude")
    lng = raw.get("longitude")
    if lat is None or lng is None:
        return None
    try:
        lat_f, lng_f = float(lat), float(lng)
    except (TypeError, ValueError):
        return None
    if not (-90 <= lat_f <= 90 and -180 <= lng_f <= 180):
        return None

    tz_name = tz_finder.timezone_at(lng=lng_f, lat=lat_f)
    if tz_name is None:
        return None
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        return None

    dt = _build_birth_datetime(
        raw.get("date") or "",
        raw.get("time") or "",
        tz,
        default_noon_if_no_time=True,
    )
    if dt is None:
        return None

    try:
        chart = BaziCalculator.from_datetime(dt, lat_f, lng_f)
    except (ValueError, Exception):
        return None

    chart_dict = chart._to_json_dict()
    return {
        "chart": chart_dict,
        "name": raw.get("name", ""),
        "biography": raw.get("biography", ""),
        "date": raw.get("date", ""),
        "time": raw.get("time", ""),
        "location": raw.get("location", ""),
        "latitude": lat_f,
        "longitude": lng_f,
        "rodden_rating": raw.get("rodden_rating", ""),
        "category": raw.get("category", ""),
        "source": raw.get("source", ""),
        "source_url": raw.get("source_url", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Astrodatabank raw JSONL to BaziChart JSONL (true solar time)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/astrodatabank_raw.jsonl",
        help="Input raw JSONL path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/astrodatabank_bazi.jsonl",
        help="Output BaziChart JSONL path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tz_finder = TimezoneFinder()
    converted = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as f_in:
        with open(output_path, "w", encoding="utf-8") as f_out:
            for i, line in enumerate(f_in):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                record = convert_row(raw, tz_finder)
                if record is None:
                    skipped += 1
                    continue
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                converted += 1
                if (converted + skipped) % 100 == 0:
                    print(f"Processed {converted + skipped} rows, converted {converted}...", end="\r")

    print(f"\nDone: {converted} converted, {skipped} skipped. Output: {output_path}")


if __name__ == "__main__":
    main()
