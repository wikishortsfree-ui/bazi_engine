"""
Tests for scripts/astrodatabank_to_bazi.py.
Validates raw → BaziChart conversion with true solar time and output schema.
"""

import json
import tempfile
from pathlib import Path

import pytest

# Run from repo root so scripts/ and bazi_engine are importable
import sys
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timezonefinder import TimezoneFinder

from scripts.astrodatabank_to_bazi import convert_row, main


def test_convert_row_produces_valid_chart():
    """Convert one raw record; chart round-trips and has expected pillars for known birth."""
    raw = {
        "name": "Test Person",
        "date": "24 April 2003",
        "time": "21:20",
        "latitude": 37.354107,
        "longitude": -121.955238,
        "location": "San Jose, CA",
        "rodden_rating": "A",
        "category": "Technology",
        "biography": "Short bio.",
        "source": "",
        "source_url": "https://example.com",
    }
    tz_finder = TimezoneFinder()
    record = convert_row(raw, tz_finder)
    assert record is not None
    assert record["name"] == "Test Person"
    assert record["biography"] == "Short bio."
    assert "chart" in record
    from bazi_engine.models import BaziChart
    chart = BaziChart.from_json(json.dumps(record["chart"], ensure_ascii=False))
    assert chart.year_pillar.stem.value + chart.year_pillar.branch.value == "癸未"
    assert chart.month_pillar.stem.value + chart.month_pillar.branch.value == "丙辰"
    assert chart.day_pillar.stem.value + chart.day_pillar.branch.value == "丁卯"
    assert chart.hour_pillar.stem.value + chart.hour_pillar.branch.value == "庚戌"


def test_convert_row_missing_time_uses_noon():
    """When time is missing, conversion uses noon (12:00) and still produces a chart."""
    raw = {
        "name": "No Time",
        "date": "1 January 1990",
        "time": "",
        "latitude": 39.9042,
        "longitude": 116.4074,
        "location": "Beijing",
        "rodden_rating": "A",
        "category": "Politics",
        "biography": "",
        "source": "",
        "source_url": "",
    }
    tz_finder = TimezoneFinder()
    record = convert_row(raw, tz_finder)
    assert record is not None
    assert "chart" in record
    meta = record["chart"].get("metadata", {})
    assert "birth_datetime" in meta
    assert "T12:00" in meta["birth_datetime"] or "T12:" in meta["birth_datetime"]


def test_convert_row_invalid_date_returns_none():
    """Invalid or missing date yields None."""
    raw = {
        "name": "Bad",
        "date": "not-a-date",
        "time": "12:00",
        "latitude": 0.0,
        "longitude": 0.0,
        "location": "",
        "rodden_rating": "A",
        "category": "",
        "biography": "",
        "source": "",
        "source_url": "",
    }
    tz_finder = TimezoneFinder()
    assert convert_row(raw, tz_finder) is None


def test_convert_row_missing_coords_returns_none():
    """Missing latitude or longitude yields None."""
    tz_finder = TimezoneFinder()
    raw = {
        "name": "No Coords",
        "date": "1 Jan 2000",
        "time": "12:00",
        "latitude": 40.0,
        "longitude": None,
        "location": "",
        "rodden_rating": "A",
        "category": "",
        "biography": "",
        "source": "",
        "source_url": "",
    }
    assert convert_row(raw, tz_finder) is None


def test_main_empty_input_produces_empty_output():
    """Running main on empty JSONL writes empty output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = Path(tmpdir) / "empty.jsonl"
        out = Path(tmpdir) / "out.jsonl"
        inp.write_text("")
        old_argv = sys.argv
        try:
            sys.argv = ["", "--input", str(inp), "--output", str(out)]
            main()
        finally:
            sys.argv = old_argv
        assert out.exists()
        assert out.read_text().strip() == ""


def test_main_one_valid_line_produces_one_output_line():
    """One valid raw line produces one Bazi JSONL line."""
    raw_line = json.dumps({
        "name": "One",
        "date": "15 August 1985",
        "time": "14:30",
        "latitude": 51.5074,
        "longitude": -0.1278,
        "location": "London",
        "rodden_rating": "A",
        "category": "Arts",
        "biography": "Bio",
        "source": "",
        "source_url": "",
    }, ensure_ascii=False) + "\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = Path(tmpdir) / "one.jsonl"
        out = Path(tmpdir) / "bazi.jsonl"
        inp.write_text(raw_line)
        old_argv = sys.argv
        try:
            sys.argv = ["", "--input", str(inp), "--output", str(out)]
            main()
        finally:
            sys.argv = old_argv
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["name"] == "One"
        assert "chart" in rec
        assert "metadata" in rec["chart"]
        assert "natal_pillars" in rec["chart"]
