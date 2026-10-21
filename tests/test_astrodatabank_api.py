"""
Tests for scripts/astrodatabank_api.py.
Validates coordinate parsing and output schema expected by phase2 data plan.
"""

import json
import tempfile
from unittest.mock import patch

import pytest

# Import from script (same process)
from scripts.astrodatabank_api import (
    parse_latitude,
    parse_longitude,
    collect_astrodatabank_dataset,
    save_and_report,
)


def test_parse_longitude():
    assert parse_longitude("122w25") == pytest.approx(-122.4167, abs=1e-3)
    assert parse_longitude("116e23") == pytest.approx(116.3833, abs=1e-3)
    assert parse_longitude("0e0") == 0.0
    assert parse_longitude("") is None
    assert parse_longitude("invalid") is None
    assert parse_longitude("122w60") is None  # minutes >= 60
    assert parse_longitude("200e0") is None   # > 180
    assert parse_longitude("181w0") is None   # > 180


def test_parse_latitude():
    assert parse_latitude("37n47") == pytest.approx(37.7833, abs=1e-3)
    assert parse_latitude("31s56") == pytest.approx(-31.9333, abs=1e-3)
    assert parse_latitude("") is None
    assert parse_latitude("37x47") is None  # invalid direction
    assert parse_latitude("91n0") is None   # > 90
    assert parse_latitude("37n60") is None   # minutes >= 60


def test_collect_output_schema_phase2():
    """Output must match phase2 plan: name, date, time, coords, location, rodden, category, biography, source, source_url."""
    category_members = {
        "query": {
            "categorymembers": [
                {"title": "Steve_Jobs"},
            ]
        }
    }
    birth_data = {
        "name": "Steve Jobs",
        "date": "24 February 1955",
        "time": "19:15",
        "place": "San Francisco, California",
        "long": "122w25",
        "lat": "37n47",
        "rodden": "AA",
        "source": "BC/BR in hand",
    }
    bio_html = "<p>American entrepreneur.</p>"

    def mock_api_get(params, verify_ssl=True):
        if params.get("list") == "categorymembers":
            return category_members
        if params.get("prop") == "wikitext":
            wikitext = "".join(f"|{k}={v}\n" for k, v in birth_data.items())
            return {"parse": {"wikitext": {"*": wikitext}}}
        if params.get("prop") == "text":
            return {"parse": {"text": {"*": bio_html}}}
        return {}

    with patch("scripts.astrodatabank_api._api_get", side_effect=mock_api_get):
        charts = collect_astrodatabank_dataset(target_count=1, min_rodden="A", verify_ssl=True)

    assert len(charts) == 1
    chart = charts[0]
    required = (
        "name", "date", "time", "longitude", "latitude", "location",
        "rodden_rating", "category", "biography", "source", "source_url",
    )
    for key in required:
        assert key in chart, f"Missing required field: {key}"
    assert chart["rodden_rating"] in ("AA", "A")
    assert -180 <= chart["longitude"] <= 180
    assert -90 <= chart["latitude"] <= 90
    assert "astro-databank" in chart["source_url"]
    assert " " not in chart["source_url"] or "_" in chart["source_url"]


def test_save_and_report_atomic_write():
    charts = [
        {
            "name": "Test",
            "date": "1 Jan 2000",
            "time": "12:00",
            "longitude": 0.0,
            "latitude": 0.0,
            "location": "Nowhere",
            "rodden_rating": "A",
            "category": "Technology",
            "biography": "",
            "source": "",
            "source_url": "https://www.astro.com/astro-databank/Test",
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = f"{tmpdir}/out.jsonl"
        save_and_report(charts, out)
        with open(out, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        loaded = json.loads(lines[0])
        assert loaded["name"] == "Test"
        assert loaded["longitude"] == 0.0
