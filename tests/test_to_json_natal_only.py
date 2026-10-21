"""Unit tests for BaziChart.to_json() with natal-only charts (no dynamic)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from bazi_engine import BaziCalculator


def test_to_json_returns_valid_json():
    """to_json() returns a string that parses as valid JSON."""
    tz = ZoneInfo("America/Los_Angeles")
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=tz)
    chart = BaziCalculator.from_datetime(dt, 37.35, -121.96)
    s = chart.to_json(indent=2)
    data = json.loads(s)
    assert isinstance(data, dict)


def test_to_json_has_required_keys():
    """Parsed JSON contains all required top-level keys."""
    tz = ZoneInfo("America/Los_Angeles")
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=tz)
    chart = BaziCalculator.from_datetime(dt, 37.35, -121.96)
    data = json.loads(chart.to_json(indent=2))
    required = [
        "metadata",
        "natal_pillars",
        "day_master",
        "ten_gods",
        "hidden_stems",
        "elemental_strengths",
        "useful_god",
        "interactions",
        "dynamic",
    ]
    for key in required:
        assert key in data, f"Missing key: {key}"


def test_to_json_chinese_characters_preserved():
    """Chinese characters (stems, branches, elements) are preserved, not escaped."""
    tz = ZoneInfo("America/Los_Angeles")
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=tz)
    chart = BaziCalculator.from_datetime(dt, 37.35, -121.96)
    s = chart.to_json(indent=2)
    # ensure_ascii=False should keep 癸, 未, 丙, etc.
    assert "癸" in s
    assert "未" in s
    assert "丙" in s
    assert "辰" in s
    assert "丁" in s
    assert "卯" in s
    data = json.loads(s)
    assert data["natal_pillars"]["year"]["stem"] == "癸"
    assert data["natal_pillars"]["year"]["branch"] == "未"
    assert data["day_master"]["element"] == "火"


def test_to_json_dynamic_null_when_no_temporal():
    """When chart has no temporal context, dynamic is null."""
    tz = ZoneInfo("America/Los_Angeles")
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=tz)
    chart = BaziCalculator.from_datetime(dt, 37.35, -121.96)
    assert chart.temporal_context is None
    data = json.loads(chart.to_json(indent=2))
    assert data["dynamic"] is None


def test_to_json_metadata_structure():
    """Metadata contains birth_datetime, timezone_offset, calculation_timestamp, calculator_version."""
    tz = timezone(timedelta(hours=8))  # fixed UTC+8
    dt = datetime(1990, 5, 15, 14, 30, 0, tzinfo=tz)
    chart = BaziCalculator.from_datetime(dt, 39.9, 116.4)
    data = json.loads(chart.to_json(indent=2))
    meta = data["metadata"]
    assert "birth_datetime" in meta
    assert "1990-05-15" in meta["birth_datetime"]
    assert meta["timezone_offset"] == 8
    assert "calculation_timestamp" in meta
    assert "calculator_version" in meta
    assert meta["calculator_version"] == "0.1.0"


def test_to_json_natal_pillars_structure():
    """Natal pillars have stem, branch, element_stem, element_branch per position."""
    tz = ZoneInfo("UTC")
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=tz)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    data = json.loads(chart.to_json(indent=2))
    for pos in ("year", "month", "day", "hour"):
        p = data["natal_pillars"][pos]
        assert "stem" in p and "branch" in p
        assert "element_stem" in p and "element_branch" in p
        assert len(p["stem"]) == 1 and len(p["branch"]) == 1


def test_to_json_elemental_strengths_sum_near_one():
    """Elemental strengths (wood, fire, earth, metal, water) sum to approximately 1.0."""
    tz = ZoneInfo("UTC")
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=tz)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    data = json.loads(chart.to_json(indent=2))
    es = data["elemental_strengths"]
    total = sum(es[k] for k in ("wood", "fire", "earth", "metal", "water"))
    assert abs(total - 1.0) < 1e-6


def test_to_json_compact_indent_none():
    """to_json(indent=None) produces compact single-line JSON."""
    tz = ZoneInfo("UTC")
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=tz)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    s = chart.to_json(indent=None)
    data = json.loads(s)
    assert data["metadata"]["calculator_version"] == "0.1.0"
    assert "\n" not in s or s.count("\n") < 2  # compact
