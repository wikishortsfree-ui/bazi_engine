"""Tests for scripts/astrodatabank_extract_events.py (event extraction and 流年/流月 chart building)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bazi_engine.models import BaziChart


# ---------------------------------------------------------------------------
# _record_key / _load_done_keys
# ---------------------------------------------------------------------------


def test_record_key_extracts_tuple():
    from scripts.astrodatabank_extract_events import _record_key

    rec = {"name": "Doe, Jane", "date": "1 January 1990", "time": "14:30", "chart": {}}
    assert _record_key(rec) == ("Doe, Jane", "1 January 1990", "14:30")


def test_record_key_missing_fields_default_empty():
    from scripts.astrodatabank_extract_events import _record_key

    assert _record_key({}) == ("", "", "")
    assert _record_key({"name": "X"}) == ("X", "", "")


def test_load_done_keys_reads_existing_output(tmp_path: Path):
    from scripts.astrodatabank_extract_events import _load_done_keys

    p = tmp_path / "events.jsonl"
    rows = [
        {"name": "A", "date": "1 Jan 2000", "time": "12:00", "events": []},
        {"name": "B", "date": "2 Feb 2001", "time": "06:00", "events": [{"year": 2010}]},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    keys = _load_done_keys(p)
    assert len(keys) == 2
    assert ("A", "1 Jan 2000", "12:00") in keys
    assert ("B", "2 Feb 2001", "06:00") in keys


def test_load_done_keys_empty_when_no_file(tmp_path: Path):
    from scripts.astrodatabank_extract_events import _load_done_keys

    assert _load_done_keys(tmp_path / "nonexistent.jsonl") == set()


# ---------------------------------------------------------------------------
# _call_qwen_extract_events
# ---------------------------------------------------------------------------


def test_call_qwen_extract_events_empty_bio_returns_empty_list():
    """Empty or whitespace biography yields no API call and returns []."""
    from scripts.astrodatabank_extract_events import _call_qwen_extract_events

    assert _call_qwen_extract_events("") == []
    assert _call_qwen_extract_events("   ") == []


def test_build_chart_with_dynamic_produces_valid_chart_with_dynamic():
    """_build_chart_with_dynamic returns chart dict with non-null dynamic, loadable by BaziChart.from_json."""
    from scripts.astrodatabank_extract_events import _build_chart_with_dynamic

    data_dir = Path(__file__).resolve().parents[1] / "data"
    bazi_path = data_dir / "astrodatabank_bazi_test.jsonl"
    if not bazi_path.exists():
        pytest.skip("data/astrodatabank_bazi_test.jsonl not found")
    with open(bazi_path, encoding="utf-8") as f:
        line = f.readline()
    if not line.strip():
        pytest.skip("astrodatabank_bazi_test.jsonl is empty")
    record = json.loads(line)
    chart_dict = record.get("chart")
    if not chart_dict:
        pytest.skip("test record has no chart")
    result = _build_chart_with_dynamic(chart_dict, event_year=2020, event_month=6, gender=1)
    assert result is not None
    assert result.get("dynamic") is not None
    chart = BaziChart.from_json(json.dumps(result))
    assert chart.temporal_context is not None
    assert chart.cross_interactions is not None


def test_build_chart_with_dynamic_liunian_only():
    """_build_chart_with_dynamic with event_month=None still produces valid dynamic (流年 only)."""
    from scripts.astrodatabank_extract_events import _build_chart_with_dynamic

    data_dir = Path(__file__).resolve().parents[1] / "data"
    bazi_path = data_dir / "astrodatabank_bazi_test.jsonl"
    if not bazi_path.exists():
        pytest.skip("data/astrodatabank_bazi_test.jsonl not found")
    with open(bazi_path, encoding="utf-8") as f:
        line = f.readline()
    if not line.strip():
        pytest.skip("astrodatabank_bazi_test.jsonl is empty")
    record = json.loads(line)
    chart_dict = record.get("chart")
    if not chart_dict:
        pytest.skip("test record has no chart")
    result = _build_chart_with_dynamic(chart_dict, event_year=2015, event_month=None, gender=0)
    assert result is not None
    assert result.get("dynamic") is not None
    chart = BaziChart.from_json(json.dumps(result))
    assert chart.temporal_context is not None
    assert chart.temporal_context.liunian is not None
    assert chart.temporal_context.liunian.year == 2015
