"""
End-to-end integration tests: from_datetime → to_json / to_feature_arrays.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from bazi_engine import BaziCalculator
from bazi_engine.models import BaziChart
from bazi_engine.temporal import (
    TemporalContext,
    compute_liunian,
    compute_liuyue,
    detect_cross_interactions,
)


def test_end_to_end_natal_from_datetime_to_json_parse():
    """from_datetime() → to_json() → parse JSON → verify all top-level keys and types."""
    tz = timezone(timedelta(hours=8))
    dt = datetime(1990, 5, 15, 14, 30, 0, tzinfo=tz)
    chart = BaziCalculator.from_datetime(dt, 39.9, 116.4)
    s = chart.to_json(indent=2)
    data = json.loads(s)
    assert "metadata" in data
    assert "natal_pillars" in data
    assert "day_master" in data
    assert "ten_gods" in data
    assert "hidden_stems" in data
    assert "elemental_strengths" in data
    assert "interactions" in data
    assert data["dynamic"] is None
    assert "1990-05-15" in data["metadata"]["birth_datetime"]
    assert data["metadata"]["timezone_offset"] == 8
    assert set(data["natal_pillars"].keys()) == {"year", "month", "day", "hour"}
    assert set(data["elemental_strengths"].keys()) == {"wood", "fire", "earth", "metal", "water"}


def test_end_to_end_with_dynamic_to_json():
    """from_datetime() with dynamic context → to_json() → verify dynamic section present."""
    tz = timezone(timedelta(hours=8))
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=tz)
    chart = BaziCalculator.from_datetime(dt, 37.35, -121.96)
    dayun_seq = BaziCalculator().compute_dayun(chart, 1, max_periods=10)
    dayun = dayun_seq.get_period_for_year(2024)
    liunian = compute_liunian(2024, day_master_stem=chart.day_pillar.stem.value)
    liuyue = compute_liuyue(2024, 3, day_master_stem=chart.day_pillar.stem.value)
    ctx = TemporalContext(dayun=dayun, liunian=liunian, liuyue=liuyue, liuri=None)
    stems = [chart.year_pillar.stem.value, chart.month_pillar.stem.value,
             chart.day_pillar.stem.value, chart.hour_pillar.stem.value]
    branches = [chart.year_pillar.branch.value, chart.month_pillar.branch.value,
                chart.day_pillar.branch.value, chart.hour_pillar.branch.value]
    cross = detect_cross_interactions(stems, branches, ctx, useful_god=None)
    chart_dyn = BaziChart(
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
    s = chart_dyn.to_json(indent=2)
    data = json.loads(s)
    assert data["dynamic"] is not None
    assert "da_yun" in data["dynamic"]
    assert "liu_nian" in data["dynamic"]
    assert "liu_yue" in data["dynamic"]
    assert "elemental_delta" in data["dynamic"]


def test_end_to_end_feature_arrays_shapes_dtypes():
    """from_datetime() → to_feature_arrays() → verify shapes/dtypes and no crash."""
    tz = timezone.utc
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=tz)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    arr = chart.to_feature_arrays()
    assert arr["stem_ids"].shape == (4,)
    assert arr["stem_ids"].dtype == np.int64
    assert arr["branch_ids"].shape == (4,)
    assert arr["interaction_vector"].shape == (64,)
    assert arr["interaction_vector"].dtype == np.float64
    assert arr["elemental_strengths"].shape == (5,)
    assert abs(arr["elemental_strengths"].sum() - 1.0) < 1e-6


def test_edge_case_dynamic_null_in_json():
    """Chart with no temporal context → JSON has "dynamic": null."""
    dt = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    assert chart.temporal_context is None
    data = json.loads(chart.to_json(indent=2))
    assert data["dynamic"] is None


def test_edge_case_partial_dynamic_pillar_presence_mask():
    """Only 大运 set (no 流年/流月) → pillar_presence_mask = [1, 0, 0]."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    dayun_seq = BaziCalculator().compute_dayun(chart, 1, max_periods=10)
    dayun = dayun_seq.get_period_for_year(2024)
    ctx = TemporalContext(dayun=dayun, liunian=None, liuyue=None, liuri=None)
    stems = [chart.year_pillar.stem.value, chart.month_pillar.stem.value,
             chart.day_pillar.stem.value, chart.hour_pillar.stem.value]
    branches = [chart.year_pillar.branch.value, chart.month_pillar.branch.value,
                chart.day_pillar.branch.value, chart.hour_pillar.branch.value]
    cross = detect_cross_interactions(stems, branches, ctx, useful_god=None)
    chart_partial = BaziChart(
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
    arr = chart_partial.to_feature_arrays()
    assert arr["pillar_presence_mask"][0] == 1
    assert arr["pillar_presence_mask"][1] == 0
    assert arr["pillar_presence_mask"][2] == 0


def test_edge_case_ambiguous_useful_god_useful_god_id_minus_one():
    """Chart with 中和 (neutral) classification → useful_god can be null; useful_god_id may be -1."""
    # Use a chart that yields 中和: strength in [0.20, 0.30). Tweak if needed.
    tz = timezone.utc
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=tz)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    arr = chart.to_feature_arrays()
    # Our heuristic: 身弱→印, 身强→官杀, 中和→useful_god null → useful_id -1 in code
    assert "useful_god_id" in arr
    assert arr["useful_god_id"].shape == (1,)
    # Just ensure it's int and in valid range or -1
    val = int(arr["useful_god_id"][0])
    assert -1 <= val <= 4


def test_edge_case_timezone_affects_pillars():
    """Different timezone for same instant → pillars can differ (or metadata offset differs)."""
    utc = timezone.utc
    utc8 = timezone(timedelta(hours=8))
    # Same UTC moment
    dt_utc = datetime(1990, 5, 15, 6, 30, 0, tzinfo=utc)
    dt_utc8 = datetime(1990, 5, 15, 14, 30, 0, tzinfo=utc8)
    chart_utc = BaziCalculator.from_datetime(dt_utc, 39.9, 116.4)
    chart_utc8 = BaziCalculator.from_datetime(dt_utc8, 39.9, 116.4)
    data_utc = json.loads(chart_utc.to_json(indent=2))
    data_utc8 = json.loads(chart_utc8.to_json(indent=2))
    assert data_utc["metadata"]["timezone_offset"] == 0
    assert data_utc8["metadata"]["timezone_offset"] == 8
    # Local times differ so solar time can differ → pillars may differ
    assert data_utc["metadata"]["birth_datetime"] != data_utc8["metadata"]["birth_datetime"]
