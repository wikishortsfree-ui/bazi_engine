"""Unit tests for BaziChart.to_json() with dynamic (temporal) context."""
from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from bazi_engine import BaziCalculator
from bazi_engine.models import BaziChart
from bazi_engine.temporal import (
    DaYun,
    LiuNian,
    LiuYue,
    TemporalContext,
    compute_liunian,
    compute_liuyue,
    detect_cross_interactions,
)


def _chart_with_dynamic(
    birth_dt: datetime,
    lat: float,
    lng: float,
    current_year: int,
    current_month: int,
    gender: int = 1,
):
    """Build chart and attach temporal context + cross_interactions for current date."""
    chart = BaziCalculator.from_datetime(birth_dt, lat, lng)
    dayun_seq = BaziCalculator().compute_dayun(chart, gender, max_periods=10)
    dayun = dayun_seq.get_period_for_year(current_year)
    liunian = compute_liunian(current_year, day_master_stem=chart.day_pillar.stem.value)
    liuyue = compute_liuyue(
        current_year, current_month, day_master_stem=chart.day_pillar.stem.value
    )
    ctx = TemporalContext(dayun=dayun, liunian=liunian, liuyue=liuyue, liuri=None)
    stems, branches = [chart.year_pillar.stem.value, chart.month_pillar.stem.value,
                       chart.day_pillar.stem.value, chart.hour_pillar.stem.value], \
                      [chart.year_pillar.branch.value, chart.month_pillar.branch.value,
                       chart.day_pillar.branch.value, chart.hour_pillar.branch.value]
    useful = None
    if chart.day_pillar.stem.element:
        useful = chart.day_pillar.stem.element
    cross = detect_cross_interactions(stems, branches, ctx, useful_god=useful)
    return BaziChart(
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


def test_to_json_dynamic_section_present():
    """When temporal_context and cross_interactions are set, dynamic is an object."""
    tz = ZoneInfo("America/Los_Angeles")
    birth_dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=tz)
    chart = _chart_with_dynamic(birth_dt, 37.35, -121.96, 2024, 3, gender=1)
    data = json.loads(chart.to_json(indent=2))
    assert data["dynamic"] is not None
    assert isinstance(data["dynamic"], dict)


def test_to_json_dynamic_contains_da_yun_liu_nian_liu_yue():
    """Dynamic section includes da_yun, liu_nian, liu_yue when present."""
    tz = ZoneInfo("America/Los_Angeles")
    birth_dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=tz)
    chart = _chart_with_dynamic(birth_dt, 37.35, -121.96, 2024, 3, gender=1)
    data = json.loads(chart.to_json(indent=2))
    dyn = data["dynamic"]
    assert "da_yun" in dyn
    assert dyn["da_yun"] is not None
    assert "stem" in dyn["da_yun"] and "branch" in dyn["da_yun"]
    assert "liu_nian" in dyn
    assert dyn["liu_nian"]["year"] == 2024
    assert "liu_yue" in dyn
    assert dyn["liu_yue"]["year"] == 2024 and dyn["liu_yue"]["month"] == 3


def test_to_json_dynamic_cross_natal_temporal_interactions():
    """Dynamic section includes cross_natal_temporal_interactions."""
    tz = ZoneInfo("America/Los_Angeles")
    birth_dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=tz)
    chart = _chart_with_dynamic(birth_dt, 37.35, -121.96, 2024, 3, gender=1)
    data = json.loads(chart.to_json(indent=2))
    dyn = data["dynamic"]
    assert "cross_natal_temporal_interactions" in dyn
    cross = dyn["cross_natal_temporal_interactions"]
    assert "natal_vs_dayun" in cross
    assert "natal_vs_liunian" in cross
    assert "natal_vs_liuyue" in cross
    assert "dayun_vs_liunian" in cross


def test_to_json_dynamic_elemental_delta():
    """Dynamic section includes elemental_delta (wood, fire, earth, metal, water)."""
    tz = ZoneInfo("America/Los_Angeles")
    birth_dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=tz)
    chart = _chart_with_dynamic(birth_dt, 37.35, -121.96, 2024, 3, gender=1)
    data = json.loads(chart.to_json(indent=2))
    dyn = data["dynamic"]
    assert "elemental_delta" in dyn
    delta = dyn["elemental_delta"]
    for key in ("wood", "fire", "earth", "metal", "water"):
        assert key in delta
        assert isinstance(delta[key], (int, float))
