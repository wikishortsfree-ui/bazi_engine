"""Unit tests for BaziChart.to_feature_arrays() with dynamic (temporal) context."""
from __future__ import annotations

from datetime import datetime, timezone

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


def _chart_with_dynamic(
    birth_dt: datetime,
    lat: float,
    lng: float,
    current_year: int,
    current_month: int,
    gender: int = 1,
) -> BaziChart:
    """Build chart with temporal_context and cross_interactions set."""
    chart = BaziCalculator.from_datetime(birth_dt, lat, lng)
    dayun_seq = BaziCalculator().compute_dayun(chart, gender, max_periods=10)
    dayun = dayun_seq.get_period_for_year(current_year)
    liunian = compute_liunian(current_year, day_master_stem=chart.day_pillar.stem.value)
    liuyue = compute_liuyue(
        current_year, current_month, day_master_stem=chart.day_pillar.stem.value
    )
    ctx = TemporalContext(dayun=dayun, liunian=liunian, liuyue=liuyue, liuri=None)
    stems = [chart.year_pillar.stem.value, chart.month_pillar.stem.value,
             chart.day_pillar.stem.value, chart.hour_pillar.stem.value]
    branches = [chart.year_pillar.branch.value, chart.month_pillar.branch.value,
                chart.day_pillar.branch.value, chart.hour_pillar.branch.value]
    cross = detect_cross_interactions(stems, branches, ctx, useful_god=None)
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


def test_to_feature_arrays_dynamic_keys_present():
    """When temporal context and cross_interactions are set, dynamic keys are present."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = _chart_with_dynamic(dt, 0, 0, 2024, 3, gender=1)
    arr = chart.to_feature_arrays()
    dynamic_keys = [
        "dayun_stem_id", "dayun_branch_id", "liunian_stem_id", "liunian_branch_id",
        "liuyue_stem_id", "liuyue_branch_id", "pillar_presence_mask",
        "dynamic_interaction_vector", "elemental_delta", "useful_god_active",
    ]
    for key in dynamic_keys:
        assert key in arr, f"Missing dynamic key: {key}"
    # Architecture: 60 binary + 5 elemental_delta + 1 useful_god_active = 66
    assert arr["dynamic_interaction_vector"].shape == (66,), "dynamic_interaction_vector must be 66-dim"
    assert arr["dynamic_interaction_vector"].dtype == np.float64


def test_to_feature_arrays_pillar_presence_mask():
    """pillar_presence_mask is [has_dayun, has_liunian, has_liuyue] (1 or 0)."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = _chart_with_dynamic(dt, 0, 0, 2024, 3, gender=1)
    arr = chart.to_feature_arrays()
    mask = arr["pillar_presence_mask"]
    assert mask.shape == (3,)
    assert mask.dtype == np.int64
    assert np.all((mask == 0) | (mask == 1))
    # We set dayun, liunian, liuyue so all three should be 1
    assert mask[0] == 1 and mask[1] == 1 and mask[2] == 1


def test_to_feature_arrays_partial_dynamic_only_dayun():
    """When only dayun is set (no liunian/liuyue), pillar_presence_mask reflects that."""
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
    chart_with_dy = BaziChart(
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
    arr = chart_with_dy.to_feature_arrays()
    mask = arr["pillar_presence_mask"]
    assert mask[0] == 1  # dayun present
    assert mask[1] == 0  # liunian absent
    assert mask[2] == 0  # liuyue absent
