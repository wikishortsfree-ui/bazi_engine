"""Unit tests for BaziChart.to_input_features(task_name) and to_ground_truth(task_name)."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from bazi_engine import BaziCalculator
from bazi_engine.models import BaziChart, _compute_useful_god
from bazi_engine.strength import compute_day_master_strength
from bazi_engine.temporal import (
    TemporalContext,
    compute_liunian,
    compute_liuyue,
    detect_cross_interactions,
)


def _chart_with_dynamic(birth_dt, lat: float = 0, lng: float = 0, year: int = 2024, month: int = 3):
    """Build a chart with temporal_context and cross_interactions."""
    chart = BaziCalculator.from_datetime(birth_dt, lat, lng)
    dayun_seq = BaziCalculator().compute_dayun(chart, 1, max_periods=10)
    dayun = dayun_seq.get_period_for_year(year)
    liunian = compute_liunian(year, day_master_stem=chart.day_pillar.stem.value)
    liuyue = compute_liuyue(year, month, day_master_stem=chart.day_pillar.stem.value)
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


def test_to_input_features_base_keys():
    """to_input_features returns only minimal base keys for natal tasks (no ground truth)."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    for task in ("ten_god", "element_strength", "interaction", "useful_god"):
        inp = chart.to_input_features(task)
        assert "stem_ids" in inp and inp["stem_ids"].shape == (4,)
        assert "branch_ids" in inp and inp["branch_ids"].shape == (4,)
        assert "stem_elements" in inp and inp["stem_elements"].shape == (4,)
        assert "stem_polarities" in inp and inp["stem_polarities"].shape == (4,)
        assert "pillar_positions" in inp and inp["pillar_positions"].shape == (4,)
        assert "ten_god_ids" not in inp
        assert "interaction_vector" not in inp
        assert "elemental_strengths" not in inp
        assert "useful_god_id" not in inp


def test_to_input_features_cross_temporal_adds_dynamic():
    """to_input_features('cross_temporal') adds temporal slot IDs and pillar_presence_mask."""
    dt = datetime(1995, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = _chart_with_dynamic(dt, 0, 0, 2024, 3)
    inp = chart.to_input_features("cross_temporal")
    assert "dayun_stem_id" in inp and inp["dayun_stem_id"].shape == (1,)
    assert "dayun_branch_id" in inp
    assert "liunian_stem_id" in inp and "liunian_branch_id" in inp
    assert "liuyue_stem_id" in inp and "liuyue_branch_id" in inp
    assert "pillar_presence_mask" in inp and inp["pillar_presence_mask"].shape == (3,)
    assert "dynamic_interaction_vector" not in inp


def test_to_input_features_cross_temporal_raises_without_temporal():
    """to_input_features('cross_temporal') raises when chart has no temporal_context."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    with pytest.raises(ValueError, match="temporal_context"):
        chart.to_input_features("cross_temporal")


def test_to_ground_truth_ten_god():
    """to_ground_truth('ten_god') returns ten_god_ids shape (4,)."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    gt = chart.to_ground_truth("ten_god")
    assert "ten_god_ids" in gt and gt["ten_god_ids"].shape == (4,)
    assert gt["ten_god_ids"].dtype == np.int64


def test_to_ground_truth_element_strength():
    """to_ground_truth('element_strength') returns elemental_strengths (5,) summing to 1."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    gt = chart.to_ground_truth("element_strength")
    assert "elemental_strengths" in gt and gt["elemental_strengths"].shape == (5,)
    assert abs(gt["elemental_strengths"].sum() - 1.0) < 1e-6


def test_to_ground_truth_interaction():
    """to_ground_truth('interaction') returns interaction_vector (64,)."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    gt = chart.to_ground_truth("interaction")
    assert "interaction_vector" in gt and gt["interaction_vector"].shape == (64,)
    assert np.all((gt["interaction_vector"] >= 0) & (gt["interaction_vector"] <= 1))


def test_to_ground_truth_useful_god():
    """to_ground_truth('useful_god') returns useful_god_id (1,) in 0..5."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    gt = chart.to_ground_truth("useful_god")
    assert "useful_god_id" in gt and gt["useful_god_id"].shape == (1,)
    assert gt["useful_god_id"].dtype == np.int64
    assert 0 <= gt["useful_god_id"].item() <= 5


def test_to_ground_truth_cross_temporal():
    """to_ground_truth('cross_temporal') returns dynamic_interaction_vector (66,)."""
    dt = datetime(1995, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = _chart_with_dynamic(dt, 0, 0, 2024, 3)
    gt = chart.to_ground_truth("cross_temporal")
    assert "dynamic_interaction_vector" in gt and gt["dynamic_interaction_vector"].shape == (66,)


def test_to_ground_truth_cross_temporal_raises_without_cross_interactions():
    """to_ground_truth('cross_temporal') raises when chart has no cross_interactions."""
    dt = datetime(1995, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    with pytest.raises(ValueError, match="cross_interactions"):
        chart.to_ground_truth("cross_temporal")


def test_input_and_ground_truth_disjoint():
    """Input keys and ground-truth keys must not overlap for any task."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    for task in ("ten_god", "element_strength", "interaction", "useful_god"):
        inp = chart.to_input_features(task)
        gt = chart.to_ground_truth(task)
        assert set(inp.keys()).isdisjoint(set(gt.keys())), f"Overlap for task {task}"


def test_to_ground_truth_multi_returns_all_four():
    """to_ground_truth('multi') returns merged dict with all four natal target keys."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    gt = chart.to_ground_truth("multi")
    assert "ten_god_ids" in gt and gt["ten_god_ids"].shape == (4,)
    assert "elemental_strengths" in gt and gt["elemental_strengths"].shape == (5,)
    assert "interaction_vector" in gt and gt["interaction_vector"].shape == (64,)
    assert "useful_god_id" in gt and gt["useful_god_id"].shape == (1,)


def test_to_input_features_unknown_task_raises():
    """to_input_features raises for unknown task_name."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    with pytest.raises(ValueError, match="Unknown task_name"):
        chart.to_input_features("unknown")


def test_to_ground_truth_unknown_task_raises():
    """to_ground_truth raises for unknown task_name."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    with pytest.raises(ValueError, match="Unknown task_name"):
        chart.to_ground_truth("unknown")
