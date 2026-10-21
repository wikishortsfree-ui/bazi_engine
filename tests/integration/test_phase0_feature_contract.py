"""
Phase 0 calculator contract: BaziChart.to_feature_arrays() keys and shapes.

Ensures encoder input contract is stable for Task 1.2+. Natal and dynamic
signatures must match ChartDataset and encoders (NatalEncoder, DynamicEncoder).
"""
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


# --- Natal contract (all charts) ---
NATAL_KEYS = [
    "stem_ids",
    "branch_ids",
    "ten_god_stem_ids",
    "stem_elements",
    "stem_polarities",
    "pillar_positions",
    "branch_elements",
    "branch_polarities",
    "ten_god_branch_ids",
    "interaction_vector",
    "elemental_strengths",
    "day_master_strength",
    "useful_god_id",
]
NATAL_SHAPES = {
    "stem_ids": (4,),
    "branch_ids": (4,),
    "ten_god_stem_ids": (4,),
    "stem_elements": (4,),
    "stem_polarities": (4,),
    "pillar_positions": (4,),
    "branch_elements": (4,),
    "branch_polarities": (4,),
    "ten_god_branch_ids": (12,),
    "interaction_vector": (64,),
    "elemental_strengths": (5,),
    "day_master_strength": (1,),
    "useful_god_id": (1,),
}
INTERACTION_VECTOR_DIM = 64
ELEMENTAL_STRENGTHS_DIM = 5


# --- Dynamic contract (when temporal_context and cross_interactions set) ---
DYNAMIC_KEYS = [
    "dayun_stem_id",
    "dayun_branch_id",
    "liunian_stem_id",
    "liunian_branch_id",
    "liuyue_stem_id",
    "liuyue_branch_id",
    "pillar_presence_mask",
    "dynamic_interaction_vector",
]
DYNAMIC_SHAPES = {
    "dayun_stem_id": (1,),
    "dayun_branch_id": (1,),
    "liunian_stem_id": (1,),
    "liunian_branch_id": (1,),
    "liuyue_stem_id": (1,),
    "liuyue_branch_id": (1,),
    "pillar_presence_mask": (3,),
    "dynamic_interaction_vector": (66,),  # 60 binary + 5 elemental_delta + 1 useful_god_active
}
DYNAMIC_INTERACTION_DIM = 66


def _chart_with_dynamic(
    birth_dt: datetime,
    lat: float,
    lng: float,
    current_year: int,
    current_month: int,
    gender: int = 1,
) -> BaziChart:
    chart = BaziCalculator.from_datetime(birth_dt, lat, lng)
    dayun_seq = BaziCalculator().compute_dayun(chart, gender, max_periods=10)
    dayun = dayun_seq.get_period_for_year(current_year)
    liunian = compute_liunian(current_year, day_master_stem=chart.day_pillar.stem.value)
    liuyue = compute_liuyue(
        current_year, current_month, day_master_stem=chart.day_pillar.stem.value,
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


def test_natal_contract_keys_and_shapes():
    """Natal: all required keys present and shapes match encoder expectations."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    arr = chart.to_feature_arrays()

    for key in NATAL_KEYS:
        assert key in arr, f"Missing natal key: {key}"
    for key, shape in NATAL_SHAPES.items():
        assert arr[key].shape == shape, f"{key}: expected {shape}, got {arr[key].shape}"
    assert arr["interaction_vector"].dtype == np.float64
    assert arr["interaction_vector"].shape[0] == INTERACTION_VECTOR_DIM
    assert arr["elemental_strengths"].shape[0] == ELEMENTAL_STRENGTHS_DIM
    assert arr["useful_god_id"].dtype == np.int64
    # useful_god_id: 0-4 (element-based) or -1 if ambiguous; vocab 6 in encoders
    assert arr["useful_god_id"].shape == (1,)
    # ten_god_stem_ids: 0-10 (日主=10 for day stem), no -1 for day master in current contract
    assert arr["ten_god_stem_ids"].shape == (4,)


def test_dynamic_contract_keys_and_shapes():
    """Dynamic: when temporal + cross_interactions set, dynamic keys and shapes match."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = _chart_with_dynamic(dt, 0, 0, 2024, 3, gender=1)
    arr = chart.to_feature_arrays()

    for key in DYNAMIC_KEYS:
        assert key in arr, f"Missing dynamic key: {key}"
    for key, shape in DYNAMIC_SHAPES.items():
        assert arr[key].shape == shape, f"{key}: expected {shape}, got {arr[key].shape}"
    assert arr["dynamic_interaction_vector"].shape[0] == DYNAMIC_INTERACTION_DIM
    assert arr["pillar_presence_mask"].shape == (3,)  # [dayun, liunian, liuyue]


def test_natal_contract_dtypes():
    """Natal: ID arrays int64, float arrays float64."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    arr = chart.to_feature_arrays()

    id_keys = [
        "stem_ids", "branch_ids", "ten_god_stem_ids", "stem_elements", "stem_polarities",
        "pillar_positions", "branch_elements", "branch_polarities", "ten_god_branch_ids",
        "useful_god_id",
    ]
    for key in id_keys:
        assert arr[key].dtype == np.int64, f"{key} should be int64"
    assert arr["interaction_vector"].dtype == np.float64
    assert arr["elemental_strengths"].dtype == np.float64
    assert arr["day_master_strength"].dtype == np.float64
