"""Unit tests for BaziChart.to_feature_arrays() with natal-only charts."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from bazi_engine import BaziCalculator
from bazi_engine.encodings import STEM_TO_ID


def test_to_feature_arrays_required_keys():
    """All required natal keys are present."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    arr = chart.to_feature_arrays()
    required = [
        "stem_ids", "branch_ids", "pillar_positions", "stem_elements", "branch_elements",
        "stem_polarities", "branch_polarities", "ten_god_stem_ids", "ten_god_branch_ids",
        "interaction_vector", "elemental_strengths", "day_master_strength", "useful_god_id",
    ]
    for key in required:
        assert key in arr, f"Missing key: {key}"


def test_to_feature_arrays_shapes_and_dtypes():
    """Array shapes and dtypes match plan (int64 vs float64)."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    arr = chart.to_feature_arrays()
    assert arr["stem_ids"].shape == (4,)
    assert arr["stem_ids"].dtype == np.int64
    assert arr["branch_ids"].shape == (4,)
    assert arr["branch_ids"].dtype == np.int64
    assert arr["pillar_positions"].shape == (4,)
    assert arr["ten_god_branch_ids"].shape == (12,)
    assert arr["interaction_vector"].shape == (64,)
    assert arr["interaction_vector"].dtype == np.float64
    assert arr["elemental_strengths"].shape == (5,)
    assert arr["elemental_strengths"].dtype == np.float64
    assert arr["day_master_strength"].shape == (1,)
    assert arr["useful_god_id"].shape == (1,)
    assert arr["useful_god_id"].dtype == np.int64


def test_to_feature_arrays_stem_ids_chart_癸未():
    """For 2003-04-24 chart, year pillar is 癸未 so stem_ids[0] == STEM_TO_ID['癸']."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    arr = chart.to_feature_arrays()
    assert arr["stem_ids"][0] == STEM_TO_ID["癸"]
    assert arr["branch_ids"][0] == 7  # 未


def test_to_feature_arrays_interaction_vector_64():
    """interaction_vector has shape (64,) and values in [0, 1]."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    arr = chart.to_feature_arrays()
    v = arr["interaction_vector"]
    assert v.shape == (64,)
    assert np.all((v >= 0) & (v <= 1))


def test_to_feature_arrays_elemental_strengths_sum_one():
    """elemental_strengths (wood, fire, earth, metal, water) sum to approximately 1.0."""
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    arr = chart.to_feature_arrays()
    total = arr["elemental_strengths"].sum()
    assert abs(total - 1.0) < 1e-6
