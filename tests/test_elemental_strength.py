"""
Unit tests for the hidden-stem-weighted elemental strength module (bazi_engine.strength.elemental_strength).

These tests target the full implementation that uses 4 visible stems (1.0 each) and 4 branches
via 藏干 weight tables. The existing test_element_strength.py covers the same API with simpler
assertions; here we add edge cases and structural guarantees.
"""
import pytest

from bazi_engine.strength import (
    classify_day_master,
    compute_day_master_strength,
    compute_element_strengths,
)
from bazi_engine.enums import Branch
from bazi_engine.tables import get_hidden_stems


# --- Pure-chart and normalization ---


def test_all_wood_stems_and_branches():
    """If all 4 stems are 甲 (Wood) and all 4 branches are 卯 (pure 乙 Wood hidden stem), Wood fraction is 1.0 and others 0.0."""
    stems = ["甲", "甲", "甲", "甲"]
    branches = ["卯", "卯", "卯", "卯"]  # 卯 = 乙 only (1.0 木 each)
    result = compute_element_strengths(stems, branches)
    assert result["木"] == pytest.approx(1.0)
    assert result["火"] == pytest.approx(0.0)
    assert result["土"] == pytest.approx(0.0)
    assert result["金"] == pytest.approx(0.0)
    assert result["水"] == pytest.approx(0.0)


def test_output_always_sums_to_one():
    """For an arbitrary valid chart, sum(result.values()) is approximately 1.0."""
    stems = ["癸", "丙", "丁", "辛"]
    branches = ["未", "辰", "卯", "亥"]
    result = compute_element_strengths(stems, branches)
    assert sum(result.values()) == pytest.approx(1.0)


def test_all_five_keys_always_present():
    """Even for a chart with no Metal at all, the 金 key must be in the result dict (value may be 0.0)."""
    stems = ["甲", "甲", "甲", "甲"]
    branches = ["卯", "卯", "卯", "卯"]  # no Metal
    result = compute_element_strengths(stems, branches)
    assert "金" in result
    assert result["金"] == pytest.approx(0.0)
    assert set(result.keys()) == {"木", "火", "土", "金", "水"}


def test_known_chart_directional_correctness():
    """For 癸未 丙辰 丁卯 辛亥 (day master 丁, Fire): Fire has non-zero fraction; Metal and Water have non-zero (辛, 癸, 亥/辰)."""
    stems = ["癸", "丙", "丁", "辛"]
    branches = ["未", "辰", "卯", "亥"]
    result = compute_element_strengths(stems, branches)
    assert result["火"] > 0.0
    assert result["金"] > 0.0
    assert result["水"] > 0.0


# --- Invalid inputs ---


def test_invalid_stem_raises():
    """Passing an unknown string as a stem (e.g. '甲甲') raises ValueError."""
    with pytest.raises(ValueError):
        compute_element_strengths(["甲", "甲甲", "丙", "丁"], ["寅", "卯", "辰", "巳"])


def test_invalid_branch_raises():
    """Passing an unknown string as a branch raises ValueError."""
    with pytest.raises(ValueError):
        compute_element_strengths(["甲", "乙", "丙", "丁"], ["寅", "卯", "辰", "亥亥"])


def test_wrong_length_raises():
    """Passing only 3 stems or 3 branches raises ValueError."""
    with pytest.raises(ValueError, match="must each have 4 items"):
        compute_element_strengths(["甲", "乙", "丙"], ["寅", "卯", "辰", "巳"])
    with pytest.raises(ValueError, match="must each have 4 items"):
        compute_element_strengths(["甲", "乙", "丙", "丁"], ["寅", "卯", "辰"])


# --- Weight table consistency ---


def test_weight_tables_sum_to_one():
    """For each of the 12 branches, hidden stem weights from tables sum to 1.0 (within floating point tolerance)."""
    for branch in Branch:
        weights = [w for _, w in get_hidden_stems(branch)]
        assert sum(weights) == pytest.approx(1.0), f"Branch {branch.value}: weights sum to {sum(weights)}"


# --- Day master strength and classification ---

# Test chart: 癸未 丙辰 丁卯 辛亥 (day master 丁 = Fire)
_STEMS_癸未_丙辰_丁卯_辛亥 = ["癸", "丙", "丁", "辛"]
_BRANCHES_癸未_丙辰_丁卯_辛亥 = ["未", "辰", "卯", "亥"]


def test_day_master_strength_is_float_in_range():
    """For the test chart (癸未 丙辰 丁卯 辛亥), the result of compute_day_master_strength is a float in [0.0, 1.0]."""
    strength = compute_day_master_strength(_STEMS_癸未_丙辰_丁卯_辛亥, _BRANCHES_癸未_丙辰_丁卯_辛亥)
    assert isinstance(strength, float)
    assert 0.0 <= strength <= 1.0


def test_day_master_strength_matches_element_fraction():
    """The day master strength for 丁 (Fire) should equal compute_element_strengths(stems, branches)['火']."""
    stems = _STEMS_癸未_丙辰_丁卯_辛亥
    branches = _BRANCHES_癸未_丙辰_丁卯_辛亥
    strength = compute_day_master_strength(stems, branches)
    fire_fraction = compute_element_strengths(stems, branches)["火"]
    assert strength == pytest.approx(fire_fraction)


def test_day_master_strength_explicit_override():
    """Calling compute_day_master_strength(stems, branches, day_master_stem='壬') should return the Water (水) fraction."""
    stems = _STEMS_癸未_丙辰_丁卯_辛亥
    branches = _BRANCHES_癸未_丙辰_丁卯_辛亥
    water_strength = compute_day_master_strength(stems, branches, day_master_stem="壬")
    strengths = compute_element_strengths(stems, branches)
    assert water_strength == pytest.approx(strengths["水"])
    assert water_strength != pytest.approx(strengths["火"])


def test_classify_strong_day_master():
    """Chart where the day master's element dominates → classification is '身强'."""
    # All stems and branches Fire: 丁 in day position, rest Fire too; branches 午 = Fire
    stems = ["丙", "丙", "丁", "丙"]
    branches = ["午", "午", "午", "午"]
    assert classify_day_master(stems, branches) == "身强"


def test_classify_weak_day_master():
    """Chart where the day master's element has a very small fraction → classification is '身弱'."""
    # Day master 丁 (Fire) at stems[2]; minimal Fire elsewhere (mostly Wood)
    stems = ["甲", "乙", "丁", "甲"]
    branches = ["卯", "卯", "卯", "卯"]  # all 乙 Wood
    assert classify_day_master(stems, branches) == "身弱"


def test_classify_balanced_day_master():
    """Chart where the day master's element fraction is between weak and strong thresholds → '中和'."""
    # Day master 丁 (Fire) at stems[2]; 2 Fire stems + 6 Wood (2 stems + 4 branches 卯) → Fire = 2/8 = 0.25 ∈ [0.20, 0.30)
    stems = ["丙", "甲", "丁", "甲"]
    branches = ["卯", "卯", "卯", "卯"]
    assert classify_day_master(stems, branches) == "中和"


def test_invalid_day_master_stem_raises():
    """Passing an unknown string as day_master_stem raises ValueError."""
    with pytest.raises(ValueError, match="day_master_stem must be a known heavenly stem"):
        compute_day_master_strength(
            _STEMS_癸未_丙辰_丁卯_辛亥,
            _BRANCHES_癸未_丙辰_丁卯_辛亥,
            day_master_stem="unknown",
        )
