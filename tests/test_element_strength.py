"""
Unit tests for elemental strength computation (4 stems @ 1.0 + 4 branches via 藏干 weights).
"""
import pytest

from bazi_engine.strength import (
    classify_day_master,
    compute_day_master_strength,
    compute_element_strengths,
)


def test_strength_all_wood_sums_to_one():
    """4 木 stems + 寅寅卯卯 (寅 has 木0.6+火0.3+土0.1, 卯 木1.0) → 木 dominant, sum 1.0."""
    stems = ["甲", "甲", "乙", "乙"]  # 4 木
    branches = ["寅", "寅", "卯", "卯"]  # 寅: 木0.6,火0.3,土0.1 each; 卯: 木1.0 each
    strengths = compute_element_strengths(stems, branches)
    # 木 = 4 + 1.2 + 2 = 7.2, 火 = 0.6, 土 = 0.2 → normalized
    assert strengths["木"] == pytest.approx(0.9)
    assert strengths["火"] == pytest.approx(0.075)
    assert strengths["土"] == pytest.approx(0.025)
    assert strengths["金"] == 0.0
    assert strengths["水"] == 0.0
    assert sum(strengths.values()) == pytest.approx(1.0)


def test_strength_half_wood_half_fire():
    """2木2火 stems + 寅卯巳午 branches (hidden stems) → 木≈火≈0.45."""
    stems = ["甲", "乙", "丙", "丁"]  # 木木火火
    branches = ["寅", "卯", "巳", "午"]  # 寅:木0.6+火0.3+土0.1, 卯:木1, 巳:火0.6+金0.3+土0.1, 午:火0.7+土0.3
    strengths = compute_element_strengths(stems, branches)
    assert strengths["木"] == pytest.approx(0.45)
    assert strengths["火"] == pytest.approx(0.45)
    assert strengths["土"] == pytest.approx(0.0625)
    assert strengths["金"] == pytest.approx(0.0375)
    assert strengths["水"] == 0.0
    assert sum(strengths.values()) == pytest.approx(1.0)


def test_strength_known_fractions():
    """土土金金 stems + 亥子寅巳 branches (藏干 weights) → known normalized fractions."""
    stems = ["己", "戊", "庚", "辛"]  # 土土金金
    branches = ["亥", "子", "寅", "巳"]  # 亥:水0.7+木0.3, 子:水1, 寅:木0.6+火0.3+土0.1, 巳:火0.6+金0.3+土0.1
    strengths = compute_element_strengths(stems, branches)
    assert strengths["土"] == pytest.approx(0.275)
    assert strengths["金"] == pytest.approx(0.2875)
    assert strengths["水"] == pytest.approx(0.2125)
    assert strengths["木"] == pytest.approx(0.1125)
    assert strengths["火"] == pytest.approx(0.1125)
    assert sum(strengths.values()) == pytest.approx(1.0)


def test_strength_always_sums_to_one():
    """Any valid chart yields fractions summing to 1.0."""
    stems = ["癸", "丙", "丁", "辛"]
    branches = ["未", "辰", "卯", "亥"]
    strengths = compute_element_strengths(stems, branches)
    assert sum(strengths.values()) == pytest.approx(1.0)
    assert set(strengths.keys()) == {"木", "火", "土", "金", "水"}


def test_strength_invalid_length_raises():
    """Non-4 stems or branches raises ValueError."""
    with pytest.raises(ValueError, match="must each have 4 items"):
        compute_element_strengths(["甲", "乙"], ["寅", "卯", "辰", "巳"])
    with pytest.raises(ValueError, match="must each have 4 items"):
        compute_element_strengths(["甲", "乙", "丙", "丁"], ["寅", "卯"])


def test_strength_unknown_stem_raises():
    """Unknown stem raises ValueError."""
    with pytest.raises(ValueError):
        compute_element_strengths(["甲", "X", "丙", "丁"], ["寅", "卯", "辰", "巳"])


def test_strength_unknown_branch_raises():
    """Unknown branch raises ValueError."""
    with pytest.raises(ValueError):
        compute_element_strengths(["甲", "乙", "丙", "丁"], ["寅", "卯", "辰", "X"])


# --- compute_day_master_strength ---


def test_day_master_strength_inferred_from_stems_index_2():
    """Day master is stems[2]; strength is that element's normalized share."""
    stems = ["甲", "甲", "乙", "乙"]  # Day stem 乙 → 木
    branches = ["寅", "寅", "卯", "卯"]
    strength = compute_day_master_strength(stems, branches)
    expected = compute_element_strengths(stems, branches)["木"]
    assert strength == pytest.approx(expected)
    assert 0.0 <= strength <= 1.0


def test_day_master_strength_explicit_stem():
    """Explicit day_master_stem overrides stems[2]."""
    stems = ["甲", "甲", "乙", "乙"]
    branches = ["寅", "寅", "卯", "卯"]
    # 丙 → 火; 火 share is small in this chart
    strength = compute_day_master_strength(stems, branches, day_master_stem="丙")
    expected = compute_element_strengths(stems, branches)["火"]
    assert strength == pytest.approx(expected)


def test_day_master_strength_unknown_stem_raises():
    """Invalid day_master_stem raises ValueError."""
    with pytest.raises(ValueError, match="known heavenly stem"):
        compute_day_master_strength(
            ["甲", "乙", "丙", "丁"],
            ["寅", "卯", "辰", "巳"],
            day_master_stem="X",
        )


# --- classify_day_master ---


def test_classify_day_master_身强():
    """Percentile >= strong_threshold (0.30) → 身强."""
    stems = ["甲", "甲", "乙", "乙"]
    branches = ["寅", "寅", "卯", "卯"]  # 木 dominant (~0.9)
    assert classify_day_master(stems, branches) == "身强"


def test_classify_day_master_身弱():
    """Percentile < weak_threshold (0.20) → 身弱."""
    stems = ["庚", "辛", "乙", "丙"]  # Day 乙 木; chart heavy 金/火
    branches = ["申", "酉", "巳", "午"]
    assert classify_day_master(stems, branches) == "身弱"


def test_classify_day_master_中和():
    """Percentile in [weak_threshold, strong_threshold) → 中和."""
    # 己戊庚辛 / 亥子寅巳 → 金 ≈ 0.2875 (test_strength_known_fractions); day stem 庚 → 金
    stems = ["己", "戊", "庚", "辛"]
    branches = ["亥", "子", "寅", "巳"]
    assert classify_day_master(stems, branches) == "中和"


def test_classify_day_master_custom_thresholds():
    """Custom strong/weak thresholds are respected."""
    stems = ["甲", "甲", "乙", "乙"]
    branches = ["寅", "寅", "卯", "卯"]
    assert classify_day_master(stems, branches, strong_threshold=0.95) == "中和"
    assert classify_day_master(stems, branches, weak_threshold=0.10) == "身强"
