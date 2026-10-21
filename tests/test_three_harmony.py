"""Tests for 三合局 / 半合 (three harmony) detection."""
import pytest

from bazi_engine.interactions.three_harmony import detect_three_harmony


def test_branches_must_have_four_items():
    """Input must be exactly 4 branches."""
    with pytest.raises(ValueError, match="must have 4 items"):
        detect_three_harmony(["子", "丑"])
    with pytest.raises(ValueError, match="must have 4 items"):
        detect_three_harmony(["申", "子", "辰"])


def test_no_combo():
    """Chart with no 三合 or 半合 returns empty list."""
    # At most one from each group: 申(Water), 寅(Fire), 巳(Metal), 亥(Wood)
    branches = ["申", "寅", "巳", "亥"]
    assert detect_three_harmony(branches) == []


def test_full_three_harmony():
    """All three members of a 三合 group present → combo_type='full', has_central=True."""
    # Water 三合: 申, 子, 辰 at positions 0,1,2; 午 at 3
    branches = ["申", "子", "辰", "午"]
    got = detect_three_harmony(branches)
    assert len(got) == 1
    assert got[0]["element"] == "Water"
    assert got[0]["combo_type"] == "full"
    assert got[0]["has_central"] is True
    assert set(got[0]["present_branches"]) == {"申", "子", "辰"}
    assert sorted(got[0]["positions"]) == [0, 1, 2]


def test_half_combo_with_central():
    """Exactly two members present including the central branch → 半合, has_central=True."""
    # Water 半合: 申 and 子 (central) at 0,1; no 辰
    branches = ["申", "子", "寅", "卯"]
    got = detect_three_harmony(branches)
    assert len(got) == 1
    assert got[0]["element"] == "Water"
    assert got[0]["combo_type"] == "half"
    assert got[0]["has_central"] is True
    assert set(got[0]["present_branches"]) == {"申", "子"}
    assert sorted(got[0]["positions"]) == [0, 1]


def test_half_combo_without_central():
    """Exactly two members present but central missing → 半合, has_central=False."""
    # Water 半合: 申 and 辰 (no 子)
    branches = ["申", "寅", "辰", "卯"]
    got = detect_three_harmony(branches)
    assert len(got) == 1
    assert got[0]["element"] == "Water"
    assert got[0]["combo_type"] == "half"
    assert got[0]["has_central"] is False
    assert set(got[0]["present_branches"]) == {"申", "辰"}
    assert sorted(got[0]["positions"]) == [0, 2]
