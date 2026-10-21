"""Tests for 地支相刑 (earthly branch punishments) detection."""
import pytest

from bazi_engine.interactions.punishments import detect_punishments


def test_branches_must_have_four_items():
    """Branches must be exactly 4 items."""
    with pytest.raises(ValueError, match="branches must have 4 items"):
        detect_punishments(["寅", "巳"])
    with pytest.raises(ValueError, match="branches must have 4 items"):
        detect_punishments(["寅", "巳", "申", "子", "丑"])


def test_no_punishments():
    """Chart with no 刑 returns empty list."""
    # No 无恩 (need 2+ of 寅巳申): only 申. No 持势 (need 2+ of 丑戌未): only 戌.
    # No 无礼 (need both 子卯): only 子. No 自刑 (need duplicate 辰午酉亥): 午 and 子, no duplicate.
    branches = ["申", "戌", "午", "子"]
    assert detect_punishments(branches) == []


def test_ungrateful_partial_two_of_three():
    """无恩之刑: 2 of 寅巳申 present → one result, is_full False."""
    branches = ["寅", "巳", "子", "午"]
    got = detect_punishments(branches)
    assert len(got) == 1
    r = got[0]
    assert r["punishment_type"] == "无恩之刑"
    assert r["members_present"] == ["寅", "巳"]
    assert r["positions"] == [0, 1]
    assert r["is_full"] is False


def test_ungrateful_full_three_present():
    """无恩之刑: all 3 of 寅巳申 present → is_full True."""
    branches = ["寅", "巳", "申", "子"]
    got = detect_punishments(branches)
    assert len(got) == 1
    r = got[0]
    assert r["punishment_type"] == "无恩之刑"
    assert r["members_present"] == ["寅", "巳", "申"]
    assert r["positions"] == [0, 1, 2]
    assert r["is_full"] is True


def test_ungrateful_positions_out_of_order():
    """无恩之刑: members_present and positions in cycle order 寅→巳→申."""
    branches = ["申", "子", "寅", "巳"]  # 申@0, 寅@2, 巳@3
    got = detect_punishments(branches)
    assert len(got) == 1
    r = got[0]
    assert r["members_present"] == ["寅", "巳", "申"]
    assert r["positions"] == [2, 3, 0]


def test_bullying_partial_two_of_three():
    """持势之刑: 2 of 丑戌未 present → one result, is_full False."""
    branches = ["丑", "戌", "子", "寅"]
    got = detect_punishments(branches)
    assert len(got) == 1
    r = got[0]
    assert r["punishment_type"] == "持势之刑"
    assert r["members_present"] == ["丑", "戌"]
    assert r["positions"] == [0, 1]
    assert r["is_full"] is False


def test_bullying_full_three_present():
    """持势之刑: all 3 of 丑戌未 present → is_full True."""
    branches = ["丑", "戌", "未", "子"]
    got = detect_punishments(branches)
    assert len(got) == 1
    r = got[0]
    assert r["punishment_type"] == "持势之刑"
    assert r["members_present"] == ["丑", "戌", "未"]
    assert r["positions"] == [0, 1, 2]
    assert r["is_full"] is True


def test_disrespectful_both_present():
    """无礼之刑: 子 and 卯 both present → branch_a 子, branch_b 卯, pos_a, pos_b."""
    branches = ["子", "卯", "寅", "午"]
    got = detect_punishments(branches)
    assert len(got) == 1
    r = got[0]
    assert r["punishment_type"] == "无礼之刑"
    assert r["branch_a"] == "子" and r["branch_b"] == "卯"
    assert r["pos_a"] == 0 and r["pos_b"] == 1


def test_disrespectful_positions_swapped():
    """无礼之刑: 卯 before 子 still reports branch_a 子, branch_b 卯 with correct positions."""
    branches = ["卯", "丑", "子", "午"]  # only 无礼 (子卯); no 无恩/持势 triplets
    got = detect_punishments(branches)
    disrespectful = [r for r in got if r["punishment_type"] == "无礼之刑"]
    assert len(disrespectful) == 1
    r = disrespectful[0]
    assert r["branch_a"] == "子" and r["branch_b"] == "卯"
    assert r["pos_a"] == 2 and r["pos_b"] == 0


def test_self_punishment_same_branch_twice():
    """自刑: same branch (辰/午/酉/亥) in two positions → one dict per such branch."""
    branches = ["辰", "子", "辰", "丑"]  # 辰 at 0 and 2
    got = detect_punishments(branches)
    assert len(got) == 1
    r = got[0]
    assert r["punishment_type"] == "自刑"
    assert r["branch"] == "辰"
    assert r["positions"] == [0, 2]


def test_self_punishment_four_positions_same_branch():
    """自刑: same self-punishment branch in 3 positions still reported once with all positions."""
    branches = ["午", "午", "午", "子"]
    got = detect_punishments(branches)
    assert len(got) == 1
    r = got[0]
    assert r["punishment_type"] == "自刑"
    assert r["branch"] == "午"
    assert r["positions"] == [0, 1, 2]


def test_self_punishment_single_appearance_not_flagged():
    """自刑 (strict): 辰/午/酉/亥 appearing once only is not flagged."""
    branches = ["辰", "丑", "寅", "亥"]  # 辰 once, 亥 once
    got = detect_punishments(branches)
    assert len(got) == 0


def test_partial_triplet_one_member_no_ungrateful():
    """Only 1 of 寅巳申 present → no 无恩之刑."""
    branches = ["寅", "子", "午", "亥"]
    got = detect_punishments(branches)
    assert not any(r["punishment_type"] == "无恩之刑" for r in got)


def test_partial_triplet_one_member_no_bullying():
    """Only 1 of 丑戌未 present → no 持势之刑."""
    branches = ["丑", "子", "寅", "卯"]
    got = detect_punishments(branches)
    assert not any(r["punishment_type"] == "持势之刑" for r in got)


def test_no_disrespectful_with_only_zi():
    """Only 子 (no 卯) → no 无礼之刑."""
    branches = ["子", "丑", "寅", "午"]
    got = detect_punishments(branches)
    assert not any(r["punishment_type"] == "无礼之刑" for r in got)


def test_chart_with_multiple_punishment_types():
    """Chart with 无恩之刑 (2) and 无礼之刑 returns two dicts."""
    branches = ["寅", "巳", "子", "卯"]
    got = detect_punishments(branches)
    assert len(got) == 2
    types = {r["punishment_type"] for r in got}
    assert "无恩之刑" in types and "无礼之刑" in types


def test_chart_three_types():
    """Chart with 无恩之刑, 持势之刑, and 无礼之刑 (e.g. 寅巳丑戌子卯 + 2 more)."""
    # 寅巳申 = 无恩 full; 丑戌未 = 持势 full; 子卯 = 无礼. Need 4 branches: 寅,巳,子,卯 → 无恩(2)+无礼
    # For 3 types in 4 pillars: 无恩 needs 寅巳申 (3), 持势 needs 丑戌未 (3), 无礼 needs 子卯 (2). Total 3+3+2=8 slots but we have 4. So we can have at most 2 types in one 4-pillar chart (e.g. 无恩 partial + 无礼, or 持势 partial + 无礼). So test two types.
    # Add 自刑: 辰辰 + 子卯 → 自刑 + 无礼. So 辰, 辰, 子, 卯 → 自刑 (辰) + 无礼 (子卯). Good.
    branches = ["辰", "辰", "子", "卯"]
    got = detect_punishments(branches)
    assert len(got) == 2
    types = {r["punishment_type"] for r in got}
    assert "自刑" in types and "无礼之刑" in types


def test_self_punishment_two_different_branches_duplicated():
    """Two different 自刑 branches each appearing twice → two dicts."""
    branches = ["辰", "午", "辰", "午"]
    got = detect_punishments(branches)
    assert len(got) == 2
    by_branch = {r["branch"]: r for r in got}
    assert "辰" in by_branch and "午" in by_branch
    assert by_branch["辰"]["positions"] == [0, 2]
    assert by_branch["午"]["positions"] == [1, 3]
