"""Tests for 地支相破 (branch six destructions) detection."""
import pytest

from bazi_engine.interactions.destructions import detect_branch_destructions


def test_branches_must_have_four_items():
    """Input must be exactly 4 branches."""
    with pytest.raises(ValueError, match="must have 4 items"):
        detect_branch_destructions(["子", "酉"])
    with pytest.raises(ValueError, match="must have 4 items"):
        detect_branch_destructions(["子", "丑", "寅", "卯", "辰"])


def test_no_destruction():
    """Chart with no 相破 pair returns empty list."""
    # 子丑寅卯: no pair is among 子酉, 午卯, 寅亥, 巳申, 丑辰, 未戌
    branches = ["子", "丑", "寅", "卯"]
    assert detect_branch_destructions(branches) == []


def test_one_destruction():
    """Chart with exactly one 相破 pair returns one dict with branch_a, branch_b, pos_a, pos_b."""
    # 子酉 at positions 0, 1
    branches = ["子", "酉", "寅", "卯"]
    got = detect_branch_destructions(branches)
    assert len(got) == 1
    assert got[0]["branch_a"] == "子"
    assert got[0]["branch_b"] == "酉"
    assert got[0]["pos_a"] == 0
    assert got[0]["pos_b"] == 1


def test_one_destruction_other_positions():
    """One 相破 pair at (0, 2) reports correct positions."""
    branches = ["午", "寅", "卯", "辰"]  # 午卯 at 0, 2
    got = detect_branch_destructions(branches)
    assert len(got) == 1
    assert got[0]["branch_a"] == "午"
    assert got[0]["branch_b"] == "卯"
    assert got[0]["pos_a"] == 0
    assert got[0]["pos_b"] == 2


def test_two_destructions():
    """Chart with two 相破 pairs returns two dicts."""
    # 子酉 at (0, 1), 丑辰 at (2, 3)
    branches = ["子", "酉", "丑", "辰"]
    got = detect_branch_destructions(branches)
    assert len(got) == 2
    by_pos = {(r["pos_a"], r["pos_b"]): r for r in got}
    assert (0, 1) in by_pos
    assert by_pos[(0, 1)]["branch_a"] == "子"
    assert by_pos[(0, 1)]["branch_b"] == "酉"
    assert (2, 3) in by_pos
    assert by_pos[(2, 3)]["branch_a"] == "丑"
    assert by_pos[(2, 3)]["branch_b"] == "辰"


def test_all_six_destruction_pair_types():
    """Each of the 6 相破 pair types is detected."""
    pairs = [
        ["子", "酉", "寅", "卯"],
        ["午", "卯", "子", "寅"],
        ["寅", "亥", "子", "卯"],
        ["巳", "申", "子", "寅"],
        ["丑", "辰", "子", "寅"],
        ["未", "戌", "子", "寅"],
    ]
    for branches in pairs:
        got = detect_branch_destructions(branches)
        assert len(got) >= 1, f"Expected at least one destruction for {branches}"
        assert got[0]["pos_a"] < got[0]["pos_b"]
        assert (got[0]["branch_a"], got[0]["branch_b"]) in {
            ("子", "酉"), ("午", "卯"), ("寅", "亥"), ("巳", "申"), ("丑", "辰"), ("未", "戌"),
        }


def test_destruction_overlap_with_six_combination():
    """寅亥 and 巳申 are both 相破 and 六合; destruction is still reported (consolidation elsewhere)."""
    # 寅亥: 相破 and 六合 → we only assert destruction is detected here
    branches_yinhai = ["寅", "亥", "子", "卯"]
    got_yinhai = detect_branch_destructions(branches_yinhai)
    assert len(got_yinhai) == 1
    assert (got_yinhai[0]["branch_a"], got_yinhai[0]["branch_b"]) in {("寅", "亥"), ("亥", "寅")}
    # 巳申: 相破 and 六合
    branches_shenshen = ["巳", "申", "子", "寅"]
    got_shenshen = detect_branch_destructions(branches_shenshen)
    assert len(got_shenshen) == 1
    assert (got_shenshen[0]["branch_a"], got_shenshen[0]["branch_b"]) in {("巳", "申"), ("申", "巳")}
