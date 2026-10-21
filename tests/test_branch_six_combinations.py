"""Tests for 地支六合 (branch six combinations) detection."""
import pytest

from bazi_engine.interactions.branch_six_combinations import detect_branch_six_combinations


def test_branches_must_have_four_items():
    """Input must be exactly 4 branches."""
    with pytest.raises(ValueError, match="must have 4 items"):
        detect_branch_six_combinations(["子", "丑"])
    with pytest.raises(ValueError, match="must have 4 items"):
        detect_branch_six_combinations(["子", "丑", "寅", "卯", "辰"])


def test_no_combination():
    """Chart with no 六合 pair returns empty list."""
    # 子寅卯辰: no pair is among 子丑, 寅亥, 卯戌, 辰酉, 巳申, 午未
    branches = ["子", "寅", "卯", "辰"]
    assert detect_branch_six_combinations(branches) == []


def test_one_combination():
    """Chart with exactly one 六合 pair returns one dict."""
    # 子丑 at positions 0,1 → 土
    branches = ["子", "丑", "寅", "卯"]
    got = detect_branch_six_combinations(branches)
    assert len(got) == 1
    assert got[0]["branch_a"] == "子"
    assert got[0]["branch_b"] == "丑"
    assert got[0]["pos_a"] == 0
    assert got[0]["pos_b"] == 1
    assert got[0]["result_element"] == "土"


def test_one_combination_reversed_positions():
    """One 六合 pair with branches in reverse position order still reports pos_a < pos_b."""
    # 丑 at 0, 子 at 2 → 子丑→土 at (0, 2)
    branches = ["丑", "寅", "子", "卯"]
    got = detect_branch_six_combinations(branches)
    assert len(got) == 1
    assert got[0]["branch_a"] == "丑"
    assert got[0]["branch_b"] == "子"
    assert got[0]["pos_a"] == 0
    assert got[0]["pos_b"] == 2
    assert got[0]["result_element"] == "土"


def test_two_simultaneous_combinations():
    """Chart with two 六合 pairs returns two dicts."""
    # 子丑 at (0,1), 寅亥 at (2,3)
    branches = ["子", "丑", "寅", "亥"]
    got = detect_branch_six_combinations(branches)
    assert len(got) == 2
    by_pos = {(r["pos_a"], r["pos_b"]): r for r in got}
    assert (0, 1) in by_pos
    assert (2, 3) in by_pos
    assert by_pos[(0, 1)]["branch_a"] == "子"
    assert by_pos[(0, 1)]["branch_b"] == "丑"
    assert by_pos[(0, 1)]["result_element"] == "土"
    assert by_pos[(2, 3)]["branch_a"] == "寅"
    assert by_pos[(2, 3)]["branch_b"] == "亥"
    assert by_pos[(2, 3)]["result_element"] == "木"


def test_all_six_pair_types():
    """Each of the 6 pair types is detected with correct result element."""
    pairs_and_elements = [
        (["子", "丑", "寅", "卯"], 0, 1, "土"),
        (["寅", "亥", "子", "卯"], 0, 1, "木"),
        (["卯", "戌", "子", "寅"], 0, 1, "火"),
        (["辰", "酉", "子", "寅"], 0, 1, "金"),
        (["巳", "申", "子", "寅"], 0, 1, "水"),
        (["午", "未", "子", "寅"], 0, 1, "土"),
    ]
    for branches, pos_a, pos_b, result_element in pairs_and_elements:
        got = detect_branch_six_combinations(branches)
        assert len(got) >= 1, f"Expected at least one combo for {branches}"
        first = got[0]
        assert first["pos_a"] == pos_a and first["pos_b"] == pos_b
        assert first["result_element"] == result_element
