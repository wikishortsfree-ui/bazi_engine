"""Tests for 地支六害 (branch six harms) detection."""
import pytest

from bazi_engine.interactions.harms import detect_branch_harms


def test_branches_must_have_four_items():
    """Input must be exactly 4 branches."""
    with pytest.raises(ValueError, match="must have 4 items"):
        detect_branch_harms(["子", "未"])
    with pytest.raises(ValueError, match="must have 4 items"):
        detect_branch_harms(["子", "丑", "寅", "卯", "辰"])


def test_no_harm():
    """Chart with no 六害 pair returns empty list."""
    # 子丑寅卯: no pair is among 子未, 丑午, 寅巳, 卯辰, 申亥, 酉戌
    branches = ["子", "丑", "寅", "卯"]
    assert detect_branch_harms(branches) == []


def test_one_harm():
    """Chart with exactly one 六害 pair returns one dict."""
    # 子未 at positions 0, 1
    branches = ["子", "未", "寅", "卯"]
    got = detect_branch_harms(branches)
    assert len(got) == 1
    assert got[0]["branch_a"] == "子"
    assert got[0]["branch_b"] == "未"
    assert got[0]["pos_a"] == 0
    assert got[0]["pos_b"] == 1


def test_one_harm_reversed_positions():
    """One 六害 pair with branches in reverse position order still reports pos_a < pos_b."""
    # 未 at 0, 子 at 2 → 子未 at (0, 2)
    branches = ["未", "寅", "子", "卯"]
    got = detect_branch_harms(branches)
    assert len(got) == 1
    assert got[0]["branch_a"] == "未"
    assert got[0]["branch_b"] == "子"
    assert got[0]["pos_a"] == 0
    assert got[0]["pos_b"] == 2


def test_two_simultaneous_harms():
    """Chart with two 六害 pairs returns two dicts."""
    # 子未 at (0, 1), 寅巳 at (2, 3)
    branches = ["子", "未", "寅", "巳"]
    got = detect_branch_harms(branches)
    assert len(got) == 2
    by_pos = {(r["pos_a"], r["pos_b"]): r for r in got}
    assert (0, 1) in by_pos
    assert (2, 3) in by_pos
    assert by_pos[(0, 1)]["branch_a"] == "子"
    assert by_pos[(0, 1)]["branch_b"] == "未"
    assert by_pos[(2, 3)]["branch_a"] == "寅"
    assert by_pos[(2, 3)]["branch_b"] == "巳"


def test_all_six_harm_pair_types():
    """Each of the 6 harm pair types is detected with correct positions."""
    # 子未, 丑午, 寅巳, 卯辰, 申亥, 酉戌
    cases = [
        (["子", "未", "寅", "卯"], 0, 1, "子", "未"),
        (["丑", "午", "寅", "卯"], 0, 1, "丑", "午"),
        (["寅", "巳", "子", "卯"], 0, 1, "寅", "巳"),
        (["卯", "辰", "子", "寅"], 0, 1, "卯", "辰"),
        (["申", "亥", "子", "寅"], 0, 1, "申", "亥"),
        (["酉", "戌", "子", "寅"], 0, 1, "酉", "戌"),
    ]
    for branches, pos_a, pos_b, br_a, br_b in cases:
        got = detect_branch_harms(branches)
        assert len(got) >= 1, f"Expected at least one harm for {branches}"
        first = got[0]
        assert first["pos_a"] == pos_a and first["pos_b"] == pos_b
        assert {first["branch_a"], first["branch_b"]} == {br_a, br_b}


def test_harm_at_non_adjacent_positions():
    """六害 pair at (0, 3) is detected."""
    branches = ["子", "寅", "卯", "未"]
    got = detect_branch_harms(branches)
    assert len(got) == 1
    assert got[0]["pos_a"] == 0
    assert got[0]["pos_b"] == 3
    assert {got[0]["branch_a"], got[0]["branch_b"]} == {"子", "未"}
