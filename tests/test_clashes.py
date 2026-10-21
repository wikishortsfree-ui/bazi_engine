"""Tests for 天干相冲 (stem clashes) and 地支六冲 (branch clashes) detection."""
import pytest

from bazi_engine.interactions.clashes import (
    detect_branch_clashes,
    detect_stem_clashes,
)


def test_stems_must_have_four_items():
    """Stems must be exactly 4 items."""
    with pytest.raises(ValueError, match="stems must have 4 items"):
        detect_stem_clashes(["甲", "乙"])
    with pytest.raises(ValueError, match="stems must have 4 items"):
        detect_stem_clashes(["甲", "乙", "丙", "丁", "戊"])


def test_branches_must_have_four_items():
    """Branches must be exactly 4 items."""
    with pytest.raises(ValueError, match="branches must have 4 items"):
        detect_branch_clashes(["子", "午"])
    with pytest.raises(ValueError, match="branches must have 4 items"):
        detect_branch_clashes(["子", "丑", "寅", "卯", "辰"])


def test_no_stem_clash():
    """Chart with no 相冲 pair returns empty list."""
    # 甲己丙丁: 甲己合, 丙丁 no clash with others
    stems = ["甲", "己", "丙", "丁"]
    assert detect_stem_clashes(stems) == []


def test_stem_clash_one():
    """One 相冲 pair (甲庚) at Year and Day returns one dict with positions and labels."""
    stems = ["甲", "乙", "庚", "辛"]  # 甲庚 at 0,2; 乙辛 at 1,3
    got = detect_stem_clashes(stems)
    assert len(got) == 2
    by_pair = {(r["item_a"], r["item_b"]): r for r in got}
    assert ("甲", "庚") in by_pair
    assert ("乙", "辛") in by_pair
    r_ag = by_pair[("甲", "庚")]
    assert r_ag["pos_a"] == 0 and r_ag["pos_b"] == 2
    assert r_ag["pos_a_label"] == "year" and r_ag["pos_b_label"] == "day"
    r_yx = by_pair[("乙", "辛")]
    assert r_yx["pos_a"] == 1 and r_yx["pos_b"] == 3
    assert r_yx["pos_a_label"] == "month" and r_yx["pos_b_label"] == "hour"


def test_stem_clash_canonical_order():
    """Stem clash reports item_a, item_b in canonical order (甲 before 庚)."""
    stems = ["庚", "乙", "丙", "甲"]  # 甲 at 3, 庚 at 0 → 甲庚
    got = detect_stem_clashes(stems)
    assert len(got) == 1
    assert got[0]["item_a"] == "甲" and got[0]["item_b"] == "庚"
    assert got[0]["pos_a"] == 3 and got[0]["pos_b"] == 0
    assert got[0]["pos_a_label"] == "hour" and got[0]["pos_b_label"] == "year"


def test_stem_clash_wu_jia_ji_yi():
    """戊甲 and 己乙 are detected as stem clashes (相冲)."""
    # 戊甲: 戊 at 0, 甲 at 2
    stems_wu_jia = ["戊", "乙", "甲", "辛"]
    got = detect_stem_clashes(stems_wu_jia)
    by_pair = {(r["item_a"], r["item_b"]): r for r in got}
    assert ("戊", "甲") in by_pair
    assert by_pair[("戊", "甲")]["pos_a"] == 0 and by_pair[("戊", "甲")]["pos_b"] == 2
    # 己乙: 己 at 1, 乙 at 3
    stems_ji_yi = ["丙", "己", "壬", "乙"]
    got2 = detect_stem_clashes(stems_ji_yi)
    by_pair2 = {(r["item_a"], r["item_b"]): r for r in got2}
    assert ("己", "乙") in by_pair2
    assert by_pair2[("己", "乙")]["pos_a"] == 1 and by_pair2[("己", "乙")]["pos_b"] == 3


def test_no_branch_clash():
    """Chart with no 六冲 pair returns empty list."""
    branches = ["子", "丑", "寅", "卯"]
    assert detect_branch_clashes(branches) == []


def test_branch_clash_one():
    """One 六冲 pair (子午) at Year and Month returns one dict with positions and labels."""
    branches = ["子", "午", "寅", "卯"]
    got = detect_branch_clashes(branches)
    assert len(got) == 1
    assert got[0]["item_a"] == "子" and got[0]["item_b"] == "午"
    assert got[0]["pos_a"] == 0 and got[0]["pos_b"] == 1
    assert got[0]["pos_a_label"] == "year" and got[0]["pos_b_label"] == "month"


def test_branch_clash_multiple():
    """Chart with two 六冲 pairs returns two dicts."""
    branches = ["子", "午", "丑", "未"]  # 子午 at (0,1), 丑未 at (2,3)
    got = detect_branch_clashes(branches)
    assert len(got) == 2
    by_pos = {(r["pos_a"], r["pos_b"]): r for r in got}
    assert (0, 1) in by_pos and by_pos[(0, 1)]["item_a"] == "子" and by_pos[(0, 1)]["item_b"] == "午"
    assert (2, 3) in by_pos and by_pos[(2, 3)]["item_a"] == "丑" and by_pos[(2, 3)]["item_b"] == "未"


def test_branch_clash_all_six_types():
    """Each of the 6 六冲 pair types is detected."""
    pairs = [
        ["子", "午", "寅", "卯"],
        ["丑", "未", "子", "寅"],
        ["寅", "申", "子", "丑"],
        ["卯", "酉", "子", "丑"],
        ["辰", "戌", "子", "丑"],
        ["巳", "亥", "子", "丑"],
    ]
    for branches in pairs:
        got = detect_branch_clashes(branches)
        assert len(got) >= 1, f"Expected at least one clash for {branches}"
        assert got[0]["pos_a"] < got[0]["pos_b"]
        assert (got[0]["item_a"], got[0]["item_b"]) in {
            ("子", "午"), ("丑", "未"), ("寅", "申"), ("卯", "酉"), ("辰", "戌"), ("巳", "亥"),
        }
