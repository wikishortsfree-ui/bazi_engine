"""Unit tests for 天干合 (stem combinations) detection — detect_stem_combinations."""
import pytest

from bazi_engine.interactions.stem_combinations import detect_stem_combinations


def test_stems_must_have_four_items():
    """Stems must be exactly 4 items."""
    with pytest.raises(ValueError, match="stems must have 4 items"):
        detect_stem_combinations(["甲", "己"])
    with pytest.raises(ValueError, match="stems must have 4 items"):
        detect_stem_combinations(["甲", "己", "丙", "丁", "戊"])


def test_no_combination_returns_empty():
    """Chart with no 天干合 pair returns empty list."""
    stems = ["甲", "丙", "戊", "庚"]  # no pair from (甲己, 乙庚, 丙辛, 丁壬, 戊癸)
    assert detect_stem_combinations(stems) == []


# --- Five combination pairs individually ---

@pytest.mark.parametrize(
    "stems,stem_a,stem_b,result_element",
    [
        (["甲", "己", "丙", "丁"], "甲", "己", "土"),
        (["乙", "庚", "丙", "丁"], "乙", "庚", "金"),
        (["丙", "辛", "甲", "丁"], "丙", "辛", "水"),
        (["丁", "壬", "甲", "丙"], "丁", "壬", "木"),
        (["戊", "癸", "甲", "丙"], "戊", "癸", "火"),
    ],
)
def test_each_combination_pair_individually(stems, stem_a, stem_b, result_element):
    """Each of the five 天干合 pairs is detected with correct result element."""
    got = detect_stem_combinations(stems)
    assert len(got) == 1
    c = got[0]
    assert c["stem_a"] == stem_a and c["stem_b"] == stem_b
    assert c["result_element"] == result_element
    assert c["pos_a"] == 0 and c["pos_b"] == 1
    assert c["is_pure"] is True


def test_is_pure_true_when_no_blocking():
    """When blocked_positions is None or empty, combinations are reported as is_pure=True."""
    stems = ["甲", "己", "丙", "丁"]
    no_block = detect_stem_combinations(stems)
    empty_block = detect_stem_combinations(stems, blocked_positions=set())
    assert len(no_block) == 1 and no_block[0]["is_pure"] is True
    assert len(empty_block) == 1 and empty_block[0]["is_pure"] is True


def test_is_pure_false_when_blocked_positions_provided():
    """When either position is in blocked_positions, is_pure is False."""
    stems = ["甲", "己", "丙", "丁"]  # 甲己 at pos 0, 1
    blocked_both = detect_stem_combinations(stems, blocked_positions={0, 1})
    assert len(blocked_both) == 1
    assert blocked_both[0]["is_pure"] is False

    blocked_one = detect_stem_combinations(stems, blocked_positions={0})
    assert len(blocked_one) == 1
    assert blocked_one[0]["is_pure"] is False

    blocked_other = detect_stem_combinations(stems, blocked_positions={1})
    assert len(blocked_other) == 1
    assert blocked_other[0]["is_pure"] is False


def test_multiple_simultaneous_combinations():
    """Chart with multiple 天干合 pairs returns one record per pair."""
    # 甲己 at (0,1), 乙庚 at (2,3)
    stems = ["甲", "己", "乙", "庚"]
    got = detect_stem_combinations(stems)
    assert len(got) == 2
    by_pair = {(c["stem_a"], c["stem_b"]): c for c in got}
    assert ("甲", "己") in by_pair
    assert ("乙", "庚") in by_pair
    assert by_pair[("甲", "己")]["pos_a"] == 0 and by_pair[("甲", "己")]["pos_b"] == 1
    assert by_pair[("甲", "己")]["result_element"] == "土"
    assert by_pair[("乙", "庚")]["pos_a"] == 2 and by_pair[("乙", "庚")]["pos_b"] == 3
    assert by_pair[("乙", "庚")]["result_element"] == "金"
    assert all(c["is_pure"] is True for c in got)


def test_multiple_combinations_with_partial_blocking():
    """Multiple combinations with blocked_positions set some is_pure to False."""
    stems = ["甲", "己", "乙", "庚"]  # 甲己 (0,1), 乙庚 (2,3)
    # Block positions 0 and 2 → 甲己 and 乙庚 both impure
    got = detect_stem_combinations(stems, blocked_positions={0, 2})
    assert len(got) == 2
    assert all(c["is_pure"] is False for c in got)
    # Block only 1 → 甲己 impure, 乙庚 pure
    got2 = detect_stem_combinations(stems, blocked_positions={1})
    assert len(got2) == 2
    by_pair = {(c["stem_a"], c["stem_b"]): c for c in got2}
    assert by_pair[("甲", "己")]["is_pure"] is False
    assert by_pair[("乙", "庚")]["is_pure"] is True


def test_first_occurrence_per_stem_used():
    """When a stem appears in multiple positions, first occurrence is used for the pair."""
    # 甲 at 0 and 2, 己 at 1 → one 甲己 combo at (0, 1)
    stems = ["甲", "己", "甲", "庚"]
    got = detect_stem_combinations(stems)
    assert len(got) == 1
    assert got[0]["stem_a"] == "甲" and got[0]["stem_b"] == "己"
    assert got[0]["pos_a"] == 0 and got[0]["pos_b"] == 1
