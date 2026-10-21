"""
Tests for hidden stem (藏干) weight tables in bazi_engine.tables: get_hidden_stems and HIDE_GAN_WEIGHTS.
"""
import math

import pytest

from bazi_engine.enums import Branch, Stem
from bazi_engine.tables import HIDE_GAN_WEIGHTS, get_hidden_stems

# All 12 earthly branches (by value)
_ZHI = [b.value for b in Branch]


def test_weights_sum_to_one_per_branch():
    """Every branch's hidden stem weights sum to exactly 1.0."""
    for branch in Branch:
        pairs = get_hidden_stems(branch)
        total = sum(w for _, w in pairs)
        assert math.isclose(total, 1.0), f"Branch {branch.value}: weights sum to {total}"


def test_stem_and_weight_lengths_match():
    """Stem list and weight list have the same length for each branch in HIDE_GAN_WEIGHTS."""
    for zhi, pairs in HIDE_GAN_WEIGHTS.items():
        stems, weights = zip(*pairs) if pairs else ([], [])
        assert len(stems) == len(weights), f"Mismatch for {zhi}"


def test_get_hidden_stems_single():
    """Single hidden stem branch (子, 卯, 酉) returns one (Stem, 1.0) tuple."""
    for branch, expected_stem in [(Branch.子, Stem.癸), (Branch.卯, Stem.乙), (Branch.酉, Stem.辛)]:
        result = get_hidden_stems(branch)
        assert result == [(expected_stem, 1.0)]


def test_get_hidden_stems_two():
    """Two hidden stems (午, 亥) return 0.7 / 0.3."""
    result = get_hidden_stems(Branch.午)
    assert result == [(Stem.丁, 0.7), (Stem.己, 0.3)]
    result = get_hidden_stems(Branch.亥)
    assert result == [(Stem.壬, 0.7), (Stem.甲, 0.3)]


def test_get_hidden_stems_three():
    """Three hidden stems (e.g. 丑, 寅) return 0.6 / 0.3 / 0.1."""
    result = get_hidden_stems(Branch.丑)
    assert result == [(Stem.己, 0.6), (Stem.癸, 0.3), (Stem.辛, 0.1)]
    result = get_hidden_stems(Branch.寅)
    assert result == [(Stem.甲, 0.6), (Stem.丙, 0.3), (Stem.戊, 0.1)]


def test_invalid_branch_raises():
    """Invalid branch string when constructing Branch raises."""
    with pytest.raises(ValueError):
        Branch("X")
    with pytest.raises(ValueError):
        Branch("")


def test_all_stems_are_standard_ten():
    """All stems returned by get_hidden_stems are from the ten heavenly stems (Stem enum)."""
    standard = set(Stem)
    for branch in Branch:
        for stem, _ in get_hidden_stems(branch):
            assert stem in standard, f"Branch {branch.value}: invalid stem {stem}"


def test_hide_gan_weights_matches_get_hidden_stems():
    """HIDE_GAN_WEIGHTS (string table) is consistent with get_hidden_stems (Branch -> Stem, float)."""
    for branch in Branch:
        table_pairs = HIDE_GAN_WEIGHTS[branch.value]
        api_pairs = get_hidden_stems(branch)
        assert len(table_pairs) == len(api_pairs), f"Length mismatch for {branch.value}"
        for (gan_s, w_s), (stem, w_f) in zip(table_pairs, api_pairs):
            assert stem.value == gan_s, f"Stem mismatch for {branch.value}: {stem.value} vs {gan_s}"
            assert w_f == w_s, f"Weight mismatch for {branch.value}: {w_f} vs {w_s}"
