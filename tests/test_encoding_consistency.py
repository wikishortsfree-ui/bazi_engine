"""Unit tests for encoding tables: round-trip and polarity."""
from __future__ import annotations

import pytest

from bazi_engine.encodings import (
    BRANCH_TO_ID,
    ID_TO_BRANCH,
    ID_TO_STEM,
    STEM_TO_ID,
    branch_polarity,
    stem_polarity,
)


def test_stem_round_trip():
    """Stem ID → stem character via reverse lookup matches STEM_TO_ID."""
    for char, idx in STEM_TO_ID.items():
        assert ID_TO_STEM[idx] == char


def test_branch_round_trip():
    """Branch ID → branch character via reverse lookup matches BRANCH_TO_ID."""
    for char, idx in BRANCH_TO_ID.items():
        assert ID_TO_BRANCH[idx] == char


def test_stem_polarity_甲_yang():
    """甲 is yang (index 0, even) → polarity 1."""
    assert stem_polarity("甲") == 1


def test_stem_polarity_乙_yin():
    """乙 is yin (index 1, odd) → polarity 0."""
    assert stem_polarity("乙") == 0


def test_branch_polarity():
    """Branches: even index = yang(1), odd = yin(0)."""
    assert branch_polarity("子") == 1  # 0
    assert branch_polarity("丑") == 0  # 1
