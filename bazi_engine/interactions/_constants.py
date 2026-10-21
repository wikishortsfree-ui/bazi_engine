"""
Shared constants and helpers for 4-pillar interaction detection.
"""
from __future__ import annotations

import itertools

from .._constants import PILLAR_COUNT, POSITION_LABELS
from ..elements import ELEMENT_GENERATES

# Elemental strength threshold for 合化 transformation (from strength_explanation.md)
TRANSFORMATION_STRENGTH_THRESHOLD = 0.30

def iter_position_pairs():
    """Yield (i, j) for each pair of pillar indices (i < j)."""
    return itertools.combinations(range(PILLAR_COUNT), 2)


def require_four_pillars(items: list, name: str) -> None:
    """Raise ValueError if len(items) != 4."""
    if len(items) != PILLAR_COUNT:
        raise ValueError(
            f"{name} must have {PILLAR_COUNT} items (Year, Month, Day, Hour), got {len(items)}"
        )


def build_pair_lookup(
    items: list[dict], key_a: str, key_b: str
) -> dict[frozenset, dict]:
    """Build frozenset({a, b}) -> dict for O(1) lookup by pair. Used by to_binary_vector."""
    return {frozenset([d[key_a], d[key_b]]): d for d in items}
