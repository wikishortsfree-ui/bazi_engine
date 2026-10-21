"""
地支六害 (earthly branch six harms) detection.

The six fixed harm pairs:
  子未, 丑午, 寅巳, 卯辰, 申亥, 酉戌

TODO: 害 is sometimes said to arise when a 六合 is broken by a clash; harms and
six-combinations are mutually exclusive for some pairs. The blocking logic
(when to suppress 害 if 六合 is present, or vice versa) will be handled in the
InteractionResult consolidation step.
"""
from __future__ import annotations

from bazi_engine.interactions._constants import iter_position_pairs, require_four_pillars

# All 6 harm pairs; immutable frozenset for order-independent lookup
BRANCH_SIX_HARMS: frozenset[frozenset[str]] = frozenset({
    frozenset({"子", "未"}),
    frozenset({"丑", "午"}),
    frozenset({"寅", "巳"}),
    frozenset({"卯", "辰"}),
    frozenset({"申", "亥"}),
    frozenset({"酉", "戌"}),
})


def _is_harm_pair(branch_a: str, branch_b: str) -> bool:
    """Return True if (branch_a, branch_b) is a 六害 pair."""
    return frozenset({branch_a, branch_b}) in BRANCH_SIX_HARMS


def detect_branch_harms(branches: list[str]) -> list[dict]:
    """
    Detect 地支六害 across the 4 natal branch positions.

    Positions are 0=Year, 1=Month, 2=Day, 3=Hour. Input must have 4 items.
    Returns one dict per harm pair with branch_a, branch_b, pos_a, pos_b.
    """
    require_four_pillars(branches, "branches")
    result: list[dict] = []
    for i, j in iter_position_pairs():
        if _is_harm_pair(branches[i], branches[j]):
            result.append({
                "branch_a": branches[i],
                "branch_b": branches[j],
                "pos_a": i,
                "pos_b": j,
            })
    return result
