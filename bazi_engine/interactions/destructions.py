"""
地支相破 (earthly branch six destructions) detection.

The 6 destruction pairs: 子酉, 午卯, 寅亥, 巳申, 丑辰, 未戌.

Note: 寅亥 and 巳申 also appear in the six-combinations (六合) list. A destruction
and a combination between the same pair is considered to cancel or weaken both;
resolution is handled in the InteractionResult consolidation step.
"""
from __future__ import annotations

from bazi_engine.interactions._constants import iter_position_pairs, require_four_pillars

# All 6 相破 branch pairs (canonical order for consistent reporting).
BRANCH_DESTRUCTION_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("子", "酉"),
    ("午", "卯"),
    ("寅", "亥"),
    ("巳", "申"),
    ("丑", "辰"),
    ("未", "戌"),
})


def _is_branch_destruction(a: str, b: str) -> bool:
    return (a, b) in BRANCH_DESTRUCTION_PAIRS or (b, a) in BRANCH_DESTRUCTION_PAIRS


def detect_branch_destructions(branches: list[str]) -> list[dict]:
    """
    Detect 地支相破 (six destructions) across the 4 natal branch positions.

    Positions are 0=Year, 1=Month, 2=Day, 3=Hour. Input must have 4 items.
    Returns a list of dicts with keys: branch_a, branch_b, pos_a, pos_b.
    """
    require_four_pillars(branches, "branches")
    result: list[dict] = []
    for i, j in iter_position_pairs():
        if _is_branch_destruction(branches[i], branches[j]):
            result.append({
                "branch_a": branches[i],
                "branch_b": branches[j],
                "pos_a": i,
                "pos_b": j,
            })
    return result
