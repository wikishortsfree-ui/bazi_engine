"""
地支六合 (earthly branch six combinations) detection.

The six fixed pairs and their result elements:
  子丑→土, 寅亥→木, 卯戌→火, 辰酉→金, 巳申→水, 午未→土
"""
from __future__ import annotations

from bazi_engine.interactions._constants import iter_position_pairs, require_four_pillars

# Canonical (branch_a, branch_b) -> result_element; lookup supports either order
BRANCH_SIX_PAIRS: dict[tuple[str, str], str] = {
    ("子", "丑"): "土",
    ("寅", "亥"): "木",
    ("卯", "戌"): "火",
    ("辰", "酉"): "金",
    ("巳", "申"): "水",
    ("午", "未"): "土",
}


def _lookup_result(branch_a: str, branch_b: str) -> str | None:
    """Return result element if (branch_a, branch_b) is a 六合 pair, else None."""
    return BRANCH_SIX_PAIRS.get((branch_a, branch_b)) or BRANCH_SIX_PAIRS.get((branch_b, branch_a))


def detect_branch_six_combinations(branches: list[str]) -> list[dict]:
    """
    Detect 地支六合 across the 4 natal branch positions.

    Positions are 0=Year, 1=Month, 2=Day, 3=Hour. Input must have 4 items.
    Returns one dict per matching pair with branch_a, branch_b, pos_a, pos_b, result_element.
    """
    require_four_pillars(branches, "branches")
    result: list[dict] = []
    for i, j in iter_position_pairs():
        elem = _lookup_result(branches[i], branches[j])
        if elem is not None:
            result.append({
                "branch_a": branches[i],
                "branch_b": branches[j],
                "pos_a": i,
                "pos_b": j,
                "result_element": elem,
            })
    return result
