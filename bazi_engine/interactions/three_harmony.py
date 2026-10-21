"""
三合局 (three harmony) and 半合 (half-combo) detection.

The four 三合 groups and their central (main) branch:
  Water: 申, 子, 辰  (central: 子)
  Fire:  寅, 午, 戌  (central: 午)
  Metal: 巳, 酉, 丑  (central: 酉)
  Wood:  亥, 卯, 未  (central: 卯)
"""
from __future__ import annotations

from bazi_engine.interactions._constants import require_four_pillars

# element -> (triple of branches, central branch)
THREE_HARMONY_GROUPS: dict[str, tuple[tuple[str, str, str], str]] = {
    "Water": (("申", "子", "辰"), "子"),
    "Fire": (("寅", "午", "戌"), "午"),
    "Metal": (("巳", "酉", "丑"), "酉"),
    "Wood": (("亥", "卯", "未"), "卯"),
}


def detect_three_harmony(branches: list[str]) -> list[dict]:
    """
    Detect 三合局 (full) and 半合 (half) across the 4 natal branch positions.

    Positions are 0=Year, 1=Month, 2=Day, 3=Hour. Input must have 4 items.
    For each of the four 三合 groups (Water, Fire, Metal, Wood):
    - If all 3 members present → full 三合 (combo_type='full').
    - If exactly 2 members present → 半合 (combo_type='half'); has_central is True
      only when one of the two is the group's central branch.

    Returns a list of dicts with: element, combo_type, has_central,
    present_branches, positions (indices into branches).
    """
    require_four_pillars(branches, "branches")
    result: list[dict] = []
    for element, ((a, b, c), central) in THREE_HARMONY_GROUPS.items():
        triple = {a, b, c}
        present_branches: list[str] = []
        positions: list[int] = []
        for i, br in enumerate(branches):
            if br in triple:
                present_branches.append(br)
                positions.append(i)
        count = len(present_branches)
        if count == 3:
            result.append({
                "element": element,
                "combo_type": "full",
                "has_central": True,
                "present_branches": present_branches,
                "positions": positions,
            })
        elif count == 2:
            result.append({
                "element": element,
                "combo_type": "half",
                "has_central": central in present_branches,
                "present_branches": present_branches,
                "positions": positions,
            })
    return result
