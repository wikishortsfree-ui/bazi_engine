"""
地支相刑 (earthly branch punishments) detection.

Four punishment categories:
1. 无恩之刑 (ungrateful): cyclic triplet 寅→巳→申→寅. Flag if 2 or 3 present.
2. 持势之刑 (bullying): cyclic triplet 丑→戌→未→丑. Flag if 2 or 3 present.
3. 无礼之刑 (disrespectful): mutual pair 子↔卯. Flag if both present.
4. 自刑 (self-punishment): 辰, 午, 酉, 亥 — flag only if same branch appears twice (strict).
"""
from __future__ import annotations

from bazi_engine.interactions._constants import require_four_pillars

# Cyclic triplets (order 寅→巳→申 and 丑→戌→未 for consistent reporting)
UNGRATEFUL_TRIPLET: tuple[str, str, str] = ("寅", "巳", "申")  # 无恩之刑
BULLYING_TRIPLET: tuple[str, str, str] = ("丑", "戌", "未")   # 持势之刑

# Mutual pair
DISRESPECTFUL_PAIR: tuple[str, str] = ("子", "卯")  # 无礼之刑, canonical order

# Self-punishment branches (自刑): must appear twice to trigger
SELF_PUNISHMENT_BRANCHES: frozenset[str] = frozenset({"辰", "午", "酉", "亥"})


def _check_cyclic_triplet(
    branches: list[str],
    triplet: tuple[str, str, str],
    punishment_type: str,
) -> dict | None:
    """If 2 or 3 of the triplet are present, return a result dict; else None."""
    members_present: list[str] = []
    positions: list[int] = []
    for b in triplet:  # preserve cycle order
        for i, br in enumerate(branches):
            if br == b:
                members_present.append(b)
                positions.append(i)
                break
    count = len(members_present)
    if count < 2:
        return None
    return {
        "punishment_type": punishment_type,
        "members_present": members_present,
        "positions": positions,
        "is_full": count == 3,
    }


def _check_disrespectful(branches: list[str]) -> dict | None:
    """If both 子 and 卯 are present, return result dict; else None."""
    pos_zi = next((i for i, b in enumerate(branches) if b == "子"), None)
    pos_mao = next((i for i, b in enumerate(branches) if b == "卯"), None)
    if pos_zi is None or pos_mao is None:
        return None
    return {
        "punishment_type": "无礼之刑",
        "branch_a": "子",
        "branch_b": "卯",
        "pos_a": pos_zi,
        "pos_b": pos_mao,
    }


def _check_self_punishments(branches: list[str]) -> list[dict]:
    """For each of 辰, 午, 酉, 亥, if it appears in two or more positions, return one dict."""
    result: list[dict] = []
    for branch in SELF_PUNISHMENT_BRANCHES:
        positions = [i for i, b in enumerate(branches) if b == branch]
        if len(positions) >= 2:
            result.append({
                "punishment_type": "自刑",
                "branch": branch,
                "positions": positions,
            })
    return result


def detect_punishments(branches: list[str]) -> list[dict]:
    """
    Detect 地支相刑 (earthly branch punishments) across the 4 natal branch positions.

    Positions are 0=Year, 1=Month, 2=Day, 3=Hour. Input must have 4 items.

    Returns a list of dicts:
    - Cyclic (无恩之刑, 持势之刑): punishment_type, members_present, positions, is_full.
    - 无礼之刑: punishment_type, branch_a, branch_b, pos_a, pos_b.
    - 自刑: punishment_type, branch, positions (only when same branch appears twice).
    """
    require_four_pillars(branches, "branches")
    cyclic_ungrateful = _check_cyclic_triplet(branches, UNGRATEFUL_TRIPLET, "无恩之刑")
    cyclic_bullying = _check_cyclic_triplet(branches, BULLYING_TRIPLET, "持势之刑")
    disrespectful = _check_disrespectful(branches)
    result = list(filter(None, [cyclic_ungrateful, cyclic_bullying, disrespectful]))
    result.extend(_check_self_punishments(branches))
    return result
