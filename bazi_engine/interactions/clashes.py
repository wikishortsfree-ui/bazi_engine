"""
天干相冲 (heavenly stem clashes) and 地支六冲 (earthly branch clashes) detection.

Stem clashes (相冲): Yang vs Yang, Yin vs Yin across the 5-element cycle.
  Pairs (6): 甲庚, 乙辛, 丙壬, 丁癸, 戊甲, 己乙.

Branch clashes (六冲): 子午, 丑未, 寅申, 卯酉, 辰戌, 巳亥.
"""
from __future__ import annotations

from bazi_engine.interactions._constants import (
    POSITION_LABELS,
    iter_position_pairs,
    require_four_pillars,
)

# Canonical (stem_a, stem_b) for the 6 相冲 pairs.
STEM_CLASH_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("甲", "庚"),
    ("乙", "辛"),
    ("丙", "壬"),
    ("丁", "癸"),
    ("戊", "甲"),
    ("己", "乙"),
})

# All 6 六冲 branch pairs (canonical order for consistent reporting).
BRANCH_CLASH_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("子", "午"),
    ("丑", "未"),
    ("寅", "申"),
    ("卯", "酉"),
    ("辰", "戌"),
    ("巳", "亥"),
})


def _is_pair(a: str, b: str, pairs: frozenset[tuple[str, str]]) -> bool:
    return (a, b) in pairs or (b, a) in pairs


def _canonical_pair(a: str, b: str, pairs: frozenset[tuple[str, str]]) -> tuple[str, str]:
    """Return (item_a, item_b) in canonical order (match the given pairs set)."""
    return (a, b) if (a, b) in pairs else (b, a)


def detect_stem_clashes(stems: list[str]) -> list[dict]:
    """
    Detect 天干相冲 (heavenly stem clashes) across the 4 stem positions.

    Positions are 0=Year, 1=Month, 2=Day, 3=Hour. Input must have 4 items.
    All 6 pairs are used: 甲庚, 乙辛, 丙壬, 丁癸, 戊甲, 己乙.

    Returns a list of dicts with: item_a, item_b (canonical order), pos_a, pos_b
    (integer indices), pos_a_label, pos_b_label (e.g. 'year', 'month').
    """
    require_four_pillars(stems, "stems")
    result: list[dict] = []
    for i, j in iter_position_pairs():
        a, b = stems[i], stems[j]
        if _is_pair(a, b, STEM_CLASH_PAIRS):
            item_a, item_b = _canonical_pair(a, b, STEM_CLASH_PAIRS)
            pos_a = i if stems[i] == item_a else j
            pos_b = j if pos_a == i else i
            result.append({
                "item_a": item_a,
                "item_b": item_b,
                "pos_a": pos_a,
                "pos_b": pos_b,
                "pos_a_label": POSITION_LABELS[pos_a],
                "pos_b_label": POSITION_LABELS[pos_b],
            })
    return result


def detect_branch_clashes(branches: list[str]) -> list[dict]:
    """
    Detect 地支六冲 (six branch clashes) across the 4 branch positions.

    Positions are 0=Year, 1=Month, 2=Day, 3=Hour. Input must have 4 items.
    All 6 pairings are checked: 子午, 丑未, 寅申, 卯酉, 辰戌, 巳亥.

    Returns a list of dicts with: item_a, item_b (canonical order), pos_a, pos_b
    (integer indices), pos_a_label, pos_b_label (e.g. 'year', 'month').
    """
    require_four_pillars(branches, "branches")
    result: list[dict] = []
    for i, j in iter_position_pairs():
        a, b = branches[i], branches[j]
        if _is_pair(a, b, BRANCH_CLASH_PAIRS):
            item_a, item_b = _canonical_pair(a, b, BRANCH_CLASH_PAIRS)
            pos_a = i if branches[i] == item_a else j
            pos_b = j if pos_a == i else i
            result.append({
                "item_a": item_a,
                "item_b": item_b,
                "pos_a": pos_a,
                "pos_b": pos_b,
                "pos_a_label": POSITION_LABELS[pos_a],
                "pos_b_label": POSITION_LABELS[pos_b],
            })
    return result
