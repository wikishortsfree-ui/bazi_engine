"""
Consolidated interaction result and detection entry point.

Aggregates all 8 interaction types (天干合, 六合, 三合, 天干冲, 六冲, 六害, 刑, 破)
and applies blocking/interference logic between them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bazi_engine.interactions._constants import (
    ELEMENT_GENERATES,
    PILLAR_COUNT,
    TRANSFORMATION_STRENGTH_THRESHOLD,
    build_pair_lookup,
)
from bazi_engine.interactions.branch_six_combinations import (
    BRANCH_SIX_PAIRS,
    detect_branch_six_combinations,
)
from bazi_engine.interactions.clashes import (
    BRANCH_CLASH_PAIRS,
    STEM_CLASH_PAIRS,
    detect_branch_clashes,
    detect_stem_clashes,
)
from bazi_engine.interactions.destructions import (
    BRANCH_DESTRUCTION_PAIRS,
    detect_branch_destructions,
)
from bazi_engine.interactions.harms import detect_branch_harms
from bazi_engine.interactions.punishments import detect_punishments
from bazi_engine.interactions.stem_combinations import (
    STEM_COMBINATION_PAIRS,
    detect_stem_combinations,
)
from bazi_engine.interactions.three_harmony import (
    THREE_HARMONY_GROUPS,
    detect_three_harmony,
)
from bazi_engine.enums import Branch
from bazi_engine.strength import compute_element_strengths


def _month_supports_element(month_branch: str, result_element: str) -> bool:
    """True if month branch supports result element: equal or 生 (generates). Seasonal support for 合化."""
    month_el = Branch(month_branch).element
    return month_el == result_element or ELEMENT_GENERATES.get(month_el) == result_element


def _check_transformation(
    result_element: str,
    month_branch: str,
    element_strengths: dict[str, float],
    threshold: float,
    no_clash_flag: bool,
) -> bool:
    """True if 化 conditions are met: seasonal support, strength >= threshold, and no clash."""
    if not no_clash_flag:
        return False
    if not _month_supports_element(month_branch, result_element):
        return False
    # Boundary is inclusive: exactly at threshold (e.g. 0.30) activates transformation.
    return element_strengths.get(result_element, 0.0) >= threshold


@dataclass
class InteractionResult:
    """Result of all stem/branch interactions for a 4-pillar chart."""

    stem_combinations: list[dict]
    branch_six_combinations: list[dict]
    three_harmony: list[dict]
    stem_clashes: list[dict]
    branch_clashes: list[dict]
    harms: list[dict]
    punishments: list[dict]
    destructions: list[dict]

    def to_binary_vector(self) -> np.ndarray:
        """
        Return a flat binary numpy array of all interaction features in fixed order.

        Layout is frozen for model compatibility. Full spec (position encoding + 64-dim
        ordering, handling of variable-length lists): docs/binary_vector_and_positions_spec.md

        Vector layout (encoder input spec):
        [0:5]    stem_combinations present (5 pairs: 甲己, 乙庚, 丙辛, 丁壬, 戊癸)
        [5:10]   stem_combinations is_pure (same order)
        [10:16]  branch_six_combinations present (6 pairs: 子丑, 寅亥, 卯戌, 辰酉, 巳申, 午未)
        [16:22]  branch_six_combinations is_blocked (same order)
        [22:26]  three_harmony full (4 elements: Water, Fire, Metal, Wood)
        [26:30]  three_harmony half (same order)
        [30:36]  stem_clashes present (6 pairs: 甲庚, 乙辛, 丙壬, 丁癸, 戊甲, 己乙; order = sorted(STEM_CLASH_PAIRS))
        [36:42]  branch_clashes present (6 pairs: 子午, 丑未, 寅申, 卯酉, 辰戌, 巳亥)
        [42:48]  harms present (6 pairs: 子未, 丑午, 寅巳, 卯辰, 申亥, 酉戌)
        [48:52]  punishments present (4 types: 无恩之刑, 持势之刑, 无礼之刑, 自刑)
        [52:58]  destructions present (6 pairs: 子酉, 午卯, 寅亥, 巳申, 丑辰, 未戌)
        [58:64]  destructions is_weakened (same order)
        Total: 64 dimensions.
        """
        stem_pairs_order = list(STEM_COMBINATION_PAIRS.keys())
        branch_six_order = list(BRANCH_SIX_PAIRS.keys())
        stem_clash_order = sorted(STEM_CLASH_PAIRS)
        branch_clash_order = sorted(BRANCH_CLASH_PAIRS)
        harm_pairs_order = [
            ("子", "未"), ("丑", "午"), ("寅", "巳"), ("卯", "辰"), ("申", "亥"), ("酉", "戌"),
        ]
        destruction_order = sorted(BRANCH_DESTRUCTION_PAIRS)
        harmony_elements_order = list(THREE_HARMONY_GROUPS.keys())
        punishment_types = ["无恩之刑", "持势之刑", "无礼之刑", "自刑"]

        stem_comb_lookup = build_pair_lookup(self.stem_combinations, "stem_a", "stem_b")
        branch_six_lookup = build_pair_lookup(self.branch_six_combinations, "branch_a", "branch_b")
        stem_clash_lookup = build_pair_lookup(self.stem_clashes, "item_a", "item_b")
        branch_clash_lookup = build_pair_lookup(self.branch_clashes, "item_a", "item_b")
        harms_lookup = build_pair_lookup(self.harms, "branch_a", "branch_b")
        destructions_lookup = build_pair_lookup(self.destructions, "branch_a", "branch_b")
        harmony_by_element = {d["element"]: d for d in self.three_harmony}
        punishment_types_present = {
            d.get("punishment_type") for d in self.punishments if d.get("punishment_type")
        }

        vec = np.zeros(64, dtype=np.float64)

        # Stem combinations present + is_pure
        for i, (sa, sb) in enumerate(stem_pairs_order):
            d = stem_comb_lookup.get(frozenset({sa, sb}))
            if d is not None:
                vec[i] = 1.0
                if d.get("is_pure", True):
                    vec[5 + i] = 1.0

        # Branch six present + is_blocked
        for i, (ba, bb) in enumerate(branch_six_order):
            d = branch_six_lookup.get(frozenset({ba, bb}))
            if d is not None:
                vec[10 + i] = 1.0
                if d.get("is_blocked", False):
                    vec[16 + i] = 1.0

        # Three harmony full + half
        for i, elem in enumerate(harmony_elements_order):
            d = harmony_by_element.get(elem)
            if d is not None:
                if d.get("combo_type") == "full":
                    vec[22 + i] = 1.0
                elif d.get("combo_type") == "half":
                    vec[26 + i] = 1.0

        # Stem clashes
        for i, (sa, sb) in enumerate(stem_clash_order):
            if stem_clash_lookup.get(frozenset({sa, sb})) is not None:
                vec[30 + i] = 1.0

        # Branch clashes
        for i, (ba, bb) in enumerate(branch_clash_order):
            if branch_clash_lookup.get(frozenset({ba, bb})) is not None:
                vec[36 + i] = 1.0

        # Harms
        for i, (ba, bb) in enumerate(harm_pairs_order):
            if harms_lookup.get(frozenset({ba, bb})) is not None:
                vec[42 + i] = 1.0

        # Punishments (by type)
        for i, ptype in enumerate(punishment_types):
            if ptype in punishment_types_present:
                vec[48 + i] = 1.0

        # Destructions present + is_weakened
        for i, (ba, bb) in enumerate(destruction_order):
            d = destructions_lookup.get(frozenset({ba, bb}))
            if d is not None:
                vec[52 + i] = 1.0
                if d.get("is_weakened", False):
                    vec[58 + i] = 1.0

        return vec


def detect_all_interactions(stems: list[str], branches: list[str]) -> InteractionResult:
    """
    Run all 8 interaction detectors and apply blocking/interference logic.

    Blocking rules:
    - If a branch is in a 冲 (clash), any 六合 involving that branch is marked is_blocked=True.
    - If a branch pair is in 六合, the 破 (destruction) for that pair is marked is_weakened=True.
    - If a stem position is in a stem clash, that stem's 天干合 is marked is_pure=False.
    """
    if len(stems) != PILLAR_COUNT or len(branches) != PILLAR_COUNT:
        raise ValueError("stems and branches must each have 4 items (Year, Month, Day, Hour)")

    # 1) Raw detection
    stem_clashes = detect_stem_clashes(stems)
    stem_clash_positions = {pos for r in stem_clashes for pos in (r["pos_a"], r["pos_b"])}
    stem_combinations = detect_stem_combinations(
        stems, blocked_positions=stem_clash_positions
    )

    branch_clashes = detect_branch_clashes(branches)
    branch_six_combinations = detect_branch_six_combinations(branches)
    three_harmony = detect_three_harmony(branches)
    harms = detect_branch_harms(branches)
    punishments = detect_punishments(branches)
    destructions = detect_branch_destructions(branches)

    # 2) Branches involved in 冲 → mark 六合 is_blocked (new dicts to avoid mutating detector output)
    branch_clash_branches = {b for r in branch_clashes for b in (r["item_a"], r["item_b"])}
    branch_six_with_blocked = [
        {**d, "is_blocked": d["branch_a"] in branch_clash_branches or d["branch_b"] in branch_clash_branches}
        for d in branch_six_combinations
    ]

    # 3) Elemental strength and 合化 (transformation_active)
    element_strengths = compute_element_strengths(stems, branches)
    month_branch = branches[1]
    stem_combinations_with_transformation = [
        {
            **d,
            "transformation_active": _check_transformation(
                d["result_element"],
                month_branch,
                element_strengths,
                TRANSFORMATION_STRENGTH_THRESHOLD,
                d.get("is_pure", True),
            ),
        }
        for d in stem_combinations
    ]
    branch_six_with_transformation = [
        {
            **d,
            "transformation_active": _check_transformation(
                d["result_element"],
                month_branch,
                element_strengths,
                TRANSFORMATION_STRENGTH_THRESHOLD,
                not d["is_blocked"],
            ),
        }
        for d in branch_six_with_blocked
    ]

    # 4) 冲 breaks 三合: any 三合 (full or half) whose present_branches include a clashed branch gets blocked_by_clash=True
    three_harmony_with_blocked = [
        {**d, "blocked_by_clash": any(b in branch_clash_branches for b in d["present_branches"])}
        for d in three_harmony
    ]

    # 5) 三合 absorbs 六合: 六合 pairs that use any branch of a non-blocked full 三合 get absorbed=True
    branches_in_non_blocked_full_sanhe = set()
    for d in three_harmony_with_blocked:
        if d.get("combo_type") == "full" and not d.get("blocked_by_clash", False):
            branches_in_non_blocked_full_sanhe.update(d["present_branches"])
    branch_six_final = [
        {
            **d,
            "absorbed": d["branch_a"] in branches_in_non_blocked_full_sanhe
            or d["branch_b"] in branches_in_non_blocked_full_sanhe,
        }
        for d in branch_six_with_transformation
    ]

    # 6) 六合 pairs → mark 破 is_weakened for same branch pair (new dicts to avoid mutating detector output)
    six_combo_pairs = {frozenset((d["branch_a"], d["branch_b"])) for d in branch_six_combinations}
    destructions_with_weakened = [
        {**d, "is_weakened": frozenset((d["branch_a"], d["branch_b"])) in six_combo_pairs}
        for d in destructions
    ]

    return InteractionResult(
        stem_combinations=stem_combinations_with_transformation,
        branch_six_combinations=branch_six_final,
        three_harmony=three_harmony_with_blocked,
        stem_clashes=stem_clashes,
        branch_clashes=branch_clashes,
        harms=harms,
        punishments=punishments,
        destructions=destructions_with_weakened,
    )
