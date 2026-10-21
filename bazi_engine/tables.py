# bazi_engine/tables.py
# Static lookup tables for hidden stems (藏干) with standard 子平 weights.
# Stem lists match lunar-python's LunarUtil.ZHI_HIDE_GAN order: 主气 → 余气 → 杂气
# Weights follow mainstream 子平 methodology. Documented choice — do not change
# without updating the elemental strength engine in Task 0.4.

from __future__ import annotations

from .elements import ELEMENT_CONTROLS_ME, ELEMENT_CONTROLS, ELEMENT_GENERATES
from .enums import Branch, Stem, TenGod

HIDE_GAN_WEIGHTS: dict[str, list[tuple[str, float]]] = {
    "子": [("癸", 1.0)],
    "丑": [("己", 0.6), ("癸", 0.3), ("辛", 0.1)],
    "寅": [("甲", 0.6), ("丙", 0.3), ("戊", 0.1)],
    "卯": [("乙", 1.0)],
    "辰": [("戊", 0.6), ("乙", 0.3), ("癸", 0.1)],
    "巳": [("丙", 0.6), ("庚", 0.3), ("戊", 0.1)],
    "午": [("丁", 0.7), ("己", 0.3)],
    "未": [("己", 0.6), ("丁", 0.3), ("乙", 0.1)],
    "申": [("庚", 0.6), ("壬", 0.3), ("戊", 0.1)],
    "酉": [("辛", 1.0)],
    "戌": [("戊", 0.6), ("辛", 0.3), ("丁", 0.1)],
    "亥": [("壬", 0.7), ("甲", 0.3)],
}

_HIDDEN_STEMS: dict[Branch, tuple[tuple[Stem, float], ...]] = {
    b: tuple((Stem(gan), w) for gan, w in HIDE_GAN_WEIGHTS.get(b.value, []))
    for b in Branch
}


def get_hidden_stems(branch: Branch) -> list[tuple[Stem, float]]:
    """
    Return hidden stems (藏干) and their 子平 weights for the given branch.
    Order: 主气 → 余气 → 杂气, matching lunar_python's ZHI_HIDE_GAN.
    """
    return list(_HIDDEN_STEMS[branch])


# 十神: used by temporal.py for 流年/流月/流日 (O(1)); same 生克/阴阳 rules as lunar_python.
# 十神: 生/克 relative to 我 (day master). Uses centralized element relations from elements.py.


def get_ten_god_for_stem(day_master: Stem, other_stem: Stem) -> TenGod:
    """
    Ten god (十神) of other_stem relative to day_master (日主).
    Same element: 同阴阳→比肩, 异→劫财. 我生: 食神/伤官. 生我: 偏印/正印. 我克: 偏财/正财. 克我: 七杀/正官.
    """
    el_d, el_o = day_master.element, other_stem.element
    same_polarity = day_master.polarity == other_stem.polarity
    if el_d == el_o:
        return TenGod.比肩 if same_polarity else TenGod.劫财
    if ELEMENT_GENERATES[el_d] == el_o:
        return TenGod.食神 if same_polarity else TenGod.伤官
    if ELEMENT_GENERATES[el_o] == el_d:
        return TenGod.偏印 if same_polarity else TenGod.正印
    if ELEMENT_CONTROLS[el_d] == el_o:
        return TenGod.偏财 if same_polarity else TenGod.正财
    if ELEMENT_CONTROLS_ME[el_d] == el_o:
        return TenGod.七杀 if same_polarity else TenGod.正官
    raise AssertionError(f"invalid stem relation: day={day_master.value} other={other_stem.value}")


def get_ten_god_for_branch(day_master: Stem, branch: Branch) -> list[TenGod]:
    """Ten gods for the hidden stems (藏干) in branch, 主气→余气→杂气 order."""
    return [get_ten_god_for_stem(day_master, s) for s, _ in get_hidden_stems(branch)]