"""
Temporal structures for bāzì (八字): Da Yun (大运), Liu Ri (流日), and related sequences.
Wraps lunar-python's Yun/DaYun API into structured Python objects.
Ten god (十神) when day_master_stem is provided uses tables.get_ten_god_for_stem/branch
(same 生克/阴阳 rules as lunar_python; O(1) instead of date search).
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, replace
from typing import List, Optional

from lunar_python import Lunar, Solar

from .enums import Branch, Stem, TenGod
from .interactions.branch_six_combinations import BRANCH_SIX_PAIRS
from .interactions.clashes import BRANCH_CLASH_PAIRS, STEM_CLASH_PAIRS
from .interactions.harms import BRANCH_SIX_HARMS
from .interactions.punishments import (
    BULLYING_TRIPLET,
    DISRESPECTFUL_PAIR,
    SELF_PUNISHMENT_BRANCHES,
    UNGRATEFUL_TRIPLET,
)
from .interactions.stem_combinations import STEM_COMBINATION_PAIRS
from ._constants import POSITION_LABELS
from .elements import ELEMENT_CONTROLS_ME
from .strength import (
    ELEMENTS,
    compute_element_strengths,
    compute_element_strengths_pillars,
)
from .tables import get_ten_god_for_branch, get_ten_god_for_stem


def _solar_after_lichun(year: int) -> Solar:
    """Solar date on or after 立春 of the given solar year (for 流年 year pillar)."""
    lunar = Lunar.fromYmd(year, 1, 1)
    table = lunar.getJieQiTable()
    lichun = table.get("立春") or table.get("LI_CHUN")
    if lichun is not None:
        return lichun
    return Solar.fromYmdHms(year, 2, 5, 12, 0, 0)


def _ten_god_for_pillar(
    day_master_stem: Optional[str], stem: str, branch: str
) -> tuple[Optional[TenGod], Optional[List[TenGod]]]:
    """Ten god for pillar (stem, branch) relative to day_master. O(1) via tables (same rules as lunar_python)."""
    if not day_master_stem or not stem or not branch:
        return None, None
    try:
        dm = Stem(day_master_stem)
        s = Stem(stem)
        b = Branch(branch)
    except (ValueError, KeyError):
        return None, None
    return get_ten_god_for_stem(dm, s), get_ten_god_for_branch(dm, b)


@dataclass
class LiuNian:
    """流年: yearly luck pillar for a given Gregorian year (year pillar from 立春)."""

    year: int
    stem: str
    branch: str
    ten_god_stem: Optional[TenGod] = None
    ten_god_branch: Optional[List[TenGod]] = None

    @property
    def ganzi(self) -> str:
        """Stem+branch as single string (e.g. '甲辰')."""
        return f"{self.stem}{self.branch}"


@dataclass
class LiuYue:
    """流月: monthly luck pillar for a given Gregorian year/month (month pillar by 节; year by 立春)."""

    year: int
    month: int
    stem: str
    branch: str
    lunar_month: int
    year_stem: str
    year_branch: str
    ten_god_stem: Optional[TenGod] = None
    ten_god_branch: Optional[List[TenGod]] = None

    @property
    def ganzi(self) -> str:
        """Stem+branch as single string (e.g. '丙辰')."""
        return f"{self.stem}{self.branch}"


def compute_liunian(year: int, *, day_master_stem: Optional[str] = None) -> LiuNian:
    """
    Compute 流年 (year pillar) for a Gregorian year.

    Uses a Solar date on or after 立春 of that year so the year pillar follows
    节气 (e.g. 2024 → 甲辰). If day_master_stem is provided, fills ten_god via tables (O(1)).
    """
    solar = _solar_after_lichun(year)
    lunar = solar.getLunar()
    ec = lunar.getEightChar()
    y = ec.getYear()
    stem, branch = (y[0], y[1]) if len(y) >= 2 else ("", "")
    ten_god_stem, ten_god_branch = _ten_god_for_pillar(day_master_stem, stem, branch)
    return LiuNian(
        year=year,
        stem=stem,
        branch=branch,
        ten_god_stem=ten_god_stem,
        ten_god_branch=ten_god_branch,
    )


def compute_liuyue(
    year: int, month: int, *, day_master_stem: Optional[str] = None
) -> LiuYue:
    """
    Compute 流月 (month pillar) for a Gregorian year/month.

    Uses the 15th of the month as a mid-month date so the month pillar respects
    solar term (节) boundaries. Year pillar at that date (立春-based) is also
    returned as year_stem/year_branch (e.g. Jan 2024 before 立春 → 癸卯).
    If day_master_stem is provided, fills ten_god via tables (O(1)).
    """
    solar = Solar.fromYmdHms(year, month, 15, 12, 0, 0)
    lunar = solar.getLunar()
    ec = lunar.getEightChar()
    m = ec.getMonth()
    y = ec.getYear()
    stem, branch = (m[0], m[1]) if len(m) >= 2 else ("", "")
    year_stem, year_branch = (y[0], y[1]) if len(y) >= 2 else ("", "")
    ten_god_stem, ten_god_branch = _ten_god_for_pillar(day_master_stem, stem, branch)
    return LiuYue(
        year=year,
        month=month,
        stem=stem,
        branch=branch,
        lunar_month=int(lunar.getMonth()),
        year_stem=year_stem,
        year_branch=year_branch,
        ten_god_stem=ten_god_stem,
        ten_god_branch=ten_god_branch,
    )


@dataclass
class LiuRi:
    """流日: daily stem-branch for a given calendar date."""

    date: datetime.date
    stem: str  # 天干 e.g. '丁'
    branch: str  # 地支 e.g. '卯'
    ten_god_stem: Optional[TenGod] = None
    ten_god_branch: Optional[List[TenGod]] = None

    @property
    def ganzi(self) -> str:
        """Stem+branch as single string (e.g. '丁卯')."""
        return f"{self.stem}{self.branch}"

    @property
    def year(self) -> int:
        return self.date.year

    @property
    def month(self) -> int:
        return self.date.month

    @property
    def day(self) -> int:
        return self.date.day


def compute_liuri(
    year: int, month: int, day: int, *, day_master_stem: Optional[str] = None
) -> LiuRi:
    """
    Compute 流日 (daily stem-branch) for a Gregorian date.

    Uses lunar-python: Solar date → Lunar → day stem (getDayGan) and day branch (getDayZhi).
    If day_master_stem is provided, fills ten_god_stem and ten_god_branch.
    """
    solar = Solar.fromYmd(year, month, day)
    lunar = solar.getLunar()
    stem = lunar.getDayGan()
    branch = lunar.getDayZhi()
    ten_god_stem, ten_god_branch = _ten_god_for_pillar(day_master_stem, stem, branch)
    return LiuRi(
        date=datetime.date(year, month, day),
        stem=stem,
        branch=branch,
        ten_god_stem=ten_god_stem,
        ten_god_branch=ten_god_branch,
    )


def compute_liuri_from_date(
    date: datetime.date, *, day_master_stem: Optional[str] = None
) -> LiuRi:
    """Convenience: compute 流日 from a datetime.date. Optional day_master_stem for ten god."""
    return compute_liuri(date.year, date.month, date.day, day_master_stem=day_master_stem)


@dataclass
class DaYun:
    """Represents one 10-year luck period (大运)."""

    stem: str  # e.g. '乙'
    branch: str  # e.g. '卯'
    start_year: int  # Gregorian year when period begins
    end_year: int  # Gregorian year when period ends
    start_age: int  # Person's age at start
    end_age: int  # Person's age at end (inclusive)
    index: int  # 0-based index in the sequence

    @property
    def gan_zhi(self) -> str:
        """Stem+branch as single string (e.g. '乙卯')."""
        return f"{self.stem}{self.branch}"


class DaYunSequence:
    """Manages the full sequence of Da Yun periods for a chart."""

    def __init__(self, periods: list[DaYun], gender: int):
        self.periods = periods
        self.gender = gender  # 0=female, 1=male

    def get_period_for_year(self, year: int) -> Optional[DaYun]:
        """Get the active Da Yun period for a given Gregorian year."""
        for period in self.periods:
            if period.start_year <= year <= period.end_year:
                return period
        return None

    def get_period_for_age(self, age: int) -> Optional[DaYun]:
        """Get the Da Yun period for a given age (person's age in years)."""
        for period in self.periods:
            if period.start_age <= age <= period.end_age:
                return period
        return None


@dataclass
class TemporalContext:
    """Context for temporal pillars (大运, 流年, 流月, 流日) when evaluating a chart. All optional."""

    dayun: Optional[DaYun] = None  # 大运: current 10-year luck pillar
    liunian: Optional[LiuNian] = None  # 流年: yearly pillar
    liuyue: Optional[LiuYue] = None  # 流月: monthly pillar
    liuri: Optional[LiuRi] = None  # 流日: daily stem-branch

    def get_presence_mask(self) -> dict[str, bool]:
        """Return a dict of boolean presence flags for each temporal pillar (dayun, liunian, liuyue, liuri)."""
        return {
            "dayun": self.dayun is not None,
            "liunian": self.liunian is not None,
            "liuyue": self.liuyue is not None,
            "liuri": self.liuri is not None,
        }


def compute_elemental_delta(
    natal_stems: List[str],
    natal_branches: List[str],
    temporal_context: TemporalContext,
) -> dict[str, float]:
    """
    Compute change in elemental strength when temporal pillars are added to the natal chart.

    Uses compute_element_strengths for natal (4 pillars), then combines natal + temporal
    pillars into extended stem/branch lists and computes extended strengths. Returns
    delta dict: {element: extended_fraction - natal_fraction} for 木/火/土/金/水.
    """
    if len(natal_stems) != 4 or len(natal_branches) != 4:
        raise ValueError("natal_stems and natal_branches must each have 4 items")
    natal = compute_element_strengths(natal_stems, natal_branches)
    extended_stems = list(natal_stems)
    extended_branches = list(natal_branches)
    for pillar in (
        temporal_context.dayun,
        temporal_context.liunian,
        temporal_context.liuyue,
        temporal_context.liuri,
    ):
        if pillar is not None:
            extended_stems.append(pillar.stem)
            extended_branches.append(pillar.branch)
    if len(extended_stems) == 4:
        extended = natal
    else:
        extended = compute_element_strengths_pillars(extended_stems, extended_branches)
    return {e: extended[e] - natal[e] for e in ELEMENTS}


def _stem_combination_result(stem_a: str, stem_b: str) -> Optional[str]:
    """Return result element if (stem_a, stem_b) is a 天干合 pair, else None. Reuses STEM_COMBINATION_PAIRS."""
    return STEM_COMBINATION_PAIRS.get((stem_a, stem_b)) or STEM_COMBINATION_PAIRS.get((stem_b, stem_a))


def _is_stem_clash(a: str, b: str) -> bool:
    """True if (a, b) is a 天干相冲 pair. Reuses STEM_CLASH_PAIRS."""
    return (a, b) in STEM_CLASH_PAIRS or (b, a) in STEM_CLASH_PAIRS


def _branch_six_result(branch_a: str, branch_b: str) -> Optional[str]:
    """Return result element if (branch_a, branch_b) is 六合, else None. Reuses BRANCH_SIX_PAIRS."""
    return BRANCH_SIX_PAIRS.get((branch_a, branch_b)) or BRANCH_SIX_PAIRS.get((branch_b, branch_a))


def _is_branch_clash(a: str, b: str) -> bool:
    """True if (a, b) is 六冲. Reuses BRANCH_CLASH_PAIRS."""
    return (a, b) in BRANCH_CLASH_PAIRS or (b, a) in BRANCH_CLASH_PAIRS


def _is_branch_harm(a: str, b: str) -> bool:
    """True if (a, b) is 六害. Reuses BRANCH_SIX_HARMS."""
    return frozenset({a, b}) in BRANCH_SIX_HARMS


def _branch_punishment_kind(branch_a: str, branch_b: str) -> Optional[str]:
    """
    If the two branches participate in a 刑 (punishment), return the type; else None.
    Handles 无礼之刑 (子卯), and 自刑 (same branch in 辰午酉亥). Cyclic triplets need 2+ members:
    if both branches are from the same triplet, return that type.
    """
    if branch_a == branch_b and branch_a in SELF_PUNISHMENT_BRANCHES:
        return "自刑"
    if frozenset({branch_a, branch_b}) == frozenset(DISRESPECTFUL_PAIR):
        return "无礼之刑"
    for triplet, ptype in ((UNGRATEFUL_TRIPLET, "无恩之刑"), (BULLYING_TRIPLET, "持势之刑")):
        if branch_a in triplet and branch_b in triplet:
            return ptype
    return None


def _stem_element(stem_char: str) -> Optional[str]:
    """Return 五行 element for a stem character, or None if unknown."""
    try:
        return Stem(stem_char).element
    except (ValueError, KeyError):
        return None


def _branch_element(branch_char: str) -> Optional[str]:
    """Return 五行 element for a branch character (main qi 主气), or None if unknown."""
    try:
        return Branch(branch_char).element
    except (ValueError, KeyError):
        return None


def check_useful_god_activation(
    useful_god: str,
    elemental_delta: dict[str, float],
    cross_interactions: "CrossInteractionResult",
) -> tuple[str, str]:
    """
    Determine if the natal useful god (用神) is active, suppressed, or neutral in the current temporal context.

    V1: useful_god must be an element (木/火/土/金/水). Ten-god names require day master and are not mapped here.
    - If the useful element has a cross-clash (克) with its controller → suppressed.
    - Else delta > 0.05 → active, delta < -0.05 → suppressed, else → neutral.

    Returns (status, explanation) where status is 'active' | 'suppressed' | 'neutral'.
    """
    if useful_god not in ELEMENTS:
        return (
            "neutral",
            f"useful_god must be an element (木/火/土/金/水) for v1; got {useful_god!r}",
        )
    element = useful_god
    controller = ELEMENT_CONTROLS_ME[element]
    delta = elemental_delta.get(element, 0.0)

    def _clash_suppresses() -> tuple[bool, str]:
        # Stem clashes: natal_*_stem_clashes, dayun_liunian, dayun_liuyue, liunian_liuyue
        stem_clash_lists = [
            cross_interactions.natal_dayun_stem_clashes,
            cross_interactions.natal_liunian_stem_clashes,
            cross_interactions.natal_liuyue_stem_clashes,
            cross_interactions.natal_liuri_stem_clashes,
            cross_interactions.dayun_liunian_stem_clashes,
            cross_interactions.dayun_liuyue_stem_clashes,
            cross_interactions.liunian_liuyue_stem_clashes,
        ]
        for lst in stem_clash_lists:
            for entry in lst:
                a = entry.get("item_a") or entry.get("stem_a")
                b = entry.get("item_b") or entry.get("stem_b")
                if a and b:
                    ea, eb = _stem_element(a), _stem_element(b)
                    if (ea, eb) == (element, controller) or (ea, eb) == (controller, element):
                        return True, f"天干相冲 {a}{b}（{element}被{controller}克）"
        # Branch conflicts (clash only): natal_*_branch_conflicts with kind "clash", dayun_liunian_branch_clashes
        branch_clash_entries = []
        for lst in [
            cross_interactions.natal_dayun_branch_conflicts,
            cross_interactions.natal_liunian_branch_conflicts,
            cross_interactions.natal_liuyue_branch_conflicts,
            cross_interactions.natal_liuri_branch_conflicts,
        ]:
            for entry in lst:
                if entry.get("kind") == "clash":
                    branch_clash_entries.append(entry)
        for lst in (
            cross_interactions.dayun_liunian_branch_clashes,
            cross_interactions.dayun_liuyue_branch_clashes,
            cross_interactions.liunian_liuyue_branch_clashes,
        ):
            for entry in lst:
                branch_clash_entries.append({
                    "branch_a": entry.get("item_a") or entry.get("branch_a"),
                    "branch_b": entry.get("item_b") or entry.get("branch_b"),
                })
        for entry in branch_clash_entries:
            a = entry.get("branch_a") or entry.get("item_a")
            b = entry.get("branch_b") or entry.get("item_b")
            if a and b:
                ea, eb = _branch_element(a), _branch_element(b)
                if (ea, eb) == (element, controller) or (ea, eb) == (controller, element):
                    return True, f"地支六冲 {a}{b}（{element}被{controller}克）"
        return False, ""

    suppressed_by_clash, clash_explanation = _clash_suppresses()
    if suppressed_by_clash:
        return ("suppressed", clash_explanation)

    if delta > 0.05:
        return ("active", f"用神{element}得令，元素增量 {delta:.3f} > 0.05")
    if delta < -0.05:
        return ("suppressed", f"用神{element}失令，元素增量 {delta:.3f} < -0.05")
    return ("neutral", f"用神{element}变化不大，元素增量 {delta:.3f} 在 [-0.05, 0.05]")


@dataclass
class CrossInteractionResult:
    """
    Result of cross-context interactions: natal pillars vs temporal (dayun/liunian/liuyue/liuri)
    and optionally dayun vs liunian.

    16 list fields: 4 interaction types × 4 cross-contexts (natal-dayun, natal-liunian, natal-liuyue, natal-liuri).
    Types: stem_combinations, stem_clashes, branch_combinations (六合), branch_conflicts (冲/害/刑).
    Plus 4 lists for dayun-liunian (when both present). Plus 4 boolean summary flags.
    """

    # Natal vs dayun (4 types)
    natal_dayun_stem_combinations: List[dict]
    natal_dayun_stem_clashes: List[dict]
    natal_dayun_branch_combinations: List[dict]
    natal_dayun_branch_conflicts: List[dict]
    # Natal vs liunian (4 types)
    natal_liunian_stem_combinations: List[dict]
    natal_liunian_stem_clashes: List[dict]
    natal_liunian_branch_combinations: List[dict]
    natal_liunian_branch_conflicts: List[dict]
    # Natal vs liuyue (4 types)
    natal_liuyue_stem_combinations: List[dict]
    natal_liuyue_stem_clashes: List[dict]
    natal_liuyue_branch_combinations: List[dict]
    natal_liuyue_branch_conflicts: List[dict]
    # Natal vs liuri (4 types)
    natal_liuri_stem_combinations: List[dict]
    natal_liuri_stem_clashes: List[dict]
    natal_liuri_branch_combinations: List[dict]
    natal_liuri_branch_conflicts: List[dict]
    # Dayun vs liunian (4 types)
    dayun_liunian_stem_combinations: List[dict]
    dayun_liunian_stem_clashes: List[dict]
    dayun_liunian_branch_combinations: List[dict]
    dayun_liunian_branch_clashes: List[dict]
    # Dayun vs liuyue (4 types; empty when liuyue absent)
    dayun_liuyue_stem_combinations: List[dict]
    dayun_liuyue_stem_clashes: List[dict]
    dayun_liuyue_branch_combinations: List[dict]
    dayun_liuyue_branch_clashes: List[dict]
    # Liunian vs liuyue (4 types; empty when liuyue absent)
    liunian_liuyue_stem_combinations: List[dict]
    liunian_liuyue_stem_clashes: List[dict]
    liunian_liuyue_branch_combinations: List[dict]
    liunian_liuyue_branch_clashes: List[dict]

    has_natal_dayun_interactions: bool
    has_natal_liunian_interactions: bool
    has_natal_liuyue_interactions: bool
    has_natal_liuri_interactions: bool

    elemental_delta: dict[str, float]  # {element: extended - natal} 木/火/土/金/水
    total_shift_magnitude: float  # sum of absolute deltas

    useful_god_activation: Optional[str] = None  # 'active' | 'suppressed' | 'neutral' when useful_god provided
    useful_god_analysis: Optional[str] = None  # explanation from check_useful_god_activation


def detect_cross_interactions(
    natal_stems: List[str],
    natal_branches: List[str],
    temporal_context: TemporalContext,
    *,
    useful_god: Optional[str] = None,
) -> CrossInteractionResult:
    """
    Detect interactions between natal pillars and temporal pillars (and dayun vs liunian).

    For each present temporal pillar (dayun, liunian, liuyue, liuri), checks every natal position
    for: stem combinations and clashes; branch 六合, 六冲, 六害, 刑. Populates the corresponding
    CrossInteractionResult lists and sets summary flags. Reuses constants from bazi_engine.interactions.
    """
    if len(natal_stems) != 4 or len(natal_branches) != 4:
        raise ValueError("natal_stems and natal_branches must each have 4 items (Year, Month, Day, Hour)")

    # Initialize all 20 lists + 4 flags
    def empty_lists_4():
        return ([], [], [], [])

    nd_sc, nd_ss, nd_bc, nd_bf = empty_lists_4()
    nn_sc, nn_ss, nn_bc, nn_bf = empty_lists_4()
    ny_sc, ny_ss, ny_bc, ny_bf = empty_lists_4()
    nr_sc, nr_ss, nr_bc, nr_bf = empty_lists_4()
    dn_sc, dn_ss, dn_bc, dn_cl = [], [], [], []
    dy_ly_sc, dy_ly_ss, dy_ly_bc, dy_ly_cl = [], [], [], []
    ln_ly_sc, ln_ly_ss, ln_ly_bc, ln_ly_cl = [], [], [], []

    def check_natal_vs_temporal(
        t_stem: str,
        t_branch: str,
        ctx_name: str,
        out_sc: List,
        out_ss: List,
        out_bc: List,
        out_bf: List,
    ) -> None:
        for pos in range(4):
            s, b = natal_stems[pos], natal_branches[pos]
            # Stem combination
            res = _stem_combination_result(s, t_stem)
            if res is not None:
                out_sc.append({
                    "natal_pos": pos,
                    "pos_label": POSITION_LABELS[pos],
                    "stem_a": s,
                    "stem_b": t_stem,
                    "result_element": res,
                    "context": ctx_name,
                })
            # Stem clash
            if _is_stem_clash(s, t_stem):
                out_ss.append({
                    "natal_pos": pos,
                    "pos_label": POSITION_LABELS[pos],
                    "item_a": s if (s, t_stem) in STEM_CLASH_PAIRS else t_stem,
                    "item_b": t_stem if (s, t_stem) in STEM_CLASH_PAIRS else s,
                    "context": ctx_name,
                })
            # Branch 六合
            elem = _branch_six_result(b, t_branch)
            if elem is not None:
                out_bc.append({
                    "natal_pos": pos,
                    "pos_label": POSITION_LABELS[pos],
                    "branch_a": b,
                    "branch_b": t_branch,
                    "result_element": elem,
                    "context": ctx_name,
                })
            # Branch clash
            if _is_branch_clash(b, t_branch):
                out_bf.append({
                    "kind": "clash",
                    "natal_pos": pos,
                    "pos_label": POSITION_LABELS[pos],
                    "branch_a": b,
                    "branch_b": t_branch,
                    "context": ctx_name,
                })
            # Branch harm
            if _is_branch_harm(b, t_branch):
                out_bf.append({
                    "kind": "harm",
                    "natal_pos": pos,
                    "pos_label": POSITION_LABELS[pos],
                    "branch_a": b,
                    "branch_b": t_branch,
                    "context": ctx_name,
                })
            # Branch punishment
            pkind = _branch_punishment_kind(b, t_branch)
            if pkind is not None:
                out_bf.append({
                    "kind": "punishment",
                    "punishment_type": pkind,
                    "natal_pos": pos,
                    "pos_label": POSITION_LABELS[pos],
                    "branch_a": b,
                    "branch_b": t_branch,
                    "context": ctx_name,
                })

    # Extract temporal pillars and run natal vs temporal
    if temporal_context.dayun is not None:
        dy = temporal_context.dayun
        check_natal_vs_temporal(dy.stem, dy.branch, "dayun", nd_sc, nd_ss, nd_bc, nd_bf)
    if temporal_context.liunian is not None:
        ln = temporal_context.liunian
        check_natal_vs_temporal(ln.stem, ln.branch, "liunian", nn_sc, nn_ss, nn_bc, nn_bf)
    if temporal_context.liuyue is not None:
        ly = temporal_context.liuyue
        check_natal_vs_temporal(ly.stem, ly.branch, "liuyue", ny_sc, ny_ss, ny_bc, ny_bf)
    if temporal_context.liuri is not None:
        lr = temporal_context.liuri
        check_natal_vs_temporal(lr.stem, lr.branch, "liuri", nr_sc, nr_ss, nr_bc, nr_bf)

    # Dayun vs liunian (single pair each)
    if temporal_context.dayun is not None and temporal_context.liunian is not None:
        ds, db = temporal_context.dayun.stem, temporal_context.dayun.branch
        ls, lb = temporal_context.liunian.stem, temporal_context.liunian.branch
        sc_res = _stem_combination_result(ds, ls)
        if sc_res is not None:
            dn_sc.append({"stem_a": ds, "stem_b": ls, "result_element": sc_res})
        if _is_stem_clash(ds, ls):
            dn_ss.append({
                "item_a": ds if (ds, ls) in STEM_CLASH_PAIRS else ls,
                "item_b": ls if (ds, ls) in STEM_CLASH_PAIRS else ds,
            })
        bc_res = _branch_six_result(db, lb)
        if bc_res is not None:
            dn_bc.append({"branch_a": db, "branch_b": lb, "result_element": bc_res})
        if _is_branch_clash(db, lb):
            dn_cl.append({
                "item_a": db if (db, lb) in BRANCH_CLASH_PAIRS else lb,
                "item_b": lb if (db, lb) in BRANCH_CLASH_PAIRS else db,
            })

    # Dayun vs liuyue (single pair each; when both present)
    if temporal_context.dayun is not None and temporal_context.liuyue is not None:
        ds, db = temporal_context.dayun.stem, temporal_context.dayun.branch
        ys, yb = temporal_context.liuyue.stem, temporal_context.liuyue.branch
        sc_res = _stem_combination_result(ds, ys)
        if sc_res is not None:
            dy_ly_sc.append({"stem_a": ds, "stem_b": ys, "result_element": sc_res})
        if _is_stem_clash(ds, ys):
            dy_ly_ss.append({
                "item_a": ds if (ds, ys) in STEM_CLASH_PAIRS else ys,
                "item_b": ys if (ds, ys) in STEM_CLASH_PAIRS else ds,
            })
        bc_res = _branch_six_result(db, yb)
        if bc_res is not None:
            dy_ly_bc.append({"branch_a": db, "branch_b": yb, "result_element": bc_res})
        if _is_branch_clash(db, yb):
            dy_ly_cl.append({
                "item_a": db if (db, yb) in BRANCH_CLASH_PAIRS else yb,
                "item_b": yb if (db, yb) in BRANCH_CLASH_PAIRS else db,
            })

    # Liunian vs liuyue (single pair each; when both present)
    if temporal_context.liunian is not None and temporal_context.liuyue is not None:
        ls, lb = temporal_context.liunian.stem, temporal_context.liunian.branch
        ys, yb = temporal_context.liuyue.stem, temporal_context.liuyue.branch
        sc_res = _stem_combination_result(ls, ys)
        if sc_res is not None:
            ln_ly_sc.append({"stem_a": ls, "stem_b": ys, "result_element": sc_res})
        if _is_stem_clash(ls, ys):
            ln_ly_ss.append({
                "item_a": ls if (ls, ys) in STEM_CLASH_PAIRS else ys,
                "item_b": ys if (ls, ys) in STEM_CLASH_PAIRS else ls,
            })
        bc_res = _branch_six_result(lb, yb)
        if bc_res is not None:
            ln_ly_bc.append({"branch_a": lb, "branch_b": yb, "result_element": bc_res})
        if _is_branch_clash(lb, yb):
            ln_ly_cl.append({
                "item_a": lb if (lb, yb) in BRANCH_CLASH_PAIRS else yb,
                "item_b": yb if (lb, yb) in BRANCH_CLASH_PAIRS else lb,
            })

    def any_interactions(*lists) -> bool:
        return any(len(lst) > 0 for lst in lists)

    delta = compute_elemental_delta(natal_stems, natal_branches, temporal_context)
    total_shift_magnitude = sum(abs(delta[e]) for e in ELEMENTS)

    result = CrossInteractionResult(
        natal_dayun_stem_combinations=nd_sc,
        natal_dayun_stem_clashes=nd_ss,
        natal_dayun_branch_combinations=nd_bc,
        natal_dayun_branch_conflicts=nd_bf,
        natal_liunian_stem_combinations=nn_sc,
        natal_liunian_stem_clashes=nn_ss,
        natal_liunian_branch_combinations=nn_bc,
        natal_liunian_branch_conflicts=nn_bf,
        natal_liuyue_stem_combinations=ny_sc,
        natal_liuyue_stem_clashes=ny_ss,
        natal_liuyue_branch_combinations=ny_bc,
        natal_liuyue_branch_conflicts=ny_bf,
        natal_liuri_stem_combinations=nr_sc,
        natal_liuri_stem_clashes=nr_ss,
        natal_liuri_branch_combinations=nr_bc,
        natal_liuri_branch_conflicts=nr_bf,
        dayun_liunian_stem_combinations=dn_sc,
        dayun_liunian_stem_clashes=dn_ss,
        dayun_liunian_branch_combinations=dn_bc,
        dayun_liunian_branch_clashes=dn_cl,
        dayun_liuyue_stem_combinations=dy_ly_sc,
        dayun_liuyue_stem_clashes=dy_ly_ss,
        dayun_liuyue_branch_combinations=dy_ly_bc,
        dayun_liuyue_branch_clashes=dy_ly_cl,
        liunian_liuyue_stem_combinations=ln_ly_sc,
        liunian_liuyue_stem_clashes=ln_ly_ss,
        liunian_liuyue_branch_combinations=ln_ly_bc,
        liunian_liuyue_branch_clashes=ln_ly_cl,
        has_natal_dayun_interactions=any_interactions(nd_sc, nd_ss, nd_bc, nd_bf),
        has_natal_liunian_interactions=any_interactions(nn_sc, nn_ss, nn_bc, nn_bf),
        has_natal_liuyue_interactions=any_interactions(ny_sc, ny_ss, ny_bc, ny_bf),
        has_natal_liuri_interactions=any_interactions(nr_sc, nr_ss, nr_bc, nr_bf),
        elemental_delta=delta,
        total_shift_magnitude=total_shift_magnitude,
    )

    if useful_god is not None:
        status, explanation = check_useful_god_activation(useful_god, delta, result)
        result = replace(result, useful_god_activation=status, useful_god_analysis=explanation)

    return result
