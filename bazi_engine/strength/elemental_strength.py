"""
Elemental strength aggregation: 4 visible stems (1.0 each) + 4 branches via 藏干 weights.

Uses hidden-stem weight tables from bazi_engine.tables. Returns normalized fractions
for 木/火/土/金/水 summing to 1.0. All five keys are always present.
"""
from __future__ import annotations

from bazi_engine.enums import Branch, Stem
from bazi_engine.tables import get_hidden_stems

ELEMENTS = ("木", "火", "土", "金", "水")


def compute_element_strengths(stems: list[str], branches: list[str]) -> dict[str, float]:
    """
    Compute total elemental weight from 4 stems and 4 branches, then normalize to sum 1.0.

    - Each visible stem contributes 1.0 to its element.
    - Each branch contributes via its hidden stems (藏干) and their fractional weights.

    Input order: Year, Month, Day, Hour. Raises ValueError if length is not 4 or
    any stem/branch is unknown. Returns dict with keys 木, 火, 土, 金, 水; all five
    keys are always present (absent elements are 0.0 before normalization).
    """
    if len(stems) != 4 or len(branches) != 4:
        raise ValueError("stems and branches must each have 4 items")

    totals: dict[str, float] = {e: 0.0 for e in ELEMENTS}

    for s in stems:
        stem_enum = Stem(s)  # raises ValueError if unknown
        totals[stem_enum.element] += 1.0

    for b in branches:
        branch_enum = Branch(b)  # raises ValueError if unknown
        for hidden_stem, weight in get_hidden_stems(branch_enum):
            totals[hidden_stem.element] += weight

    total = sum(totals.values())
    if total == 0:
        return {e: 0.2 for e in ELEMENTS}
    return {e: totals[e] / total for e in ELEMENTS}


def compute_element_strengths_pillars(
    stems: list[str], branches: list[str]
) -> dict[str, float]:
    """
    Compute normalized elemental strengths for N pillars (same logic as compute_element_strengths).

    Used for extended charts (e.g. natal + temporal pillars). stems and branches must have
    the same length (any length >= 1). Returns dict with keys 木, 火, 土, 金, 水 summing to 1.0.
    """
    if len(stems) != len(branches) or len(stems) == 0:
        raise ValueError("stems and branches must have the same non-zero length")

    totals: dict[str, float] = {e: 0.0 for e in ELEMENTS}
    for s in stems:
        stem_enum = Stem(s)
        totals[stem_enum.element] += 1.0
    for b in branches:
        branch_enum = Branch(b)
        for hidden_stem, weight in get_hidden_stems(branch_enum):
            totals[hidden_stem.element] += weight
    total = sum(totals.values())
    if total == 0:
        return {e: 0.2 for e in ELEMENTS}
    return {e: totals[e] / total for e in ELEMENTS}


def compute_day_master_strength(
    stems: list[str],
    branches: list[str],
    day_master_stem: str | None = None,
) -> float:
    """
    Return the day master's elemental strength as a fraction in [0.0, 1.0].

    If day_master_stem is not provided, it is inferred from stems[2] (the Day stem,
    in Year/Month/Day/Hour order). The day master's element is looked up via Stem(...).element; the strength is the fraction of total elemental weight
    belonging to that element (the normalized value from compute_element_strengths).

    Raises ValueError if day_master_stem (or stems[2] when inferred) is not a
    known heavenly stem.
    """
    if day_master_stem is None:
        day_master_stem = stems[2]
    try:
        element = Stem(day_master_stem).element
    except ValueError:
        raise ValueError(
            f"day_master_stem must be a known heavenly stem, got {day_master_stem!r}"
        ) from None
    strengths = compute_element_strengths(stems, branches)
    return strengths[element]


def classify_day_master(
    stems: list[str],
    branches: list[str],
    strong_threshold: float = 0.30,
    weak_threshold: float = 0.20,
) -> str:
    """
    Classify the day master as 身强, 中和, or 身弱 by elemental strength percentile.

    - Returns '身强' if the day master's elemental percentile >= strong_threshold.
    - Returns '身弱' if the percentile < weak_threshold.
    - Returns '中和' otherwise.

    The thresholds are configurable. The default values (0.30 / 0.20) are a
    simplified starting point; the full useful god engine (Task 0.5) will apply
    seasonal corrections on top of this.
    """
    percentile = compute_day_master_strength(stems, branches)
    if percentile >= strong_threshold:
        return "身强"
    if percentile < weak_threshold:
        return "身弱"
    return "中和"
