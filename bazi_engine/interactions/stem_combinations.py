"""
天干合 (heavenly stem combinations) detection.

The five fixed pairs and their resulting elements:
  甲己→土, 乙庚→金, 丙辛→水, 丁壬→木, 戊癸→火
"""
from __future__ import annotations

from bazi_engine.interactions._constants import require_four_pillars

# (stem_a, stem_b) -> result_element; order is canonical for reporting
STEM_COMBINATION_PAIRS: dict[tuple[str, str], str] = {
    ("甲", "己"): "土",
    ("乙", "庚"): "金",
    ("丙", "辛"): "水",
    ("丁", "壬"): "木",
    ("戊", "癸"): "火",
}


def detect_stem_combinations(
    stems: list[str],
    *,
    blocked_positions: set[int] | None = None,
) -> list[dict]:
    """
    Detect 天干合 (heavenly stem combinations) across a list of stem strings.

    Positions are 0=Year, 1=Month, 2=Day, 3=Hour. Input must have 4 items.

    For each of the five pairs (甲己, 乙庚, 丙辛, 丁壬, 戊癸), if both stems
    appear in the list, one combination record is emitted with the two
    positional indices and the resulting element.

    is_pure is False if either position is in blocked_positions (e.g. involved
    in a clash 冲 or another combination). Callers will wire the blocking set
    during InteractionResult consolidation; for now pass None for default
    (all combinations reported as pure).
    """
    require_four_pillars(stems, "stems")
    blocked = blocked_positions if blocked_positions is not None else set()

    # Build position index per stem (first occurrence only for canonical pair)
    stem_to_positions: dict[str, list[int]] = {}
    for i, s in enumerate(stems):
        stem_to_positions.setdefault(s, []).append(i)

    result: list[dict] = []
    for (stem_a, stem_b), result_element in STEM_COMBINATION_PAIRS.items():
        positions_a = stem_to_positions.get(stem_a, [])
        positions_b = stem_to_positions.get(stem_b, [])
        if not positions_a or not positions_b:
            continue
        # One combination per pair: use first occurrence of each stem
        pos_a = positions_a[0]
        pos_b = positions_b[0]
        if pos_a == pos_b:
            continue
        is_pure = pos_a not in blocked and pos_b not in blocked
        result.append({
            "stem_a": stem_a,
            "stem_b": stem_b,
            "pos_a": pos_a,
            "pos_b": pos_b,
            "result_element": result_element,
            "is_pure": is_pure,
        })
    return result


if __name__ == "__main__":
    # Chart 癸丙丁辛 (Year=癸, Month=丙, Day=丁, Hour=辛).
    # Some traditions expect no combinations here; by strict presence we detect 丙辛→水.
    chart1 = ["癸", "丙", "丁", "辛"]
    combos1 = detect_stem_combinations(chart1)
    print("Chart 癸丙丁辛 (no combinations expected in some traditions):", chart1)
    print("Combinations:", combos1)
    print()

    # Synthetic chart that triggers at least one combination: 甲己→土 at 0,1
    chart2 = ["甲", "己", "丙", "丁"]
    combos2 = detect_stem_combinations(chart2)
    print("Chart 甲己丙丁 (one combination: 甲己→土):", chart2)
    print("Combinations:", combos2)
    combos2_blocked = detect_stem_combinations(chart2, blocked_positions={0, 1})
    print("With blocked_positions={0, 1} (is_pure=False):", combos2_blocked)
