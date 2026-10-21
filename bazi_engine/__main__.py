"""
CLI for single-chart spot-checking and accuracy verification.

Usage:
  python -m bazi_engine --date "YYYY-MM-DD" [HH:MM:SS] --tz 8 [--lat 39.9 --lng 116.4]
  python -m bazi_engine --date "1990-05-15 14:30:00" --tz 8 --format full
  python -m bazi_engine --date "1990-05-15 14:30:00" --tz 8 --format full --current-date 2024-06-01 --gender 1

Formats:
  pretty  - Summary: pillars, day master, 用神, element strengths, interaction counts (default).
  full    - All details for accuracy check: metadata (incl. true solar time), 藏干, full interactions, optional 大运/流年/流月.
  json    - Full chart as JSON (same as to_json()).
  arrays  - Feature array keys and shapes (encoder input).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from . import BaziCalculator
from .models import BaziChart
from .temporal import (
    TemporalContext,
    compute_liunian,
    compute_liuyue,
    detect_cross_interactions,
)

DEFAULT_LAT, DEFAULT_LNG = 39.9, 116.4


def _parse_date(s: str, tz_offset: int) -> datetime:
    """Parse date/time string and attach timezone."""
    s = s.strip()
    tz = timezone(timedelta(hours=tz_offset))
    if " " in s:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    else:
        dt = datetime.strptime(s, "%Y-%m-%d")
    return dt.replace(tzinfo=tz)


def _format_pretty(chart: BaziChart) -> str:
    """Pretty-print chart with Unicode box drawing and strength bars."""
    lines = []
    lines.append("八字命盘")
    lines.append("━" * 36)
    tz_off = chart.birth_dt.utcoffset()
    tz_str = f"UTC{tz_off.total_seconds()/3600:+.0f}" if tz_off else "UTC"
    lines.append(f"出生时间: {chart.birth_dt.strftime('%Y-%m-%d %H:%M:%S')} ({tz_str})")
    lines.append("")
    lines.append("四柱:")
    for label, pillar in [
        ("年柱", chart.year_pillar),
        ("月柱", chart.month_pillar),
        ("日柱", chart.day_pillar),
        ("时柱", chart.hour_pillar),
    ]:
        elem = f"({pillar.stem.element}{pillar.branch.element})"
        tg = pillar.ten_god_stem.value
        lines.append(f"  {label}: {pillar}  {elem}  {tg}")
    lines.append("")

    stems, branches = chart._stems_and_branches()
    from .strength import compute_day_master_strength, classify_day_master
    strength = compute_day_master_strength(stems, branches)
    classification = classify_day_master(stems, branches)
    dm = chart.day_pillar.stem
    lines.append(f"日主: {dm.value}{dm.element}  {classification} (强度: {strength:.2f})")

    useful = _compute_useful_god_str(chart)
    if useful:
        lines.append(f"用神: {useful}")
    lines.append("")

    from .strength import ELEMENTS, compute_element_strengths
    el_strengths = compute_element_strengths(stems, branches)
    elem_labels = {"木": "木", "火": "火", "土": "土", "金": "金", "水": "水"}
    lines.append("五行强度:")
    for e in ELEMENTS:
        pct = el_strengths[e] * 100
        bar_len = int(round(pct / 10))
        bar = "█" * bar_len + "░" * (10 - bar_len)
        lines.append(f"  {elem_labels[e]}: {bar} {pct:.1f}%")
    lines.append("")

    from .interactions import detect_all_interactions
    ir = detect_all_interactions(stems, branches)
    lines.append("相互关系:")
    lines.append(f"  ✓ 天干合: {len(ir.stem_combinations)} 组" if ir.stem_combinations else "  ✓ 天干合: 无")
    lines.append(f"  ✓ 天干冲: {len(ir.stem_clashes)} 组" if ir.stem_clashes else "  ✓ 天干冲: 无")
    lines.append(f"  ✓ 地支六合: {len(ir.branch_six_combinations)} 组" if ir.branch_six_combinations else "  ✓ 地支六合: 无")
    lines.append(f"  ✓ 地支六冲: {len(ir.branch_clashes)} 组" if ir.branch_clashes else "  ✓ 地支六冲: 无")
    lines.append(f"  ✓ 三合局: {len(ir.three_harmony)}" if ir.three_harmony else "  ✓ 三合局: 无")
    lines.append(f"  ✓ 六害: {len(ir.harms)} 组" if ir.harms else "  ✓ 六害: 无")
    lines.append(f"  ✓ 相刑: {len(ir.punishments)}" if ir.punishments else "  ✓ 相刑: 无")
    lines.append(f"  ✓ 相破: {len(ir.destructions)} 组" if ir.destructions else "  ✓ 相破: 无")

    return "\n".join(lines)


def _compute_useful_god_str(chart: BaziChart) -> str:
    from .models import _compute_useful_god
    from .strength import compute_day_master_strength
    stems, branches = chart._stems_and_branches()
    pct = compute_day_master_strength(stems, branches)
    ug = _compute_useful_god(pct, chart.day_pillar.stem.element)
    if not ug:
        return ""
    return f"{ug.get('primary', '')} ({ug.get('primary_element', '')})"


def _format_arrays(chart: BaziChart) -> str:
    """Print keys and shapes of to_feature_arrays()."""
    arr = chart.to_feature_arrays()
    lines = ["Feature arrays:"]
    for k in sorted(arr.keys()):
        v = arr[k]
        lines.append(f"  {k}: {v.dtype} {v.shape}")
    return "\n".join(lines)


def _format_full(chart: BaziChart) -> str:
    """Full descriptive output of everything we generate, for accuracy checking."""
    lines = []
    data = chart._to_json_dict()

    # --- Metadata (include solar_dt which is not in JSON metadata) ---
    lines.append("=== Metadata ===")
    meta = data["metadata"]
    lines.append(f"  birth_datetime:     {meta['birth_datetime']}")
    lines.append(f"  timezone_offset:    UTC{meta['timezone_offset']:+.0f} hours")
    lines.append(f"  location:           lat={meta['lat']}, lng={meta['lng']}")
    lines.append(f"  true_solar_time:    {chart.solar_dt.strftime('%Y-%m-%d %H:%M:%S')} (local, used for 月柱)")
    lines.append(f"  sect (流派):        {meta['sect']} (1=陰曆, 2=盲派)")
    lines.append("")

    # --- Natal pillars + elements + 十神 ---
    lines.append("=== Four Pillars (四柱) ===")
    pillars = data["natal_pillars"]
    ten_gods = data["ten_gods"]
    for pos, key in enumerate(["year", "month", "day", "hour"]):
        p = pillars[key]
        tg_stem = ten_gods[f"{key}_stem"]
        tg_branch = ten_gods[f"{key}_branch"]
        lines.append(f"  {key}: {p['stem']}{p['branch']}  ({p['element_stem']}{p['element_branch']})  天干十神: {tg_stem}  地支藏干十神: {tg_branch}")
    lines.append("")

    # --- Hidden stems (藏干) per branch ---
    lines.append("=== Hidden Stems (藏干) by branch ===")
    for key in ["year_branch", "month_branch", "day_branch", "hour_branch"]:
        hiddens = data["hidden_stems"][key]
        branch_key = key.replace("_branch", "")
        branch_char = pillars[branch_key]["branch"]
        parts = [f"{h['stem']}({h['element']}) w={h['weight']} {h['ten_god']}" for h in hiddens]
        lines.append(f"  {branch_key} {branch_char}: " + "  ".join(parts))
    lines.append("")

    # --- Day master & 用神 ---
    lines.append("=== Day Master & 用神 ===")
    dm = data["day_master"]
    lines.append(f"  day_master: {dm['stem']} ({dm['element']})  strength_percentile: {dm['strength_percentile']:.4f}  classification: {dm['classification']}")
    ug = data.get("useful_god")
    if ug:
        lines.append(f"  useful_god: primary={ug.get('primary')} ({ug.get('primary_element')})  methodology: {ug.get('methodology', '')}")
        lines.append(f"    favorable_elements: {ug.get('favorable_elements', [])}  unfavorable: {ug.get('unfavorable_elements', [])}")
    else:
        lines.append("  useful_god: (none / 中和)")
    lines.append("")

    # --- Elemental strengths ---
    lines.append("=== Elemental Strengths (五行强度) ===")
    for eng, val in data["elemental_strengths"].items():
        pct = val * 100
        bar_len = int(round(pct / 10))
        bar = "█" * bar_len + "░" * (10 - bar_len)
        lines.append(f"  {eng}: {bar} {pct:.2f}%")
    lines.append("")

    # --- Natal interactions (full detail with positions) ---
    lines.append("=== Natal Interactions (detailed) ===")
    ir = data["interactions"]
    _append_interaction_lines(lines, ir)
    lines.append("")

    # --- Dynamic (大运/流年/流月 + cross-interactions) ---
    dyn = data.get("dynamic")
    if dyn:
        lines.append("=== Dynamic (大运 / 流年 / 流月) ===")
        if "da_yun" in dyn:
            dy = dyn["da_yun"]
            lines.append(f"  大运: {dy['stem']}{dy.get('branch', '')}  years {dy['start_year']}-{dy['end_year']}  (age {dy['start_age']}-{dy['end_age']})")
        if "liu_nian" in dyn:
            ln = dyn["liu_nian"]
            lines.append(f"  流年: {ln['year']} {ln['stem']}{ln['branch']}")
        if "liu_yue" in dyn:
            ly = dyn["liu_yue"]
            lines.append(f"  流月: {ly['year']}-{ly['month']:02d} {ly['stem']}{ly['branch']}")
        lines.append("")
        lines.append("=== Cross-Context Interactions (all pairs) ===")
        cross = dyn.get("cross_natal_temporal_interactions", {})
        all_pairs = [
            ("natal vs 大运", "natal_vs_dayun"),
            ("natal vs 流年", "natal_vs_liunian"),
            ("natal vs 流月", "natal_vs_liuyue"),
            ("大运 vs 流年", "dayun_vs_liunian"),
            ("大运 vs 流月", "dayun_vs_liuyue"),
            ("流年 vs 流月", "liunian_vs_liuyue"),
        ]
        for ctx_name, ctx_key in all_pairs:
            ctx = cross.get(ctx_key, {})
            if not ctx:
                continue
            cl = ctx.get("clashes") or []
            comb = ctx.get("combinations") or []
            bc = ctx.get("branch_combinations") or []
            harms = ctx.get("harms") or []
            if cl or comb or bc or harms:
                lines.append(f"  {ctx_name}: stem_clashes={len(cl)}  stem_combinations={len(comb)}  branch_combinations={len(bc)}  branch_harms={len(harms)}")
                if cl:
                    for c in cl:
                        lines.append(f"    stem_clash: {c.get('stem_a')}-{c.get('stem_b')}")
                if comb:
                    for c in comb:
                        lines.append(f"    stem_combination: {c.get('stem_a')}+{c.get('stem_b')} → {c.get('result_element', '')}")
                if bc:
                    for b in bc:
                        lines.append(f"    branch_combination: {b.get('branch_a')}-{b.get('branch_b')}")
                if harms:
                    for h in harms:
                        lines.append(f"    branch_harm: {h.get('branch_a')}-{h.get('branch_b')}")
        if "elemental_delta" in dyn:
            lines.append("  elemental_delta (temporal shift): " + str(dyn["elemental_delta"]))
        if "useful_god_activation" in dyn:
            uga = dyn["useful_god_activation"]
            lines.append(f"  useful_god_activation: is_active={uga.get('is_active')}  mechanism: {uga.get('mechanism', '')}")
        lines.append("")

    return "\n".join(lines)


def _append_interaction_lines(lines: list, ir: dict) -> None:
    """Append readable interaction details to lines."""
    sc = ir.get("stem_combinations") or []
    for r in sc:
        pos_l = r.get("position_labels", [])
        pl = "–".join(pos_l) if pos_l else ""
        lines.append(f"  天干合: {pl}  {r.get('stems', [])} → {r.get('result_element', '')}  is_pure={r.get('is_pure', True)}  transformation_active={r.get('transformation_active', False)}")
    if not sc:
        lines.append("  天干合: (none)")

    ss = ir.get("stem_clashes") or []
    for r in ss:
        pos_l = r.get("position_labels", [])
        pl = "–".join(pos_l) if pos_l else ""
        lines.append(f"  天干冲: {pl}  {r.get('stems', [])}")
    if not ss:
        lines.append("  天干冲: (none)")

    bc = ir.get("branch_six_combinations") or []
    for r in bc:
        pos_l = r.get("position_labels", [])
        pl = "–".join(pos_l) if pos_l else ""
        lines.append(f"  地支六合: {pl}  {r.get('branches', [])} → {r.get('result_element', '')}  is_blocked={r.get('is_blocked', False)}")
    if not bc:
        lines.append("  地支六合: (none)")

    cl = ir.get("branch_clashes") or []
    for r in cl:
        pos_l = r.get("position_labels", [])
        pl = "–".join(pos_l) if pos_l else ""
        lines.append(f"  地支六冲: {pl}  {r.get('branches', [])}")
    if not cl:
        lines.append("  地支六冲: (none)")

    th = ir.get("three_harmony") or []
    for r in th:
        lines.append(f"  三合: element={r.get('element')}  type={r.get('combo_type')}  branches={r.get('present_branches')}  positions={r.get('positions')}  blocked_by_clash={r.get('blocked_by_clash', False)}")
    if not th:
        lines.append("  三合局: (none)")

    harms = ir.get("harms") or []
    for r in harms:
        pos_l = r.get("position_labels", [])
        pl = "–".join(pos_l) if pos_l else ""
        lines.append(f"  六害: {pl}  {r.get('branches', [])}")
    if not harms:
        lines.append("  六害: (none)")

    pun = ir.get("punishments") or []
    for r in pun:
        lines.append(f"  相刑: type={r.get('punishment_type')}  positions={r.get('positions')}  members={r.get('members_present')}")
    if not pun:
        lines.append("  相刑: (none)")

    dest = ir.get("destructions") or []
    for r in dest:
        pos_l = r.get("position_labels", [])
        pl = "–".join(pos_l) if pos_l else ""
        lines.append(f"  相破: {pl}  {r.get('branches', [])}  is_weakened={r.get('is_weakened', False)}")
    if not dest:
        lines.append("  相破: (none)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bazi chart calculator CLI")
    parser.add_argument("--date", required=True, help="Birth datetime (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--tz", type=int, default=8, help="Timezone offset in hours (default 8)")
    parser.add_argument(
        "--format",
        choices=("pretty", "json", "arrays", "full"),
        default="pretty",
        help="Output format: pretty (summary), full (all details for accuracy check), json, arrays (default: pretty)",
    )
    parser.add_argument(
        "--current-date",
        default=None,
        help="Date for dynamic 大运/流年/流月 (YYYY-MM-DD); enables dynamic in output",
    )
    parser.add_argument(
        "--gender",
        type=int,
        default=1,
        choices=(0, 1),
        help="Gender for 大运 direction: 0=female, 1=male (default 1, used with --current-date)",
    )
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Latitude")
    parser.add_argument("--lng", type=float, default=DEFAULT_LNG, help="Longitude")
    args = parser.parse_args()

    try:
        dt = _parse_date(args.date, args.tz)
    except ValueError as e:
        print(f"Error: invalid date: {e}", file=sys.stderr)
        return 1

    chart = BaziCalculator.from_datetime(dt, args.lat, args.lng)

    if args.current_date:
        try:
            cur = datetime.strptime(args.current_date, "%Y-%m-%d").date()
        except ValueError:
            print(f"Error: invalid --current-date {args.current_date!r}", file=sys.stderr)
            return 1
        dayun_seq = BaziCalculator().compute_dayun(chart, args.gender, max_periods=10)
        dayun = dayun_seq.get_period_for_year(cur.year)
        liunian = compute_liunian(cur.year, day_master_stem=chart.day_pillar.stem.value)
        liuyue = compute_liuyue(cur.year, cur.month, day_master_stem=chart.day_pillar.stem.value)
        ctx = TemporalContext(dayun=dayun, liunian=liunian, liuyue=liuyue, liuri=None)
        stems = [chart.year_pillar.stem.value, chart.month_pillar.stem.value,
                 chart.day_pillar.stem.value, chart.hour_pillar.stem.value]
        branches = [chart.year_pillar.branch.value, chart.month_pillar.branch.value,
                   chart.day_pillar.branch.value, chart.hour_pillar.branch.value]
        cross = detect_cross_interactions(stems, branches, ctx, useful_god=None)
        chart = BaziChart(
            birth_dt=chart.birth_dt,
            solar_dt=chart.solar_dt,
            lat=chart.lat,
            lng=chart.lng,
            year_pillar=chart.year_pillar,
            month_pillar=chart.month_pillar,
            day_pillar=chart.day_pillar,
            hour_pillar=chart.hour_pillar,
            sect=chart.sect,
            temporal_context=ctx,
            cross_interactions=cross,
        )

    if args.format == "pretty":
        print(_format_pretty(chart))
    elif args.format == "full":
        print(_format_full(chart))
    elif args.format == "json":
        print(chart.to_json(indent=2))
    else:
        print(_format_arrays(chart))
    return 0


if __name__ == "__main__":
    sys.exit(main())
