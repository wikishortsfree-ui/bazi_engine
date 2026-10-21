#!/usr/bin/env python3
"""
Generate stratified synthetic Bazi charts for Phase 2 data curation.

Uses acceptance sampling so that:
- Day master (天干) ~uniform across 10 stems
- Month branch (月支) ~uniform across 12 branches
- Day-master strength: oversample 身强 and 身弱
- Charts with 刑/破/害 (punishments, destructions, harms) oversampled
- Useful god (用神) diversity

A configurable fraction of charts include temporal_context + cross_interactions
(大运/流年/流月) for 流年/流月 training.

Output: JSONL with one object per line: {"birth_datetime": "...", "chart": {...}}.
Chart is full BaziChart JSON (including optional "dynamic" for temporal subset).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Project root for imports when run as script
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from bazi_engine import BaziCalculator
from bazi_engine.enums import Branch
from bazi_engine.interactions import detect_all_interactions
from bazi_engine.models import BaziChart
from bazi_engine.strength import classify_day_master, compute_day_master_strength
from bazi_engine.temporal import (
    TemporalContext,
    compute_liunian,
    compute_liuyue,
    detect_cross_interactions,
)

DATE_MIN = 1920
DATE_MAX = 2020
TZ_OFFSET_MIN = -12
TZ_OFFSET_MAX = 14
LAT_MIN = -60
LAT_MAX = 60
LNG_MIN = -180
LNG_MAX = 180

DAY_MASTERS = tuple("甲乙丙丁戊己庚辛壬癸")
MONTH_BRANCHES = tuple(b.value for b in Branch)
STRENGTH_LABELS = ("身强", "身弱", "中和")
USEFUL_GOD_LABELS = ("比劫", "食伤", "财", "官杀", "印", "ambiguous")


def _random_datetime(rng: random.Random) -> datetime:
    year = rng.randint(DATE_MIN, DATE_MAX)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    offset = rng.randint(TZ_OFFSET_MIN, TZ_OFFSET_MAX)
    dt = datetime(year, month, day, hour, minute, second)
    return dt.replace(tzinfo=timezone(timedelta(hours=offset)))


def _random_lat_lng(rng: random.Random) -> tuple[float, float]:
    lat = round(rng.uniform(LAT_MIN, LAT_MAX), 6)
    lng = round(rng.uniform(LNG_MIN, LNG_MAX), 6)
    return lat, lng


def _useful_god_class(chart: BaziChart) -> str:
    from bazi_engine.models import _compute_useful_god

    stems = [
        chart.year_pillar.stem.value,
        chart.month_pillar.stem.value,
        chart.day_pillar.stem.value,
        chart.hour_pillar.stem.value,
    ]
    branches = [
        chart.year_pillar.branch.value,
        chart.month_pillar.branch.value,
        chart.day_pillar.branch.value,
        chart.hour_pillar.branch.value,
    ]
    pct = compute_day_master_strength(stems, branches)
    elem = chart.day_pillar.stem.element
    ug = _compute_useful_god(pct, elem)
    if ug is None or ug.get("is_ambiguous", True):
        return "ambiguous"
    primary = ug.get("primary", "")
    return primary if primary in ("比劫", "食伤", "财", "官杀", "印") else "ambiguous"


def _has_punishment_harm_destruction(chart: BaziChart) -> bool:
    stems = [
        chart.year_pillar.stem.value,
        chart.month_pillar.stem.value,
        chart.day_pillar.stem.value,
        chart.hour_pillar.stem.value,
    ]
    branches = [
        chart.year_pillar.branch.value,
        chart.month_pillar.branch.value,
        chart.day_pillar.branch.value,
        chart.hour_pillar.branch.value,
    ]
    ir = detect_all_interactions(stems, branches)
    return (
        len(ir.punishments) > 0
        or len(ir.harms) > 0
        or len(ir.destructions) > 0
    )


def _chart_strata(chart: BaziChart) -> dict[str, str | bool]:
    stems = [
        chart.year_pillar.stem.value,
        chart.month_pillar.stem.value,
        chart.day_pillar.stem.value,
        chart.hour_pillar.stem.value,
    ]
    branches = [
        chart.year_pillar.branch.value,
        chart.month_pillar.branch.value,
        chart.day_pillar.branch.value,
        chart.hour_pillar.branch.value,
    ]
    return {
        "day_master": chart.day_pillar.stem.value,
        "month_branch": chart.month_pillar.branch.value,
        "strength": classify_day_master(stems, branches),
        "has_interaction": _has_punishment_harm_destruction(chart),
        "useful_god": _useful_god_class(chart),
    }


def _build_chart_with_dynamic(
    chart: BaziChart,
    flowing_year: int,
    flowing_month: int,
    gender: int,
) -> BaziChart:
    dayun_seq = BaziCalculator().compute_dayun(chart, gender, max_periods=10)
    dayun = dayun_seq.get_period_for_year(flowing_year)
    liunian = compute_liunian(flowing_year, day_master_stem=chart.day_pillar.stem.value)
    liuyue = compute_liuyue(
        flowing_year,
        flowing_month,
        day_master_stem=chart.day_pillar.stem.value,
    )
    ctx = TemporalContext(dayun=dayun, liunian=liunian, liuyue=liuyue, liuri=None)
    stems = [
        chart.year_pillar.stem.value,
        chart.month_pillar.stem.value,
        chart.day_pillar.stem.value,
        chart.hour_pillar.stem.value,
    ]
    branches = [
        chart.year_pillar.branch.value,
        chart.month_pillar.branch.value,
        chart.day_pillar.branch.value,
        chart.hour_pillar.branch.value,
    ]
    cross = detect_cross_interactions(stems, branches, ctx, useful_god=None)
    return BaziChart(
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


def _targets(total: int) -> dict[str, dict[str, int]]:
    t_dm = total // 10
    t_mb = total // 12
    t_strong = int(total * 0.30)
    t_weak = int(total * 0.30)
    t_mid = total - t_strong - t_weak
    t_interaction = int(total * 0.25)
    t_ug = total // 6
    return {
        "day_master": {dm: t_dm for dm in DAY_MASTERS},
        "month_branch": {mb: t_mb for mb in MONTH_BRANCHES},
        "strength": {"身强": t_strong, "身弱": t_weak, "中和": t_mid},
        "has_interaction": {True: t_interaction, False: total - t_interaction},
        "useful_god": {ug: t_ug for ug in USEFUL_GOD_LABELS},
    }


def _underfilled(
    counts: dict[str, dict[str, int]],
    targets: dict[str, dict[str, int]],
) -> bool:
    for dim, target in targets.items():
        for k, t in target.items():
            if counts[dim].get(k, 0) < t:
                return True
    return False


def _helps_fill(
    strata: dict[str, str | bool],
    counts: dict[str, dict[str, int]],
    targets: dict[str, dict[str, int]],
) -> bool:
    for dim in ("day_master", "month_branch", "strength", "has_interaction", "useful_god"):
        k = strata[dim]
        if counts[dim].get(k, 0) < targets[dim].get(k, 0):
            return True
    return False


def generate_stratified_charts(
    num_charts: int,
    *,
    rng: random.Random,
    dynamic_fraction: float = 0.2,
    max_attempts_ratio: int = 30,
) -> list[dict]:
    """
    Generate num_charts with stratified diversity; a fraction include temporal context.

    Uses acceptance sampling: after num_charts are written, continue until
    all dimension targets are met (day_master, month_branch, strength, interaction, useful_god).
    """
    targets = _targets(num_charts)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    results: list[dict] = []
    total_attempts = num_charts * max_attempts_ratio
    attempts = 0
    while len(results) < num_charts or _underfilled(counts, targets):
        if attempts >= total_attempts:
            break
        attempts += 1
        dt = _random_datetime(rng)
        lat, lng = _random_lat_lng(rng)
        try:
            chart = BaziCalculator.from_datetime(dt, lat, lng)
        except Exception:
            continue
        strata = _chart_strata(chart)
        accept = False
        if len(results) < num_charts:
            accept = True
        elif _helps_fill(strata, counts, targets):
            accept = True
        if not accept:
            continue
        include_dynamic = rng.random() < dynamic_fraction
        if include_dynamic:
            gender = rng.choice((0, 1))
            fy = rng.randint(DATE_MIN, DATE_MAX)
            fm = rng.randint(1, 12)
            chart = _build_chart_with_dynamic(chart, fy, fm, gender)
        chart_dict = chart._to_json_dict()
        birth_iso = chart.birth_dt.isoformat()
        results.append({"birth_datetime": birth_iso, "chart": chart_dict})
        for dim, key in strata.items():
            counts[dim][key] = counts[dim].get(key, 0) + 1

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate stratified synthetic Bazi charts for Phase 2 (synthetic_charts.jsonl).",
    )
    parser.add_argument(
        "--num-charts",
        type=int,
        default=10_000,
        metavar="N",
        help="Target number of charts (default: 10000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic_charts.jsonl"),
        metavar="PATH",
        help="Output JSONL path (default: data/synthetic_charts.jsonl)",
    )
    parser.add_argument(
        "--dynamic-fraction",
        type=float,
        default=0.2,
        metavar="F",
        help="Fraction of charts with 大运/流年/流月 for 流年/流月 tasks (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-attempts-ratio",
        type=int,
        default=30,
        metavar="R",
        help="Max sampling attempts = num_charts * R (default: 30)",
    )
    args = parser.parse_args()

    if args.num_charts < 1:
        print("Error: --num-charts must be >= 1", file=sys.stderr)
        return 1
    if not 0 <= args.dynamic_fraction <= 1:
        print("Error: --dynamic-fraction must be between 0 and 1", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    results = generate_stratified_charts(
        args.num_charts,
        rng=rng,
        dynamic_fraction=args.dynamic_fraction,
        max_attempts_ratio=args.max_attempts_ratio,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} charts to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
