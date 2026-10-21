"""
Factual consistency checks for Phase 2 curated records (chart vs reasoning/output).

No LLM or remote API calls: chart JSON plus regex/heuristic text checks only.

Compares ten-gods, day-master strength classification, natal interactions, and
parsed 用神 claims against the structured chart JSON from BaziChart.to_json().

Usage:
  python scripts/factual_check.py --input data/qwen_raw.jsonl -o data/qwen_fact_checked.jsonl
  python scripts/factual_check.py -i data/qwen_raw.jsonl --print-summary
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bazi_engine.interactions.branch_six_combinations import BRANCH_SIX_PAIRS
from bazi_engine.interactions.clashes import BRANCH_CLASH_PAIRS, STEM_CLASH_PAIRS
from bazi_engine.interactions.destructions import BRANCH_DESTRUCTION_PAIRS
from bazi_engine.interactions.harms import BRANCH_SIX_HARMS
from bazi_engine.interactions.stem_combinations import STEM_COMBINATION_PAIRS

# Explicit 十神 (excluding 日主; 日主 is always allowed in prose).
_TEN_GOD_NAMES = (
    "比肩",
    "劫财",
    "食神",
    "伤官",
    "偏财",
    "正财",
    "七杀",
    "正官",
    "偏印",
    "正印",
)
_TEN_GOD_PATTERN = re.compile("|".join(re.escape(x) for x in sorted(_TEN_GOD_NAMES, key=len, reverse=True)))

_COMPOSITE_CHECKS: tuple[tuple[str, frozenset[str]], ...] = (
    ("官杀", frozenset({"正官", "七杀"})),
    ("财星", frozenset({"偏财", "正财"})),
    ("食伤", frozenset({"食神", "伤官"})),
    ("比劫", frozenset({"比肩", "劫财"})),
    ("印星", frozenset({"偏印", "正印"})),
)

_PUNISHMENT_LABELS = ("无恩之刑", "持势之刑", "无礼之刑", "自刑")

# ---------------------------------------------------------------------------
# Context-aware negation / hypothetical detection
# ---------------------------------------------------------------------------
# Prefixes immediately before a term that indicate absence (longest first).
_NEGATION_PREFIXES: tuple[str, ...] = (
    "无明现", "并非", "没有", "全无", "不见", "难见", "无", "缺", "乏",
)
# Suffixes immediately after a term that indicate absence.
_NEGATION_SUFFIXES: tuple[str, ...] = (
    "皆不显", "皆不现", "皆不透",
    "不显", "不透", "不现", "全缺", "缺失", "缺位", "缺乏",
    "为0", "为零", "不见", "不足",
    "皆弱", "皆微", "皆缺", "皆空",
    "难显", "难聚", "难发", "难得", "难求",
    "暗藏", "虚浮", "更弱", "微弱", "弱隐", "亦无",
    "缺",
)
# Single characters that signal absence when found near the match.
_ABSENCE_CHARS = frozenset("无缺不没")
# Keywords that signal the sentence is about a future transit, not natal chart.
_TRANSIT_KEYWORDS: tuple[str, ...] = (
    "大运", "流年", "流月", "若行", "若走", "若逢", "若遇", "若得",
)
# 用神 context: discussing what the chart needs, not what it has.
_YONGSHEN_KEYWORDS: tuple[str, ...] = (
    "用神取", "用神为", "用神明确取", "用神首取",
)


def _is_excused_context(
    text: str,
    start: int,
    end: int,
    *,
    prefix_window: int = 10,
    suffix_window: int = 8,
    transit_window: int = 30,
) -> bool:
    """Return True if the match at text[start:end] sits in negation or transit context."""
    pre = text[max(0, start - prefix_window) : start]
    suf = text[end : min(len(text), end + suffix_window)]

    for p in _NEGATION_PREFIXES:
        if pre.endswith(p):
            return True
    for s in _NEGATION_SUFFIXES:
        if suf.startswith(s):
            return True

    # Broader absence-word scan: any 无/缺/不/没 within a tight window.
    near_pre = text[max(0, start - 6) : start]
    near_suf = text[end : min(len(text), end + 6)]
    if _ABSENCE_CHARS & set(near_pre + near_suf):
        return True

    # Parenthetical explanation of an element, e.g. "无木（官杀）".
    paren_pre = text[max(0, start - 4) : start]
    if "（" in paren_pre or "(" in paren_pre:
        return True

    broad_pre = text[max(0, start - transit_window) : start]
    for kw in _TRANSIT_KEYWORDS:
        if kw in broad_pre:
            return True
    for kw in _YONGSHEN_KEYWORDS:
        if kw in broad_pre:
            return True

    return False


def _has_positive_mention(text: str, term: str) -> bool:
    """True if *term* appears at least once in *text* outside negation/transit context."""
    for m in re.finditer(re.escape(term), text):
        if not _is_excused_context(text, m.start(), m.end()):
            return True
    return False

_USEFUL_PRIMARY_ALIASES: dict[str, frozenset[str]] = {
    "印": frozenset({"偏印", "正印", "印"}),
    "比劫": frozenset({"比肩", "劫财", "比劫"}),
    "财": frozenset({"偏财", "正财", "财", "财星"}),
    "食伤": frozenset({"食神", "伤官", "食伤"}),
    "官杀": frozenset({"正官", "七杀", "官杀"}),
}


def _chart_dict(chart_val: Any) -> dict[str, Any] | None:
    if chart_val is None:
        return None
    if isinstance(chart_val, dict):
        return chart_val
    if isinstance(chart_val, str):
        try:
            return json.loads(chart_val)
        except json.JSONDecodeError:
            return None
    return None


def _collect_ten_gods_from_chart(chart: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    tg = chart.get("ten_gods") or {}
    for key in ("year_stem", "month_stem", "day_stem", "hour_stem"):
        v = tg.get(key)
        if isinstance(v, str) and v:
            out.add(v)
    for key in ("year_branch", "month_branch", "day_branch", "hour_branch"):
        for x in tg.get(key) or []:
            if isinstance(x, str) and x:
                out.add(x)
    hidden = chart.get("hidden_stems") or {}
    for _k, stems in hidden.items():
        if not isinstance(stems, list):
            continue
        for item in stems:
            if isinstance(item, dict):
                g = item.get("ten_god")
                if isinstance(g, str) and g:
                    out.add(g)
    return out


def _branch_pair_sets(interactions: dict[str, Any]) -> dict[str, Any]:
    """Normalize natal interaction branch/stem pairs for lookup."""

    def pair_from_branches(item: dict[str, Any]) -> frozenset[str] | None:
        br = item.get("branches") or item.get("members_present")
        if isinstance(br, list) and len(br) >= 2:
            return frozenset({str(br[0]), str(br[1])})
        if isinstance(br, list) and len(br) == 1 and len(item.get("positions", [])) >= 2:
            return None
        return None

    def pair_from_stems(item: dict[str, Any]) -> frozenset[str] | None:
        st = item.get("stems")
        if isinstance(st, list) and len(st) >= 2:
            return frozenset({str(st[0]), str(st[1])})
        ia, ib = item.get("item_a"), item.get("item_b")
        if ia and ib:
            return frozenset({str(ia), str(ib)})
        return None

    clashes: set[frozenset[str]] = set()
    for item in interactions.get("branch_clashes") or []:
        p = pair_from_branches(item)
        if p and len(p) == 2:
            clashes.add(p)
    six: set[frozenset[str]] = set()
    for item in interactions.get("branch_six_combinations") or []:
        p = pair_from_branches(item)
        if p and len(p) == 2:
            six.add(p)
    harms: set[frozenset[str]] = set()
    for item in interactions.get("harms") or []:
        p = pair_from_branches(item)
        if p and len(p) == 2:
            harms.add(p)
    dest: set[frozenset[str]] = set()
    for item in interactions.get("destructions") or []:
        p = pair_from_branches(item)
        if p and len(p) == 2:
            dest.add(p)
    stem_clash: set[frozenset[str]] = set()
    for item in interactions.get("stem_clashes") or []:
        p = pair_from_stems(item)
        if p and len(p) == 2:
            stem_clash.add(p)
    stem_comb: set[frozenset[str]] = set()
    for item in interactions.get("stem_combinations") or []:
        p = pair_from_stems(item)
        if p and len(p) == 2:
            stem_comb.add(p)
    punish_types: set[str] = set()
    for item in interactions.get("punishments") or []:
        t = item.get("punishment_type")
        if isinstance(t, str) and t:
            punish_types.add(t)
    return {
        "branch_clashes": clashes,
        "branch_six": six,
        "harms": harms,
        "destructions": dest,
        "stem_clashes": stem_clash,
        "stem_combinations": stem_comb,
        "punishment_types": punish_types,
    }


def _mentions_branch_pair(text: str, a: str, b: str, kind: str) -> bool:
    """Heuristic: natal pair + 冲/合/害/破 style verbs."""
    if a not in text or b not in text:
        return False
    patterns = {
        "clash": (
            rf"{re.escape(a)}{re.escape(b)}冲",
            rf"{re.escape(b)}{re.escape(a)}冲",
            rf"{re.escape(a)}与{re.escape(b)}冲",
            rf"{re.escape(b)}与{re.escape(a)}冲",
            rf"{re.escape(a)}{re.escape(b)}相冲",
            rf"{re.escape(b)}{re.escape(a)}相冲",
            rf"{re.escape(a)}冲{re.escape(b)}",
            rf"{re.escape(b)}冲{re.escape(a)}",
        ),
        "six": (
            rf"{re.escape(a)}{re.escape(b)}合",
            rf"{re.escape(b)}{re.escape(a)}合",
            rf"{re.escape(a)}与{re.escape(b)}合",
            rf"{re.escape(b)}与{re.escape(a)}合",
            rf"{re.escape(a)}{re.escape(b)}六合",
        ),
        "harm": (
            rf"{re.escape(a)}{re.escape(b)}害",
            rf"{re.escape(b)}{re.escape(a)}害",
            rf"{re.escape(a)}与{re.escape(b)}害",
            rf"{re.escape(b)}与{re.escape(a)}害",
        ),
        "destruction": (
            rf"{re.escape(a)}{re.escape(b)}破",
            rf"{re.escape(b)}{re.escape(a)}破",
            rf"{re.escape(a)}与{re.escape(b)}破",
            rf"{re.escape(b)}与{re.escape(a)}破",
        ),
    }
    for pat in patterns.get(kind, ()):
        if re.search(pat, text):
            return True
    return False


def _mentions_stem_pair(text: str, a: str, b: str, kind: str) -> bool:
    if a not in text or b not in text:
        return False
    if kind == "clash":
        pats = (
            rf"{re.escape(a)}{re.escape(b)}冲",
            rf"{re.escape(b)}{re.escape(a)}冲",
            rf"{re.escape(a)}与{re.escape(b)}冲",
            rf"{re.escape(a)}冲{re.escape(b)}",
        )
    elif kind == "combine":
        pats = (
            rf"{re.escape(a)}{re.escape(b)}合",
            rf"{re.escape(b)}{re.escape(a)}合",
            rf"{re.escape(a)}与{re.escape(b)}合",
        )
    else:
        pats = ()
    return any(re.search(p, text) for p in pats)


def _positive_strength_claim(text: str, term: str) -> bool:
    """True if `term` (身强/身弱) appears with a positive (non-negated) claim."""
    for m in re.finditer(re.escape(term), text):
        i = m.start()
        if i > 0 and text[i - 1] in "不非莫难略偏稍":
            continue
        if i > 1 and text[i - 2 : i] in ("并非", "不是", "不算", "没有"):
            continue
        return True
    return False


def _check_interactions(text: str, pair_sets: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for a, b in BRANCH_CLASH_PAIRS:
        if _mentions_branch_pair(text, a, b, "clash"):
            if frozenset({a, b}) not in pair_sets["branch_clashes"]:
                issues.append(
                    {
                        "type": "phantom_interaction",
                        "message": f"text claims branch clash {a}{b} not in chart natal branch_clashes",
                    }
                )
    for a, b in BRANCH_SIX_PAIRS:
        if _mentions_branch_pair(text, a, b, "six"):
            if frozenset({a, b}) not in pair_sets["branch_six"]:
                issues.append(
                    {
                        "type": "phantom_interaction",
                        "message": f"text claims 六合 {a}{b} not in chart natal branch_six_combinations",
                    }
                )
    for pair in BRANCH_SIX_HARMS:
        a, b = tuple(sorted(pair, key=lambda x: "子丑寅卯辰巳午未申酉戌亥".index(x)))
        if _mentions_branch_pair(text, a, b, "harm"):
            if frozenset({a, b}) not in pair_sets["harms"]:
                issues.append(
                    {
                        "type": "phantom_interaction",
                        "message": f"text claims 六害 {a}{b} not in chart natal harms",
                    }
                )
    for a, b in BRANCH_DESTRUCTION_PAIRS:
        if _mentions_branch_pair(text, a, b, "destruction"):
            if frozenset({a, b}) not in pair_sets["destructions"]:
                issues.append(
                    {
                        "type": "phantom_interaction",
                        "message": f"text claims 相破 {a}{b} not in chart natal destructions",
                    }
                )
    for a, b in STEM_CLASH_PAIRS:
        if _mentions_stem_pair(text, a, b, "clash"):
            if frozenset({a, b}) not in pair_sets["stem_clashes"]:
                issues.append(
                    {
                        "type": "phantom_interaction",
                        "message": f"text claims stem clash {a}{b} not in chart natal stem_clashes",
                    }
                )
    for a, b in STEM_COMBINATION_PAIRS:
        if _mentions_stem_pair(text, a, b, "combine"):
            if frozenset({a, b}) not in pair_sets["stem_combinations"]:
                issues.append(
                    {
                        "type": "phantom_interaction",
                        "message": f"text claims stem combination {a}{b} not in chart natal stem_combinations",
                    }
                )
    for label in _PUNISHMENT_LABELS:
        if label in text and label not in pair_sets["punishment_types"]:
            issues.append(
                {
                    "type": "phantom_interaction",
                    "message": f"text mentions punishment {label} not present in chart natal punishments",
                }
            )
    return issues


def _normalize_useful_token(raw: str) -> str | None:
    s = raw.strip()
    for primary, aliases in _USEFUL_PRIMARY_ALIASES.items():
        if s in aliases:
            return primary
    if s in _USEFUL_PRIMARY_ALIASES:
        return s
    return None


def _check_useful_god(text: str, chart: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    ug = chart.get("useful_god")
    if not isinstance(ug, dict) or ug.get("is_ambiguous") or not ug.get("primary"):
        return issues
    chart_primary = str(ug["primary"])
    for m in re.finditer(
        r"用神\s*[{是为以：:]\s*([^\s，。；;\n]{1,8})",
        text,
    ):
        token = _normalize_useful_token(m.group(1))
        if token is None:
            continue
        if token != chart_primary:
            issues.append(
                {
                    "type": "useful_god_mismatch",
                    "message": f"text 用神 implies {token!r} but chart useful_god.primary is {chart_primary!r}",
                }
            )
    return issues


def check_factual_consistency(
    record: dict[str, Any],
    *,
    check_interactions: bool = True,
) -> list[dict[str, str]]:
    """
    Return a list of issue dicts with keys type, message.
    Empty list means no issues detected (not a proof of correctness).
    """
    issues: list[dict[str, str]] = []
    chart = _chart_dict(record.get("chart"))
    if chart is None:
        return [{"type": "chart_parse", "message": "missing or invalid chart"}]

    parts = [
        str(record.get("reasoning") or ""),
        str(record.get("output") or ""),
        str(record.get("query") or ""),
    ]
    text = "\n".join(parts)

    allowed_tg = _collect_ten_gods_from_chart(chart)
    for comp, needed in _COMPOSITE_CHECKS:
        if comp not in text:
            continue
        if not (allowed_tg & needed) and _has_positive_mention(text, comp):
            issues.append(
                {
                    "type": "composite_ten_god",
                    "message": f"text uses {comp!r} but chart has no matching ten-god in {sorted(needed)}",
                }
            )

    phantom_tg: set[str] = set()
    for m in _TEN_GOD_PATTERN.finditer(text):
        name = m.group(0)
        if name == "日主":
            continue
        if name not in allowed_tg and name not in phantom_tg:
            if not _is_excused_context(text, m.start(), m.end()):
                phantom_tg.add(name)
                issues.append(
                    {
                        "type": "phantom_ten_god",
                        "message": f"text mentions {name} not present on chart ten_gods/hidden_stems",
                    }
                )

    dm = chart.get("day_master") or {}
    classification = dm.get("classification")
    if isinstance(classification, str):
        strong = _positive_strength_claim(text, "身强")
        weak = _positive_strength_claim(text, "身弱")
        if strong and not weak and classification == "身弱":
            issues.append(
                {
                    "type": "strength_mismatch",
                    "message": "text asserts 身强 but chart day_master.classification is 身弱",
                }
            )
        if weak and not strong and classification == "身强":
            issues.append(
                {
                    "type": "strength_mismatch",
                    "message": "text asserts 身弱 but chart day_master.classification is 身强",
                }
            )

    if check_interactions:
        interactions = chart.get("interactions") or {}
        if isinstance(interactions, dict):
            pair_sets = _branch_pair_sets(interactions)
            issues.extend(_check_interactions(text, pair_sets))

    issues.extend(_check_useful_god(text, chart))
    return issues


def _annotate_record(record: dict[str, Any], issues: list[dict[str, str]]) -> dict[str, Any]:
    out = dict(record)
    out["factual_issues"] = issues
    out["factual_ok"] = len(issues) == 0
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Factual consistency check for Phase 2 JSONL records.")
    p.add_argument("--input", "-i", type=Path, required=True, help="Input JSONL path")
    p.add_argument("--output", "-o", type=Path, default=None, help="Output JSONL (default: stdout)")
    p.add_argument("--print-summary", action="store_true", help="Print counts to stderr")
    p.add_argument(
        "--no-interactions",
        action="store_true",
        help="Skip natal interaction phantom checks (noisier on free-form text)",
    )
    args = p.parse_args(argv)

    total = 0
    with_issues = 0
    by_type: dict[str, int] = {}

    out_f = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                issues = check_factual_consistency(
                    rec,
                    check_interactions=not args.no_interactions,
                )
                total += 1
                if issues:
                    with_issues += 1
                    for it in issues:
                        by_type[it["type"]] = by_type.get(it["type"], 0) + 1
                out_f.write(json.dumps(_annotate_record(rec, issues), ensure_ascii=False) + "\n")
    finally:
        if args.output:
            out_f.close()

    if args.print_summary:
        print(
            f"factual_check: lines={total} with_issues={with_issues} by_type={by_type}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
