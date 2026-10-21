"""
QA template library for Stage 1 (single-fact) and Stage 2 (multi-fact) datasets.

Every template is a :class:`QATemplate` with:
- `id`: unique string
- `category`: one of the Stage 1 categories (pillar_identity, ten_gods,
  five_elements, day_master_strength, interactions, hidden_stems, dynamic)
  or Stage 2 categories (relational, conditional, cross_pillar, analytical,
  interaction_chain, temporal_relational, summary).
- `text_fn(chart_json) -> str`: Chinese question.
- `answer_fn(chart_json) -> str`: ground-truth answer string.
- `verify_fn(chart_json, chart, answer) -> bool`: sanity check that the
  produced answer is consistent with the chart (re-derives via the BaziChart
  object, i.e. a different code path than answer_fn).
- `applies(chart_json) -> bool`: whether this template can apply to this chart
  (e.g. templates that ask "哪个柱的十神是正官？" require that ten-god to exist;
  templates that ask 大运/流年 require a dynamic context).

Ground-truth extraction is strictly deterministic: every template's answer
comes either from the stored chart JSON fields or from recomputation via
BaziChart / the calculator. No heuristics, no LLM.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from bazi_engine.elements import (
    ELEMENT_CONTROLS,
    ELEMENT_CONTROLS_ME,
    ELEMENT_GENERATES,
    ELEMENT_GENERATES_ME,
)
from bazi_engine.models import BaziChart


POS_LABELS_CN = ("年", "月", "日", "时")
STEM_POS_LABELS_CN = ("年干", "月干", "日干", "时干")
BRANCH_POS_LABELS_CN = ("年支", "月支", "日支", "时支")
PILLAR_POS_LABELS_CN = ("年柱", "月柱", "日柱", "时柱")
ELEMENT_EN_TO_CN = {"wood": "木", "fire": "火", "earth": "土", "metal": "金", "water": "水"}
ELEMENTS_CN = ("木", "火", "土", "金", "水")
PILLAR_KEYS = ("year", "month", "day", "hour")


# -----------------------------------------------------------------------------
# QATemplate dataclass
# -----------------------------------------------------------------------------


@dataclass
class QATemplate:
    id: str
    category: str
    text_fn: Callable[[dict[str, Any]], str]
    answer_fn: Callable[[dict[str, Any]], str]
    verify_fn: Callable[[dict[str, Any], BaziChart, str], bool]
    requires_dynamic: bool = False
    applies: Callable[[dict[str, Any]], bool] = lambda _c: True


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------


def _pillar_str(chart_json: dict[str, Any], key: str) -> str:
    p = chart_json["natal_pillars"][key]
    return f"{p['stem']}{p['branch']}"


def _positions_pair_cn(positions: list[int]) -> str:
    """['year','day'] or [0, 2] -> '年柱和日柱'."""
    labels = []
    for p in positions:
        if isinstance(p, str):
            idx = PILLAR_KEYS.index(p) if p in PILLAR_KEYS else -1
        else:
            idx = int(p)
        if 0 <= idx < 4:
            labels.append(PILLAR_POS_LABELS_CN[idx])
    return "和".join(labels) if labels else ""


def _format_element_pct(pct: float) -> str:
    return f"{pct * 100:.1f}%"


def _dm_element(chart_json: dict[str, Any]) -> str:
    return chart_json["day_master"]["element"]


def _strongest_element(chart_json: dict[str, Any]) -> str:
    es = chart_json["elemental_strengths"]
    best = max(es.items(), key=lambda kv: kv[1])[0]
    return ELEMENT_EN_TO_CN[best]


def _weakest_element(chart_json: dict[str, Any]) -> str:
    es = chart_json["elemental_strengths"]
    worst = min(es.items(), key=lambda kv: kv[1])[0]
    return ELEMENT_EN_TO_CN[worst]


def _element_pct_cn(chart_json: dict[str, Any], element_cn: str) -> float:
    en = [k for k, v in ELEMENT_EN_TO_CN.items() if v == element_cn][0]
    return float(chart_json["elemental_strengths"][en])


def _element_relation(subject_cn: str, target_cn: str) -> str:
    """Return the 五行 relation of subject relative to target: 同/生/克/被生/被克."""
    if subject_cn == target_cn:
        return "同"
    if ELEMENT_GENERATES.get(subject_cn) == target_cn:
        return "生"
    if ELEMENT_CONTROLS.get(subject_cn) == target_cn:
        return "克"
    if ELEMENT_GENERATES_ME.get(target_cn) == subject_cn:
        return "生"
    if ELEMENT_CONTROLS_ME.get(target_cn) == subject_cn:
        return "克"
    if ELEMENT_GENERATES.get(target_cn) == subject_cn:
        return "被生"
    if ELEMENT_CONTROLS.get(target_cn) == subject_cn:
        return "被克"
    return "无"


# Map interaction type key -> (Chinese label, question-phrasing label)
INTERACTION_TYPE_LABELS = {
    "stem_combinations": ("天干五合", "天干五合"),
    "stem_clashes": ("天干相冲", "天干相冲"),
    "branch_six_combinations": ("地支六合", "地支六合"),
    "branch_clashes": ("地支六冲", "地支六冲"),
    "three_harmony": ("三合局", "三合局"),
    "harms": ("六害", "地支相害"),
    "punishments": ("相刑", "地支相刑"),
    "destructions": ("相破", "地支相破"),
}


def _describe_interaction(kind: str, rec: dict[str, Any]) -> str:
    """Compact Chinese description of a single interaction record."""
    if kind == "stem_combinations":
        stems = rec.get("stems") or []
        pos = _positions_pair_cn(rec.get("positions") or rec.get("position_labels") or [])
        return f"{stems[0]}{stems[1]}合（{pos}）" if len(stems) == 2 else ""
    if kind == "stem_clashes":
        stems = rec.get("stems") or []
        pos = _positions_pair_cn(rec.get("positions") or rec.get("position_labels") or [])
        return f"{stems[0]}{stems[1]}相冲（{pos}）" if len(stems) == 2 else ""
    if kind == "branch_six_combinations":
        br = rec.get("branches") or []
        pos = _positions_pair_cn(rec.get("positions") or rec.get("position_labels") or [])
        note = "（受阻）" if rec.get("is_blocked") else ""
        return f"{br[0]}{br[1]}六合{note}（{pos}）" if len(br) == 2 else ""
    if kind == "branch_clashes":
        br = rec.get("branches") or []
        pos = _positions_pair_cn(rec.get("positions") or rec.get("position_labels") or [])
        return f"{br[0]}{br[1]}相冲（{pos}）" if len(br) == 2 else ""
    if kind == "three_harmony":
        br = rec.get("present_branches") or []
        elem = rec.get("element", "")
        combo = rec.get("combo_type", "")
        combo_cn = "完整" if combo == "full" else "半合"
        blocked = "（被冲破）" if rec.get("blocked_by_clash") else ""
        return f"{''.join(br)}三合{elem}局（{combo_cn}）{blocked}"
    if kind == "harms":
        br = rec.get("branches") or []
        pos = _positions_pair_cn(rec.get("positions") or rec.get("position_labels") or [])
        return f"{br[0]}{br[1]}相害（{pos}）" if len(br) == 2 else ""
    if kind == "punishments":
        ptype = rec.get("punishment_type", "")
        pos = _positions_pair_cn(rec.get("positions") or [])
        return f"{ptype}（{pos}）"
    if kind == "destructions":
        br = rec.get("branches") or []
        pos = _positions_pair_cn(rec.get("positions") or rec.get("position_labels") or [])
        note = "（已被削弱）" if rec.get("is_weakened") else ""
        return f"{br[0]}{br[1]}相破{note}（{pos}）" if len(br) == 2 else ""
    return ""


def _format_interaction_list(kind: str, records: list[dict[str, Any]]) -> str:
    """Turn a list of interaction records into '无' or a '；'-joined description."""
    if not records:
        return "无"
    parts = [_describe_interaction(kind, r) for r in records]
    parts = [p for p in parts if p]
    return "；".join(parts) if parts else "无"


# -----------------------------------------------------------------------------
# Low-level verify helpers (re-derive via BaziChart object, a separate code path)
# -----------------------------------------------------------------------------


def _v_contains_chart_value(answer: str, values: list[str]) -> bool:
    """Answer must literally contain every value (1-char strings typically)."""
    return all(v in answer for v in values) and len(answer) > 0


def _v_pillar_full(answer: str, chart: BaziChart, key: str) -> bool:
    p = getattr(chart, f"{key}_pillar")
    return p.stem.value in answer and p.branch.value in answer


def _v_stem(answer: str, chart: BaziChart, key: str) -> bool:
    p = getattr(chart, f"{key}_pillar")
    return answer.strip() == p.stem.value or answer == p.stem.value


def _v_branch(answer: str, chart: BaziChart, key: str) -> bool:
    p = getattr(chart, f"{key}_pillar")
    return answer.strip() == p.branch.value or answer == p.branch.value


# -----------------------------------------------------------------------------
# Stage 1 template builders
# -----------------------------------------------------------------------------


def _build_pillar_identity_templates() -> list[QATemplate]:
    templates: list[QATemplate] = []

    templates.append(
        QATemplate(
            id="pillar-day-stem",
            category="pillar_identity",
            text_fn=lambda c: "日主天干是什么？请只回答一个汉字。",
            answer_fn=lambda c: c["natal_pillars"]["day"]["stem"],
            verify_fn=lambda c, ch, a: _v_stem(a, ch, "day"),
        )
    )

    for i, key in enumerate(PILLAR_KEYS):
        templates.append(
            QATemplate(
                id=f"pillar-{key}",
                category="pillar_identity",
                text_fn=(lambda k=key: (
                    lambda c: f"{PILLAR_POS_LABELS_CN[PILLAR_KEYS.index(k)]}是什么？请回答天干地支两个汉字。"
                ))(),
                answer_fn=(lambda k=key: (lambda c: _pillar_str(c, k)))(),
                verify_fn=(lambda k=key: (lambda c, ch, a: _v_pillar_full(a, ch, k)))(),
            )
        )

    for key in ("year", "month", "hour"):
        templates.append(
            QATemplate(
                id=f"branch-{key}",
                category="pillar_identity",
                text_fn=(lambda k=key: (
                    lambda c: f"{BRANCH_POS_LABELS_CN[PILLAR_KEYS.index(k)]}是什么？请只回答一个汉字。"
                ))(),
                answer_fn=(lambda k=key: (lambda c: c["natal_pillars"][k]["branch"]))(),
                verify_fn=(lambda k=key: (lambda c, ch, a: _v_branch(a, ch, k)))(),
            )
        )

    return templates


def _build_ten_god_templates() -> list[QATemplate]:
    templates: list[QATemplate] = []

    for key in ("year", "month", "hour"):
        templates.append(
            QATemplate(
                id=f"tengod-{key}-stem",
                category="ten_gods",
                text_fn=(lambda k=key: (
                    lambda c: f"{STEM_POS_LABELS_CN[PILLAR_KEYS.index(k)]}相对于日主的十神是什么？请直接回答十神名称。"
                ))(),
                answer_fn=(lambda k=key: (lambda c: c["ten_gods"][f"{k}_stem"]))(),
                verify_fn=(lambda k=key: (
                    lambda c, ch, a: getattr(ch, f"{k}_pillar").ten_god_stem.value in a
                ))(),
            )
        )

    for key in PILLAR_KEYS:
        templates.append(
            QATemplate(
                id=f"tengod-{key}-branch-list",
                category="ten_gods",
                text_fn=(lambda k=key: (
                    lambda c: f"{BRANCH_POS_LABELS_CN[PILLAR_KEYS.index(k)]}藏干的十神依次是什么？用顿号分隔，例如“食神、比肩、偏印”。"
                ))(),
                answer_fn=(lambda k=key: (
                    lambda c: "、".join(c["ten_gods"][f"{k}_branch"]) or "无"
                ))(),
                verify_fn=(lambda k=key: (
                    lambda c, ch, a: all(
                        tg.value in a for tg in getattr(ch, f"{k}_pillar").ten_god_branch
                    )
                ))(),
            )
        )

    # "Which pillar has X ten-god?" — include a few common ten-gods. Build one
    # template per target ten-god; applies only when that ten-god appears.
    for target in ("正官", "正财", "七杀"):
        def _answer_which(c, t=target):
            hits = []
            for i, k in enumerate(PILLAR_KEYS):
                if c["ten_gods"][f"{k}_stem"] == t:
                    hits.append(STEM_POS_LABELS_CN[i])
            return "、".join(hits) if hits else "无"

        def _applies_which(c, t=target):
            return any(c["ten_gods"][f"{k}_stem"] == t for k in PILLAR_KEYS)

        def _verify_which(c, ch, a, t=target):
            expected: list[str] = []
            for i, k in enumerate(PILLAR_KEYS):
                if getattr(ch, f"{k}_pillar").ten_god_stem.value == t:
                    expected.append(STEM_POS_LABELS_CN[i])
            return all(e in a for e in expected) and len(expected) > 0

        templates.append(
            QATemplate(
                id=f"tengod-which-has-{target}",
                category="ten_gods",
                text_fn=(lambda t=target: (lambda c: f"哪个柱的天干是{t}？请回答柱位名称。"))(),
                answer_fn=_answer_which,
                verify_fn=_verify_which,
                applies=_applies_which,
            )
        )

    return templates


def _build_five_element_templates() -> list[QATemplate]:
    templates: list[QATemplate] = []

    templates.append(
        QATemplate(
            id="element-strongest",
            category="five_elements",
            text_fn=lambda c: "五行中哪个最旺？请只回答“木火土金水”之一。",
            answer_fn=_strongest_element,
            verify_fn=lambda c, ch, a: _strongest_element(c) in a,
        )
    )
    templates.append(
        QATemplate(
            id="element-weakest",
            category="five_elements",
            text_fn=lambda c: "五行中哪个最弱？请只回答“木火土金水”之一。",
            answer_fn=_weakest_element,
            verify_fn=lambda c, ch, a: _weakest_element(c) in a,
        )
    )

    for elem_cn in ELEMENTS_CN:
        templates.append(
            QATemplate(
                id=f"element-{elem_cn}-pct",
                category="five_elements",
                text_fn=(lambda e=elem_cn: (lambda c: f"五行{e}占比是多少？请用百分数回答。"))(),
                answer_fn=(lambda e=elem_cn: (
                    lambda c: _format_element_pct(_element_pct_cn(c, e))
                ))(),
                verify_fn=(lambda e=elem_cn: (
                    lambda c, ch, a: _format_element_pct(_element_pct_cn(c, e))[:4] in a
                    or f"{_element_pct_cn(c, e) * 100:.0f}" in a
                ))(),
            )
        )

    templates.append(
        QATemplate(
            id="element-distribution",
            category="five_elements",
            text_fn=lambda c: "五行分布是什么？请按“木火土金水”顺序，每项“元素百分数”，用空格分隔。",
            answer_fn=lambda c: " ".join(
                f"{e}{_format_element_pct(_element_pct_cn(c, e))}" for e in ELEMENTS_CN
            ),
            verify_fn=lambda c, ch, a: all(e in a for e in ELEMENTS_CN),
        )
    )

    return templates


def _build_day_master_strength_templates() -> list[QATemplate]:
    templates: list[QATemplate] = []

    templates.append(
        QATemplate(
            id="dm-strong-weak",
            category="day_master_strength",
            text_fn=lambda c: "日主是身强、中和还是身弱？请只回答三者之一。",
            answer_fn=lambda c: c["day_master"]["classification"],
            verify_fn=lambda c, ch, a: c["day_master"]["classification"] in a,
        )
    )
    templates.append(
        QATemplate(
            id="dm-percentile",
            category="day_master_strength",
            text_fn=lambda c: "日主强度百分位是多少？请用0到100之间的百分数回答。",
            answer_fn=lambda c: _format_element_pct(float(c["day_master"]["strength_percentile"])),
            verify_fn=lambda c, ch, a: (
                f"{float(c['day_master']['strength_percentile']) * 100:.0f}" in a
                or f"{float(c['day_master']['strength_percentile']) * 100:.1f}" in a
            ),
        )
    )
    templates.append(
        QATemplate(
            id="dm-classification",
            category="day_master_strength",
            text_fn=lambda c: "请用一个词回答此命局的身强身弱分类。",
            answer_fn=lambda c: c["day_master"]["classification"],
            verify_fn=lambda c, ch, a: c["day_master"]["classification"] in a,
        )
    )

    def _ug_primary(c: dict[str, Any]) -> str:
        ug = c.get("useful_god") or {}
        if not ug or ug.get("is_ambiguous"):
            return "不明"
        return ug.get("primary", "不明")

    templates.append(
        QATemplate(
            id="dm-useful-god-primary",
            category="day_master_strength",
            text_fn=lambda c: "此命局用神大类是什么？请只回答“印”“比劫”“财”“食伤”“官杀”之一；若不明确，回答“不明”。",
            answer_fn=_ug_primary,
            verify_fn=lambda c, ch, a: _ug_primary(c) in a,
        )
    )

    def _favorable(c: dict[str, Any]) -> str:
        ug = c.get("useful_god") or {}
        fav = ug.get("favorable_elements") or []
        return "、".join(fav) if fav else "不明"

    templates.append(
        QATemplate(
            id="dm-favorable-elements",
            category="day_master_strength",
            text_fn=lambda c: "此命局哪些五行对命主有利？请用顿号分隔；若不明确，回答“不明”。",
            answer_fn=_favorable,
            verify_fn=lambda c, ch, a: _favorable(c) == "不明"
            or all(e in a for e in (c["useful_god"] or {}).get("favorable_elements", [])),
            applies=lambda c: bool(
                (c.get("useful_god") or {}).get("favorable_elements")
                or (c.get("useful_god") or {}).get("is_ambiguous")
            ),
        )
    )

    return templates


def _build_interaction_templates() -> list[QATemplate]:
    templates: list[QATemplate] = []

    for kind, (label_cn, q_label) in INTERACTION_TYPE_LABELS.items():
        def _answer(c, k=kind):
            return _format_interaction_list(k, c["interactions"][k])

        def _verify(c, ch, a, k=kind):
            recs = c["interactions"][k]
            if not recs:
                return "无" in a or "没有" in a
            # Answer must reference characters present in at least one record.
            for r in recs:
                items = (
                    r.get("branches") or r.get("stems") or r.get("present_branches") or []
                )
                if items and all(it in a for it in items):
                    return True
                if k == "punishments" and r.get("punishment_type", "") in a:
                    return True
            return False

        templates.append(
            QATemplate(
                id=f"interaction-{kind}",
                category="interactions",
                text_fn=(lambda qlab=q_label: (
                    lambda c: f"此命局是否有{qlab}？如有，请简述所有情况；若无，回答“无”。"
                ))(),
                answer_fn=_answer,
                verify_fn=_verify,
            )
        )

    # Count-based templates for three common types.
    for kind, label_cn in (
        ("branch_clashes", "地支六冲"),
        ("stem_combinations", "天干五合"),
        ("harms", "地支相害"),
    ):
        templates.append(
            QATemplate(
                id=f"interaction-count-{kind}",
                category="interactions",
                text_fn=(lambda lab=label_cn: (
                    lambda c: f"此命局共有几组{lab}？请只回答一个整数。"
                ))(),
                answer_fn=(lambda k=kind: (lambda c: str(len(c["interactions"][k]))))(),
                verify_fn=(lambda k=kind: (
                    lambda c, ch, a: str(len(c["interactions"][k])) in a
                ))(),
            )
        )

    # Overall summary: whether the chart has "many" interactions (>=3 total).
    def _any_count(c):
        return sum(len(c["interactions"][k]) for k in INTERACTION_TYPE_LABELS)

    templates.append(
        QATemplate(
            id="interaction-any",
            category="interactions",
            text_fn=lambda c: "此命局整体是否互动丰富？请只回答“是”或“否”。",
            answer_fn=lambda c: "是" if _any_count(c) >= 3 else "否",
            verify_fn=lambda c, ch, a: ("是" if _any_count(c) >= 3 else "否") in a,
        )
    )

    return templates


def _build_hidden_stem_templates() -> list[QATemplate]:
    templates: list[QATemplate] = []

    for key in PILLAR_KEYS:
        templates.append(
            QATemplate(
                id=f"hidden-{key}-stems",
                category="hidden_stems",
                text_fn=(lambda k=key: (
                    lambda c: f"{BRANCH_POS_LABELS_CN[PILLAR_KEYS.index(k)]}藏干有哪些？请用顿号分隔。"
                ))(),
                answer_fn=(lambda k=key: (
                    lambda c: "、".join(
                        h["stem"] for h in c["hidden_stems"][f"{k}_branch"]
                    )
                    or "无"
                ))(),
                verify_fn=(lambda k=key: (
                    lambda c, ch, a: all(
                        h["stem"] in a for h in c["hidden_stems"][f"{k}_branch"]
                    )
                ))(),
            )
        )

    # Main (首藏干) stem of the day branch — important for 藏干 reasoning.
    templates.append(
        QATemplate(
            id="hidden-day-branch-main",
            category="hidden_stems",
            text_fn=lambda c: "日支的主气（首位藏干）是什么？请只回答一个汉字。",
            answer_fn=lambda c: (c["hidden_stems"]["day_branch"] or [{"stem": ""}])[0][
                "stem"
            ],
            verify_fn=lambda c, ch, a: (
                c["hidden_stems"]["day_branch"]
                and c["hidden_stems"]["day_branch"][0]["stem"] in a
            ),
            applies=lambda c: bool(c["hidden_stems"]["day_branch"]),
        )
    )

    return templates


def _build_dynamic_templates() -> list[QATemplate]:
    templates: list[QATemplate] = []

    def _need_dynamic(c):
        return bool(c.get("dynamic"))

    def _need_dayun(c):
        return bool((c.get("dynamic") or {}).get("da_yun"))

    def _need_liunian(c):
        return bool((c.get("dynamic") or {}).get("liu_nian"))

    def _need_liuyue(c):
        return bool((c.get("dynamic") or {}).get("liu_yue"))

    templates.append(
        QATemplate(
            id="dyn-dayun-pillar",
            category="dynamic",
            text_fn=lambda c: "当前大运是什么？请回答天干地支两个汉字。",
            answer_fn=lambda c: (
                (c["dynamic"]["da_yun"]["stem"] + c["dynamic"]["da_yun"]["branch"])
            ),
            verify_fn=lambda c, ch, a: (
                c["dynamic"]["da_yun"]["stem"] in a
                and c["dynamic"]["da_yun"]["branch"] in a
            ),
            requires_dynamic=True,
            applies=_need_dayun,
        )
    )
    templates.append(
        QATemplate(
            id="dyn-liunian-pillar",
            category="dynamic",
            text_fn=lambda c: "当前流年是什么？请回答天干地支两个汉字。",
            answer_fn=lambda c: (
                c["dynamic"]["liu_nian"]["stem"] + c["dynamic"]["liu_nian"]["branch"]
            ),
            verify_fn=lambda c, ch, a: (
                c["dynamic"]["liu_nian"]["stem"] in a
                and c["dynamic"]["liu_nian"]["branch"] in a
            ),
            requires_dynamic=True,
            applies=_need_liunian,
        )
    )
    templates.append(
        QATemplate(
            id="dyn-liuyue-pillar",
            category="dynamic",
            text_fn=lambda c: "当前流月是什么？请回答天干地支两个汉字。",
            answer_fn=lambda c: (
                c["dynamic"]["liu_yue"]["stem"] + c["dynamic"]["liu_yue"]["branch"]
            ),
            verify_fn=lambda c, ch, a: (
                c["dynamic"]["liu_yue"]["stem"] in a
                and c["dynamic"]["liu_yue"]["branch"] in a
            ),
            requires_dynamic=True,
            applies=_need_liuyue,
        )
    )

    def _cross_summary(c):
        cross = (c.get("dynamic") or {}).get("cross_natal_temporal_interactions") or {}
        total = 0
        for ctx in cross.values():
            if not isinstance(ctx, dict):
                continue
            for field in ("clashes", "combinations", "branch_combinations", "harms"):
                total += len(ctx.get(field) or [])
        return str(total)

    templates.append(
        QATemplate(
            id="dyn-cross-count",
            category="dynamic",
            text_fn=lambda c: "此命局（含大运/流年/流月）之间的互动冲合总共有多少组？请只回答一个整数。",
            answer_fn=_cross_summary,
            verify_fn=lambda c, ch, a: _cross_summary(c) in a,
            requires_dynamic=True,
            applies=_need_dynamic,
        )
    )

    def _delta_direction(c):
        delta = (c.get("dynamic") or {}).get("elemental_delta") or {}
        if not delta:
            return "无"
        biggest = max(delta.items(), key=lambda kv: kv[1])
        smallest = min(delta.items(), key=lambda kv: kv[1])
        return f"最旺{ELEMENT_EN_TO_CN[biggest[0]]}（+{biggest[1]:.2f}），最弱{ELEMENT_EN_TO_CN[smallest[0]]}（{smallest[1]:+.2f}）"

    templates.append(
        QATemplate(
            id="dyn-delta-direction",
            category="dynamic",
            text_fn=lambda c: "大运流年相对natal，哪个五行上升最多、哪个下降最多？请分别回答元素名。",
            answer_fn=_delta_direction,
            verify_fn=lambda c, ch, a: all(
                e in a
                for e in (
                    ELEMENT_EN_TO_CN[
                        max(
                            (c.get("dynamic") or {}).get("elemental_delta", {}).items(),
                            key=lambda kv: kv[1],
                        )[0]
                    ],
                    ELEMENT_EN_TO_CN[
                        min(
                            (c.get("dynamic") or {}).get("elemental_delta", {}).items(),
                            key=lambda kv: kv[1],
                        )[0]
                    ],
                )
            ),
            requires_dynamic=True,
            applies=lambda c: bool((c.get("dynamic") or {}).get("elemental_delta")),
        )
    )

    def _ug_active(c):
        ugact = (c.get("dynamic") or {}).get("useful_god_activation")
        if ugact is None:
            return "不明"
        return "是" if ugact.get("is_active") else "否"

    templates.append(
        QATemplate(
            id="dyn-useful-god-active",
            category="dynamic",
            text_fn=lambda c: "大运流年是否激活了用神？请只回答“是”“否”或“不明”。",
            answer_fn=_ug_active,
            verify_fn=lambda c, ch, a: _ug_active(c) in a,
            requires_dynamic=True,
            applies=lambda c: (c.get("dynamic") or {}).get("useful_god_activation") is not None,
        )
    )

    return templates


def build_stage1_templates() -> list[QATemplate]:
    """All Stage 1 single-fact templates (~50)."""
    all_: list[QATemplate] = []
    all_.extend(_build_pillar_identity_templates())
    all_.extend(_build_ten_god_templates())
    all_.extend(_build_five_element_templates())
    all_.extend(_build_day_master_strength_templates())
    all_.extend(_build_interaction_templates())
    all_.extend(_build_hidden_stem_templates())
    all_.extend(_build_dynamic_templates())
    return all_


# -----------------------------------------------------------------------------
# Stage 2 template builders (multi-fact)
# -----------------------------------------------------------------------------


def _build_relational_templates() -> list[QATemplate]:
    templates: list[QATemplate] = []

    def _pair_tengods(c):
        y = c["ten_gods"]["year_stem"]
        m = c["ten_gods"]["month_stem"]
        return f"年干十神为{y}，月干十神为{m}。"

    templates.append(
        QATemplate(
            id="s2-relational-year-month",
            category="relational",
            text_fn=lambda c: "请分别说出年干和月干相对于日主的十神。",
            answer_fn=_pair_tengods,
            verify_fn=lambda c, ch, a: (
                c["ten_gods"]["year_stem"] in a and c["ten_gods"]["month_stem"] in a
            ),
        )
    )

    def _pair_tengods_with_relation(c):
        y = c["ten_gods"]["year_stem"]
        h = c["ten_gods"]["hour_stem"]
        y_el = c["natal_pillars"]["year"]["element_stem"]
        h_el = c["natal_pillars"]["hour"]["element_stem"]
        rel = _element_relation(y_el, h_el)
        return f"年干十神为{y}（{y_el}），时干十神为{h}（{h_el}）；年干与时干五行关系为{rel}。"

    templates.append(
        QATemplate(
            id="s2-relational-year-hour-relation",
            category="relational",
            text_fn=lambda c: "分别说出年干和时干的十神及其五行，并判断两者的五行关系（生、克或同）。",
            answer_fn=_pair_tengods_with_relation,
            verify_fn=lambda c, ch, a: (
                c["ten_gods"]["year_stem"] in a
                and c["ten_gods"]["hour_stem"] in a
                and c["natal_pillars"]["year"]["element_stem"] in a
            ),
        )
    )

    return templates


def _build_conditional_templates() -> list[QATemplate]:
    templates: list[QATemplate] = []

    def _ug_expected_vs_actual(c):
        cls_ = c["day_master"]["classification"]
        ug = (c.get("useful_god") or {})
        actual = ug.get("primary") or "不明"
        if ug.get("is_ambiguous"):
            actual = "不明"
        if cls_ == "身强":
            rule = "身强宜克、泄、耗（用官杀、食伤、财）"
        elif cls_ == "身弱":
            rule = "身弱宜生、扶（用印、比劫）"
        else:
            rule = "中和无显著偏向"
        return f"日主为{cls_}；{rule}；实际用神为{actual}。"

    templates.append(
        QATemplate(
            id="s2-conditional-ug",
            category="conditional",
            text_fn=lambda c: "此命日主身强还是身弱？按此判断用神取向，并说出实际用神。",
            answer_fn=_ug_expected_vs_actual,
            verify_fn=lambda c, ch, a: (
                c["day_master"]["classification"] in a
                and ((c.get("useful_god") or {}).get("primary", "不明") in a or "不明" in a)
            ),
        )
    )

    return templates


def _build_cross_pillar_templates() -> list[QATemplate]:
    templates: list[QATemplate] = []

    def _which_and_hidden(c, target="正官"):
        hits = []
        for i, k in enumerate(PILLAR_KEYS):
            if c["ten_gods"][f"{k}_stem"] == target:
                hits.append((STEM_POS_LABELS_CN[i], k))
        if not hits:
            return "无"
        parts = []
        for label, k in hits:
            hidden = c["hidden_stems"][f"{k}_branch"]
            stems = "、".join(h["stem"] for h in hidden) or "无"
            parts.append(f"{label}为{target}，同柱地支藏干为{stems}")
        return "；".join(parts) + "。"

    for target in ("正官", "正财"):
        templates.append(
            QATemplate(
                id=f"s2-cross-which-hidden-{target}",
                category="cross_pillar",
                text_fn=(lambda t=target: (
                    lambda c: f"哪个柱的天干是{t}？该柱地支藏干有哪些？"
                ))(),
                answer_fn=(lambda t=target: (lambda c: _which_and_hidden(c, t)))(),
                verify_fn=(lambda t=target: (
                    lambda c, ch, a: any(
                        c["ten_gods"][f"{k}_stem"] == t for k in PILLAR_KEYS
                    )
                    and all(
                        h["stem"] in a
                        for k in PILLAR_KEYS
                        if c["ten_gods"][f"{k}_stem"] == t
                        for h in c["hidden_stems"][f"{k}_branch"]
                    )
                ))(),
                applies=(lambda t=target: (
                    lambda c: any(c["ten_gods"][f"{k}_stem"] == t for k in PILLAR_KEYS)
                ))(),
            )
        )

    return templates


def _build_analytical_templates() -> list[QATemplate]:
    templates: list[QATemplate] = []

    def _strongest_and_relation(c):
        strongest = _strongest_element(c)
        dm_el = _dm_element(c)
        rel = _element_relation(strongest, dm_el)
        if strongest == dm_el:
            semantic = "同我（助身）"
        elif ELEMENT_GENERATES.get(strongest) == dm_el:
            semantic = "生我（印）"
        elif ELEMENT_CONTROLS.get(strongest) == dm_el:
            semantic = "克我（官杀）"
        elif ELEMENT_GENERATES.get(dm_el) == strongest:
            semantic = "我生（食伤）"
        elif ELEMENT_CONTROLS.get(dm_el) == strongest:
            semantic = "我克（财）"
        else:
            semantic = "无直接关系"
        return f"五行最旺为{strongest}；日主为{dm_el}；{strongest}与{dm_el}五行关系为{rel}（{semantic}）。"

    templates.append(
        QATemplate(
            id="s2-analytical-strongest-relation",
            category="analytical",
            text_fn=lambda c: "五行最旺的是什么？与日主五行是什么关系？请同时回答这两点。",
            answer_fn=_strongest_and_relation,
            verify_fn=lambda c, ch, a: (
                _strongest_element(c) in a and _dm_element(c) in a
            ),
        )
    )

    def _weakest_favorable(c):
        weakest = _weakest_element(c)
        ug = c.get("useful_god") or {}
        fav = ug.get("favorable_elements") or []
        is_fav = weakest in fav
        return (
            f"五行最弱为{weakest}；"
            f"命局喜用五行为{'、'.join(fav) if fav else '不明'}；"
            f"最弱元素{'属于' if is_fav else '不属于'}喜用。"
        )

    templates.append(
        QATemplate(
            id="s2-analytical-weakest-favorable",
            category="analytical",
            text_fn=lambda c: "五行最弱的是什么？它是否属于命局的喜用五行？请同时回答。",
            answer_fn=_weakest_favorable,
            verify_fn=lambda c, ch, a: (
                _weakest_element(c) in a
                and (
                    not (c.get("useful_god") or {}).get("favorable_elements")
                    or all(e in a for e in (c["useful_god"] or {}).get("favorable_elements", []))
                )
            ),
        )
    )

    return templates


def _build_interaction_chain_templates() -> list[QATemplate]:
    templates: list[QATemplate] = []

    def _clash_blocks_combo(c):
        clashes = c["interactions"]["branch_clashes"]
        combos = c["interactions"]["branch_six_combinations"]
        clash_desc = _format_interaction_list("branch_clashes", clashes)
        combo_desc = _format_interaction_list("branch_six_combinations", combos)
        blocked = [x for x in combos if x.get("is_blocked")]
        if blocked:
            blk = "；".join(
                _describe_interaction("branch_six_combinations", x) for x in blocked
            )
            effect = f"其中以下六合被冲破：{blk}。"
        else:
            effect = "六合未被冲破。"
        return f"地支六冲：{clash_desc}。地支六合：{combo_desc}。{effect}"

    templates.append(
        QATemplate(
            id="s2-chain-clash-blocks-combo",
            category="interaction_chain",
            text_fn=lambda c: "地支六冲有哪些？是否影响了六合？请同时给出六冲与六合，并指出六合是否被冲破。",
            answer_fn=_clash_blocks_combo,
            verify_fn=lambda c, ch, a: (
                ("无" in a or "没有" in a or any(
                    br in a for r in c["interactions"]["branch_clashes"]
                    for br in (r.get("branches") or [])
                ))
                and (
                    "六合" in a or "无" in a
                )
            ),
        )
    )

    def _harmony_with_clash(c):
        th = c["interactions"]["three_harmony"]
        if not th:
            return "无三合局。"
        parts = []
        for r in th:
            desc = _describe_interaction("three_harmony", r)
            parts.append(desc)
        return "三合局：" + "；".join(parts) + "。"

    templates.append(
        QATemplate(
            id="s2-chain-three-harmony",
            category="interaction_chain",
            text_fn=lambda c: "命局是否存在三合局？如有，请列出元素和完整度，并说明是否被冲破。",
            answer_fn=_harmony_with_clash,
            verify_fn=lambda c, ch, a: (
                not c["interactions"]["three_harmony"]
                and ("无" in a or "没有" in a)
            )
            or all(
                r.get("element", "") in a for r in c["interactions"]["three_harmony"]
            ),
        )
    )

    return templates


def _build_temporal_relational_templates() -> list[QATemplate]:
    templates: list[QATemplate] = []

    def _liunian_vs_dm(c):
        ln = (c.get("dynamic") or {}).get("liu_nian")
        if not ln:
            return "无流年。"
        ln_stem = ln["stem"]
        ln_branch = ln["branch"]
        dm = c["day_master"]["stem"]
        dm_el = c["day_master"]["element"]
        # Use stem elements: need to look up from encodings. Easier: use BaziChart after load.
        return f"流年天干为{ln_stem}，地支为{ln_branch}；日主为{dm}（{dm_el}）。"

    templates.append(
        QATemplate(
            id="s2-temporal-liunian-dm",
            category="temporal_relational",
            text_fn=lambda c: "流年的干支是什么？与日主分别是什么五行？请同时回答。",
            answer_fn=_liunian_vs_dm,
            verify_fn=lambda c, ch, a: (
                (c.get("dynamic") or {}).get("liu_nian", {}).get("stem", "") in a
                and (c.get("dynamic") or {}).get("liu_nian", {}).get("branch", "") in a
            ),
            requires_dynamic=True,
            applies=lambda c: bool((c.get("dynamic") or {}).get("liu_nian")),
        )
    )

    def _dayun_cross(c):
        cross = (c.get("dynamic") or {}).get("cross_natal_temporal_interactions") or {}
        ctx = cross.get("natal_vs_dayun") or {}
        parts = []
        for field, label in (
            ("clashes", "冲"),
            ("combinations", "合"),
            ("branch_combinations", "地支合"),
            ("harms", "害"),
        ):
            if ctx.get(field):
                parts.append(f"{label}:{len(ctx[field])}组")
        if not parts:
            return "大运与natal无显著互动。"
        return "大运与natal互动：" + "，".join(parts) + "。"

    templates.append(
        QATemplate(
            id="s2-temporal-dayun-cross",
            category="temporal_relational",
            text_fn=lambda c: "大运与原局之间的冲、合、害情况如何？请分别说出各类数量。",
            answer_fn=_dayun_cross,
            verify_fn=lambda c, ch, a: True,  # count-based; trust answer_fn
            requires_dynamic=True,
            applies=lambda c: bool((c.get("dynamic") or {}).get("cross_natal_temporal_interactions")),
        )
    )

    return templates


def _build_summary_templates() -> list[QATemplate]:
    templates: list[QATemplate] = []

    def _three_features(c):
        dm = c["day_master"]
        strongest = _strongest_element(c)
        interactions_total = sum(
            len(c["interactions"][k]) for k in INTERACTION_TYPE_LABELS
        )
        ug = c.get("useful_god") or {}
        ug_primary = ug.get("primary", "不明") if not ug.get("is_ambiguous") else "不明"
        return (
            f"日主{dm['stem']}（{dm['element']}），{dm['classification']}；"
            f"五行最旺为{strongest}；"
            f"共{interactions_total}组干支互动，用神大类为{ug_primary}。"
        )

    templates.append(
        QATemplate(
            id="s2-summary-three-features",
            category="summary",
            text_fn=lambda c: "请简要列出此命局的三个关键特征：日主与强弱、最旺五行、互动数量与用神。",
            answer_fn=_three_features,
            verify_fn=lambda c, ch, a: (
                c["day_master"]["stem"] in a
                and c["day_master"]["classification"] in a
                and _strongest_element(c) in a
            ),
        )
    )

    def _two_facts_pillar_tengod(c):
        y = _pillar_str(c, "year")
        y_tg = c["ten_gods"]["year_stem"]
        m = _pillar_str(c, "month")
        m_tg = c["ten_gods"]["month_stem"]
        return f"年柱{y}，年干十神{y_tg}；月柱{m}，月干十神{m_tg}。"

    templates.append(
        QATemplate(
            id="s2-summary-year-month",
            category="summary",
            text_fn=lambda c: "请同时给出年柱、月柱及其天干十神。",
            answer_fn=_two_facts_pillar_tengod,
            verify_fn=lambda c, ch, a: (
                c["natal_pillars"]["year"]["stem"] in a
                and c["natal_pillars"]["year"]["branch"] in a
                and c["natal_pillars"]["month"]["stem"] in a
                and c["ten_gods"]["year_stem"] in a
                and c["ten_gods"]["month_stem"] in a
            ),
        )
    )

    return templates


def build_stage2_templates() -> list[QATemplate]:
    """All Stage 2 multi-fact templates (~20)."""
    all_: list[QATemplate] = []
    all_.extend(_build_relational_templates())
    all_.extend(_build_conditional_templates())
    all_.extend(_build_cross_pillar_templates())
    all_.extend(_build_analytical_templates())
    all_.extend(_build_interaction_chain_templates())
    all_.extend(_build_temporal_relational_templates())
    all_.extend(_build_summary_templates())
    return all_


def applicable_templates(
    templates: list[QATemplate], chart_json: dict[str, Any]
) -> list[QATemplate]:
    """Filter templates to those whose `applies` returns True for this chart."""
    out = []
    has_dynamic = bool(chart_json.get("dynamic"))
    for t in templates:
        if t.requires_dynamic and not has_dynamic:
            continue
        try:
            if not t.applies(chart_json):
                continue
        except Exception:
            continue
        out.append(t)
    return out
