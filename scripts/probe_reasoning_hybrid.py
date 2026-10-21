#!/usr/bin/env python3
"""
Reasoning probe for the hybrid-prompt projection (plan §6, §7, §8).

Unlike ``probe_factual_64token.py`` (single-fact recall), each question here
requires combining ≥2 facts from ``BaziChart.to_prompt_text()`` — the kind of
synthesis the ship gate actually cares about. Output is generations + chart
context dumped to JSON and a side-by-side Markdown rubric sheet, ready for
manual scoring against plan §8's 4-axis rubric (structural accuracy /
reasoning coherence / interpretive depth / Chinese fluency, 1-5 each).

Usage (from repo root, Phase 2 env):

    # Hybrid full
    python scripts/probe_reasoning_hybrid.py \\
        --hybrid_chart_text \\
        --projection_checkpoint checkpoints/phase3_prep/qa_hybrid_v1/checkpoint_best.npz \\
        --output_json reports/probe_reasoning_hybrid_step800_full.json

    # §7 ablation: same prompt, virtual tokens zeroed
    python scripts/probe_reasoning_hybrid.py \\
        --hybrid_chart_text --zero_virtual_tokens \\
        --projection_checkpoint checkpoints/phase3_prep/qa_hybrid_v1/checkpoint_best.npz \\
        --output_json reports/probe_reasoning_hybrid_step800_textonly.json

The Markdown sheet is auto-written next to ``--output_json`` with the same
stem and ``.md`` suffix.

Question design notes
---------------------
- Categories cover the five reasoning patterns we expect Qwen to handle on
  top of the chart text block:
  A. Ten-god derivation (stem + DM stem → ten-god rule)
  B. Useful-god rationale (DM strength + element imbalance → 用神)
  C. Interaction impact (interaction parties + DM context → effect)
  D. Day-master strength reasoning (得令/得地/得势 + supporting/draining factors)
  E. Composite character / fortune inference (multi-fact → free-form synthesis)
- Each ``ReasoningQuestion`` exposes ``context_fn(chart) -> dict`` listing the
  ground-truth facts the answer should reference. The Markdown sheet shows
  these alongside the prediction so a human scorer can grade in one pass
  without re-deriving the facts themselves.
- This is the v1 question set; refinement is expected after first scoring
  pass (plan §6: "Reasoning-probe authoring is the highest-skill subtask;
  partner with project owner for question design.").
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Reuse loader and generator from the categorical probe so behavior stays in lockstep.
from scripts.probe_factual_64token import (  # noqa: E402
    generate_from_chart,
    load_bazi_llm,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reasoning probe for the hybrid-prompt projection (§6/§7).",
    )
    p.add_argument(
        "--test_data",
        type=Path,
        default=_root / "data" / "phase2_test.jsonl",
    )
    p.add_argument("--n_virtual_tokens", type=int, default=63)
    p.add_argument("--max_length", type=int, default=960)
    p.add_argument(
        "--llm_path",
        type=Path,
        default=_root / "models" / "qwen2.5-7b-instruct-mlx-4bit",
    )
    p.add_argument("--natal_mode", choices=("encoder", "embedding"), default="encoder")
    p.add_argument(
        "--natal_weights",
        type=Path,
        default=_root / "checkpoints" / "natal_encoder_mlx.npz",
    )
    p.add_argument(
        "--dynamic_weights",
        type=Path,
        default=_root / "checkpoints" / "dynamic_encoder_mlx.npz",
    )
    p.add_argument(
        "--projection_checkpoint",
        type=Path,
        required=True,
        help="Hybrid-trained projection checkpoint (.npz).",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=400,
        help="Reasoning answers need room; default 400 covers a paragraph.",
    )
    p.add_argument(
        "--num_charts",
        type=int,
        default=5,
        help="Number of distinct charts to probe (each runs the full question set).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--hybrid_chart_text",
        action="store_true",
        help="Match training format: prepend chart.to_prompt_text() to each question.",
    )
    p.add_argument(
        "--zero_virtual_tokens",
        action="store_true",
        help="§7 ablation: zero out the projection's virtual tokens at inference.",
    )
    p.add_argument(
        "--output_json",
        type=Path,
        default=_root / "reports" / "probe_reasoning_hybrid_results.json",
    )
    return p.parse_args()


@dataclass
class ReasoningQuestion:
    id: str
    category: str
    text_fn: Callable[[Any], str]
    context_fn: Callable[[Any], Dict[str, Any]]
    skip_if: Callable[[Any], bool] = field(default=lambda c: False)
    rationale: str = ""


def _build_questions() -> List[ReasoningQuestion]:
    """Draft v1 reasoning question set (20 questions across 5 categories).

    Each question is parametric over a chart so ground-truth facts can be
    auto-surfaced into the rubric sheet.
    """
    from bazi_engine.models import (
        compute_day_master_strength,
        compute_element_strengths,
        classify_day_master,
        _compute_useful_god,
    )
    from bazi_engine.interactions import detect_all_interactions

    PILLAR_LABELS = ("年", "月", "日", "时")

    def _pillars(chart):
        return (chart.year_pillar, chart.month_pillar, chart.day_pillar, chart.hour_pillar)

    def _ten_gods(chart):
        out = {}
        for lbl, p in zip(PILLAR_LABELS, _pillars(chart)):
            tg = p.ten_god_stem.value if p.ten_god_stem else "—"
            out[lbl] = tg
        return out

    def _strengths(chart):
        stems, branches = chart._stems_and_branches()
        return compute_element_strengths(stems, branches)

    def _classification(chart):
        stems, branches = chart._stems_and_branches()
        return classify_day_master(stems, branches)

    def _useful_god(chart):
        stems, branches = chart._stems_and_branches()
        pct = compute_day_master_strength(stems, branches)
        ug = _compute_useful_god(pct, chart.day_pillar.stem.element)
        return ug or {}

    def _interactions(chart):
        stems, branches = chart._stems_and_branches()
        return detect_all_interactions(stems, branches)

    def _strongest(chart):
        s = _strengths(chart)
        return max(s.items(), key=lambda kv: kv[1])

    def _weakest(chart):
        s = _strengths(chart)
        return min(s.items(), key=lambda kv: kv[1])

    questions: List[ReasoningQuestion] = []

    # ---------- A. Ten-god derivation (4) -----------------------------------
    def a1_text(c):
        return (
            f"日主天干是{c.day_pillar.stem.value}（五行属{c.day_pillar.stem.element}）。"
            f"年干是{c.year_pillar.stem.value}（五行属{c.year_pillar.stem.element}）。"
            "请说明年干对日主的十神关系是什么，并给出推导过程（结合五行生克和阴阳）。"
        )

    def a1_ctx(c):
        tgs = _ten_gods(c)
        return {
            "day_stem": c.day_pillar.stem.value,
            "day_element": c.day_pillar.stem.element,
            "year_stem": c.year_pillar.stem.value,
            "year_element": c.year_pillar.stem.element,
            "expected_ten_god": tgs["年"],
        }

    questions.append(ReasoningQuestion(
        id="A1-tengod-derivation-year",
        category="ten_god_derivation",
        text_fn=a1_text,
        context_fn=a1_ctx,
        rationale="Ten-god rule from DM stem + year stem (≥2 facts).",
    ))

    def a2_text(c):
        return (
            "请按重要性排序四柱天干对日主的十神关系，简述每柱十神的命理含义"
            "（不超过150字）。"
        )

    def a2_ctx(c):
        tgs = _ten_gods(c)
        return {"ten_gods_per_pillar": tgs, "day_stem": c.day_pillar.stem.value}

    questions.append(ReasoningQuestion(
        id="A2-tengod-rank-meaning",
        category="ten_god_derivation",
        text_fn=a2_text,
        context_fn=a2_ctx,
        rationale="Combines all 4 stem identities + DM into ranked synthesis.",
    ))

    def a3_text(c):
        return (
            "在四柱天干中，哪一个对日主的克泄作用最强？请说明依据"
            "（指出该天干、其十神类型、以及与日主的五行关系）。"
        )

    def a3_ctx(c):
        tgs = _ten_gods(c)
        depleting_tg = {"七杀", "正官", "正财", "偏财", "伤官", "食神"}
        depleting_pillars = [
            (lbl, getattr(c, f"{name}_pillar").stem.value, tgs[lbl])
            for lbl, name in zip(PILLAR_LABELS, ("year", "month", "day", "hour"))
            if tgs[lbl] in depleting_tg
        ]
        return {
            "day_stem": c.day_pillar.stem.value,
            "day_element": c.day_pillar.stem.element,
            "ten_gods_per_pillar": tgs,
            "depleting_pillars": depleting_pillars,
        }

    questions.append(ReasoningQuestion(
        id="A3-strongest-depleting-stem",
        category="ten_god_derivation",
        text_fn=a3_text,
        context_fn=a3_ctx,
        rationale="Requires comparing all stems against DM (≥2 facts + ranking).",
    ))

    def a4_text(c):
        tgs = _ten_gods(c)
        target_tg = tgs["时"]
        return (
            f"时干的十神是{target_tg}。这个十神出现在时柱代表什么含义？"
            "请结合时柱所代表的人生领域作答（不超过150字）。"
        )

    def a4_ctx(c):
        tgs = _ten_gods(c)
        return {
            "hour_stem": c.hour_pillar.stem.value,
            "hour_ten_god": tgs["时"],
            "day_stem": c.day_pillar.stem.value,
        }

    questions.append(ReasoningQuestion(
        id="A4-hour-pillar-tengod-meaning",
        category="ten_god_derivation",
        text_fn=a4_text,
        context_fn=a4_ctx,
        rationale="Ten-god + pillar-domain knowledge (子女/晚年).",
    ))

    # ---------- B. Useful-god rationale (4) ---------------------------------
    def b1_text(c):
        return (
            "请综合日主强弱、五行分布和主要交互，说明这个八字的用神是什么，"
            "并给出推导依据（不超过200字）。"
        )

    def b1_ctx(c):
        ug = _useful_god(c)
        return {
            "day_master_classification": _classification(c),
            "element_strengths": {k: round(v, 3) for k, v in _strengths(c).items()},
            "useful_god": ug.get("primary", "不明"),
            "useful_god_element": ug.get("primary_element", "不明"),
        }

    questions.append(ReasoningQuestion(
        id="B1-useful-god-rationale",
        category="useful_god",
        text_fn=b1_text,
        context_fn=b1_ctx,
        rationale="Plan §6 example: '为什么用神是土？'",
    ))

    def b2_text(c):
        return (
            "这个八字最忌讳哪个五行？为什么？"
            "请结合日主强弱和已经偏旺的五行说明（不超过150字）。"
        )

    def b2_ctx(c):
        strongest_el, strongest_v = _strongest(c)
        return {
            "day_master_classification": _classification(c),
            "strongest_element": strongest_el,
            "strongest_pct": round(strongest_v, 3),
            "useful_god_element": _useful_god(c).get("primary_element", "不明"),
        }

    questions.append(ReasoningQuestion(
        id="B2-most-harmful-element",
        category="useful_god",
        text_fn=b2_text,
        context_fn=b2_ctx,
        rationale="Combines DM strength + most-strong element.",
    ))

    def b3_text(c):
        ug = _useful_god(c)
        ug_el = ug.get("primary_element", "土")
        return (
            f"如果这个人走{ug_el}运，会对命局产生什么影响？"
            "请说明哪些方面会被加强、哪些会被削弱（不超过150字）。"
        )

    def b3_ctx(c):
        return {
            "useful_god_element": _useful_god(c).get("primary_element", "不明"),
            "useful_god_label": _useful_god(c).get("primary", "不明"),
            "day_master_classification": _classification(c),
        }

    questions.append(ReasoningQuestion(
        id="B3-luck-cycle-impact",
        category="useful_god",
        text_fn=b3_text,
        context_fn=b3_ctx,
        rationale="Useful god + DM strength → counterfactual luck cycle.",
    ))

    def b4_text(c):
        return (
            "命局中的用神是否得到生扶？是否被克泄？"
            "请指出生扶或克泄用神的具体干支（不超过150字）。"
        )

    def b4_ctx(c):
        return {
            "useful_god_element": _useful_god(c).get("primary_element", "不明"),
            "element_strengths": {k: round(v, 3) for k, v in _strengths(c).items()},
            "ten_gods_per_pillar": _ten_gods(c),
        }

    questions.append(ReasoningQuestion(
        id="B4-useful-god-purity",
        category="useful_god",
        text_fn=b4_text,
        context_fn=b4_ctx,
        rationale="Requires identifying useful god + matching element pillars.",
    ))

    # ---------- C. Interaction impact (4) -----------------------------------
    def _has_branch_clash(c):
        return bool(_interactions(c).branch_clashes)

    def _has_stem_combo(c):
        return bool(_interactions(c).stem_combinations)

    def c1_text(c):
        ir = _interactions(c)
        if ir.branch_clashes:
            r = ir.branch_clashes[0]
            a, b = r["pos_a"], r["pos_b"]
            pa, pb = PILLAR_LABELS[a], PILLAR_LABELS[b]
            ba = _pillars(c)[a].branch.value
            bb = _pillars(c)[b].branch.value
            return (
                f"{pa}支（{ba}）和{pb}支（{bb}）形成六冲。"
                "请说明这个冲对日主和命局整体的影响（不超过200字）。"
            )
        return ""  # never used; skip_if filters

    def c1_ctx(c):
        ir = _interactions(c)
        r = ir.branch_clashes[0]
        a, b = r["pos_a"], r["pos_b"]
        return {
            "clash_pillars": (PILLAR_LABELS[a], PILLAR_LABELS[b]),
            "clash_branches": (_pillars(c)[a].branch.value, _pillars(c)[b].branch.value),
            "clash_elements": (
                _pillars(c)[a].branch.element,
                _pillars(c)[b].branch.element,
            ),
            "day_master_classification": _classification(c),
            "day_element": c.day_pillar.stem.element,
        }

    questions.append(ReasoningQuestion(
        id="C1-branch-clash-impact",
        category="interaction",
        text_fn=c1_text,
        context_fn=c1_ctx,
        skip_if=lambda c: not _has_branch_clash(c),
        rationale="Plan §6 example pattern: interaction × DM strength.",
    ))

    def c2_text(c):
        ir = _interactions(c)
        if ir.stem_combinations:
            r = ir.stem_combinations[0]
            a, b = r["pos_a"], r["pos_b"]
            pa, pb = PILLAR_LABELS[a], PILLAR_LABELS[b]
            sa = _pillars(c)[a].stem.value
            sb = _pillars(c)[b].stem.value
            return (
                f"{pa}干（{sa}）和{pb}干（{sb}）相合。"
                "请说明这个天干合对命局五行力量的影响（不超过200字）。"
            )
        return ""

    def c2_ctx(c):
        ir = _interactions(c)
        r = ir.stem_combinations[0]
        a, b = r["pos_a"], r["pos_b"]
        return {
            "combo_pillars": (PILLAR_LABELS[a], PILLAR_LABELS[b]),
            "combo_stems": (_pillars(c)[a].stem.value, _pillars(c)[b].stem.value),
            "combo_transform_to": r.get("transformed_element", r.get("element", "—")),
            "element_strengths": {k: round(v, 3) for k, v in _strengths(c).items()},
        }

    questions.append(ReasoningQuestion(
        id="C2-stem-combo-impact",
        category="interaction",
        text_fn=c2_text,
        context_fn=c2_ctx,
        skip_if=lambda c: not _has_stem_combo(c),
        rationale="Stem-combo + element-balance synthesis.",
    ))

    def c3_text(c):
        return (
            "请总体评价这个八字地支的冲合状况：偏冲突还是偏和谐？"
            "为什么？请引用具体的交互（不超过200字）。"
        )

    def c3_ctx(c):
        ir = _interactions(c)
        return {
            "branch_clashes": [tuple(sorted([r["pos_a"], r["pos_b"]])) for r in ir.branch_clashes],
            "stem_combinations": [tuple(sorted([r["pos_a"], r["pos_b"]])) for r in ir.stem_combinations],
            "branch_six_combos": [tuple(sorted([r["pos_a"], r["pos_b"]])) for r in (ir.branch_six_combinations or [])],
            "harms": [tuple(sorted([r["pos_a"], r["pos_b"]])) for r in (ir.harms or [])],
        }

    questions.append(ReasoningQuestion(
        id="C3-overall-harmony-conflict",
        category="interaction",
        text_fn=c3_text,
        context_fn=c3_ctx,
        rationale="Requires aggregating multiple interactions into overall judgment.",
    ))

    def c4_text(c):
        return (
            "如果这个八字没有目前的主要交互（冲、合、刑、害），"
            "日主的强弱判断会有什么变化？请说明你的推理（不超过200字）。"
        )

    def c4_ctx(c):
        ir = _interactions(c)
        return {
            "day_master_classification": _classification(c),
            "interactions_present": {
                "branch_clashes": len(ir.branch_clashes),
                "stem_combinations": len(ir.stem_combinations),
                "branch_six_combinations": len(ir.branch_six_combinations or []),
                "harms": len(ir.harms or []),
            },
            "element_strengths": {k: round(v, 3) for k, v in _strengths(c).items()},
        }

    questions.append(ReasoningQuestion(
        id="C4-counterfactual-no-interactions",
        category="interaction",
        text_fn=c4_text,
        context_fn=c4_ctx,
        rationale="Counterfactual reasoning: requires understanding interaction → strength mapping.",
    ))

    # ---------- D. Day-master strength reasoning (4) ------------------------
    def d1_text(c):
        return (
            "请按照得令、得地、得势三方面，分析日主的强弱"
            "（每一方面用1-2句概括即可，总长不超过200字）。"
        )

    def d1_ctx(c):
        return {
            "day_stem": c.day_pillar.stem.value,
            "day_element": c.day_pillar.stem.element,
            "month_branch": c.month_pillar.branch.value,
            "month_element": c.month_pillar.branch.element,
            "day_master_classification": _classification(c),
            "element_strengths": {k: round(v, 3) for k, v in _strengths(c).items()},
        }

    questions.append(ReasoningQuestion(
        id="D1-strength-three-axes",
        category="day_master_strength",
        text_fn=d1_text,
        context_fn=d1_ctx,
        rationale="Classical 三元 analysis: needs DM stem + month branch + ≥2 supporting facts.",
    ))

    def d2_text(c):
        return (
            f"月支是{c.month_pillar.branch.value}（五行属{c.month_pillar.branch.element}）。"
            "月令对日主是助还是泄？请说明月支本气藏干和日主的关系（不超过150字）。"
        )

    def d2_ctx(c):
        return {
            "day_stem": c.day_pillar.stem.value,
            "day_element": c.day_pillar.stem.element,
            "month_branch": c.month_pillar.branch.value,
            "month_element": c.month_pillar.branch.element,
        }

    questions.append(ReasoningQuestion(
        id="D2-month-pillar-effect",
        category="day_master_strength",
        text_fn=d2_text,
        context_fn=d2_ctx,
        rationale="Month branch element vs DM element relation.",
    ))

    def d3_text(c):
        return (
            "请列出至少两个对日主有帮扶作用的干支，并说明其作用机制；"
            "再列出至少一个克泄日主的干支（不超过200字）。"
        )

    def d3_ctx(c):
        return {
            "day_stem": c.day_pillar.stem.value,
            "day_element": c.day_pillar.stem.element,
            "ten_gods_per_pillar": _ten_gods(c),
            "all_pillars": [
                {
                    "label": lbl,
                    "stem": p.stem.value,
                    "stem_element": p.stem.element,
                    "branch": p.branch.value,
                    "branch_element": p.branch.element,
                }
                for lbl, p in zip(PILLAR_LABELS, _pillars(c))
            ],
        }

    questions.append(ReasoningQuestion(
        id="D3-supporting-vs-draining",
        category="day_master_strength",
        text_fn=d3_text,
        context_fn=d3_ctx,
        rationale="Requires scanning all 8 chars + classifying each vs DM.",
    ))

    def d4_text(c):
        return (
            "如果这个日主在五行最弱的那个季节出生，命局还能保持目前的强弱判断吗？"
            "请说明你的推理（不超过150字）。"
        )

    def d4_ctx(c):
        we, wv = _weakest(c)
        return {
            "day_master_classification": _classification(c),
            "day_element": c.day_pillar.stem.element,
            "weakest_element": we,
            "weakest_pct": round(wv, 3),
        }

    questions.append(ReasoningQuestion(
        id="D4-counterfactual-season",
        category="day_master_strength",
        text_fn=d4_text,
        context_fn=d4_ctx,
        rationale="Counterfactual: DM element + season change.",
    ))

    # ---------- E. Composite character / fortune inference (4) --------------
    def e1_text(c):
        return (
            "综合日主类型、用神、和主要交互，请描述这个人的性格倾向"
            "（约200字，重点指出至少两条与命局对应的特质）。"
        )

    def e1_ctx(c):
        return {
            "day_master_classification": _classification(c),
            "day_stem": c.day_pillar.stem.value,
            "day_element": c.day_pillar.stem.element,
            "useful_god": _useful_god(c).get("primary", "不明"),
            "ten_gods_per_pillar": _ten_gods(c),
        }

    questions.append(ReasoningQuestion(
        id="E1-personality-synthesis",
        category="composite",
        text_fn=e1_text,
        context_fn=e1_ctx,
        rationale="Multi-fact → personality. Tests interpretive depth.",
    ))

    def e2_text(c):
        return (
            "从财星（正财、偏财）的位置和强弱来看，这个人的财富指标如何？"
            "请引用具体的柱位作答（不超过200字）。"
        )

    def e2_ctx(c):
        tgs = _ten_gods(c)
        wealth_pillars = [(lbl, tg) for lbl, tg in tgs.items() if tg in ("正财", "偏财")]
        return {
            "day_master_classification": _classification(c),
            "ten_gods_per_pillar": tgs,
            "wealth_pillars": wealth_pillars,
            "element_strengths": {k: round(v, 3) for k, v in _strengths(c).items()},
        }

    questions.append(ReasoningQuestion(
        id="E2-wealth-indicator",
        category="composite",
        text_fn=e2_text,
        context_fn=e2_ctx,
        rationale="Wealth-star location + DM strength → wealth interpretation.",
    ))

    def e3_text(c):
        return (
            "从官杀星（正官、七杀）的位置和与用神的关系来看，"
            "这个人的事业方向倾向于什么？为什么？（不超过200字）"
        )

    def e3_ctx(c):
        tgs = _ten_gods(c)
        officer_pillars = [(lbl, tg) for lbl, tg in tgs.items() if tg in ("正官", "七杀")]
        return {
            "day_master_classification": _classification(c),
            "ten_gods_per_pillar": tgs,
            "officer_pillars": officer_pillars,
            "useful_god": _useful_god(c).get("primary", "不明"),
        }

    questions.append(ReasoningQuestion(
        id="E3-career-from-officer",
        category="composite",
        text_fn=e3_text,
        context_fn=e3_ctx,
        rationale="Officer-star position + useful-god relation → career.",
    ))

    def e4_text(c):
        return (
            "请指出这个八字最大的优势和最大的隐患，各用一两句话说明依据，"
            "总长不超过200字。"
        )

    def e4_ctx(c):
        ir = _interactions(c)
        return {
            "day_master_classification": _classification(c),
            "useful_god": _useful_god(c).get("primary", "不明"),
            "strongest_element": _strongest(c)[0],
            "weakest_element": _weakest(c)[0],
            "interactions_brief": {
                "branch_clashes": len(ir.branch_clashes),
                "stem_combinations": len(ir.stem_combinations),
            },
        }

    questions.append(ReasoningQuestion(
        id="E4-strength-vulnerability",
        category="composite",
        text_fn=e4_text,
        context_fn=e4_ctx,
        rationale="Forces explicit identification of an asset and a liability.",
    ))

    return questions


def _to_mx_input(item):
    """Reshape Phase2ChartDataset item into (natal_batch, dynamic_batch, cross_batch)."""
    natal_np = item["natal_inputs"]
    dynamic_np = item["dynamic_inputs"]
    cross_np = item.get("cross_features")
    natal_batch = {k: v[None, ...] for k, v in natal_np.items()}
    if dynamic_np is not None:
        dynamic_batch = {k: v[None, ...] for k, v in dynamic_np.items()}
    else:
        dynamic_batch = None
    cross_batch = cross_np[None, ...] if cross_np is not None else None
    return natal_batch, dynamic_batch, cross_batch


def _chart_from_record(rec):
    from bazi_engine.models import BaziChart

    return BaziChart.from_json(json.dumps(rec["chart"], ensure_ascii=False))


def _build_full_query(chart, question_text: str, hybrid: bool) -> str:
    if hybrid:
        chart_text = chart.to_prompt_text()
        return f"{chart_text}\n\n问题：{question_text}\n答："
    return (
        "请根据以下八字回答问题。\n"
        f"问题：{question_text}\n"
        "请给出有理有据的解释。"
    )


def _write_markdown_sheet(out_path: Path, results: list, args) -> None:
    """Render predictions + chart context + rubric placeholder into markdown."""
    lines: list[str] = []
    lines.append(f"# Reasoning probe: {args.projection_checkpoint.name}")
    lines.append("")
    lines.append(
        f"- hybrid_chart_text: **{args.hybrid_chart_text}**"
        f"  zero_virtual_tokens: **{args.zero_virtual_tokens}**"
    )
    lines.append(f"- charts: {args.num_charts}  questions/chart: see below")
    lines.append("")
    lines.append(
        "Score each prediction on plan §8's rubric (1-5 each), then sum and average:"
    )
    lines.append("")
    lines.append(
        "| Axis | Description |\n"
        "|---|---|\n"
        "| structural | Uses the right facts from the chart correctly |\n"
        "| reasoning | Logical chain connects the facts coherently |\n"
        "| depth | Goes beyond surface; demonstrates interpretation |\n"
        "| fluency | Natural, idiomatic 简体中文 |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    by_chart: dict[int, list] = {}
    for r in results:
        by_chart.setdefault(r["chart_index"], []).append(r)

    for chart_idx, recs in by_chart.items():
        chart_name = recs[0].get("chart_name") or f"chart#{chart_idx}"
        lines.append(f"## {chart_name} (chart_index={chart_idx})")
        lines.append("")
        lines.append("**Chart text block (the model sees this when --hybrid_chart_text):**")
        lines.append("")
        lines.append("```")
        lines.append(recs[0]["chart_text"])
        lines.append("```")
        lines.append("")

        for r in recs:
            lines.append(f"### {r['question_id']}  [{r['category']}]")
            lines.append("")
            lines.append(f"_Tests:_ {r['rationale']}")
            lines.append("")
            lines.append("**Question:**")
            lines.append(r["question"])
            lines.append("")
            lines.append("**Ground-truth facts the answer should use:**")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(r["context"], ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
            lines.append("**Prediction:**")
            lines.append("")
            lines.append("> " + r["prediction"].replace("\n", "\n> "))
            lines.append("")
            lines.append(
                "Scores: structural=__ /5  reasoning=__ /5  depth=__ /5  fluency=__ /5"
            )
            lines.append("")
        lines.append("---")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_probe(args) -> dict:
    import numpy as np

    from bazi_model.data.phase2_dataset import Phase2ChartDataset

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading test data from {args.test_data} ...")
    dataset = Phase2ChartDataset(args.test_data)
    n_charts_total = len(dataset)
    if n_charts_total == 0:
        raise RuntimeError("Phase 2 test dataset is empty.")

    print("Loading BaziLLM ...")
    model = load_bazi_llm(args)

    questions = _build_questions()
    print(f"Question set: {len(questions)} questions across categories.")

    chart_indices = random.sample(range(n_charts_total), args.num_charts)
    print(f"Selected {len(chart_indices)} charts: {chart_indices}")

    results: list = []
    skipped = 0
    for chart_idx in chart_indices:
        rec = dataset._records[chart_idx]
        item = dataset[chart_idx]
        chart = _chart_from_record(rec)
        chart_text = chart.to_prompt_text()
        natal_batch, dynamic_batch, cross_batch = _to_mx_input(item)

        for q in questions:
            if q.skip_if(chart):
                skipped += 1
                continue
            question_text = q.text_fn(chart)
            full_query = _build_full_query(chart, question_text, args.hybrid_chart_text)
            try:
                ctx = q.context_fn(chart)
            except Exception as e:
                ctx = {"_context_error": repr(e)}
            pred = generate_from_chart(
                model,
                natal_batch,
                dynamic_batch,
                full_query,
                cross_features=cross_batch,
                max_tokens=args.max_tokens,
                zero_virtual_tokens=args.zero_virtual_tokens,
            )
            print(f"  [{q.category}] {q.id}  chart={chart_idx}  pred_len={len(pred)}")
            results.append({
                "chart_index": int(chart_idx),
                "chart_name": rec.get("name", ""),
                "chart_text": chart_text,
                "question_id": q.id,
                "category": q.category,
                "rationale": q.rationale,
                "question": question_text,
                "context": ctx,
                "prediction": pred,
            })

    print(f"\nGenerated {len(results)} predictions ({skipped} skipped due to skip_if).")

    summary = {
        "n_charts": args.num_charts,
        "n_questions_defined": len(questions),
        "n_predictions": len(results),
        "n_skipped": skipped,
        "hybrid_chart_text": bool(args.hybrid_chart_text),
        "zero_virtual_tokens": bool(args.zero_virtual_tokens),
        "checkpoint": str(args.projection_checkpoint),
    }
    return {"summary": summary, "results": results}


def main() -> int:
    args = _parse_args()

    try:
        import mlx.core as mx  # noqa: F401
        import mlx.nn as nn  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError as e:
        print("Error: missing MLX / numpy dependencies.", file=sys.stderr)
        print(f"ImportError: {e}", file=sys.stderr)
        return 1

    out = run_probe(args)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results written to {args.output_json}")

    md_path = args.output_json.with_suffix(".md")
    _write_markdown_sheet(md_path, out["results"], args)
    print(f"Rubric scoring sheet written to {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
