"""
Phase 2 prompt builder: chart features → structured Chinese prompts for LLM
interpretation generation (Qwen-Plus / Claude).

Given a BaziChart JSON dict (as from BaziChart.to_json()), builds:
  1. Structured chart feature text (pillars, ten gods, hidden stems, elemental
     strengths, useful god, interactions).
  2. Temporal feature text for 流年/流月 (大运, 流年, 流月, cross interactions,
     elemental delta, useful god activation).
  3. Per-task system + user prompts for 本命分析, 流年, 流月, 八字问答.
  4. Optional classical excerpt stub (for future RAG integration).

Output names align with Phase 2 pipeline: query, reasoning, output.

Usage (as module):
    from scripts.phase2_prompts import build_prompt
    result = build_prompt(chart_dict, "本命分析")
    # result = {"system_prompt": ..., "user_prompt": ..., "task_type": ..., "query": ...}

Usage (CLI demo):
    python scripts/phase2_prompts.py --input data/astrodatabank_bazi.jsonl --task 本命分析 --limit 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(_REPO_ROOT))

from bazi_engine._constants import POSITION_LABELS
from scripts.prepare_phase2_dataset import TASK_TYPES

_ELEMENT_EN_TO_ZH = {"wood": "木", "fire": "火", "earth": "土", "metal": "金", "water": "水"}
_POSITION_LABELS_ZH = {"year": "年柱", "month": "月柱", "day": "日柱", "hour": "时柱"}

# ── system prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一位资深八字命理师，精通子平真诠、滴天髓、穷通宝鉴等经典命理典籍。"
    "请根据下方提供的八字盘面数据（四柱、十神、藏干、五行力量、用神、合冲刑害等），"
    "进行专业的命理分析。\n\n"
    "要求：\n"
    "1. 用 JSON 格式回复，包含两个字段：\n"
    '   - "reasoning"：推理过程（300–600字），引用具体的十神、五行、合冲等数据，'
    "说明判断依据。\n"
    '   - "output"：最终解读（200–500字），分为「命理分析」和「实用建议」两段。\n'
    "2. 不得编造盘面中不存在的十神、合冲、刑害关系。\n"
    "3. 引用五行力量数值时使用原始百分比。\n"
    "4. 使用传统命理术语（用神、忌神、十神、刑冲合害、身强身弱等），"
    "但解释需通俗易懂。\n"
    "5. 仅输出 JSON，不要在 JSON 之外添加任何文字。"
)

SYSTEM_PROMPT_QA = (
    "你是一位资深八字命理师，精通子平真诠、滴天髓、穷通宝鉴等经典命理典籍。"
    "请根据下方提供的八字盘面数据，回答用户的命理问题。\n\n"
    "要求：\n"
    "1. 用 JSON 格式回复，包含两个字段：\n"
    '   - "reasoning"：推理过程（200–500字），引用盘面中具体的十神、五行、'
    "合冲数据说明判断依据。\n"
    '   - "output"：针对问题的回答（100–400字），给出明确且实用的建议。\n'
    "2. 不得编造盘面中不存在的关系或数据。\n"
    "3. 仅输出 JSON，不要在 JSON 之外添加任何文字。"
)

# ── chart feature text ──────────────────────────────────────────────────────

def chart_features_text(chart: dict[str, Any]) -> str:
    """
    Build structured Chinese text summarizing natal chart features.

    Args:
        chart: Full BaziChart JSON dict (as from BaziChart.to_json() or a record's "chart" field).

    Returns:
        Multi-line Chinese string suitable for LLM context.
    """
    parts: list[str] = []

    natal = chart.get("natal_pillars") or {}
    dm = chart.get("day_master") or {}
    ten_gods = chart.get("ten_gods") or {}
    hidden = chart.get("hidden_stems") or {}
    el_str = chart.get("elemental_strengths") or {}
    useful = chart.get("useful_god")
    interactions = chart.get("interactions") or {}

    parts.append("【四柱】")
    for key, label in (("year", "年"), ("month", "月"), ("day", "日"), ("hour", "时")):
        p = natal.get(key) or {}
        stem, branch = p.get("stem", "?"), p.get("branch", "?")
        tg_stem = ten_gods.get(f"{key}_stem", "")
        parts.append(f"  {label}柱：{stem}{branch}（天干{stem}[{tg_stem}]，地支{branch}）")

    parts.append("")
    parts.append("【日主】")
    parts.append(
        f"  日主：{dm.get('stem', '?')}（{dm.get('element', '?')}），"
        f"强度 {dm.get('strength_percentile', '?')}，"
        f"判定：{dm.get('classification', '?')}"
    )

    parts.append("")
    parts.append("【十神（天干）】")
    for key, label in (("year", "年"), ("month", "月"), ("day", "日"), ("hour", "时")):
        tg = ten_gods.get(f"{key}_stem", "")
        parts.append(f"  {label}干：{tg}")

    parts.append("")
    parts.append("【藏干与十神（地支）】")
    for key, label in (("year", "年"), ("month", "月"), ("day", "日"), ("hour", "时")):
        branch_tg = ten_gods.get(f"{key}_branch") or []
        hs = hidden.get(f"{key}_branch") or []
        desc_parts = []
        for i, h in enumerate(hs):
            tg = branch_tg[i] if i < len(branch_tg) else "?"
            desc_parts.append(f"{h['stem']}({h['element']}, {tg}, 权重{h['weight']})")
        parts.append(f"  {label}支藏干：{'、'.join(desc_parts) if desc_parts else '无'}")

    parts.append("")
    parts.append("【五行力量】")
    for en_key in ("wood", "fire", "earth", "metal", "water"):
        zh = _ELEMENT_EN_TO_ZH.get(en_key, en_key)
        val = el_str.get(en_key, 0)
        parts.append(f"  {zh}：{val:.1%}" if isinstance(val, (int, float)) else f"  {zh}：{val}")

    parts.append("")
    parts.append("【用神】")
    if useful and not useful.get("is_ambiguous", True):
        parts.append(
            f"  用神：{useful.get('primary', '?')}"
            f"（{useful.get('primary_element', '?')}），"
            f"方法论：{useful.get('methodology', '?')}"
        )
        fav = useful.get("favorable_elements") or []
        if fav:
            parts.append(f"  喜用五行：{'、'.join(fav)}")
    else:
        parts.append("  用神：中和（无明显偏向）")

    interaction_text = _format_interactions(interactions)
    if interaction_text:
        parts.append("")
        parts.append("【命局合冲刑害破】")
        parts.append(interaction_text)

    return "\n".join(parts)


def _format_interactions(interactions: dict[str, Any]) -> str:
    """Format natal interactions into Chinese text lines."""
    lines: list[str] = []

    for sc in interactions.get("stem_combinations") or []:
        labels = sc.get("position_labels") or []
        stems = sc.get("stems") or []
        active = "化成" if sc.get("transformation_active") else "合而不化"
        lines.append(
            f"  天干五合：{'+'.join(labels)} {'+'.join(stems)} → "
            f"{sc.get('result_element', '?')}（{active}）"
        )

    for sc in interactions.get("stem_clashes") or []:
        labels = sc.get("position_labels") or []
        stems = sc.get("stems") or []
        lines.append(f"  天干相冲：{'+'.join(labels)} {'+'.join(stems)}")

    for bc in interactions.get("branch_six_combinations") or []:
        labels = bc.get("position_labels") or []
        branches = bc.get("branches") or []
        blocked = "（被冲破）" if bc.get("is_blocked") else ""
        lines.append(
            f"  地支六合：{'+'.join(labels)} {'+'.join(branches)} → "
            f"{bc.get('result_element', '?')}{blocked}"
        )

    for bc in interactions.get("branch_clashes") or []:
        labels = bc.get("position_labels") or []
        branches = bc.get("branches") or []
        lines.append(f"  地支六冲：{'+'.join(labels)} {'+'.join(branches)}")

    for th in interactions.get("three_harmony") or []:
        combo = th.get("combo_type", "?")
        combo_label = "三合" if combo == "full" else "半合"
        blocked = "（被冲破）" if th.get("blocked_by_clash") else ""
        branches = th.get("present_branches") or []
        lines.append(
            f"  {combo_label}：{'+'.join(branches)} → "
            f"{th.get('element', '?')}{blocked}"
        )

    for h in interactions.get("harms") or []:
        labels = h.get("position_labels") or []
        branches = h.get("branches") or []
        lines.append(f"  六害：{'+'.join(labels)} {'+'.join(branches)}")

    for p in interactions.get("punishments") or []:
        ptype = p.get("punishment_type", "刑")
        members = p.get("members_present") or []
        lines.append(f"  刑（{ptype}）：{'、'.join(str(m) for m in members) if members else '见上方地支'}")

    for d in interactions.get("destructions") or []:
        labels = d.get("position_labels") or []
        branches = d.get("branches") or []
        weakened = "（减弱）" if d.get("is_weakened") else ""
        lines.append(f"  破：{'+'.join(labels)} {'+'.join(branches)}{weakened}")

    return "\n".join(lines) if lines else "  无明显合冲刑害破"


# ── temporal feature text ───────────────────────────────────────────────────

def temporal_features_text(chart: dict[str, Any]) -> Optional[str]:
    """
    Build structured Chinese text for temporal context (大运/流年/流月 + cross interactions).

    Returns None if the chart has no dynamic section.
    """
    dynamic = chart.get("dynamic")
    if not dynamic:
        return None

    parts: list[str] = []

    da_yun = dynamic.get("da_yun")
    if da_yun:
        parts.append("【大运】")
        parts.append(
            f"  {da_yun.get('stem', '?')}{da_yun.get('branch', '?')}（"
            f"{da_yun.get('start_year', '?')}–{da_yun.get('end_year', '?')}年，"
            f"{da_yun.get('start_age', '?')}–{da_yun.get('end_age', '?')}岁）"
        )

    liu_nian = dynamic.get("liu_nian")
    if liu_nian:
        parts.append("【流年】")
        parts.append(
            f"  {liu_nian.get('stem', '?')}{liu_nian.get('branch', '?')}"
            f"（{liu_nian.get('year', '?')}年）"
        )

    liu_yue = dynamic.get("liu_yue")
    if liu_yue:
        parts.append("【流月】")
        parts.append(
            f"  {liu_yue.get('stem', '?')}{liu_yue.get('branch', '?')}"
            f"（{liu_yue.get('year', '?')}年{liu_yue.get('month', '?')}月）"
        )

    cross = dynamic.get("cross_natal_temporal_interactions")
    if cross:
        cross_text = _format_cross_interactions(cross)
        if cross_text:
            parts.append("")
            parts.append("【命运交互（合冲刑害）】")
            parts.append(cross_text)

    el_delta = dynamic.get("elemental_delta")
    if el_delta:
        parts.append("")
        parts.append("【五行变化量（运势 - 命局）】")
        for en_key in ("wood", "fire", "earth", "metal", "water"):
            zh = _ELEMENT_EN_TO_ZH.get(en_key, en_key)
            val = el_delta.get(en_key, 0)
            sign = "+" if val > 0 else ""
            parts.append(f"  {zh}：{sign}{val:.4f}" if isinstance(val, (int, float)) else f"  {zh}：{val}")

    uga = dynamic.get("useful_god_activation")
    if uga:
        status = "活跃" if uga.get("is_active") else "不活跃/中性"
        mechanism = uga.get("mechanism", "")
        parts.append("")
        parts.append("【用神活跃状态】")
        parts.append(f"  状态：{status}")
        if mechanism:
            parts.append(f"  原因：{mechanism}")

    return "\n".join(parts) if parts else None


def _natal_pos_label(pos: Any) -> str:
    """Return '(命局X柱)' label for a natal_pos index, or '' if pos is None/invalid."""
    if pos is None or not isinstance(pos, int) or not (0 <= pos < 4):
        return ""
    return f"(命局{_POSITION_LABELS_ZH[POSITION_LABELS[pos]]})"


def _format_cross_interactions(cross: dict[str, Any]) -> str:
    """Format cross natal-temporal interactions into Chinese text."""
    ctx_labels = {
        "natal_vs_dayun": "命局×大运",
        "natal_vs_liunian": "命局×流年",
        "natal_vs_liuyue": "命局×流月",
        "dayun_vs_liunian": "大运×流年",
        "dayun_vs_liuyue": "大运×流月",
        "liunian_vs_liuyue": "流年×流月",
    }
    lines: list[str] = []

    for ctx_key, ctx_label in ctx_labels.items():
        ctx = cross.get(ctx_key)
        if not ctx:
            continue
        ctx_lines: list[str] = []

        for c in ctx.get("combinations") or []:
            a, b = c.get("stem_a", "?"), c.get("stem_b", "?")
            res = c.get("result_element", "?")
            pos_str = _natal_pos_label(c.get("natal_pos"))
            ctx_lines.append(f"    天干合：{a}+{b}→{res}{pos_str}")

        for c in ctx.get("clashes") or []:
            a, b = c.get("stem_a", "?"), c.get("stem_b", "?")
            pos_str = _natal_pos_label(c.get("natal_pos"))
            ctx_lines.append(f"    天干冲：{a}冲{b}{pos_str}")

        for c in ctx.get("branch_combinations") or []:
            a, b = c.get("branch_a", "?"), c.get("branch_b", "?")
            pos_str = _natal_pos_label(c.get("natal_pos"))
            ctx_lines.append(f"    地支合：{a}+{b}{pos_str}")

        for c in ctx.get("harms") or []:
            a, b = c.get("branch_a", "?"), c.get("branch_b", "?")
            pos_str = _natal_pos_label(c.get("natal_pos"))
            ctx_lines.append(f"    地支害/冲/刑：{a}+{b}{pos_str}")

        if ctx_lines:
            lines.append(f"  [{ctx_label}]")
            lines.extend(ctx_lines)

    return "\n".join(lines) if lines else "  无明显交互"


# ── classical text stub ─────────────────────────────────────────────────────

def classical_excerpt_stub(chart: dict[str, Any]) -> Optional[str]:
    """
    Placeholder for classical text retrieval (RAG).

    Returns None until the classical corpus (task 2.5) is available.
    When implemented, this will query a FAISS index over chunked 滴天髓/子平真诠/
    穷通宝鉴/三命通会 and return relevant passages based on the chart's day master
    element, useful god, and interaction patterns.
    """
    return None


# ── per-task prompt builders ────────────────────────────────────────────────

_QA_TEMPLATES = [
    "此八字的事业运如何？适合从事什么行业？",
    "此八字的婚姻感情方面有什么特点？",
    "此八字的财运如何？何时有较大的财运机会？",
    "此八字的健康方面需要注意什么？",
    "此八字的学业运势如何？",
    "此八字的人际关系和贵人运如何？",
    "此八字的性格特点是什么？优缺点如何？",
    "此八字一生中最需要注意的事情是什么？",
]


def build_prompt(
    chart: dict[str, Any],
    task_type: str,
    *,
    question: Optional[str] = None,
    include_classical: bool = False,
) -> dict[str, str]:
    """
    Build a complete prompt for LLM interpretation generation.

    Args:
        chart: Full BaziChart JSON dict (from BaziChart.to_json() or a record's "chart" field).
        task_type: One of 本命分析, 流年, 流月, 八字问答.
        question: For 八字问答, the user's specific question. Ignored for other task types.
        include_classical: If True, append classical excerpt (stub for now).

    Returns:
        Dict with system_prompt, user_prompt, task_type, query.
        query = the user-facing question text (suitable for the "query" field in curated JSONL).
    """
    if task_type not in TASK_TYPES:
        raise ValueError(f"Unknown task_type: {task_type!r}; expected one of {TASK_TYPES}")

    features = chart_features_text(chart)
    classical = classical_excerpt_stub(chart) if include_classical else None

    if task_type == "本命分析":
        return _build_benming(features, classical)
    elif task_type == "流年":
        return _build_liunian(chart, features, classical)
    elif task_type == "流月":
        return _build_liuyue(chart, features, classical)
    elif task_type == "八字问答":
        return _build_qa(chart, features, classical, question)
    else:
        raise ValueError(f"Unhandled task_type: {task_type!r}")


def _build_benming(
    features: str,
    classical: Optional[str],
) -> dict[str, str]:
    query = "请对此八字进行全面的本命分析，包括性格、事业、财运、婚姻、健康等方面。"
    user_prompt = f"以下是八字盘面数据：\n\n{features}"
    if classical:
        user_prompt += f"\n\n【参考经典】\n{classical}"
    user_prompt += f"\n\n{query}"
    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "task_type": "本命分析",
        "query": query,
    }


def _build_liunian(
    chart: dict[str, Any],
    features: str,
    classical: Optional[str],
) -> dict[str, str]:
    dynamic = chart.get("dynamic") or {}
    liu_nian = dynamic.get("liu_nian") or {}
    year = liu_nian.get("year", "?")
    query = f"请分析此八字在{year}年（流年）的运势，包括事业、财运、感情、健康等方面的变化。"

    temporal = temporal_features_text(chart)
    user_prompt = f"以下是八字盘面数据：\n\n{features}"
    if temporal:
        user_prompt += f"\n\n{temporal}"
    if classical:
        user_prompt += f"\n\n【参考经典】\n{classical}"
    user_prompt += f"\n\n{query}"
    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "task_type": "流年",
        "query": query,
    }


def _build_liuyue(
    chart: dict[str, Any],
    features: str,
    classical: Optional[str],
) -> dict[str, str]:
    dynamic = chart.get("dynamic") or {}
    liu_yue = dynamic.get("liu_yue") or {}
    year = liu_yue.get("year", "?")
    month = liu_yue.get("month", "?")
    query = f"请分析此八字在{year}年{month}月（流月）的运势走向和注意事项。"

    temporal = temporal_features_text(chart)
    user_prompt = f"以下是八字盘面数据：\n\n{features}"
    if temporal:
        user_prompt += f"\n\n{temporal}"
    if classical:
        user_prompt += f"\n\n【参考经典】\n{classical}"
    user_prompt += f"\n\n{query}"
    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "task_type": "流月",
        "query": query,
    }


def _build_qa(
    chart: dict[str, Any],
    features: str,
    classical: Optional[str],
    question: Optional[str],
) -> dict[str, str]:
    query = question or "请分析此八字的整体格局和特点。"

    temporal = temporal_features_text(chart)
    user_prompt = f"以下是八字盘面数据：\n\n{features}"
    if temporal:
        user_prompt += f"\n\n{temporal}"
    if classical:
        user_prompt += f"\n\n【参考经典】\n{classical}"
    user_prompt += f"\n\n问题：{query}"
    return {
        "system_prompt": SYSTEM_PROMPT_QA,
        "user_prompt": user_prompt,
        "task_type": "八字问答",
        "query": query,
    }


# ── convenience: build from JSONL record ────────────────────────────────────

def build_prompt_from_record(
    record: dict[str, Any],
    task_type: str,
    *,
    question: Optional[str] = None,
    include_classical: bool = False,
) -> dict[str, str]:
    """
    Build prompt from a JSONL record (with "chart" field, as in selected_charts.jsonl
    or astrodatabank_bazi.jsonl).

    Handles chart being either a dict or a JSON string.
    """
    chart = record.get("chart")
    if chart is None:
        raise ValueError("Record has no 'chart' field")
    if isinstance(chart, str):
        chart = json.loads(chart)
    return build_prompt(chart, task_type, question=question, include_classical=include_classical)


# ── CLI demo ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demo: build Phase 2 prompts from chart JSONL."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/astrodatabank_bazi.jsonl"),
        help="Input JSONL with chart records (default: data/astrodatabank_bazi.jsonl)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=list(TASK_TYPES),
        default="本命分析",
        help="Task type to build prompt for (default: 本命分析)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Custom question for 八字问答 task",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of records to process (default: 1)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of records to skip (default: 0)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file (JSONL); prints to stdout if not specified",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        return 1

    records_processed = 0
    out_file = open(args.output, "w", encoding="utf-8") if args.output else None

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < args.offset:
                    continue
                if records_processed >= args.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    result = build_prompt_from_record(
                        record, args.task, question=args.question,
                    )
                except (ValueError, KeyError) as e:
                    print(f"Skipping record {i}: {e}", file=sys.stderr)
                    continue

                if out_file:
                    out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                else:
                    name = record.get("name", f"record_{i}")
                    print(f"═══ {name} ({args.task}) ═══")
                    print(f"\n[system_prompt]\n{result['system_prompt']}")
                    print(f"\n[user_prompt]\n{result['user_prompt']}")
                    print(f"\n[query]\n{result['query']}")
                    print()

                records_processed += 1
    finally:
        if out_file:
            out_file.close()

    print(f"Processed {records_processed} record(s).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
