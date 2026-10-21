"""Tests for scripts/factual_check.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _sample_chart() -> dict:
    p = ROOT / "data" / "synthetic_charts.jsonl"
    with open(p, encoding="utf-8") as f:
        line = f.readline()
    row = json.loads(line)
    return row["chart"]


def test_check_factual_phantom_ten_god() -> None:
    from scripts.factual_check import check_factual_consistency

    chart = _sample_chart()
    rec = {
        "chart": chart,
        "task_type": "本命分析",
        "query": "q",
        "reasoning": "命局中伤官透出。",
        "output": "建议：无。",
    }
    issues = check_factual_consistency(rec, check_interactions=False)
    types = {i["type"] for i in issues}
    assert "phantom_ten_god" in types


def test_phantom_ten_god_negation_excused() -> None:
    """Negated mentions ('缺伤官', '无劫财') should NOT trigger phantom_ten_god."""
    from scripts.factual_check import check_factual_consistency

    chart = _sample_chart()
    rec = {
        "chart": chart,
        "task_type": "本命分析",
        "query": "q",
        "reasoning": "命局缺伤官，亦无劫财帮身，正印不显。",
        "output": "建议补印。",
    }
    issues = check_factual_consistency(rec, check_interactions=False)
    assert not any(i["type"] == "phantom_ten_god" for i in issues)


def test_phantom_ten_god_transit_excused() -> None:
    """Transit context ('大运引动伤官') should NOT trigger phantom_ten_god."""
    from scripts.factual_check import check_factual_consistency

    chart = _sample_chart()
    rec = {
        "chart": chart,
        "task_type": "本命分析",
        "query": "q",
        "reasoning": "中年后若大运引动伤官，方有突破。",
        "output": "待流年逢劫财，可助身。",
    }
    issues = check_factual_consistency(rec, check_interactions=False)
    assert not any(i["type"] == "phantom_ten_god" for i in issues)


def test_phantom_ten_god_positive_still_flagged() -> None:
    """A genuine positive assertion of an absent ten god must still be flagged."""
    from scripts.factual_check import check_factual_consistency

    chart = _sample_chart()
    rec = {
        "chart": chart,
        "task_type": "本命分析",
        "query": "q",
        "reasoning": "伤官透干，锋芒毕露。",
        "output": "output",
    }
    issues = check_factual_consistency(rec, check_interactions=False)
    assert any(i["type"] == "phantom_ten_god" for i in issues)


def test_composite_ten_god_negation_excused() -> None:
    """Negated composite terms ('无印星', '印星不显') should not be flagged."""
    from scripts.factual_check import check_factual_consistency

    chart = _sample_chart()
    # 'chart' has no 正印 or 偏印 in allowed set? Actually it has 偏印.
    # We need a chart that truly has no members of a composite group.
    # Build a minimal chart with no 正官 or 七杀 (for 官杀 check).
    import copy
    chart2 = copy.deepcopy(chart)
    # Remove all 正官 and 七杀 from ten_gods and hidden_stems
    for k in ("year_stem", "month_stem", "day_stem", "hour_stem"):
        if chart2["ten_gods"].get(k) in ("正官", "七杀"):
            chart2["ten_gods"][k] = "日主"
    for k in ("year_branch", "month_branch", "day_branch", "hour_branch"):
        chart2["ten_gods"][k] = [
            x for x in chart2["ten_gods"].get(k, []) if x not in ("正官", "七杀")
        ]
    for branch_key, stems in chart2.get("hidden_stems", {}).items():
        for s in stems:
            if s.get("ten_god") in ("正官", "七杀"):
                s["ten_god"] = "比肩"

    rec = {
        "chart": chart2,
        "task_type": "本命分析",
        "query": "q",
        "reasoning": "原局无官杀制身。",
        "output": "output",
    }
    issues = check_factual_consistency(rec, check_interactions=False)
    assert not any(i["type"] == "composite_ten_god" for i in issues)


def test_check_factual_strength_mismatch() -> None:
    from scripts.factual_check import check_factual_consistency

    chart = _sample_chart()
    chart = json.loads(json.dumps(chart))
    chart["day_master"]["classification"] = "身强"
    rec = {
        "chart": chart,
        "task_type": "本命分析",
        "query": "q",
        "reasoning": "日主身弱，喜印比。",
        "output": "output",
    }
    issues = check_factual_consistency(rec, check_interactions=False)
    assert any(i["type"] == "strength_mismatch" for i in issues)


def test_check_factual_phantom_branch_clash() -> None:
    from scripts.factual_check import check_factual_consistency

    chart = _sample_chart()
    rec = {
        "chart": chart,
        "task_type": "本命分析",
        "query": "q",
        "reasoning": "四柱见子午冲，动象明显。",
        "output": "output",
    }
    issues = check_factual_consistency(rec, check_interactions=True)
    assert any("子午" in i.get("message", "") for i in issues)


def test_check_factual_useful_god_mismatch() -> None:
    from scripts.factual_check import check_factual_consistency

    chart = _sample_chart()
    rec = {
        "chart": chart,
        "task_type": "本命分析",
        "query": "q",
        "reasoning": "用神为印，宜补印。",
        "output": "output",
    }
    issues = check_factual_consistency(rec, check_interactions=False)
    assert any(i["type"] == "useful_god_mismatch" for i in issues)


def test_check_factual_skip_interactions() -> None:
    from scripts.factual_check import check_factual_consistency

    chart = _sample_chart()
    rec = {
        "chart": chart,
        "task_type": "本命分析",
        "query": "q",
        "reasoning": "四柱见子午冲。",
        "output": "output",
    }
    issues = check_factual_consistency(rec, check_interactions=False)
    assert not any(i["type"] == "phantom_interaction" for i in issues)


def test_invalid_chart() -> None:
    from scripts.factual_check import check_factual_consistency

    issues = check_factual_consistency({"chart": None, "reasoning": "", "output": ""})
    assert issues and issues[0]["type"] == "chart_parse"


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_cli_help(flag: str) -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "factual_check.py"), flag],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
