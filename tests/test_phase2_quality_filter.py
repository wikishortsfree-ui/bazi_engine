"""Tests for scripts/phase2_quality_filter.py."""
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
    return json.loads(line)["chart"]


def _good_record(chart: dict | None = None) -> dict:
    """Record that should pass both checks (Tier 1)."""
    chart = chart or _sample_chart()
    return {
        "chart": chart,
        "task_type": "本命分析",
        "query": "请分析此八字。",
        "reasoning": (
            "日主偏弱，月令当令，十神配置以财官为主。"
            "天干透干影响格局层次，地支藏干与天干配合决定力量消长。"
            "从格局角度，先看月令，再论用神喜忌。"
            "整体气势偏于泄耗，命局需要扶助日主之力量。"
            "喜用配合得当则运势可期，忌神制化亦需考量。"
        ),
        "output": (
            "大运进入水地，流年引动用神。"
            "流月与命局天干地支多番作用。"
            "喜忌分明时宜顺势而行。"
            "格局清纯则论贵，混杂则论劳碌。"
            "以上从八字结构作一分析，供参考。"
            "若遇刑冲合害并见，需分主次与先后。"
            "十神组合体现六亲与事业倾向。"
            "大运流年流月三者叠看，不宜单柱论断。"
            "日主强弱与月令关系紧密，不可偏废。"
            "用神得力则运势顺遂，受制则多阻滞。"
            "天干论气，地支论根，二者须参看。"
            "实践中仍要结合大运流年流月的具体引动来下结论。"
            "命局中五行流通顺畅者为佳，偏枯则多波折。"
        ),
    }


class TestScoreRecord:
    def test_clean_record_is_tier1(self) -> None:
        from scripts.phase2_quality_filter import score_record

        rec = _good_record()
        scored = score_record(rec, check_interactions=False)
        assert scored["score"] >= 80
        assert scored["tier"] == 1
        assert scored["factual_issues"] == []
        assert scored["language_issues"] == []

    def test_missing_chart_is_tier3(self) -> None:
        from scripts.phase2_quality_filter import score_record

        rec = {"chart": None, "reasoning": "", "output": ""}
        scored = score_record(rec, check_interactions=False)
        assert scored["score"] == 0
        assert scored["tier"] == 3

    def test_factual_issues_reduce_score(self) -> None:
        from scripts.phase2_quality_filter import score_record

        chart = _sample_chart()
        chart = json.loads(json.dumps(chart))
        chart["day_master"]["classification"] = "身强"
        rec = {
            "chart": chart,
            "task_type": "本命分析",
            "query": "q",
            "reasoning": "日主身弱，喜印比。用神得力，十神配合以财官为主。月令当令，格局层次可观。",
            "output": "大运流年引动用神，喜忌分明时宜顺势而行。格局清纯则论贵，十神组合体现六亲与事业倾向。大运流年流月三者叠看，不宜单柱论断。日主强弱与月令关系紧密。天干论气，地支论根，二者须参看。实践中仍要结合大运流年流月的具体引动来下结论。",
        }
        scored = score_record(rec, check_interactions=False)
        assert scored["score"] < 100
        assert any(i["type"] == "strength_mismatch" for i in scored["factual_issues"])

    def test_language_issues_reduce_score(self) -> None:
        from scripts.phase2_quality_filter import score_record

        rec = {
            "chart": _sample_chart(),
            "task_type": "本命分析",
            "query": "q",
            "reasoning": "短",
            "output": "也短",
        }
        scored = score_record(rec, check_interactions=False)
        assert scored["score"] < 100
        assert any(i["type"] == "short_reasoning" for i in scored["language_issues"])
        assert any(i["type"] == "short_output" for i in scored["language_issues"])

    def test_score_floor_is_zero(self) -> None:
        from scripts.phase2_quality_filter import compute_score

        issues = [{"type": "chart_parse"}] * 5
        assert compute_score(issues) == 0


class TestComputeScoreAndTier:
    def test_no_issues_is_100(self) -> None:
        from scripts.phase2_quality_filter import assign_tier, compute_score

        assert compute_score([]) == 100
        assert assign_tier(100) == 1

    def test_tier_boundaries(self) -> None:
        from scripts.phase2_quality_filter import assign_tier

        assert assign_tier(80) == 1
        assert assign_tier(79) == 2
        assert assign_tier(60) == 2
        assert assign_tier(59) == 3
        assert assign_tier(0) == 3


class TestCLI:
    def test_processes_jsonl(self, tmp_path: Path) -> None:
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        rec = _good_record()
        inp.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "phase2_quality_filter.py"),
                "-i", str(inp),
                "-o", str(out),
                "--print-summary",
                "--no-interactions",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert r.returncode == 0
        loaded = json.loads(out.read_text(encoding="utf-8").strip())
        assert "score" in loaded
        assert "tier" in loaded
        assert "factual_issues" in loaded
        assert "language_issues" in loaded
        assert "phase2_quality_filter" in r.stderr

    def test_tier2_list_output(self, tmp_path: Path) -> None:
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        tier2 = tmp_path / "tier2.jsonl"
        rec = {
            "chart": _sample_chart(),
            "task_type": "本命分析",
            "query": "q",
            "reasoning": "短",
            "output": "也短",
        }
        inp.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "phase2_quality_filter.py"),
                "-i", str(inp),
                "-o", str(out),
                "--tier2-list", str(tier2),
                "--no-interactions",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert r.returncode == 0
        scored = json.loads(out.read_text(encoding="utf-8").strip())
        if scored["tier"] == 2:
            tier2_row = json.loads(tier2.read_text(encoding="utf-8").strip())
            assert tier2_row["tier"] == 2
        else:
            tier2_content = tier2.read_text(encoding="utf-8").strip()
            assert tier2_content == ""

    @pytest.mark.parametrize("flag", ["--help", "-h"])
    def test_cli_help(self, flag: str) -> None:
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "phase2_quality_filter.py"), flag],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert r.returncode == 0
