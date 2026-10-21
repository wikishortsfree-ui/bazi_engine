"""Tests for scripts/language_quality_check.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_short_text_flags() -> None:
    from scripts.language_quality_check import check_language_quality

    rec = {"reasoning": "短", "output": "也短"}
    issues = check_language_quality(rec, min_reasoning_chars=100, min_output_chars=200)
    types = {i["type"] for i in issues}
    assert "short_reasoning" in types
    assert "short_output" in types


def test_generic_phrases_flag() -> None:
    from scripts.language_quality_check import check_language_quality

    filler = "。".join(["综上所述"] * 10)
    rec = {
        "reasoning": "x" * 120 + filler,
        "output": "y" * 220 + filler,
    }
    issues = check_language_quality(rec, max_generic_hits=8)
    assert any(i["type"] == "generic_phrases" for i in issues)


def test_thin_terminology() -> None:
    from scripts.language_quality_check import check_language_quality

    rec = {
        "reasoning": "a" * 150,
        "output": "b" * 250,
    }
    issues = check_language_quality(rec, min_classical_terms=2)
    assert any(i["type"] == "thin_terminology" for i in issues)


def test_ok_with_domain_terms() -> None:
    from scripts.language_quality_check import check_language_quality

    rec = {
        "reasoning": (
            "日主偏弱，月令当令，十神配置以财官为主。"
            "身强身弱需结合地支刑冲合害综合看。"
            "天干透干影响格局层次。"
            "地支藏干与天干配合决定力量消长。"
            "从格局角度，先看月令，再论用神喜忌。"
            "身强喜泄耗，身弱喜生扶，此为常法。"
            "再看地支刑冲对用神的影响。"
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
            "十神与刑冲合害宜统观。"
        ),
    }
    assert len(rec["reasoning"]) >= 100
    assert len(rec["output"]) >= 200
    issues = check_language_quality(rec)
    assert not issues


def test_repetition_flag() -> None:
    from scripts.language_quality_check import check_language_quality

    sentence = "流年不利需谨慎处理八字结构"
    rec = {
        "reasoning": "日主中和。用神。十神。",
        "output": (sentence + "。") * 10,
    }
    issues = check_language_quality(rec, max_duplicate_ratio=0.2, max_duplicate_sentence_count=3)
    assert any(i["type"] == "repetition" for i in issues)


def test_cli_processes_jsonl(tmp_path: Path) -> None:
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    row = {
        "chart": {},
        "reasoning": "x" * 120,
        "output": "y" * 220 + "用神。流年。",
    }
    inp.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "language_quality_check.py"),
            "-i",
            str(inp),
            "-o",
            str(out),
            "--print-summary",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    loaded = json.loads(out.read_text(encoding="utf-8").strip())
    assert "language_issues" in loaded
    assert "language_ok" in loaded


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_cli_help(flag: str) -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "language_quality_check.py"), flag],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
