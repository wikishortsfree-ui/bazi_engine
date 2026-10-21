"""Tests for the hybrid QA prompt assembly.

Covers the helper ``build_hybrid_qa_prompt`` and the ``hybrid_chart_text`` flag
on ``Stage1QADataset`` and ``Phase2ChartDataset``. The format and rationale are
documented in ``plans/hybrid-prompt-v1-and-lora_*.plan.md`` §4.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from bazi_engine import BaziCalculator
from bazi_model.data.phase2_dataset import (
    Phase2ChartDataset,
    build_hybrid_qa_prompt,
    phase2_collate,
)
from bazi_model.data.stage1_qa_dataset import Stage1QADataset


def _sample_chart():
    return BaziCalculator.from_datetime(
        datetime(2003, 4, 24, 21, 20, 0, tzinfo=ZoneInfo("America/Los_Angeles")),
        37.35,
        -121.96,
    )


# --- build_hybrid_qa_prompt ---------------------------------------------------


def test_build_hybrid_qa_prompt_contains_chart_text() -> None:
    """The chart's ``to_prompt_text()`` block is embedded verbatim."""
    chart = _sample_chart()
    prompt = build_hybrid_qa_prompt(chart, "年柱是什么？")
    assert chart.to_prompt_text() in prompt


def test_build_hybrid_qa_prompt_question_and_answer_markers() -> None:
    """Output ends with ``问题：{q}\\n答：`` so the join in collate yields ``答：\\n{answer}``."""
    chart = _sample_chart()
    q = "年柱是什么？"
    prompt = build_hybrid_qa_prompt(chart, q)
    assert prompt.endswith(f"问题：{q}\n答：")


def test_build_hybrid_qa_prompt_blank_line_between_chart_and_question() -> None:
    """A blank line separates the chart block from ``问题：``.

    Concretely, the chart text ends with the 主要交互 line and is followed by
    ``\\n\\n问题：…``; the double newline guarantees the visual break the LLM
    needs to realize the question starts a new section.
    """
    chart = _sample_chart()
    prompt = build_hybrid_qa_prompt(chart, "测试")
    chart_text = chart.to_prompt_text()
    assert prompt.startswith(chart_text + "\n\n问题：")


def test_build_hybrid_qa_prompt_is_deterministic() -> None:
    """Calling twice with the same args yields byte-identical output."""
    chart = _sample_chart()
    a = build_hybrid_qa_prompt(chart, "test")
    b = build_hybrid_qa_prompt(chart, "test")
    assert a == b


# --- Stage1QADataset hybrid toggle --------------------------------------------


@pytest.fixture
def stage1_qa_files(tmp_path: Path):
    """Write a tiny ``stage1_qa.jsonl`` + ``qa_charts.jsonl`` pair."""
    chart = _sample_chart()
    chart_dict = json.loads(chart.to_json(indent=None))
    charts_path = tmp_path / "qa_charts.jsonl"
    charts_path.write_text(
        json.dumps({"chart_id": "chart_0", "chart": chart_dict}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    qa_path = tmp_path / "stage1_qa.jsonl"
    rec = {
        "chart_id": "chart_0",
        "question": "年柱是什么？",
        "answer": "癸未",
        "question_category": "pillar_identity",
        "template_id": "tpl_year_pillar",
    }
    qa_path.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    return qa_path, charts_path


def test_stage1_qa_default_input_sequence_is_question(stage1_qa_files) -> None:
    """Default behavior (hybrid_chart_text=False) preserves bare-question input."""
    qa_path, charts_path = stage1_qa_files
    ds = Stage1QADataset(qa_path, charts_path)
    item = ds[0]
    assert item["input_sequence"] == "年柱是什么？"
    assert item["target_sequence"] == "癸未"


def test_stage1_qa_hybrid_input_sequence_contains_chart_block(stage1_qa_files) -> None:
    """With hybrid_chart_text=True, input_sequence includes the chart text + ``问题：…\\n答：``."""
    qa_path, charts_path = stage1_qa_files
    ds = Stage1QADataset(qa_path, charts_path, hybrid_chart_text=True)
    item = ds[0]
    chart = ds._get_chart("chart_0")
    expected = build_hybrid_qa_prompt(chart, "年柱是什么？")
    assert item["input_sequence"] == expected
    # Target stays intact so loss-mask geometry is unaffected.
    assert item["target_sequence"] == "癸未"


# --- Phase2ChartDataset hybrid toggle -----------------------------------------


@pytest.fixture
def phase2_jsonl(tmp_path: Path):
    chart = _sample_chart()
    chart_dict = json.loads(chart.to_json(indent=None))
    rec = {
        "chart": chart_dict,
        "task_type": "本命分析",
        "query": "请分析。",
        "reasoning": "根据八字结构…",
        "output": "结论。",
        "input_sequence": "[TASK:本命分析]\n请分析。",
        "target_sequence": "根据八字结构…\n结论。",
    }
    p = tmp_path / "phase2_train.jsonl"
    p.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def test_phase2_default_input_sequence_unchanged(phase2_jsonl) -> None:
    ds = Phase2ChartDataset(phase2_jsonl)
    item = ds[0]
    assert item["input_sequence"] == "[TASK:本命分析]\n请分析。"


def test_phase2_hybrid_prepends_chart_text(phase2_jsonl) -> None:
    """Hybrid mode prepends ``chart.to_prompt_text() + '\\n\\n'`` to the original prefix."""
    ds = Phase2ChartDataset(phase2_jsonl, hybrid_chart_text=True)
    item = ds[0]
    chart = _sample_chart()
    expected_prefix = chart.to_prompt_text() + "\n\n"
    assert item["input_sequence"].startswith(expected_prefix)
    assert item["input_sequence"].endswith("[TASK:本命分析]\n请分析。")


# --- Loss mask geometry sanity check ------------------------------------------


class _FixedTokenizer:
    """Tokenizer that splits on whitespace; used to verify loss-mask geometry."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 0

    def __call__(
        self,
        texts,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="np",
        add_special_tokens=True,
    ):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        encoded = [text.split() for text in texts]
        if padding == "max_length":
            seq_len = max_length
            ids = np.full((len(encoded), seq_len), self.pad_token_id, dtype=np.int32)
            for i, toks in enumerate(encoded):
                n = min(len(toks), seq_len)
                ids[i, :n] = np.arange(1, n + 1, dtype=np.int32)
            return {"input_ids": ids}
        # padding=False: HF returns a 2D ndarray of shape (batch, max_len_in_batch).
        max_n = max(len(t) for t in encoded)
        ids = np.full((len(encoded), max_n), self.pad_token_id, dtype=np.int32)
        for i, toks in enumerate(encoded):
            ids[i, : len(toks)] = np.arange(1, len(toks) + 1, dtype=np.int32)
        return {"input_ids": ids}


def test_hybrid_loss_mask_covers_only_answer_tokens(stage1_qa_files) -> None:
    """``phase2_collate`` masks everything up to ``答：`` even in hybrid mode."""
    qa_path, charts_path = stage1_qa_files
    ds = Stage1QADataset(qa_path, charts_path, hybrid_chart_text=True)
    tok = _FixedTokenizer()
    out = phase2_collate([ds[0]], tok, max_length=512, pad_token_id=0)

    # Tokens that should fall under loss mask = answer tokens only.
    answer_tokens = ds[0]["target_sequence"].split()
    assert int(out["loss_mask"].sum()) == len(answer_tokens), (
        f"loss_mask sum={int(out['loss_mask'].sum())} != answer tokens "
        f"({len(answer_tokens)})"
    )
