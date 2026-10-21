"""Tests for Phase 2 dataset and collate: phase2_dataset.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bazi_engine import BaziCalculator
from bazi_engine.models import BaziChart
from bazi_model.data.phase2_dataset import (
    EMBEDDING_NATAL_EXTRA_KEYS,
    Phase2ChartDataset,
    phase2_collate,
    get_task_weights_for_sampler,
    TASK_TYPES,
    TARGET_DISTRIBUTION,
    NATAL_INPUT_KEYS,
    DYNAMIC_INPUT_KEYS,
)


def _curated_record(chart_dict: dict, task_type: str) -> dict:
    """One phase2 record with input_sequence and target_sequence."""
    task_tokens = {"本命分析": "[TASK:本命分析]", "流年": "[TASK:流年]", "流月": "[TASK:流月]", "八字问答": "[TASK:八字问答]"}
    query = "请分析。"
    reasoning = "根据八字结构…"
    output = "结论。"
    task_token = task_tokens.get(task_type, "")
    return {
        "chart": chart_dict,
        "task_type": task_type,
        "query": query,
        "reasoning": reasoning,
        "output": output,
        "input_sequence": task_token + "\n" + query,
        "target_sequence": reasoning + "\n" + output,
    }


@pytest.fixture
def phase2_jsonl_path():
    """Write a minimal phase2_train-style JSONL (natal-only tasks)."""
    records = []
    for i, task in enumerate(["本命分析", "八字问答"]):
        dt = __import__("datetime").datetime(1990 + i, 6, 15, 12, 0, 0, tzinfo=__import__("datetime").timezone.utc)
        chart = BaziCalculator.from_datetime(dt, 0, 0)
        records.append(_curated_record(json.loads(chart.to_json(indent=None)), task))
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


class _MockTokenizer:
    """Minimal tokenizer for collate tests: fixed prefix_len and full sequence length."""

    def __init__(self, prefix_len: int = 5, full_len: int = 20, pad_id: int = 0):
        self.prefix_len = prefix_len
        self.full_len = full_len
        self.pad_token_id = pad_id
        self.eos_token_id = pad_id

    def __call__(self, texts, padding="max_length", max_length=1024, truncation=True, return_tensors="np", add_special_tokens=True):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        if padding == "max_length":
            seq_len = max_length
            input_ids = np.full((batch_size, seq_len), self.pad_token_id, dtype=np.int32)
            input_ids[:, : min(seq_len - 1, self.full_len)] = 1
            return {"input_ids": input_ids}
        # padding=False: return variable length per item (for prefix length)
        return {
            "input_ids": np.array([np.ones(self.prefix_len, dtype=np.int32) for _ in texts])
        }


def test_phase2_dataset_load(phase2_jsonl_path):
    ds = Phase2ChartDataset(phase2_jsonl_path)
    assert len(ds) == 2
    item = ds[0]
    assert "natal_inputs" in item
    assert "task_type" in item
    assert "input_sequence" in item
    assert "target_sequence" in item
    for k in NATAL_INPUT_KEYS:
        assert k in item["natal_inputs"], k
    for k in EMBEDDING_NATAL_EXTRA_KEYS:
        assert k in item["natal_inputs"], k
    assert item["natal_inputs"]["stem_ids"].shape == (4,)
    assert item["natal_inputs"]["interaction_vector"].shape == (64,)
    assert item["natal_inputs"]["elemental_strengths"].shape == (5,)
    assert item["natal_inputs"]["day_master_strength"].shape == (1,)
    assert item["natal_inputs"]["useful_god_id"].shape == (1,)
    assert item["task_type"] in TASK_TYPES


def test_phase2_collate_shapes(phase2_jsonl_path):
    ds = Phase2ChartDataset(phase2_jsonl_path)
    batch = [ds[0], ds[1]]
    tokenizer = _MockTokenizer(prefix_len=5, full_len=20, pad_id=0)
    out = phase2_collate(batch, tokenizer, max_length=32, pad_token_id=0, ignore_index=-100)
    assert out["input_ids"].shape[0] == 2
    assert out["input_ids"].shape[1] == 32
    assert out["labels"].shape == out["input_ids"].shape
    assert out["loss_mask"].shape == out["input_ids"].shape
    assert out["chart_natal_inputs"]["stem_ids"].shape[0] == 2
    for k in EMBEDDING_NATAL_EXTRA_KEYS:
        assert k in out["chart_natal_inputs"], k
        assert out["chart_natal_inputs"][k].shape[0] == 2
    assert out["chart_natal_inputs"]["interaction_vector"].shape == (2, 64)
    assert out["chart_natal_inputs"]["elemental_strengths"].shape == (2, 5)
    assert out["chart_dynamic_inputs"] is None  # natal-only batch
    assert out["loss_mask"].dtype == np.float32
    # Mask should be 1 only in the "target" region (prefix_len-1 to seq_len-2)
    assert out["loss_mask"].sum() >= 0


def test_phase2_task_weights(phase2_jsonl_path):
    ds = Phase2ChartDataset(phase2_jsonl_path)
    weights = get_task_weights_for_sampler(ds)
    assert len(weights) == len(ds)
    assert all(w > 0 for w in weights)


def test_phase2_task_types_and_distribution():
    assert set(TARGET_DISTRIBUTION.keys()) == set(TASK_TYPES)
    assert abs(sum(TARGET_DISTRIBUTION.values()) - 1.0) < 1e-6
