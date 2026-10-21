"""Tests for Stage1QADataset: join with qa_charts + category weighting."""
from __future__ import annotations

import datetime as dt
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bazi_engine import BaziCalculator
from bazi_model.data.phase2_dataset import NATAL_INPUT_KEYS, phase2_collate
from bazi_model.data.stage1_qa_dataset import (
    Stage1QADataset,
    TARGET_CATEGORY_DISTRIBUTION,
    _category_bucket,
    get_category_weights_for_sampler,
    stratified_holdout_indices,
)


@pytest.fixture
def qa_dataset_paths():
    """Build a tiny Stage 1 QA + qa_charts pair with a controlled category mix.

    Mix (12 records):
    - pillar_identity: 4
    - interactions:    4   (over-represented; sampler should down-weight)
    - ten_gods:        2
    - hidden_stems:    2   (goes to "misc" bucket)
    """
    tmp_qa = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    tmp_charts = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")

    charts: dict[str, dict] = {}
    for i in range(3):
        when = dt.datetime(1990 + i, 6, 15, 12, 0, 0, tzinfo=dt.timezone.utc)
        chart = BaziCalculator.from_datetime(when, 0, 0)
        cid = f"chart_{i}"
        chart_dict = json.loads(chart.to_json(indent=None))
        charts[cid] = chart_dict
        tmp_charts.write(json.dumps({"chart_id": cid, "chart": chart_dict}, ensure_ascii=False) + "\n")
    tmp_charts.flush()

    mix = (
        [("pillar_identity", "年柱是什么？", "甲子", "pillar-year")] * 4
        + [("interactions", "地支有没有相冲？", "无", "interaction-clash")] * 4
        + [("ten_gods", "年干十神？", "比肩", "tengod-year")] * 2
        + [("hidden_stems", "日支主气？", "己", "hidden-day-main")] * 2
    )
    for i, (cat, q, a, tid) in enumerate(mix):
        cid = f"chart_{i % 3}"
        tmp_qa.write(
            json.dumps(
                {
                    "chart_id": cid,
                    "question": q,
                    "answer": a,
                    "question_category": cat,
                    "template_id": tid,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    tmp_qa.flush()

    qa_path = Path(tmp_qa.name)
    charts_path = Path(tmp_charts.name)
    tmp_qa.close()
    tmp_charts.close()
    yield qa_path, charts_path
    qa_path.unlink(missing_ok=True)
    charts_path.unlink(missing_ok=True)


def test_stage1_qa_dataset_load(qa_dataset_paths):
    qa_path, charts_path = qa_dataset_paths
    ds = Stage1QADataset(qa_path, charts_path)
    assert len(ds) == 12
    item = ds[0]
    assert item["task_type"] == "八字问答"
    assert item["input_sequence"] == "年柱是什么？"
    assert item["target_sequence"] == "甲子"
    for k in NATAL_INPUT_KEYS:
        assert k in item["natal_inputs"], k
    assert item["natal_inputs"]["stem_ids"].shape == (4,)
    assert item["cross_features"].shape == (27,)
    assert item["category_bucket"] == "pillar_identity"


def test_stage1_qa_missing_chart_drops_record(qa_dataset_paths):
    qa_path, charts_path = qa_dataset_paths
    with open(qa_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "chart_id": "no_such_chart",
                    "question": "...",
                    "answer": "...",
                    "question_category": "pillar_identity",
                    "template_id": "x",
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    ds = Stage1QADataset(qa_path, charts_path)
    assert len(ds) == 12  # the dangling record is dropped
    chart_ids = {ds._records[i]["chart_id"] for i in range(len(ds))}
    assert "no_such_chart" not in chart_ids


def test_category_bucket_maps_unknown_to_misc():
    assert _category_bucket("pillar_identity") == "pillar_identity"
    assert _category_bucket("hidden_stems") == "misc"
    assert _category_bucket("dynamic") == "misc"


def test_target_distribution_sums_to_one():
    assert abs(sum(TARGET_CATEGORY_DISTRIBUTION.values()) - 1.0) < 1e-9


def test_category_weights_reshape_distribution(qa_dataset_paths):
    qa_path, charts_path = qa_dataset_paths
    ds = Stage1QADataset(qa_path, charts_path)
    weights = get_category_weights_for_sampler(ds)
    assert len(weights) == len(ds)
    assert all(w > 0 for w in weights)

    # After normalization, per-category mass should match the target dist.
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    buckets = [ds._records[i]["question_category"] for i in range(len(ds))]
    bucketed = [_category_bucket(b) for b in buckets]
    mass = {b: 0.0 for b in TARGET_CATEGORY_DISTRIBUTION}
    for b, pw in zip(bucketed, w):
        mass[b] += float(pw)

    # Categories absent from the fixture carry zero mass.
    for b in ("five_elements", "day_master_strength"):
        assert mass[b] == 0.0
    # Among observed buckets, mass is proportional to the target distribution.
    present = ("pillar_identity", "ten_gods", "interactions", "misc")
    present_total = sum(TARGET_CATEGORY_DISTRIBUTION[b] for b in present)
    for b in present:
        expected = TARGET_CATEGORY_DISTRIBUTION[b] / present_total
        assert abs(mass[b] - expected) < 1e-9, (b, mass[b], expected)


def test_limit_truncates_records(qa_dataset_paths):
    qa_path, charts_path = qa_dataset_paths
    ds = Stage1QADataset(qa_path, charts_path, limit=5)
    assert len(ds) == 5


def test_stratified_holdout_keeps_disjoint_split(qa_dataset_paths):
    qa_path, charts_path = qa_dataset_paths
    ds = Stage1QADataset(qa_path, charts_path)
    train_idx, val_idx = stratified_holdout_indices(ds, per_category=1, seed=0)
    assert len(set(train_idx).intersection(val_idx)) == 0
    assert len(train_idx) + len(val_idx) == len(ds)
    # One val sample per present bucket (pillar_identity, interactions, ten_gods, misc)
    assert len(val_idx) == 4


def test_collate_with_qa_items(qa_dataset_paths):
    """``phase2_collate`` accepts QA items since the item shape matches."""
    qa_path, charts_path = qa_dataset_paths
    ds = Stage1QADataset(qa_path, charts_path)
    batch = [ds[0], ds[1]]

    class _MockTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, texts, padding="max_length", max_length=1024, truncation=True,
                     return_tensors="np", add_special_tokens=True):
            if isinstance(texts, str):
                texts = [texts]
            if padding == "max_length":
                ids = np.full((len(texts), max_length), 0, dtype=np.int32)
                ids[:, : min(max_length - 1, 8)] = 1
                return {"input_ids": ids}
            return {"input_ids": np.array([np.ones(4, dtype=np.int32) for _ in texts])}

    out = phase2_collate(batch, _MockTokenizer(), max_length=16, pad_token_id=0)
    assert out["input_ids"].shape == (2, 16)
    assert out["labels"].shape == (2, 16)
    assert out["loss_mask"].shape == (2, 16)
    assert out["chart_natal_inputs"]["stem_ids"].shape[0] == 2
    assert out["chart_dynamic_inputs"] is None  # natal-only
    assert out["cross_features"].shape == (2, 27)
