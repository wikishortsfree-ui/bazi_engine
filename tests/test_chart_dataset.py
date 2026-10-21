"""Unit tests for ChartDataset: (inputs, targets) per task_name, temporal_only filtering."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from bazi_engine import BaziCalculator
from bazi_engine.models import BaziChart
from bazi_engine.temporal import (
    TemporalContext,
    compute_liunian,
    compute_liuyue,
    detect_cross_interactions,
)
from bazi_model.data import ChartDataset, chart_collate_fn_inputs_targets


def _chart_with_dynamic(birth_dt, lat: float = 0, lng: float = 0, year: int = 2024, month: int = 3):
    """Build a chart with temporal_context and cross_interactions."""
    chart = BaziCalculator.from_datetime(birth_dt, lat, lng)
    dayun_seq = BaziCalculator().compute_dayun(chart, 1, max_periods=10)
    dayun = dayun_seq.get_period_for_year(year)
    liunian = compute_liunian(year, day_master_stem=chart.day_pillar.stem.value)
    liuyue = compute_liuyue(year, month, day_master_stem=chart.day_pillar.stem.value)
    ctx = TemporalContext(dayun=dayun, liunian=liunian, liuyue=liuyue, liuri=None)
    stems = [
        chart.year_pillar.stem.value,
        chart.month_pillar.stem.value,
        chart.day_pillar.stem.value,
        chart.hour_pillar.stem.value,
    ]
    branches = [
        chart.year_pillar.branch.value,
        chart.month_pillar.branch.value,
        chart.day_pillar.branch.value,
        chart.hour_pillar.branch.value,
    ]
    cross = detect_cross_interactions(stems, branches, ctx, useful_god=None)
    return BaziChart(
        birth_dt=chart.birth_dt,
        solar_dt=chart.solar_dt,
        lat=chart.lat,
        lng=chart.lng,
        year_pillar=chart.year_pillar,
        month_pillar=chart.month_pillar,
        day_pillar=chart.day_pillar,
        hour_pillar=chart.hour_pillar,
        sect=chart.sect,
        temporal_context=ctx,
        cross_interactions=cross,
    )


@pytest.fixture
def charts_jsonl_path():
    """Write a small charts.jsonl: 3 natal-only + 2 with dynamic."""
    from datetime import datetime, timezone

    lines = []
    # Natal-only
    for i in range(3):
        dt = datetime(1990 + i, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        chart = BaziCalculator.from_datetime(dt, 0, 0)
        lines.append(chart.to_json(indent=None))
    # With dynamic
    for i in range(2):
        dt = datetime(1995 + i, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
        chart = _chart_with_dynamic(dt, 0, 0, 2024, 3)
        lines.append(chart.to_json(indent=None))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for line in lines:
            f.write(line + "\n")
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


def test_chart_dataset_keys_and_shapes_per_task(charts_jsonl_path):
    """ChartDataset returns (inputs, targets) with task-appropriate keys; no input/target overlap."""
    ds = ChartDataset(charts_jsonl_path, task_name="ten_god", temporal_only=False)
    assert len(ds) == 5

    inputs, targets = ds[0]
    assert set(inputs.keys()).isdisjoint(set(targets.keys())), "inputs and targets must not overlap"

    # Base inputs (all tasks)
    assert "stem_ids" in inputs and "branch_ids" in inputs
    assert "stem_elements" in inputs and "stem_polarities" in inputs and "pillar_positions" in inputs
    assert inputs["stem_ids"].shape == (4,)
    assert inputs["stem_ids"].dtype == torch.int64
    assert inputs["branch_ids"].shape == (4,)
    # ten_god targets
    assert "ten_god_ids" in targets
    assert targets["ten_god_ids"].shape == (4,)
    assert targets["ten_god_ids"].dtype == torch.int64


def test_chart_dataset_natal_sample_inputs_no_target_keys(charts_jsonl_path):
    """Inputs contain only base keys (no ground-truth keys)."""
    ds = ChartDataset(charts_jsonl_path, task_name="ten_god", temporal_only=False)
    inputs, targets = ds[0]
    assert "ten_god_ids" not in inputs
    assert "ten_god_ids" in targets
    assert "stem_ids" in inputs
    assert "stem_ids" not in targets


def test_chart_dataset_temporal_only_filters(charts_jsonl_path):
    """When task_name=cross_temporal and temporal_only=True, only charts with dynamic."""
    ds_all = ChartDataset(charts_jsonl_path, task_name="ten_god", temporal_only=False)
    ds_temporal = ChartDataset(
        charts_jsonl_path, task_name="cross_temporal", temporal_only=True
    )

    assert len(ds_all) == 5
    assert len(ds_temporal) == 2

    for i in range(len(ds_temporal)):
        inputs, targets = ds_temporal[i]
        assert inputs["pillar_presence_mask"].sum() >= 1
        assert inputs["dayun_stem_id"].item() >= 0 or inputs["liunian_stem_id"].item() >= 0
        assert targets["dynamic_interaction_vector"].shape == (66,)


def test_chart_dataset_collate_fn_stacks_batch(charts_jsonl_path):
    """chart_collate_fn_inputs_targets stacks (inputs, targets) into batched tensors."""
    ds = ChartDataset(charts_jsonl_path, task_name="ten_god", temporal_only=False)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, collate_fn=chart_collate_fn_inputs_targets, num_workers=0
    )
    batch_inputs, batch_targets = next(iter(loader))
    assert batch_inputs["stem_ids"].shape == (2, 4)
    assert batch_inputs["branch_ids"].shape == (2, 4)
    assert batch_targets["ten_god_ids"].shape == (2, 4)


def test_chart_dataset_requires_task_name_and_cross_temporal_requires_temporal_only(charts_jsonl_path):
    """ChartDataset requires valid task_name; cross_temporal requires temporal_only=True."""
    with pytest.raises(ValueError, match="task_name must be one of"):
        ChartDataset(charts_jsonl_path, task_name="invalid_task", temporal_only=False)
    with pytest.raises(ValueError, match="cross_temporal task requires temporal_only=True"):
        ChartDataset(charts_jsonl_path, task_name="cross_temporal", temporal_only=False)


def test_chart_dataset_from_json_roundtrip_shapes():
    """One chart: from_json(line).to_feature_arrays() matches original to_feature_arrays() shapes."""
    from datetime import datetime, timezone

    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    original = chart.to_feature_arrays()
    json_str = chart.to_json(indent=None)
    restored = BaziChart.from_json(json_str)
    restored_arrays = restored.to_feature_arrays()

    assert set(original.keys()) == set(restored_arrays.keys())
    for key in original:
        assert original[key].shape == restored_arrays[key].shape, f"Shape mismatch: {key}"
        assert original[key].dtype == restored_arrays[key].dtype, f"Dtype mismatch: {key}"
