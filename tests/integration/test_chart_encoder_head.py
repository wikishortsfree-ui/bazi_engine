"""
Integration test: ChartDataset → NatalEncoder → one head → loss.

Verifies that a real batch from ChartDataset flows through NatalEncoder and
a task head (TenGodHead) with correct shapes and that loss backward runs.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from bazi_engine import BaziCalculator
from bazi_model.data import ChartDataset, chart_collate_fn_inputs_targets
from bazi_model.encoders import NatalEncoder
from bazi_model.tasks import TenGodHead


@pytest.fixture
def charts_jsonl_path():
    """Small charts.jsonl with natal-only charts for integration test."""
    from datetime import datetime, timezone

    lines = []
    for i in range(3):
        dt = datetime(1990 + i, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        chart = BaziCalculator.from_datetime(dt, 0, 0)
        lines.append(chart.to_json(indent=None))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for line in lines:
            f.write(line + "\n")
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


def _encoder_kwargs_from_batch(batch: dict[str, torch.Tensor]) -> dict:
    """Extract minimal kwargs for NatalEncoder.forward (no ground-truth leakage)."""
    return {
        "stem_ids": batch["stem_ids"].long(),
        "branch_ids": batch["branch_ids"].long(),
        "stem_elements": batch["stem_elements"].long(),
        "stem_polarities": batch["stem_polarities"].long(),
        "pillar_positions": batch["pillar_positions"].long(),
    }


def test_chart_dataset_natal_encoder_ten_god_head_shapes_and_loss(charts_jsonl_path):
    """ChartDataset → NatalEncoder → TenGodHead: shapes and loss sanity."""
    from torch.utils.data import DataLoader

    batch_size = 2
    ds = ChartDataset(charts_jsonl_path, task_name="ten_god", temporal_only=False)
    loader = DataLoader(
        ds, batch_size=batch_size, collate_fn=chart_collate_fn_inputs_targets, num_workers=0
    )
    inputs, targets = next(iter(loader))

    encoder = NatalEncoder()
    encoder_kwargs = _encoder_kwargs_from_batch(inputs)
    natal_emb = encoder(**encoder_kwargs)

    assert natal_emb.shape == (batch_size, 256), (
        f"natal_emb shape expected (batch, 256), got {natal_emb.shape}"
    )

    head = TenGodHead(d_model=256)
    ten_god_ids = targets["ten_god_ids"].long()
    loss, logits = head.forward_loss(natal_emb, ten_god_ids)

    assert logits.shape == (batch_size, 4, 11), (
        f"logits shape expected (batch, 4, 11), got {logits.shape}"
    )
    assert loss.dim() == 0, "loss should be scalar"
    assert loss.item() >= 0.0

    loss.backward()
    # natal_emb is non-leaf; gradients flow to encoder. Verify head receives gradients.
    for name, p in head.named_parameters():
        assert p.grad is not None, f"head parameter {name} has no gradient"
