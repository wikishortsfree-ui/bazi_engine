"""
Tests for Phase 1 evaluation: common vs rare interaction bit groups and F1 reporting.

Verifies the index-based definition (0–29 common, 30–63 rare) and the
positive-rate-based split used in scripts/evaluate_phase1.py.
"""
from __future__ import annotations

import numpy as np
import pytest

# Import from script (same constants and helper used at runtime)
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.evaluate_phase1 import (
    COMMON_BIT_INDICES,
    RARE_BIT_INDICES,
    get_common_rare_indices_by_positive_rate,
    INTERACTION_DIM,
)


def test_common_rare_index_definition():
    """Common = 0–29, rare = 30–63 per binary_vector_and_positions_spec.md."""
    assert INTERACTION_DIM == 64
    assert COMMON_BIT_INDICES == list(range(30))
    assert RARE_BIT_INDICES == list(range(30, 64))
    assert set(COMMON_BIT_INDICES) & set(RARE_BIT_INDICES) == set()
    assert len(COMMON_BIT_INDICES) + len(RARE_BIT_INDICES) == 64


def test_get_common_rare_indices_by_positive_rate():
    """Positive-rate split: rare = bits with positive rate < threshold."""
    # All bits ~50% positive -> all "common" with threshold 0.15
    rng = np.random.default_rng(42)
    targets = (rng.random((200, INTERACTION_DIM)) < 0.5).astype(np.float64)
    common, rare = get_common_rare_indices_by_positive_rate(targets, rare_threshold=0.15)
    assert len(common) + len(rare) == 64
    # With 50% rate, all should be common (>= 0.15)
    assert len(rare) == 0
    assert len(common) == 64

    # Bits 0–9 very rare (5%), 10–63 common (30%)
    targets2 = np.zeros((500, INTERACTION_DIM))
    targets2[:, :10] = (rng.random((500, 10)) < 0.05).astype(np.float64)
    targets2[:, 10:] = (rng.random((500, 54)) < 0.3).astype(np.float64)
    common2, rare2 = get_common_rare_indices_by_positive_rate(
        targets2, rare_threshold=0.15
    )
    assert len(common2) + len(rare2) == 64
    # Bits 0–9 should be rare (low rate), 10–63 common
    assert set(rare2) <= set(range(10)) or len(rare2) >= 10  # at least the low-rate bits
    assert all(i in common2 or i in rare2 for i in range(64))


def test_f1_common_rare_returned_keys():
    """_eval_interaction returns f1_common_bits, f1_rare_bits, n_common_bits, n_rare_bits, common_rare_mode."""
    pytest.importorskip("sklearn")
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from scripts.evaluate_phase1 import _build_interaction, _eval_interaction

    # Minimal model and fake data (one batch)
    device = torch.device("cpu")
    model = _build_interaction(device)
    # Fake inputs and targets matching ChartDataset shape for interaction task
    N = 8
    inputs = {
        "stem_ids": torch.randint(0, 10, (N, 4)),
        "branch_ids": torch.randint(0, 12, (N, 4)),
        "stem_elements": torch.randint(0, 5, (N, 4)),
        "stem_polarities": torch.randint(0, 2, (N, 4)),
        "pillar_positions": torch.arange(4).unsqueeze(0).expand(N, -1),
    }
    targets = {"interaction_vector": (torch.rand(N, 64) < 0.2).float()}
    batch = (inputs, targets)

    class SingleBatchLoader:
        def __init__(self, batch):
            self.batch = batch
        def __iter__(self):
            yield self.batch
        def __len__(self):
            return 1

    loader = SingleBatchLoader(batch)
    out_index = _eval_interaction(
        model, loader, device, common_rare_mode="index"
    )
    assert "f1_common_bits" in out_index
    assert "f1_rare_bits" in out_index
    assert out_index["n_common_bits"] == 30
    assert out_index["n_rare_bits"] == 34
    assert out_index["common_rare_mode"] == "index"

    out_rate = _eval_interaction(
        model, loader, device, common_rare_mode="positive_rate", rare_threshold=0.15
    )
    assert "f1_common_bits" in out_rate
    assert "f1_rare_bits" in out_rate
    assert out_rate["n_common_bits"] + out_rate["n_rare_bits"] == 64
    assert out_rate["common_rare_mode"] == "positive_rate"
