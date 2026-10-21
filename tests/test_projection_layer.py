"""
Tests for bazi_model.projection.ProjectionLayer (per-slot LLaVA-style).

Input shape (batch, n_slots, d_in_per_slot); output shape
(batch, n_slots * n_virtual_per_slot, 3584).
"""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    from bazi_model.projection import (
        D_IN_PER_SLOT,
        D_LLM,
        N_SLOTS,
        N_VIRTUAL_PER_SLOT,
        ProjectionLayer,
    )
    _MLX_AVAILABLE = True
except ImportError:
    mx = None
    D_IN_PER_SLOT = D_LLM = N_SLOTS = N_VIRTUAL_PER_SLOT = None
    ProjectionLayer = None
    _MLX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _MLX_AVAILABLE, reason="mlx not available")


def test_projection_output_shape_default_batch_1():
    """Default config (7 slots * 9 virtual) yields (1, 63, 3584) for batch 1."""
    layer = ProjectionLayer()
    x = mx.zeros((1, N_SLOTS, D_IN_PER_SLOT))
    out = layer(x)
    assert out.shape == (1, N_SLOTS * N_VIRTUAL_PER_SLOT, D_LLM)
    assert layer.n_virtual_tokens == N_SLOTS * N_VIRTUAL_PER_SLOT


def test_projection_output_shape_default_batch_4():
    """Default config preserves the batch axis."""
    layer = ProjectionLayer()
    x = mx.zeros((4, N_SLOTS, D_IN_PER_SLOT))
    out = layer(x)
    assert out.shape == (4, N_SLOTS * N_VIRTUAL_PER_SLOT, D_LLM)


def test_projection_respects_n_virtual_per_slot():
    """n_slots and n_virtual_per_slot are configurable; total = product."""
    layer = ProjectionLayer(d_in_per_slot=D_IN_PER_SLOT, d_llm=D_LLM, n_slots=3, n_virtual_per_slot=2)
    x = mx.zeros((2, 3, D_IN_PER_SLOT))
    out = layer(x)
    assert out.shape == (2, 6, D_LLM)
    assert layer.n_virtual_tokens == 6


def test_projection_rejects_wrong_input_shape():
    """Mismatched n_slots in the input is a hard error rather than silent reshape."""
    layer = ProjectionLayer()
    bad = mx.zeros((1, N_SLOTS - 1, D_IN_PER_SLOT))
    with pytest.raises(ValueError):
        layer(bad)
