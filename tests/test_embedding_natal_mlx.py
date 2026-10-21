"""Tests for ``EmbeddingNatalMLX``: shape contract, per-pillar identity, and
chart-summary sensitivity to the ``interaction_vector`` slot.

These confirm the two design claims in the plan:
- swapping the day-stem id changes only ``slot[3]`` (per-pillar identity);
- toggling the natal-internal interaction vector changes ``slot[0]``
  (chart-summary slot fed by ``ChartSummaryHead``).
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx

    from bazi_model.encoders.embedding_natal_mlx import (
        EMBEDDING_NATAL_INPUT_KEYS,
        EmbeddingNatalMLX,
    )

    _MLX_AVAILABLE = True
except ImportError:
    mx = None
    EmbeddingNatalMLX = None
    EMBEDDING_NATAL_INPUT_KEYS = ()
    _MLX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _MLX_AVAILABLE, reason="mlx not available")

D_MODEL = 256


def _base_inputs(batch: int = 1) -> dict:
    return {
        "stem_ids": mx.zeros((batch, 4), dtype=mx.int32),
        "branch_ids": mx.zeros((batch, 4), dtype=mx.int32),
        "stem_elements": mx.zeros((batch, 4), dtype=mx.int32),
        "stem_polarities": mx.zeros((batch, 4), dtype=mx.int32),
        "pillar_positions": mx.broadcast_to(mx.array([[0, 1, 2, 3]]), (batch, 4)),
        "interaction_vector": mx.zeros((batch, 64)),
        "elemental_strengths": mx.zeros((batch, 5)),
        "day_master_strength": mx.zeros((batch, 1)),
        "useful_god_id": mx.full((batch,), -1, dtype=mx.int32),
    }


def test_shape_and_keys():
    module = EmbeddingNatalMLX()
    out = module(**_base_inputs(batch=2))
    assert out.shape == (2, 5, D_MODEL)
    # natal_input_keys is the 5 base + 4 chart-summary scalars
    assert set(EMBEDDING_NATAL_INPUT_KEYS) == set(module.natal_input_keys)
    assert len(module.natal_input_keys) == 9


def test_per_pillar_identity_changes_only_day_slot():
    """Two charts that differ only in day stem produce different slot 3 outputs."""
    module = EmbeddingNatalMLX()
    a = _base_inputs(batch=1)
    b = _base_inputs(batch=1)
    new_stem = mx.array([[0, 0, 7, 0]], dtype=mx.int32)
    b["stem_ids"] = new_stem

    out_a = np.array(module(**a))
    out_b = np.array(module(**b))

    # slot 3 (day pillar = position 2 -> slot index 3) must differ.
    assert not np.allclose(out_a[0, 3], out_b[0, 3])
    # Other pillar slots (1=year, 2=month, 4=hour) should be identical.
    for slot in (1, 2, 4):
        assert np.allclose(out_a[0, slot], out_b[0, slot]), f"slot {slot} changed unexpectedly"


def test_chart_summary_sensitive_to_interaction_vector():
    """Slot 0 (chart-summary) changes when ``interaction_vector`` flips a bit."""
    module = EmbeddingNatalMLX()
    a = _base_inputs(batch=1)
    b = _base_inputs(batch=1)
    iv = np.zeros((1, 64), dtype=np.float32)
    iv[0, 7] = 1.0
    b["interaction_vector"] = mx.array(iv)

    out_a = np.array(module(**a))
    out_b = np.array(module(**b))

    # Slot 0 must differ.
    assert not np.allclose(out_a[0, 0], out_b[0, 0])
    # Pillar slots 1-4 must be unchanged because they don't see the chart summary.
    for slot in (1, 2, 3, 4):
        assert np.allclose(out_a[0, slot], out_b[0, slot]), f"slot {slot} changed unexpectedly"


def test_useful_god_unknown_handled():
    """``useful_god_id == -1`` maps to the unknown vocab row without error."""
    module = EmbeddingNatalMLX()
    inputs = _base_inputs(batch=2)
    inputs["useful_god_id"] = mx.array([-1, 3], dtype=mx.int32)
    out = module(**inputs)
    assert out.shape == (2, 5, D_MODEL)
    assert not np.any(np.isnan(np.array(out)))
