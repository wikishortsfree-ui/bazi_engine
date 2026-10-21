"""Unit tests for NatalEncoder: output shape (batch, 256) and optional param count ~2M."""
from __future__ import annotations

import pytest

try:
    import torch
    from bazi_model.encoders.natal_encoder import NatalEncoder
except ImportError:
    torch = None
    NatalEncoder = None  # type: ignore[misc, assignment]

pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


def _make_natal_inputs(batch: int = 2, device: torch.device | None = None):
    """Minimal inputs only (no ten_god, interaction, strength, useful_god)."""
    if device is None:
        device = torch.device("cpu")
    return {
        "stem_ids": torch.randint(0, 10, (batch, 4), device=device, dtype=torch.long),
        "branch_ids": torch.randint(0, 12, (batch, 4), device=device, dtype=torch.long),
        "stem_elements": torch.randint(0, 5, (batch, 4), device=device, dtype=torch.long),
        "stem_polarities": torch.randint(0, 2, (batch, 4), device=device, dtype=torch.long),
        "pillar_positions": torch.tensor([[0, 1, 2, 3]] * batch, device=device, dtype=torch.long),
    }


def test_natal_encoder_output_shape():
    """Forward with batch=2 yields natal_emb shape (2, 256)."""
    batch = 2
    encoder = NatalEncoder(d_emb=32, d_model=256)
    inputs = _make_natal_inputs(batch=batch)

    out = encoder(**inputs)

    assert out.shape == (batch, 256), f"expected (2, 256), got {out.shape}"


def test_natal_encoder_parameter_count():
    """NatalEncoder parameter count is roughly 1M–2M (no interaction encoder)."""
    encoder = NatalEncoder(d_emb=32, d_model=256)
    nparams = sum(p.numel() for p in encoder.parameters())

    assert 1_000_000 <= nparams <= 2_500_000, (
        f"expected param count in [1M, 2.5M], got {nparams:,} (~{nparams/1e6:.2f}M)"
    )
