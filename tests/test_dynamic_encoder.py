"""Unit tests for DynamicEncoder: all present, 大运+流年 only, natal-only, masking correctness."""
from __future__ import annotations

import pytest

try:
    import torch
    from bazi_model.encoders.dynamic_encoder import DynamicEncoder
except ImportError:
    torch = None
    DynamicEncoder = None  # type: ignore[misc, assignment]

pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


def _make_dynamic_inputs(
    batch: int,
    dayun: bool = True,
    liunian: bool = True,
    liuyue: bool = True,
    device: torch.device | None = None,
):
    """Build inputs for DynamicEncoder. When a slot is absent, IDs are -1 and mask 0."""
    if device is None:
        device = torch.device("cpu")
    mask = torch.tensor(
        [[1 if dayun else 0, 1 if liunian else 0, 1 if liuyue else 0]] * batch,
        device=device,
        dtype=torch.long,
    )
    # Use valid IDs when present, -1 when absent
    def id_or_missing(present: bool, val: int = 0):
        return val if present else -1

    return {
        "dayun_stem_id": torch.tensor([id_or_missing(dayun)] * batch, device=device, dtype=torch.long),
        "dayun_branch_id": torch.tensor([id_or_missing(dayun)] * batch, device=device, dtype=torch.long),
        "liunian_stem_id": torch.tensor([id_or_missing(liunian, 1)] * batch, device=device, dtype=torch.long),
        "liunian_branch_id": torch.tensor([id_or_missing(liunian, 1)] * batch, device=device, dtype=torch.long),
        "liuyue_stem_id": torch.tensor([id_or_missing(liuyue, 2)] * batch, device=device, dtype=torch.long),
        "liuyue_branch_id": torch.tensor([id_or_missing(liuyue, 2)] * batch, device=device, dtype=torch.long),
        "pillar_presence_mask": mask,
    }


def test_dynamic_encoder_all_present():
    """All three temporal pillars present (大运+流年+流月) → output shape (batch, 256)."""
    batch = 2
    encoder = DynamicEncoder()
    inputs = _make_dynamic_inputs(batch, dayun=True, liunian=True, liuyue=True)

    out = encoder(**inputs)

    assert out.shape == (batch, 256), f"expected ({batch}, 256), got {out.shape}"


def test_dynamic_encoder_dayun_liunian_only():
    """Only 大运+流年 present, 流月 absent → output shape (batch, 256)."""
    batch = 2
    encoder = DynamicEncoder()
    inputs = _make_dynamic_inputs(batch, dayun=True, liunian=True, liuyue=False)

    out = encoder(**inputs)

    assert out.shape == (batch, 256)


def test_dynamic_encoder_natal_only():
    """All temporal slots absent (natal-only) → null_dynamic_emb path, output (batch, 256)."""
    batch = 2
    encoder = DynamicEncoder()
    inputs = _make_dynamic_inputs(batch, dayun=False, liunian=False, liuyue=False)

    out = encoder(**inputs)

    assert out.shape == (batch, 256)


def test_dynamic_encoder_masking_correctness():
    """When a slot is absent, changing its stem/branch IDs must not change output (no leakage)."""
    batch = 2
    encoder = DynamicEncoder()
    encoder.eval()
    # 流月 absent: mask [1, 1, 0]
    inputs_a = _make_dynamic_inputs(batch, dayun=True, liunian=True, liuyue=False)
    inputs_b = _make_dynamic_inputs(batch, dayun=True, liunian=True, liuyue=False)
    # Override liuyue IDs to different values; they must be ignored when mask[2]==0
    inputs_b["liuyue_stem_id"] = torch.tensor([5, 7], dtype=torch.long)
    inputs_b["liuyue_branch_id"] = torch.tensor([8, 10], dtype=torch.long)

    with torch.no_grad():
        out_a = encoder(**inputs_a)
        out_b = encoder(**inputs_b)

    torch.testing.assert_close(out_a, out_b, msg="Output must not depend on absent slot IDs")


def test_dynamic_encoder_batch_one_single_sample_mask():
    """Single sample with 1D pillar_presence_mask (3,) → output (1, 256)."""
    encoder = DynamicEncoder()
    inputs = _make_dynamic_inputs(1, dayun=True, liunian=True, liuyue=True)
    inputs["pillar_presence_mask"] = torch.tensor([1, 1, 1], dtype=torch.long)

    out = encoder(**inputs)

    assert out.shape == (1, 256)


def test_dynamic_encoder_parameter_count():
    """DynamicEncoder parameter count is roughly 1.5M–2.5M (sanity check)."""
    encoder = DynamicEncoder()
    nparams = sum(p.numel() for p in encoder.parameters())
    assert 1_500_000 <= nparams <= 2_500_000, (
        f"expected param count in [1.5M, 2.5M], got {nparams:,} (~{nparams/1e6:.2f}M)"
    )