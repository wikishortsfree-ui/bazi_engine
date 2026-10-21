"""Unit tests for TenGodHead: output shape (batch, 4, 11), CrossEntropy loss, backward."""
from __future__ import annotations

import pytest

try:
    import torch
    from bazi_model.tasks import TenGodHead
except ImportError:
    torch = None
    TenGodHead = None  # type: ignore[misc, assignment]

pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


def test_ten_god_head_output_shape():
    """Forward with batch=2 yields logits shape (2, 4, 11)."""
    batch = 2
    d_model = 256
    head = TenGodHead(d_model=d_model)
    natal_emb = torch.randn(batch, d_model)

    logits = head(natal_emb)

    assert logits.shape == (batch, 4, 11), f"expected (2, 4, 11), got {logits.shape}"


def test_ten_god_head_loss_forward():
    """Loss is scalar and accepts target (batch, 4) with values 0..10."""
    batch = 2
    head = TenGodHead(d_model=256)
    natal_emb = torch.randn(batch, 256)
    ten_god_stem_ids = torch.randint(0, 11, (batch, 4), dtype=torch.long)

    logits = head(natal_emb)
    loss = head.loss(logits, ten_god_stem_ids)

    assert loss.dim() == 0, "loss should be scalar"
    assert loss.item() >= 0.0


def test_ten_god_head_loss_backward():
    """Backward runs without error; gradients flow to head parameters."""
    batch = 2
    head = TenGodHead(d_model=256)
    natal_emb = torch.randn(batch, 256, requires_grad=True)
    ten_god_stem_ids = torch.randint(0, 11, (batch, 4), dtype=torch.long)

    logits = head(natal_emb)
    loss = head.loss(logits, ten_god_stem_ids)
    loss.backward()

    assert natal_emb.grad is not None
    for name, p in head.named_parameters():
        assert p.grad is not None, f"no grad for {name}"


def test_ten_god_head_forward_loss():
    """forward_loss returns (loss, logits) with correct shapes."""
    batch = 3
    head = TenGodHead(d_model=256)
    natal_emb = torch.randn(batch, 256)
    ten_god_stem_ids = torch.randint(0, 11, (batch, 4), dtype=torch.long)

    loss, logits = head.forward_loss(natal_emb, ten_god_stem_ids)

    assert loss.dim() == 0
    assert logits.shape == (batch, 4, 11)
    assert head.loss(logits, ten_god_stem_ids).allclose(loss)


def test_ten_god_head_day_position_target():
    """Target with 日主 (class 10) in position 2 is handled correctly."""
    batch = 1
    head = TenGodHead(d_model=256)
    natal_emb = torch.randn(batch, 256)
    # day position (index 2) = 日主 = 10
    ten_god_stem_ids = torch.tensor([[3, 7, 10, 1]], dtype=torch.long)  # (1, 4)

    logits = head(natal_emb)
    loss = head.loss(logits, ten_god_stem_ids)

    assert loss.dim() == 0
    assert not torch.isnan(loss) and not torch.isinf(loss)
