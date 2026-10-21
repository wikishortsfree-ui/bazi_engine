"""Unit tests for UsefulGodHead: output shape (batch, 6), CrossEntropy loss, -1→5 mapping, backward."""
from __future__ import annotations

import pytest

try:
    import torch
    from bazi_model.tasks import UsefulGodHead
except ImportError:
    torch = None
    UsefulGodHead = None  # type: ignore[misc, assignment]

pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


def test_useful_god_head_output_shape():
    """Forward with batch=2 yields logits shape (2, 6)."""
    batch = 2
    input_dim = 256
    head = UsefulGodHead(input_dim=input_dim, n_classes=6)
    natal_emb = torch.randn(batch, input_dim)

    logits = head(natal_emb)

    assert logits.shape == (batch, 6), f"expected (2, 6), got {logits.shape}"


def test_useful_god_head_loss_scalar():
    """loss(logits, useful_god_id) returns a scalar."""
    batch = 3
    head = UsefulGodHead(input_dim=256, n_classes=6)
    logits = torch.randn(batch, 6)
    # (batch, 1) targets: 0–4 or -1
    useful_god_id = torch.tensor([[0], [2], [4]], dtype=torch.long)

    loss = head.loss(logits, useful_god_id)

    assert loss.dim() == 0, "loss should be scalar"
    assert loss.item() >= 0


def test_useful_god_head_loss_ambiguous_mapped_to_five():
    """Target -1 is mapped to class 5; loss is computable and matches manual CE."""
    batch = 2
    head = UsefulGodHead(input_dim=256, n_classes=6)
    logits = torch.randn(batch, 6)
    # One unambiguous (0), one ambiguous (-1 → 5)
    useful_god_id = torch.tensor([[0], [-1]], dtype=torch.long)

    loss = head.loss(logits, useful_god_id)

    assert loss.dim() == 0
    manual_target = torch.tensor([0, 5], dtype=torch.long)
    expected = torch.nn.functional.cross_entropy(logits, manual_target)
    torch.testing.assert_close(loss, expected)


def test_useful_god_head_forward_loss():
    """forward_loss(natal_emb, useful_god_id) returns (loss, logits) and loss is scalar."""
    batch = 2
    head = UsefulGodHead(input_dim=256, n_classes=6)
    natal_emb = torch.randn(batch, 256)
    useful_god_id = torch.tensor([[1], [3]], dtype=torch.long)

    loss, logits = head.forward_loss(natal_emb, useful_god_id)

    assert logits.shape == (batch, 6)
    assert loss.dim() == 0
    torch.testing.assert_close(loss, head.loss(logits, useful_god_id))


def test_useful_god_head_backward():
    """Backward through loss runs without error."""
    batch = 2
    head = UsefulGodHead(input_dim=256, n_classes=6)
    natal_emb = torch.randn(batch, 256, requires_grad=True)
    useful_god_id = torch.tensor([[0], [4]], dtype=torch.long)

    loss, logits = head.forward_loss(natal_emb, useful_god_id)
    loss.backward()

    assert natal_emb.grad is not None
    assert natal_emb.grad.shape == natal_emb.shape
