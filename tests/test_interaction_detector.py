"""Unit tests for InteractionHead: output (batch, 64) logits, BCE + pos_weight, forward_loss and backward."""
from __future__ import annotations

import pytest

try:
    import torch
    from bazi_model.tasks.interaction_detector import InteractionHead, INTERACTION_DIM
except ImportError:
    torch = None
    InteractionHead = None  # type: ignore[misc, assignment]
    INTERACTION_DIM = 64

pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


def test_interaction_head_output_shape():
    """Forward with batch=2 yields logits shape (2, 64)."""
    batch = 2
    d_in = 256
    head = InteractionHead(d_in=d_in, d_out=INTERACTION_DIM)
    natal_emb = torch.randn(batch, d_in)

    logits = head(natal_emb)

    assert logits.shape == (batch, INTERACTION_DIM), f"expected ({batch}, {INTERACTION_DIM}), got {logits.shape}"


def test_interaction_head_loss_scalar_and_backward():
    """forward_loss returns (loss, logits) and backward runs."""
    batch = 2
    d_in = 256
    head = InteractionHead(d_in=d_in)
    natal_emb = torch.randn(batch, d_in, requires_grad=True)
    batch_dict = {
        "interaction_vector": torch.randint(0, 2, (batch, INTERACTION_DIM)).to(torch.float64),
    }

    loss, logits = head.forward_loss(batch_dict, natal_emb)

    assert loss.dim() == 0, f"expected scalar loss, got shape {loss.shape}"
    assert logits.shape == (batch, INTERACTION_DIM)
    loss.backward()
    assert natal_emb.grad is not None


def test_interaction_head_loss_with_pos_weight_tensor():
    """Loss with pos_weight tensor (64,) runs and backward works."""
    batch = 2
    d_in = 256
    pos_weight = torch.ones(INTERACTION_DIM) * 2.0
    head = InteractionHead(d_in=d_in, pos_weight=pos_weight)
    natal_emb = torch.randn(batch, d_in)
    target = torch.randint(0, 2, (batch, INTERACTION_DIM)).to(torch.float32)
    logits = head(natal_emb)

    loss = head.loss(logits, target)

    assert loss.dim() == 0
    loss.backward()


def test_interaction_head_loss_with_pos_weight_scalar():
    """Loss with pos_weight as float runs."""
    batch = 2
    head = InteractionHead(d_in=256, pos_weight=2.0)
    natal_emb = torch.randn(batch, 256)
    batch_dict = {"interaction_vector": torch.zeros(batch, INTERACTION_DIM)}

    loss, logits = head.forward_loss(batch_dict, natal_emb)

    assert loss.dim() == 0
    assert logits.shape == (batch, INTERACTION_DIM)


def test_interaction_head_sigmoid_inference():
    """Logits -> sigmoid gives values in [0, 1] and shape (batch, 64)."""
    batch = 3
    head = InteractionHead(d_in=256)
    natal_emb = torch.randn(batch, 256)
    logits = head(natal_emb)
    probs = torch.sigmoid(logits)

    assert probs.shape == (batch, INTERACTION_DIM)
    assert (probs >= 0).all() and (probs <= 1).all()
