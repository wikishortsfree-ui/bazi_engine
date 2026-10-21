"""Unit tests for ElementStrengthHead: output shape (batch, 5), MSE loss, backward."""
from __future__ import annotations

import pytest

try:
    import torch
    from bazi_model.tasks import ElementStrengthHead
except ImportError:
    torch = None
    ElementStrengthHead = None  # type: ignore[misc, assignment]

pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


def test_element_strength_head_output_shape():
    """Forward with batch=2 yields (2, 5) probabilities."""
    batch = 2
    head = ElementStrengthHead(d_natal=256, n_elements=5)
    natal_emb = torch.randn(batch, 256)

    probs = head(natal_emb)

    assert probs.shape == (batch, 5), f"expected (2, 5), got {probs.shape}"


def test_element_strength_head_predict_sum_one():
    """forward/predict return (batch, 5) with rows summing to 1."""
    batch = 3
    head = ElementStrengthHead(d_natal=256, n_elements=5)
    natal_emb = torch.randn(batch, 256)

    probs = head.predict(natal_emb)

    assert probs.shape == (batch, 5)
    row_sums = probs.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones(batch, device=probs.device))


def test_element_strength_head_loss_forward():
    """loss(pred_probs, target) returns a scalar; with ordinal_loss_weight=0 equals MSE."""
    batch = 2
    head = ElementStrengthHead(d_natal=256, n_elements=5, ordinal_loss_weight=0.0)
    natal_emb = torch.randn(batch, 256)
    pred_probs = head(natal_emb)
    target = torch.softmax(torch.randn(batch, 5), dim=-1)

    loss = head.loss(pred_probs, target)

    assert loss.dim() == 0, "loss should be scalar"
    expected = torch.nn.functional.mse_loss(pred_probs, target)
    torch.testing.assert_close(loss, expected)


def test_element_strength_head_forward_loss():
    """forward_loss(natal_emb, target) returns (loss, probs) and loss is scalar."""
    batch = 2
    head = ElementStrengthHead(d_natal=256, n_elements=5)
    natal_emb = torch.randn(batch, 256)
    target = torch.softmax(torch.randn(batch, 5), dim=-1)

    loss, probs = head.forward_loss(natal_emb, target)

    assert probs.shape == (batch, 5)
    assert loss.dim() == 0
    torch.testing.assert_close(loss, head.loss(probs, target))


def test_element_strength_head_backward():
    """Backward through loss runs without error."""
    batch = 2
    head = ElementStrengthHead(d_natal=256, n_elements=5)
    natal_emb = torch.randn(batch, 256, requires_grad=True)
    target = torch.softmax(torch.randn(batch, 5), dim=-1)

    loss, probs = head.forward_loss(natal_emb, target)
    loss.backward()

    assert natal_emb.grad is not None
    assert natal_emb.grad.shape == natal_emb.shape


def test_element_strength_head_ordinal_loss_increases_total():
    """With ordinal_loss_weight > 0, total loss >= MSE (ordinal term is non-negative)."""
    batch = 2
    head = ElementStrengthHead(d_natal=256, n_elements=5, ordinal_loss_weight=0.15)
    pred_probs = torch.softmax(torch.randn(batch, 5), dim=-1)
    target = torch.softmax(torch.randn(batch, 5), dim=-1)

    loss = head.loss(pred_probs, target)
    mse = torch.nn.functional.mse_loss(pred_probs, target)

    assert loss.dim() == 0
    assert loss >= mse - 1e-6  # ordinal term is non-negative (softplus)
