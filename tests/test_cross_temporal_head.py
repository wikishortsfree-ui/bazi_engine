"""Unit tests for CrossTemporalHead: output shape, hybrid loss, backward."""
from __future__ import annotations

import pytest

try:
    import torch
    from bazi_model.tasks.cross_temporal_detector import (
        CROSS_TEMPORAL_SEGMENT_RANGES,
        CrossTemporalHead,
        DYNAMIC_INTERACTION_DIM,
        NATAL_DIM,
        DYNAMIC_DIM,
        BINARY_INDICES,
        CONTINUOUS_INDICES,
    )
except (ImportError, AttributeError):
    torch = None
    CrossTemporalHead = None  # type: ignore[misc, assignment]

pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


def _make_realistic_target(batch: int) -> torch.Tensor:
    """Target with binary 0/1 for dims [0:60]+[65] and continuous floats for dims [60:65]."""
    target = torch.zeros(batch, DYNAMIC_INTERACTION_DIM)
    target[:, BINARY_INDICES] = torch.randint(0, 2, (batch, len(BINARY_INDICES))).float()
    target[:, CONTINUOUS_INDICES] = torch.randn(batch, len(CONTINUOUS_INDICES)) * 2.0
    return target


def test_cross_temporal_head_output_shape():
    """Forward(natal_emb, dynamic_emb) returns logits (batch, 66)."""
    batch = 4
    head = CrossTemporalHead()
    natal_emb = torch.randn(batch, NATAL_DIM)
    dynamic_emb = torch.randn(batch, DYNAMIC_DIM)

    logits = head(natal_emb, dynamic_emb)

    assert logits.shape == (batch, DYNAMIC_INTERACTION_DIM), (
        f"expected ({batch}, {DYNAMIC_INTERACTION_DIM}), got {logits.shape}"
    )


def test_cross_temporal_head_sigmoid_bounds():
    """Sigmoid(logits) yields values in [0, 1]."""
    batch = 2
    head = CrossTemporalHead()
    natal_emb = torch.randn(batch, NATAL_DIM)
    dynamic_emb = torch.randn(batch, DYNAMIC_DIM)

    logits = head(natal_emb, dynamic_emb)
    probs = torch.sigmoid(logits)

    assert (probs >= 0).all() and (probs <= 1).all()
    assert probs.shape == (batch, DYNAMIC_INTERACTION_DIM)


def test_cross_temporal_head_forward_loss_scalar():
    """forward_loss returns (loss, logits); loss is non-negative with realistic targets."""
    batch = 3
    head = CrossTemporalHead()
    natal_emb = torch.randn(batch, NATAL_DIM)
    dynamic_emb = torch.randn(batch, DYNAMIC_DIM)
    target = _make_realistic_target(batch)
    batch_dict = {"dynamic_interaction_vector": target}

    loss, logits = head.forward_loss(batch_dict, natal_emb, dynamic_emb)

    assert loss.dim() == 0, f"loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0
    assert logits.shape == (batch, DYNAMIC_INTERACTION_DIM)


def test_cross_temporal_head_loss_nonnegative_with_out_of_range_targets():
    """Loss stays non-negative even when elemental_delta targets are far from [0,1]."""
    batch = 4
    head = CrossTemporalHead()
    natal_emb = torch.randn(batch, NATAL_DIM)
    dynamic_emb = torch.randn(batch, DYNAMIC_DIM)
    target = torch.zeros(batch, DYNAMIC_INTERACTION_DIM)
    target[:, CONTINUOUS_INDICES] = torch.tensor([[-5.0, 3.0, -1.2, 0.0, 8.0]] * batch)
    batch_dict = {"dynamic_interaction_vector": target}

    loss, _ = head.forward_loss(batch_dict, natal_emb, dynamic_emb)

    assert loss.item() >= 0, f"loss should be non-negative, got {loss.item()}"


def test_cross_temporal_head_backward():
    """Loss backward runs without error with realistic targets."""
    batch = 2
    head = CrossTemporalHead()
    natal_emb = torch.randn(batch, NATAL_DIM, requires_grad=True)
    dynamic_emb = torch.randn(batch, DYNAMIC_DIM, requires_grad=True)
    target = _make_realistic_target(batch)
    batch_dict = {"dynamic_interaction_vector": target}

    loss, _ = head.forward_loss(batch_dict, natal_emb, dynamic_emb)
    loss.backward()

    assert natal_emb.grad is not None
    assert dynamic_emb.grad is not None


def test_cross_temporal_head_mlp_architecture():
    """Head uses 2-layer MLP (512 -> hidden -> 66)."""
    head = CrossTemporalHead()
    batch = 1
    natal_emb = torch.randn(batch, NATAL_DIM)
    dynamic_emb = torch.randn(batch, DYNAMIC_DIM)

    logits = head(natal_emb, dynamic_emb)

    assert logits.shape == (batch, DYNAMIC_INTERACTION_DIM)
    assert isinstance(head.mlp, torch.nn.Sequential)
    first_linear = head.mlp[0]
    assert isinstance(first_linear, torch.nn.Linear)
    assert first_linear.in_features == NATAL_DIM + DYNAMIC_DIM


def test_cross_temporal_index_coverage():
    """BINARY_INDICES + CONTINUOUS_INDICES cover all 66 dims exactly once."""
    all_indices = sorted(BINARY_INDICES + CONTINUOUS_INDICES)
    assert all_indices == list(range(DYNAMIC_INTERACTION_DIM))


def test_cross_temporal_segment_weights_equivalence():
    """CrossTemporalHead(segment_weights=ones_7) gives same loss as CrossTemporalHead() for same (logits, target)."""
    batch = 4
    ones_7 = [1.0] * len(CROSS_TEMPORAL_SEGMENT_RANGES)
    head_default = CrossTemporalHead()
    head_weighted = CrossTemporalHead(segment_weights=ones_7)
    head_weighted.load_state_dict(head_default.state_dict(), strict=False)

    natal_emb = torch.randn(batch, NATAL_DIM)
    dynamic_emb = torch.randn(batch, DYNAMIC_DIM)
    target = _make_realistic_target(batch)

    logits = head_default(natal_emb, dynamic_emb)
    loss_default = head_default.loss(logits, target, reduction="mean")
    loss_weighted = head_weighted.loss(logits, target, reduction="mean")

    torch.testing.assert_close(loss_default, loss_weighted, rtol=1e-5, atol=1e-5)


def test_cross_temporal_segment_weights_upweight_increases_loss_contribution():
    """Different segment_weights produce different losses; upweighting a segment changes the combined BCE."""
    batch = 4
    head_equal = CrossTemporalHead(segment_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    head_seg0_up = CrossTemporalHead(segment_weights=[3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    head_seg0_up.load_state_dict(head_equal.state_dict(), strict=False)
    head_seg0_up._segment_weights.data = torch.tensor(
        [3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32
    )

    # Fix seed and use data where segment 0 has different BCE from others so weights change the result
    torch.manual_seed(42)
    logits = torch.randn(batch, DYNAMIC_INTERACTION_DIM) * 0.5
    target = _make_realistic_target(batch)

    loss_equal = head_equal.loss(logits, target, reduction="mean")
    loss_seg0_up = head_seg0_up.loss(logits, target, reduction="mean")

    assert loss_equal.dim() == 0 and loss_seg0_up.dim() == 0
    assert loss_equal.item() != loss_seg0_up.item()


def test_cross_temporal_pos_weight_forward_and_loss():
    """CrossTemporalHead(pos_weight=tensor_61) runs forward and loss; higher pos_weight on one dim increases loss when that dim is positive and predicted low."""
    batch = 4
    n_bin = len(BINARY_INDICES)
    pos_weight_ones = torch.ones(n_bin, dtype=torch.float32)
    pos_weight_high_on_last = torch.ones(n_bin, dtype=torch.float32)
    pos_weight_high_on_last[-1] = 10.0

    head_default = CrossTemporalHead()
    head_pos = CrossTemporalHead(pos_weight=pos_weight_ones)
    head_pos_high = CrossTemporalHead(pos_weight=pos_weight_high_on_last)
    head_pos.load_state_dict(head_default.state_dict(), strict=False)
    head_pos_high.load_state_dict(head_default.state_dict(), strict=False)

    natal_emb = torch.randn(batch, NATAL_DIM)
    dynamic_emb = torch.randn(batch, DYNAMIC_DIM)
    target = _make_realistic_target(batch)

    logits = head_default(natal_emb, dynamic_emb)
    loss_default = head_default.loss(logits, target, reduction="mean")
    loss_pos = head_pos.loss(logits, target, reduction="mean")
    torch.testing.assert_close(loss_default, loss_pos, rtol=1e-5, atol=1e-5)

    # Make last binary dim (useful_god_active) positive and logit very negative so BCE is high for that dim
    target_positive_last = target.clone()
    target_positive_last[:, 65] = 1.0
    logits_low_last = logits.clone()
    logits_low_last[:, 65] = -5.0

    loss_high_weight = head_pos_high.loss(logits_low_last, target_positive_last, reduction="mean")
    loss_low_weight = head_pos.loss(logits_low_last, target_positive_last, reduction="mean")
    assert loss_high_weight > loss_low_weight
