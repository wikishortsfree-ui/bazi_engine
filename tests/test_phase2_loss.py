"""
Tests for Phase 2 LM loss: mask correctness and zero gradients on masked positions.

Requires: mlx. Skips when MLX is not available.
"""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    from bazi_model.training.phase2_loss import compute_lm_loss
    _MLX_AVAILABLE = True
except ImportError:
    mx = None
    compute_lm_loss = None
    _MLX_AVAILABLE = False


def _make_batch(batch: int = 2, seq_len: int = 6, vocab_size: int = 4, seed: int = 42):
    """Logits (B,S,V), labels (B,S) int32, loss_mask (B,S) 0/1. Half positions masked out."""
    rng = mx.random.key(seed)
    logits = mx.random.normal((batch, seq_len, vocab_size), key=rng).astype(mx.float32)
    rng, _ = mx.random.split(rng)
    labels = mx.random.randint(0, vocab_size, (batch, seq_len), key=rng).astype(mx.int32)
    # mask: 0 for first half of seq, 1 for second half (so loss only on second half)
    mask = mx.zeros((batch, seq_len), dtype=mx.float32)
    half = seq_len // 2
    mask = mx.concatenate([mx.zeros((batch, half)), mx.ones((batch, seq_len - half))], axis=1)
    return logits, labels, mask


@pytest.mark.skipif(not _MLX_AVAILABLE, reason="MLX not available")
def test_loss_mask_correctness():
    """Loss is unchanged when logits are modified only at positions where loss_mask=0."""
    logits, labels, loss_mask = _make_batch(batch=2, seq_len=8, vocab_size=5)
    loss1 = compute_lm_loss(logits, labels, loss_mask)

    # Perturb logits only at masked-out positions (first half)
    half = 8 // 2
    logits_perturbed = mx.array(logits)
    # In MLX arrays are immutable in the sense we need to create a new array for "in-place" change
    perturb = mx.ones_like(logits) * 100.0  # large change
    # Zero out perturb where mask=1 so we only change mask=0 positions
    mask_bc = mx.broadcast_to(loss_mask[:, :, None], logits.shape)
    perturb = perturb * (1.0 - mask_bc)
    logits_perturbed = logits + perturb

    loss2 = compute_lm_loss(logits_perturbed, labels, loss_mask)
    mx.eval(loss1, loss2)
    assert float(loss1) == pytest.approx(float(loss2), rel=1e-5, abs=1e-5), (
        "Loss should be unchanged when only masked-out positions are modified"
    )


@pytest.mark.skipif(not _MLX_AVAILABLE, reason="MLX not available")
def test_gradients_zero_on_masked_positions():
    """Gradients of loss w.r.t. logits are zero where loss_mask=0."""
    logits, labels, loss_mask = _make_batch(batch=2, seq_len=6, vocab_size=4)

    def loss_fn(logits):
        return compute_lm_loss(logits, labels, loss_mask)

    grad_fn = mx.grad(loss_fn)
    grad_logits = grad_fn(logits)
    mx.eval(grad_logits)

    # grad_logits shape (batch, seq_len, vocab_size). Where loss_mask[b,i]=0, grad_logits[b,i,:] should be 0
    half = 6 // 2
    # First half: mask=0
    grad_at_masked_out = grad_logits[:, :half, :]  # (batch, 3, vocab)
    norm_masked_out = float(mx.sum(mx.square(grad_at_masked_out)))
    assert norm_masked_out == pytest.approx(0.0, abs=1e-5), (
        "Gradient at positions with loss_mask=0 should be zero"
    )

    # Second half: mask=1 — gradient should be non-zero (at least one position)
    grad_at_masked_in = grad_logits[:, half:, :]
    norm_masked_in = float(mx.sum(mx.square(grad_at_masked_in)))
    assert norm_masked_in > 1e-6, "Gradient at positions with loss_mask=1 should be non-zero"
