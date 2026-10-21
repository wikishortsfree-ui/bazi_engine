"""
Phase 2 LM loss: masked causal cross-entropy for BaziLLM training (MLX).

Loss is computed only at positions where loss_mask == 1 (reasoning + output).
Mean over valid positions. Option A: labels are shifted target (labels[i] = token at i+1).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def compute_lm_loss(
    logits: mx.array,
    labels: mx.array,
    loss_mask: mx.array,
    *,
    axis: int = -1,
    eps: float = 1e-8,
) -> mx.array:
    """
    Causal LM cross-entropy; only positions with loss_mask=1 contribute.

    Args:
        logits: (batch, seq_len, vocab_size) from BaziLLM.
        labels: (batch, seq_len) int32, shifted target; use -100 or pad_id for ignore.
        loss_mask: (batch, seq_len) float 0/1; 1 = compute loss at this position.
        axis: Axis over which to compute softmax (default -1).
        eps: Small constant to avoid division by zero when no valid positions.

    Returns:
        Scalar loss (mean over valid positions).
    """
    # Per-position CE: (batch, seq_len)
    ce = nn.losses.cross_entropy(logits, labels, axis=axis, reduction="none")
    # Only where loss_mask == 1
    mask_f = loss_mask.astype(ce.dtype)
    masked = ce * mask_f
    n = mx.maximum(mask_f.sum(), eps)
    return masked.sum() / n
