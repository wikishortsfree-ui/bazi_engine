"""
Cross-interaction feature encoder (MLX): 27-dim feature vector → (batch, d_model).

Encodes temporal cross-interaction signals (elemental deltas, clash/combination flags,
useful-god activation) that bazi_engine computes but the dynamic encoder doesn't consume.

Features (27 floats, all zero for natal-only tasks):
  0–4:   elemental_delta (wood/fire/earth/metal/water)
  5:     total_shift_magnitude
  6–9:   natal–liunian branch conflict flags (per natal position)
  10–13: natal–liunian branch combination flags (per natal position)
  14–17: natal–dayun branch conflict flags (per natal position)
  18–21: natal–dayun branch combination flags (per natal position)
  22:    natal–liunian stem clash present
  23:    natal–liunian stem combination present
  24:    natal–dayun stem clash present
  25:    natal–dayun stem combination present
  26:    useful_god_activation (+1 active, -1 suppressed, 0 neutral/unknown)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

CROSS_FEATURE_DIM = 27
D_MODEL = 256


class CrossInteractionEncoderMLX(nn.Module):
    """MLP: 27 → hidden → d_model. Trained from scratch alongside projection."""

    def __init__(
        self,
        d_input: int = CROSS_FEATURE_DIM,
        d_hidden: int = 128,
        d_model: int = D_MODEL,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_input, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (batch, CROSS_FEATURE_DIM) float
        Returns:
            (batch, d_model)
        """
        h = nn.gelu(self.fc1(x))
        h = self.norm(h)
        return self.fc2(h)
