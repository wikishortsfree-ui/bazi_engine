"""
Cross-temporal detector: predict 66-dim dynamic interaction vector from natal + dynamic embeddings.

Input: concat(natal_emb, dynamic_emb) → (batch, 512). Use only when temporal context present.
Output: (batch, 66). Target: dynamic_interaction_vector (batch, 66).

Layout: [0:60] binary interaction bits, [60:65] elemental_delta (continuous), [65] useful_god_active (binary).
Loss: hybrid — BCE on binary dims [0:60]+[65], MSE on continuous dims [60:65].
Optional: segment_weights (7,) and pos_weight (61,) for upweighting weak segments / rare positives.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch import nn

D_NATAL = 256
D_DYNAMIC = 256
D_CONCAT = D_NATAL + D_DYNAMIC
N_DYNAMIC_DIMS = 66

# Index ranges within the 66-dim dynamic_interaction_vector.
_BINARY_INDICES = list(range(0, 60)) + [65]  # 61 binary dims
_CONTINUOUS_INDICES = list(range(60, 65))     # 5 elemental-delta dims

# Segment ranges for the 61 binary dims (for per-segment loss weighting).
# Order: natal×大运, natal×流年, natal×流月, 大运×流年, 大运×流月, 流年×流月, useful_god_active.
CROSS_TEMPORAL_SEGMENT_RANGES = [
    (0, 16),
    (16, 32),
    (32, 48),
    (48, 52),
    (52, 56),
    (56, 60),
    (60, 61),
]

# Aliases for tests / external use
NATAL_DIM = D_NATAL
DYNAMIC_DIM = D_DYNAMIC
DYNAMIC_INTERACTION_DIM = N_DYNAMIC_DIMS
BINARY_INDICES = _BINARY_INDICES
CONTINUOUS_INDICES = _CONTINUOUS_INDICES


class CrossTemporalHead(nn.Module):
    """
    Predicts 66-dim dynamic interaction vector from concatenated natal + dynamic embeddings.

    The 66-dim vector mixes binary interaction flags (dims 0-59, 65) with continuous
    elemental-delta values (dims 60-64). Loss is hybrid: BCE on binary, MSE on continuous.

    Optional segment_weights (length 7) upweight BCE contribution per segment (natal×大运,
    natal×流年, natal×流月, 大运×流年, 大运×流月, 流年×流月, useful_god_active). Optional
    pos_weight (length 61) upweights the positive class per binary dim (e.g. for rare bits).
    Both default to None for backward compatibility; no buffers are registered when unused.
    """

    def __init__(
        self,
        d_natal: int = D_NATAL,
        d_dynamic: int = D_DYNAMIC,
        n_dims: int = N_DYNAMIC_DIMS,
        mse_weight: float = 1.0,
        segment_weights: Optional[Union[Sequence[float], torch.Tensor]] = None,
        pos_weight: Optional[Union[float, torch.Tensor]] = None,
    ):
        super().__init__()
        self.d_natal = d_natal
        self.d_dynamic = d_dynamic
        self.d_concat = d_natal + d_dynamic
        self.n_dims = n_dims
        self.mse_weight = mse_weight
        hidden = self.d_concat // 2  # 256
        self.mlp = nn.Sequential(
            nn.Linear(self.d_concat, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_dims),
        )
        self.register_buffer(
            "_binary_idx", torch.tensor(_BINARY_INDICES, dtype=torch.long)
        )
        self.register_buffer(
            "_continuous_idx", torch.tensor(_CONTINUOUS_INDICES, dtype=torch.long)
        )
        if segment_weights is not None:
            w = (
                torch.tensor(segment_weights, dtype=torch.float32)
                if not isinstance(segment_weights, torch.Tensor)
                else segment_weights.to(torch.float32)
            )
            assert w.shape == (len(CROSS_TEMPORAL_SEGMENT_RANGES),), (
                f"segment_weights must have length {len(CROSS_TEMPORAL_SEGMENT_RANGES)}, got {w.shape}"
            )
            self.register_buffer("_segment_weights", w)
        if pos_weight is not None:
            pw = (
                torch.full((len(_BINARY_INDICES),), float(pos_weight), dtype=torch.float32)
                if isinstance(pos_weight, (int, float))
                else pos_weight.to(torch.float32)
            )
            assert pw.shape == (len(_BINARY_INDICES),), (
                f"pos_weight must have shape ({len(_BINARY_INDICES)},), got {pw.shape}"
            )
            self.register_buffer("_pos_weight", pw)

    def forward(
        self,
        natal_emb: torch.Tensor,
        dynamic_emb: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """
        Args:
            natal_emb: (batch, d_natal)
            dynamic_emb: (batch, d_dynamic); required for this head.
        Returns:
            Raw output (batch, 66). Binary dims are logits; continuous dims are direct predictions.
        """
        if dynamic_emb is None:
            raise ValueError("CrossTemporalHead requires dynamic_emb")
        concat = torch.cat([natal_emb, dynamic_emb], dim=-1)  # (batch, 512)
        return self.mlp(concat)

    def loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Hybrid loss: BCE on binary dims [0:60]+[65], MSE on continuous dims [60:65].
        When segment_weights is set, BCE is computed per segment and combined as a
        weighted average. When pos_weight is set, it is passed to BCE for the 61 dims.
        """
        if target.dtype != logits.dtype:
            target = target.to(logits.dtype)
        logits_bin = logits[:, self._binary_idx]
        target_bin = target[:, self._binary_idx]
        pos_w = getattr(self, "_pos_weight", None)

        segment_weights = getattr(self, "_segment_weights", None)
        if segment_weights is not None:
            # Per-segment BCE then weighted average (by segment size so all 1s matches default mean).
            bce_per_dim = nn.functional.binary_cross_entropy_with_logits(
                logits_bin, target_bin, pos_weight=pos_w, reduction="none"
            )
            bce_per_segment = []
            segment_sizes = []
            for start, end in CROSS_TEMPORAL_SEGMENT_RANGES:
                bce_per_segment.append(bce_per_dim[:, start:end].mean())
                segment_sizes.append(end - start)
            bce_per_segment = torch.stack(bce_per_segment)
            sizes = torch.tensor(
                segment_sizes, dtype=segment_weights.dtype, device=segment_weights.device
            )
            weight_times_size = segment_weights * sizes
            bce = (weight_times_size * bce_per_segment).sum() / weight_times_size.sum()
        else:
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits_bin, target_bin, pos_weight=pos_w, reduction=reduction
            )

        mse = nn.functional.mse_loss(
            logits[:, self._continuous_idx], target[:, self._continuous_idx], reduction=reduction
        )
        return bce + self.mse_weight * mse

    def forward_loss(
        self,
        batch: dict[str, torch.Tensor],
        natal_emb: torch.Tensor,
        dynamic_emb: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward + loss in one call for trainer use.

        Args:
            batch: Must contain "dynamic_interaction_vector" (batch, 66).
            natal_emb: (batch, d_natal).
            dynamic_emb: (batch, d_dynamic); required for this head.

        Returns:
            (loss_scalar, logits). Unified convention: loss first, then prediction.
        """
        logits = self.forward(natal_emb, dynamic_emb=dynamic_emb, **kwargs)
        target = batch["dynamic_interaction_vector"]
        loss = self.loss(logits, target, reduction="mean")
        return loss, logits
