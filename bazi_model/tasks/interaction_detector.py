"""
Interaction head: predict 64-dim binary interaction vector from natal embedding.

Input: natal_emb (batch, 256).
Output: logits (batch, 64); sigmoid gives probabilities. Aligns with
docs/binary_vector_and_positions_spec.md (64-dim layout).
Target: interaction_vector (batch, 64), 0.0/1.0 from ChartDataset.
Loss: BCEWithLogitsLoss with optional pos_weight for rare interaction bits.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

INTERACTION_DIM = 64


class InteractionHead(nn.Module):
    """
    Predicts 64-dim interaction presence from natal embedding.

    Forward returns logits (batch, 64). Use sigmoid for probabilities.
    Loss is BCEWithLogitsLoss; pos_weight can be (64,) or scalar to upweight
    rare positive bits (e.g. (1 - p) / p per dimension from dataset stats).
    """

    def __init__(
        self,
        d_in: int = 256,
        d_out: int = INTERACTION_DIM,
        pos_weight: Optional[Union[float, torch.Tensor]] = None,
    ):
        """
        Args:
            d_in: Natal embedding dimension (default 256).
            d_out: Output dimension (default 64, binary vector spec).
            pos_weight: Optional weight for positive class. If float, same for
                all 64 dims; if Tensor, shape (64,). Use (1-p)/p per dim from
                dataset for rare-bit upweighting. None = no weighting.
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.linear = nn.Linear(d_in, d_out)
        if pos_weight is not None:
            if isinstance(pos_weight, (int, float)):
                self.register_buffer("_pos_weight", torch.full((d_out,), float(pos_weight)))
            else:
                assert pos_weight.shape == (d_out,), f"pos_weight shape {pos_weight.shape} != ({d_out},)"
                self.register_buffer("_pos_weight", pos_weight.to(torch.float32))

    def forward(self, natal_emb: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """
        Args:
            natal_emb: (batch, d_in) natal embedding.

        Returns:
            Logits (batch, 64). Apply sigmoid for probabilities.
        """
        return self.linear(natal_emb)  # (batch, 64)

    def loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        BCEWithLogitsLoss between logits and target.

        Args:
            logits: (batch, 64) from forward().
            target: (batch, 64) 0.0/1.0 interaction_vector.
            reduction: 'mean', 'sum', or 'none'.

        Returns:
            Scalar loss (if reduction='mean' or 'sum').
        """
        if target.dtype != logits.dtype:
            target = target.to(logits.dtype)
        pos_weight = getattr(self, "_pos_weight", None)
        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
            reduction=reduction,
        )

    def forward_loss(
        self,
        batch: dict[str, torch.Tensor],
        natal_emb: torch.Tensor,
        *,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward + loss in one call for trainer use.

        Args:
            batch: Must contain "interaction_vector" (batch, 64).
            natal_emb: (batch, d_in).
            reduction: Passed to loss().

        Returns:
            (loss_scalar, logits). Unified convention: loss first, then prediction.
        """
        logits = self.forward(natal_emb)
        target = batch["interaction_vector"]
        loss = self.loss(logits, target, reduction=reduction)
        return loss, logits
