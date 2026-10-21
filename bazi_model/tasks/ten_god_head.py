"""
Ten-god classification head: predicts ten-god stem ID per pillar from natal embedding.

Output shape (batch, 4, 11): 4 pillar positions × 11 classes (比肩…正印 + 日主).
Class index 10 = 日主 (day stem only); positions use 0–9 otherwise.
Loss: CrossEntropy (optionally with class weights, label smoothing) or Focal loss.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

NUM_POSITIONS = 4
NUM_CLASSES = 11  # 0–9 ten-gods + 10 = 日主


class TenGodHead(nn.Module):
    """
    Predicts ten-god stem ID for each of the 4 pillar positions from natal_emb.

    Input: natal_emb (batch, d_model), typically d_model=256 from NatalEncoder.
    Output: logits (batch, 4, 11). Target: ten_god_stem_ids (batch, 4), values in 0..10.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_positions: int = NUM_POSITIONS,
        num_classes: int = NUM_CLASSES,
        class_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_positions = num_positions
        self.num_classes = num_classes
        self.linear = nn.Linear(d_model, num_positions * num_classes)
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight.float())
        else:
            self.class_weight = None
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma

    def forward(self, natal_emb: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """
        Args:
            natal_emb: (batch, d_model)

        Returns:
            Logits (batch, num_positions, num_classes), i.e. (batch, 4, 11).
        """
        batch = natal_emb.size(0)
        out = self.linear(natal_emb)  # (batch, 4*11)
        return out.view(batch, self.num_positions, self.num_classes)

    def loss(
        self,
        logits: torch.Tensor,
        ten_god_stem_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        CrossEntropy or Focal loss between predicted logits and target IDs.

        Args:
            logits: (batch, 4, 11)
            ten_god_stem_ids: (batch, 4), values in 0..10, long dtype.

        Returns:
            Scalar loss.
        """
        batch, num_pos, num_cls = logits.shape
        logits_flat = logits.view(batch * num_pos, num_cls)
        targets_flat = ten_god_stem_ids.reshape(batch * num_pos).long()

        if self.use_focal:
            ce = F.cross_entropy(
                logits_flat,
                targets_flat,
                reduction="none",
                weight=self.class_weight,
                label_smoothing=self.label_smoothing,
            )
            pt = torch.exp(-ce)
            focal_weight = (1 - pt) ** self.focal_gamma
            if self.class_weight is not None:
                alpha_t = self.class_weight[targets_flat]
                focal_weight = alpha_t * focal_weight
            return focal_weight.mul(ce).mean()
        return F.cross_entropy(
            logits_flat,
            targets_flat,
            weight=self.class_weight,
            label_smoothing=self.label_smoothing,
        )

    def forward_loss(
        self,
        natal_emb: torch.Tensor,
        ten_god_stem_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass and loss in one call (for trainer use).

        Returns:
            (loss, logits). Unified convention: loss first, then prediction.
        """
        logits = self.forward(natal_emb)
        loss = self.loss(logits, ten_god_stem_ids)
        return loss, logits
