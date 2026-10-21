"""
Useful god (用神) classifier: predicts 6-class label from natal_emb.

Classes 0–4: 比劫, 食伤, 财, 官杀, 印 (USEFUL_GOD_TO_ID). Class 5 = ambiguous (-1).
Target: useful_god_id (batch, 1), values 0–4 or -1; -1 is mapped to class 5 for loss.
Loss: CrossEntropy or Focal (when focal_gamma set). Class weights (sqrt-inverse freq) applied as alpha in Focal.
Train on all samples; evaluate "unambiguous accuracy" on target != 5.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

# 6 classes: 0–4 (比劫, 食伤, 财, 官杀, 印) + 5 (ambiguous)
N_CLASSES = 6
AMBIGUOUS_CLASS_ID = 5
D_NATAL = 256


class UsefulGodHead(nn.Module):
    """
    Classifies useful god (用神) from natal embedding via 2-layer MLP.
    Output: logits (batch, 6). Class 5 = ambiguous (target -1).
    """

    def __init__(
        self,
        input_dim: int = D_NATAL,
        n_classes: int = N_CLASSES,
        hidden_dim: int = 128,
        class_weight: torch.Tensor | None = None,
        focal_gamma: float | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.focal_gamma = focal_gamma
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight)
        else:
            self.class_weight: torch.Tensor | None = None

    def forward(self, natal_emb: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """
        Args:
            natal_emb: (batch, input_dim) from NatalEncoder.

        Returns:
            Logits (batch, n_classes). Class 5 = ambiguous.
        """
        return self.mlp(natal_emb)

    def loss(
        self,
        logits: torch.Tensor,
        useful_god_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        CrossEntropy or Focal: useful_god_id (batch, 1) with values 0–4 or -1.
        -1 is mapped to class index 5 (ambiguous) so all samples are used.
        When focal_gamma is set, uses Focal loss with class_weight as alpha to down-weight easy examples.

        Args:
            logits: (batch, 6).
            useful_god_id: (batch, 1), long, values in {0, 1, 2, 3, 4, -1}.

        Returns:
            Scalar loss.
        """
        squeezed = useful_god_id.squeeze(-1)
        target = torch.where(
            squeezed == -1,
            torch.full_like(squeezed, AMBIGUOUS_CLASS_ID),
            squeezed,
        )
        if self.focal_gamma is None:
            return F.cross_entropy(logits, target, weight=self.class_weight)

        # Focal: -alpha_t * (1 - p_t)^gamma * log(p_t); class_weight used as alpha
        log_p = F.log_softmax(logits, dim=-1)
        pt = log_p.exp().gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        log_pt = log_p.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        focal_weight = (1.0 - pt) ** self.focal_gamma
        if self.class_weight is not None:
            alpha = self.class_weight.to(logits.device).gather(0, target)
            focal_weight = alpha * focal_weight
        return (-focal_weight * log_pt).mean()

    def forward_loss(
        self,
        natal_emb: torch.Tensor,
        useful_god_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience: forward + loss in one call for trainer use.

        Returns:
            (loss_scalar, logits). Unified convention: loss first, then prediction.
        """
        logits = self.forward(natal_emb)
        loss = self.loss(logits, useful_god_id)
        return loss, logits
