"""
Element strength head: predicts (batch, 5) elemental strengths from natal_emb.

Target: elemental_strengths (batch, 5), float, sum 1.0 per row (ELEMENTS order:
木, 火, 土, 金, 水). Loss: MSE on probabilities + optional pairwise ranking
(ordinal) auxiliary loss. Design rationale: docs/design_decisions.md §1, §12.
"""
from __future__ import annotations

import torch
from torch import nn

N_ELEMENTS = 5
D_NATAL = 256


def _pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    margin_min: float = 0.0,
) -> torch.Tensor:
    """
    Pairwise ranking loss: penalize when pred order of (i, j) disagrees with target.

    For each pair (i, j) where target[i] > target[j], we want pred[i] > pred[j].
    Contribution = softplus(pred[j] - pred[i]). Averaged over included pairs.

    When margin_min > 0, only include pairs where target[i] - target[j] >= margin_min
    (so we skip "borderline" pairs where the true order is ambiguous).

    Args:
        pred: (batch, n_elements) predicted scores/probs.
        target: (batch, n_elements) target strengths.
        margin_min: Only count pairs with target[i] - target[j] >= margin_min (default 0).

    Returns:
        Scalar: mean loss over included ordered pairs, or 0 if none.
    """
    _, K = pred.shape
    if K < 2:
        return pred.new_zeros(())
    total = pred.new_zeros(())
    count = 0
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            # target[i] > target[j] => we want pred[i] > pred[j]
            margin = target[:, i] - target[:, j]
            mask = margin > (margin_min - 1e-8)  # include pairs with margin >= margin_min
            if mask.any():
                diff = pred[:, j] - pred[:, i]
                total = total + (nn.functional.softplus(diff) * mask.float()).sum()
                count = count + mask.sum().item()
    if count <= 0:
        return pred.new_zeros(())
    return total / count


class ElementStrengthHead(nn.Module):
    """
    Regresses natal_emb to a 5-d distribution over elements (softmax).
    Forward returns (batch, 5) probabilities that sum to 1 per row.
    Loss = MSE + ordinal_loss_weight * pairwise_ranking_loss (when weight > 0).
    """

    def __init__(
        self,
        d_natal: int = D_NATAL,
        n_elements: int = N_ELEMENTS,
        ordinal_loss_weight: float = 0.15,
        ordinal_margin_min: float = 0.0,
    ):
        super().__init__()
        self.d_natal = d_natal
        self.n_elements = n_elements
        self.ordinal_loss_weight = ordinal_loss_weight
        self.ordinal_margin_min = ordinal_margin_min
        self.linear = nn.Linear(d_natal, n_elements)

    def forward(self, natal_emb: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """
        Args:
            natal_emb: (batch, d_natal) from NatalEncoder.

        Returns:
            (batch, n_elements) probabilities (softmax), row sum 1.
        """
        logits = self.linear(natal_emb)
        return logits.softmax(dim=-1)

    def predict(self, natal_emb: torch.Tensor) -> torch.Tensor:
        """Return (batch, 5) probabilities that sum to 1 per row. Alias for forward."""
        return self.forward(natal_emb)

    def loss(
        self,
        pred_probs: torch.Tensor,
        elemental_strengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE + optional pairwise ranking (ordinal) auxiliary loss.

        Args:
            pred_probs: (batch, 5) from forward(natal_emb).
            elemental_strengths: (batch, 5), float, sum 1.0 per row.

        Returns:
            Scalar: mse + ordinal_loss_weight * pairwise_ranking_loss.
        """
        mse = nn.functional.mse_loss(pred_probs, elemental_strengths)
        if self.ordinal_loss_weight <= 0.0:
            return mse
        ordinal = _pairwise_ranking_loss(
            pred_probs, elemental_strengths, margin_min=self.ordinal_margin_min
        )
        return mse + self.ordinal_loss_weight * ordinal

    def forward_loss(
        self,
        natal_emb: torch.Tensor,
        elemental_strengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience: forward + loss in one call for trainer use.

        Returns:
            (loss_scalar, probs). Unified convention: loss first, then prediction.
        """
        probs = self.forward(natal_emb)
        loss = self.loss(probs, elemental_strengths)
        return loss, probs
