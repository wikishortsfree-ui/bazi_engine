"""
Natal encoder: combine pillar embeddings and a learned aux token into a single
[CLS] representation of size d_model (256) for task heads (ten-god, element
strength, interactions, useful god).

Accepts only minimal inputs (no ground-truth ten-god, interaction, strength,
or useful-god) to avoid data leakage. Architecture: [CLS] + 4 pillar vectors
+ 1 learned aux token (6 tokens), each projected to d_model, then 2-layer
transformer encoder; output = CLS token.
"""
from __future__ import annotations

import torch
from torch import nn

from .embeddings import PillarEmbedding

D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1


class NatalEncoder(nn.Module):
    """
    Combines PillarEmbedding (with ten_god always padded to -1) and a learned
    aux token into sequence [CLS, P0, P1, P2, P3, aux], runs a 2-layer
    transformer, and returns the CLS token as natal_emb (batch, 256).
    """

    def __init__(
        self,
        d_emb: int = 32,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.d_emb = d_emb
        self.d_model = d_model

        self.pillar_embedding = PillarEmbedding(d_emb=d_emb)
        pillar_dim = 6 * d_emb
        self.pillar_proj = nn.Linear(pillar_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.aux_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        stem_ids: torch.Tensor,
        branch_ids: torch.Tensor,
        stem_elements: torch.Tensor,
        stem_polarities: torch.Tensor,
        pillar_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns natal_emb of shape (batch, d_model). Only minimal chart inputs
        are accepted; ten_god is never fed (pillar embedding uses padding).
        """
        batch = stem_ids.size(0)
        device = stem_ids.device

        ten_god_padding = torch.full(
            (batch, 4), -1, device=device, dtype=stem_ids.dtype
        )
        pillar_emb = self.pillar_embedding(
            stem_ids=stem_ids,
            branch_ids=branch_ids,
            ten_god_stem_ids=ten_god_padding,
            stem_elements=stem_elements,
            stem_polarities=stem_polarities,
            pillar_positions=pillar_positions,
        )  # (batch, 4, 6*d_emb)

        pillars_proj = self.pillar_proj(pillar_emb)  # (batch, 4, d_model)
        cls = self.cls_token.expand(batch, -1, -1).to(device)  # (batch, 1, d_model)
        aux = self.aux_token.expand(batch, -1, -1).to(device)  # (batch, 1, d_model)

        sequence = torch.cat([cls, pillars_proj, aux], dim=1)  # (batch, 6, d_model)
        out = self.transformer(sequence)  # (batch, 6, d_model)
        return out[:, 0, :]  # (batch, d_model)
