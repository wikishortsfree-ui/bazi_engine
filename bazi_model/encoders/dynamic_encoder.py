"""
Dynamic encoder: encode temporal context (大运, 流年, 流月) into a fixed-size
dynamic_emb (batch, 256), with correct masking when some temporal pillars are absent.

Accepts only minimal inputs (no ground-truth dynamic_interaction_vector) to avoid
data leakage. Sequence: [slot0, slot1, slot2, aux] (4 tokens); 4th token is a
learned aux_token.

Inputs:
- 3 temporal slots: dayun, liunian, liuyue (stem_id, branch_id each; -1 if absent).
- pillar_presence_mask (batch, 3): 1 if present, 0 if absent.

When a slot is absent, a learnable null_dynamic_emb is used so the model can
distinguish "no temporal context" from "present but masked".
"""
from __future__ import annotations

import torch
from torch import nn

from bazi_model.encoders.embeddings import (
    BRANCH_VOCAB,
    STEM_VOCAB,
)

# Padding indices for absent stem/branch (-1 in feature arrays)
STEM_PADDING_IDX = STEM_VOCAB        # 10
BRANCH_PADDING_IDX = BRANCH_VOCAB   # 12
NUM_TEMPORAL_SLOTS = 3
D_MODEL = 256


class DynamicEncoder(nn.Module):
    """
    Encode 3 temporal pillar slots (大运, 流年, 流月) and a learned aux token into
    a single vector of size d_model (256). Uses learned null_dynamic_emb for
    absent slots. No ground-truth cross-interaction vector is fed.
    """

    def __init__(
        self,
        d_emb: int = 32,
        d_model: int = D_MODEL,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_emb = d_emb
        self.d_model = d_model
        slot_dim = 2 * d_emb  # stem + branch per slot

        # Stem/branch embeddings for temporal pillars (same vocab as PillarEmbedding)
        self.stem_emb = nn.Embedding(STEM_VOCAB + 1, d_emb, padding_idx=STEM_PADDING_IDX)
        self.branch_emb = nn.Embedding(BRANCH_VOCAB + 1, d_emb, padding_idx=BRANCH_PADDING_IDX)

        # Learned vector for absent slots (one per slot, or shared); plan says one shared
        self.null_dynamic_emb = nn.Parameter(torch.zeros(1, NUM_TEMPORAL_SLOTS, slot_dim))

        # Project each slot to d_model; 4th token is learned (no cross-interaction input)
        self.slot_proj = nn.Linear(slot_dim, d_model)
        self.aux_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Sequence: [slot0, slot1, slot2, aux] -> 4 tokens, d_model each
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Aggregate 4 tokens to single vector (batch, d_model)
        self.output_proj = nn.Linear(d_model * 4, d_model)

    def _clamp_ids(self, stem_ids: torch.Tensor, branch_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map -1 to padding index for embedding lookup; leave valid IDs unchanged."""
        stem = torch.where(
            stem_ids >= 0,
            stem_ids,
            torch.full_like(stem_ids, STEM_PADDING_IDX, device=stem_ids.device, dtype=stem_ids.dtype),
        )
        branch = torch.where(
            branch_ids >= 0,
            branch_ids,
            torch.full_like(branch_ids, BRANCH_PADDING_IDX, device=branch_ids.device, dtype=branch_ids.dtype),
        )
        return stem, branch

    def forward(
        self,
        dayun_stem_id: torch.Tensor,      # (batch,) or (batch, 1)
        dayun_branch_id: torch.Tensor,
        liunian_stem_id: torch.Tensor,
        liunian_branch_id: torch.Tensor,
        liuyue_stem_id: torch.Tensor,
        liuyue_branch_id: torch.Tensor,
        pillar_presence_mask: torch.Tensor,  # (batch, 3), 1=present, 0=absent
    ) -> torch.Tensor:
        """
        Returns dynamic_emb of shape (batch, d_model). Absent slots use
        null_dynamic_emb; 4th token is a learned aux (no ground-truth cross vector).
        """
        batch = dayun_stem_id.shape[0]
        device = dayun_stem_id.device
        slot_dim = 2 * self.d_emb

        # Flatten to (batch,) if (batch, 1)
        def flat(x: torch.Tensor) -> torch.Tensor:
            return x.view(-1) if x.dim() > 1 else x

        ds, db = flat(dayun_stem_id), flat(dayun_branch_id)
        ls, lb = flat(liunian_stem_id), flat(liunian_branch_id)
        ys, yb = flat(liuyue_stem_id), flat(liuyue_branch_id)

        ds, db = self._clamp_ids(ds, db)
        ls, lb = self._clamp_ids(ls, lb)
        ys, yb = self._clamp_ids(ys, yb)

        # (batch, 3) presence: [dayun, liunian, liuyue]
        mask = pillar_presence_mask
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(batch, -1)
        mask = mask[:, :NUM_TEMPORAL_SLOTS].to(device)

        # Slot vectors: (batch, 3, slot_dim)
        stem_embs = torch.stack([
            self.stem_emb(ds),
            self.stem_emb(ls),
            self.stem_emb(ys),
        ], dim=1)  # (batch, 3, d_emb)
        branch_embs = torch.stack([
            self.branch_emb(db),
            self.branch_emb(lb),
            self.branch_emb(yb),
        ], dim=1)  # (batch, 3, d_emb)
        slot_vecs = torch.cat([stem_embs, branch_embs], dim=-1)  # (batch, 3, slot_dim)

        # Where absent, use null_dynamic_emb
        null = self.null_dynamic_emb.expand(batch, -1, -1)  # (batch, 3, slot_dim)
        mask_exp = mask.unsqueeze(-1)  # (batch, 3, 1)
        slot_vecs = mask_exp * slot_vecs + (1 - mask_exp) * null

        # Project slots to d_model -> (batch, 3, d_model)
        slots = self.slot_proj(slot_vecs)

        # 4th token: learned aux (no ground-truth dynamic_interaction_vector)
        aux = self.aux_token.expand(batch, -1, -1).to(device)  # (batch, 1, d_model)

        # Sequence of 4 tokens: (batch, 4, d_model)
        seq = torch.cat([slots, aux], dim=1)

        out = self.transformer(seq)  # (batch, 4, d_model)

        # Aggregate to (batch, d_model)
        out_flat = out.reshape(batch, -1)
        return self.output_proj(out_flat)
