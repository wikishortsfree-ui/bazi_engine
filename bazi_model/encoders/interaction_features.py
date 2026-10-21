"""
Interaction feature encoder: chart-level binary (64), continuous (5+1), and optional
useful_god_id into a single vector of size d_interaction (e.g. 80–120).

Feature index mapping (output vector):
  - dims 0..(binary_out-1): from binary path (64 → linear → e.g. 64)
  - dims binary_out..(binary_out+cont_out-1): from continuous path (6 → linear → e.g. 24)
  - dims (binary_out+cont_out)..(d_interaction-1): from useful_god embedding (vocab 6 → small dim)
"""
from __future__ import annotations

import torch
from torch import nn

BINARY_DIM = 64
CONTINUOUS_DIM = 6  # 5 elemental_strengths + 1 day_master_strength
# Phase 0 BaziChart.to_feature_arrays() uses bazi_engine.encodings.USEFUL_GOD_TO_ID:
# 5 element-based categories (比劫/食伤/财/官杀/印) → IDs 0–4; -1 = ambiguous. Vocab 6 = 0–4 + padding index 5.
USEFUL_GOD_VOCAB = 6  # 0–4 + padding for -1 (matches Phase 0; 10-class would require engine change)


class InteractionFeatureEncoder(nn.Module):
    """
    Encode interaction_vector (64), elemental_strengths (5), day_master_strength (1),
    and optional useful_god_id into a single vector of size d_interaction.
    Output shape: (batch, d_interaction).
    """

    def __init__(
        self,
        d_interaction: int = 96,
        binary_hidden: int = 64,
        continuous_hidden: int = 24,
        useful_god_emb_dim: int = 8,
    ):
        super().__init__()
        self.d_interaction = d_interaction
        self.binary_hidden = binary_hidden
        self.continuous_hidden = continuous_hidden
        self.useful_god_emb_dim = useful_god_emb_dim

        self.binary_fc = nn.Sequential(
            nn.Linear(BINARY_DIM, binary_hidden),
            nn.ReLU(),
            nn.LayerNorm(binary_hidden),
        )
        self.continuous_fc = nn.Sequential(
            nn.Linear(CONTINUOUS_DIM, continuous_hidden),
            nn.ReLU(),
            nn.LayerNorm(continuous_hidden),
        )
        self.useful_god_emb = nn.Embedding(USEFUL_GOD_VOCAB, useful_god_emb_dim, padding_idx=5)  # -1 → 5

        d_from_enc = binary_hidden + continuous_hidden + useful_god_emb_dim
        if d_from_enc != d_interaction:
            self.proj = nn.Linear(d_from_enc, d_interaction)
        else:
            self.proj = nn.Identity()

    def forward(
        self,
        interaction_vector: torch.Tensor,   # (batch, 64)
        elemental_strengths: torch.Tensor,  # (batch, 5)
        day_master_strength: torch.Tensor,  # (batch, 1)
        useful_god_id: torch.Tensor | None = None,  # (batch,) or (batch, 1), 0–4 or -1
    ) -> torch.Tensor:
        batch = interaction_vector.size(0)
        b = self.binary_fc(interaction_vector)                    # (batch, binary_hidden)
        c = self.continuous_fc(
            torch.cat([elemental_strengths, day_master_strength], dim=-1)
        )                                                          # (batch, continuous_hidden)
        if useful_god_id is not None:
            if useful_god_id.dim() == 2:
                useful_god_id = useful_god_id.squeeze(-1)
            # -1 → 5 (padding index)
            ug = useful_god_id.clone()
            ug = torch.where(ug >= 0, ug, torch.full_like(ug, 5, device=ug.device, dtype=ug.dtype))
            u = self.useful_god_emb(ug)                            # (batch, useful_god_emb_dim)
        else:
            u = torch.zeros(
                batch, self.useful_god_emb_dim,
                device=interaction_vector.device,
                dtype=interaction_vector.dtype,
            )
        out = torch.cat([b, c, u], dim=-1)
        return self.proj(out)  # (batch, d_interaction)
