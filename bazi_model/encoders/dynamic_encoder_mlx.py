"""
Dynamic encoder (MLX): same contract as dynamic_encoder.py but implemented with mlx.nn
for Phase 2 (frozen encoder + projection + LLM on Apple Silicon).

Encodes 3 temporal slots (大运, 流年, 流月) + learned aux token into (batch, d_model).
Uses learned null_dynamic_emb for absent slots.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from bazi_model.encoders.natal_encoder_mlx import (
    TransformerEncoderLayerMLX,
    _load_encoder_weights_from_npz,
)

# Same as dynamic_encoder.py (from embeddings: STEM_VOCAB=10, BRANCH_VOCAB=12)
STEM_VOCAB = 10
BRANCH_VOCAB = 12
STEM_PADDING_IDX = 10  # STEM_VOCAB
BRANCH_PADDING_IDX = 12  # BRANCH_VOCAB
NUM_TEMPORAL_SLOTS = 3
D_MODEL = 256


class DynamicEncoderMLX(nn.Module):
    """
    Encode 3 temporal pillar slots (大运, 流年, 流月) and a learned aux token into
    a single vector of size d_model (256). Uses learned null_dynamic_emb for
    absent slots. Same contract as DynamicEncoder.
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
        slot_dim = 2 * d_emb

        self.stem_emb = nn.Embedding(STEM_VOCAB + 1, d_emb)
        self.branch_emb = nn.Embedding(BRANCH_VOCAB + 1, d_emb)
        self.null_dynamic_emb = mx.zeros((1, NUM_TEMPORAL_SLOTS, slot_dim))
        self.slot_proj = nn.Linear(slot_dim, d_model)
        self.aux_token = mx.random.normal((1, 1, d_model)) * 0.02

        self.layers = [
            TransformerEncoderLayerMLX(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
            )
            for _ in range(num_layers)
        ]
        self.output_proj = nn.Linear(d_model * 4, d_model)

    def _clamp_ids(
        self, stem_ids: mx.array, branch_ids: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Map -1 to padding index for embedding lookup."""
        stem = mx.where(
            stem_ids >= 0,
            stem_ids,
            mx.full(stem_ids.shape, STEM_PADDING_IDX),
        )
        branch = mx.where(
            branch_ids >= 0,
            branch_ids,
            mx.full(branch_ids.shape, BRANCH_PADDING_IDX),
        )
        return stem, branch

    def __call__(
        self,
        dayun_stem_id: mx.array,
        dayun_branch_id: mx.array,
        liunian_stem_id: mx.array,
        liunian_branch_id: mx.array,
        liuyue_stem_id: mx.array,
        liuyue_branch_id: mx.array,
        pillar_presence_mask: mx.array,
        *,
        rng: mx.rng.RNGState | None = None,
    ) -> mx.array:
        """
        Returns dynamic_emb of shape (batch, d_model). Same signature as DynamicEncoder.forward.
        """
        batch = dayun_stem_id.shape[0]
        slot_dim = 2 * self.d_emb

        def flat(x: mx.array) -> mx.array:
            return mx.reshape(x, (-1,)) if x.ndim > 1 else x

        ds, db = flat(dayun_stem_id), flat(dayun_branch_id)
        ls, lb = flat(liunian_stem_id), flat(liunian_branch_id)
        ys, yb = flat(liuyue_stem_id), flat(liuyue_branch_id)

        ds, db = self._clamp_ids(ds, db)
        ls, lb = self._clamp_ids(ls, lb)
        ys, yb = self._clamp_ids(ys, yb)

        mask = pillar_presence_mask
        if mask.ndim == 1:
            mask = mx.broadcast_to(mx.reshape(mask, (1, -1)), (batch, NUM_TEMPORAL_SLOTS))
        else:
            mask = mask[:, :NUM_TEMPORAL_SLOTS]

        stem_embs = mx.stack(
            [self.stem_emb(ds), self.stem_emb(ls), self.stem_emb(ys)],
            axis=1,
        )
        branch_embs = mx.stack(
            [self.branch_emb(db), self.branch_emb(lb), self.branch_emb(yb)],
            axis=1,
        )
        slot_vecs = mx.concatenate([stem_embs, branch_embs], axis=-1)

        null = mx.broadcast_to(self.null_dynamic_emb, (batch, NUM_TEMPORAL_SLOTS, slot_dim))
        mask_exp = mx.expand_dims(mask, axis=-1)
        slot_vecs = mask_exp * slot_vecs + (1 - mask_exp) * null

        slots = self.slot_proj(slot_vecs)
        aux = mx.broadcast_to(self.aux_token, (batch, 1, self.d_model))
        seq = mx.concatenate([slots, aux], axis=1)

        for layer in self.layers:
            seq = layer(seq, rng=rng)

        out_flat = mx.reshape(seq, (batch, -1))
        return self.output_proj(out_flat)

    def load_weights(self, path: str | Path) -> None:
        """Load weights from .npz produced by convert_encoders_to_mlx.py (PyTorch dynamic_encoder state_dict keys)."""
        _load_encoder_weights_from_npz(Path(path), self)
