"""
Pillar embedding (MLX): map each of the 4 pillars to a single vector by embedding
6 discrete features per pillar and concatenating. Output shape: (batch, 4, 6 * d_emb).

Vocab sizes and d_emb match bazi_model.encoders.embeddings (PyTorch) for weight conversion.
Ten_god uses an extra padding index for -1 (missing) so num_embeddings = 12; index 11 = padding.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# Vocab sizes from bazi_engine.encodings (contract); must match embeddings.py.
STEM_VOCAB = 10  # 甲=0 … 癸=9
BRANCH_VOCAB = 12  # 子=0 … 亥=11
TEN_GOD_VOCAB = 11  # 0–9 + 日主=10; we add 1 for padding (-1 → index 11)
TEN_GOD_PADDING_IDX = 11
ELEMENT_VOCAB = 5  # 木火土金水
POLARITY_VOCAB = 2  # 0=yin, 1=yang
POSITION_VOCAB = 4  # 0=year, 1=month, 2=day, 3=hour

FEATURES_PER_PILLAR = 6
PILLAR_DIM = FEATURES_PER_PILLAR  # number of embedding types concat'd per pillar


class PillarEmbedding(nn.Module):
    """
    Embed each of the 4 pillars into a single vector by looking up 6 discrete
    features per pillar and concatenating. Output (batch, 4, 6*d_emb).
    MLX implementation; no PyTorch imports.
    """

    def __init__(self, d_emb: int = 32):
        super().__init__()
        self.d_emb = d_emb
        self.stem_emb = nn.Embedding(STEM_VOCAB, d_emb)
        self.branch_emb = nn.Embedding(BRANCH_VOCAB, d_emb)
        # 12 embeddings: 0–10 valid, 11 = padding for ten_god_stem_ids == -1
        self.ten_god_emb = nn.Embedding(TEN_GOD_VOCAB + 1, d_emb)
        self.element_emb = nn.Embedding(ELEMENT_VOCAB, d_emb)
        self.polarity_emb = nn.Embedding(POLARITY_VOCAB, d_emb)
        self.position_emb = nn.Embedding(POSITION_VOCAB, d_emb)

    def __call__(
        self,
        stem_ids: mx.array,  # (batch, 4)
        branch_ids: mx.array,
        ten_god_stem_ids: mx.array,  # (batch, 4), 0–10 or -1
        stem_elements: mx.array,
        stem_polarities: mx.array,
        pillar_positions: mx.array,
    ) -> mx.array:
        """
        Returns (batch, 4, 6*d_emb). -1 in ten_god_stem_ids is mapped to
        TEN_GOD_PADDING_IDX so those positions get the padding embedding.
        """
        # Map -1 to padding index for ten_god lookup (MLX has no padding_idx)
        padding = mx.full(ten_god_stem_ids.shape, TEN_GOD_PADDING_IDX)
        tg = mx.where(ten_god_stem_ids >= 0, ten_god_stem_ids, padding)

        # Vectorized over pillars: (batch, 4) -> (batch, 4, d_emb) each, then concat
        s = self.stem_emb(stem_ids)  # (batch, 4, d_emb)
        b = self.branch_emb(branch_ids)
        t = self.ten_god_emb(tg)
        e = self.element_emb(stem_elements)
        pol = self.polarity_emb(stem_polarities)
        pos = self.position_emb(pillar_positions)
        out = mx.concatenate([s, b, t, e, pol, pos], axis=-1)  # (batch, 4, 6*d_emb)
        return out
