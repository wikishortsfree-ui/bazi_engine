"""
Embedding-only natal head (MLX): drop-in replacement for ``NatalEncoderMLX`` that
returns ``(batch, 5, d_model)`` without any transformer.

Motivation (see ``plans/embedding-natal-arch-v3_e240dad3.plan.md`` and
``scripts/_probe_natal_encoder_tokens.py``):

- A linear probe of the per-pillar tokens emitted by ``NatalEncoderMLX`` recovers
  ``(stem_id, branch_id)`` at 100% test accuracy, i.e. those tokens are already
  identity-preserving by construction. We can replace the transformer with a
  pure embedding lookup with the same identity guarantee and zero learning cost.
- ``cross_features`` is all-zero for natal-only QA charts, so the natal-internal
  64-d ``interaction_vector`` (already computed in ``BaziChart``) is currently
  invisible to the model. We surface it (and the strength scalars + useful-god
  one-hot) through a small "chart summary" slot that takes the place of the CLS
  token.

Slot layout (matches ``NatalEncoderMLX`` for drop-in compatibility):

- ``slot[:, 0]``      = ``ChartSummaryHead(interaction_vector || elemental_strengths
                                         || day_master_strength || useful_god_emb)``
- ``slot[:, 1..4]``   = per-pillar embedding sums projected to ``d_model``

The four per-pillar tokens are the elementwise sum of stem / branch / position /
element / polarity embeddings (one ``nn.Embedding`` per feature, all of width
``d_emb``), then a single ``nn.Linear(d_emb, d_model)`` shared across the four
pillars. We chose sum (not concat) to keep the head tiny; if it underfits we
swap to concat without changing the public contract.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from bazi_model.encoders.embeddings_mlx import (
    BRANCH_VOCAB,
    ELEMENT_VOCAB,
    POLARITY_VOCAB,
    POSITION_VOCAB,
    STEM_VOCAB,
)

# Useful-god vocab: ids 0–4 (比劫/食伤/财/官杀/印 per encodings.USEFUL_GOD_TO_ID)
# plus index 5 reserved for "unknown" (raw value -1).
USEFUL_GOD_VOCAB = 6
USEFUL_GOD_UNKNOWN_IDX = 5

# Default chart-summary widths.
INTERACTION_VECTOR_DIM = 64
ELEMENTAL_STRENGTHS_DIM = 5
DAY_MASTER_STRENGTH_DIM = 1


# Input keys this module expects in ``BaziLLM.__call__``'s ``chart_natal_inputs``;
# extends the base 5-key contract to also surface the chart-summary scalars.
EMBEDDING_NATAL_INPUT_KEYS = (
    "stem_ids",
    "branch_ids",
    "stem_elements",
    "stem_polarities",
    "pillar_positions",
    "interaction_vector",
    "elemental_strengths",
    "day_master_strength",
    "useful_god_id",
)


class ChartSummaryHead(nn.Module):
    """MLP from concatenated chart-summary features to ``d_model``."""

    def __init__(
        self,
        d_model: int,
        n_useful_god_emb: int,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        d_in = (
            INTERACTION_VECTOR_DIM
            + ELEMENTAL_STRENGTHS_DIM
            + DAY_MASTER_STRENGTH_DIM
            + n_useful_god_emb
        )
        if hidden is None:
            hidden = max(d_model, 128)
        self.linear1 = nn.Linear(d_in, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.linear2 = nn.Linear(hidden, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.linear1(x)
        h = nn.gelu(h)
        h = self.norm(h)
        return self.linear2(h)


class EmbeddingNatalMLX(nn.Module):
    """
    Embedding-only natal head: ``(b, 5, d_model)`` with no transformer.

    Slot 0 carries chart-summary features (interactions + element strengths +
    day-master strength + useful-god); slots 1–4 are per-pillar embedding sums.
    """

    def __init__(
        self,
        d_emb: int = 64,
        d_model: int = 256,
        n_useful_god: int = 11,
    ) -> None:
        super().__init__()
        self.d_emb = d_emb
        self.d_model = d_model
        self.n_useful_god = n_useful_god

        self.stem_emb = nn.Embedding(STEM_VOCAB, d_emb)
        self.branch_emb = nn.Embedding(BRANCH_VOCAB, d_emb)
        self.position_emb = nn.Embedding(POSITION_VOCAB, d_emb)
        self.element_emb = nn.Embedding(ELEMENT_VOCAB, d_emb)
        self.polarity_emb = nn.Embedding(POLARITY_VOCAB, d_emb)

        self.pillar_proj = nn.Linear(d_emb, d_model)

        self.useful_god_emb = nn.Embedding(USEFUL_GOD_VOCAB, n_useful_god)
        self.chart_head = ChartSummaryHead(d_model=d_model, n_useful_god_emb=n_useful_god)

    @property
    def natal_input_keys(self) -> tuple[str, ...]:
        """Keys expected in ``chart_natal_inputs`` (read by ``BaziLLM``)."""
        return EMBEDDING_NATAL_INPUT_KEYS

    def __call__(
        self,
        stem_ids: mx.array,
        branch_ids: mx.array,
        stem_elements: mx.array,
        stem_polarities: mx.array,
        pillar_positions: mx.array,
        interaction_vector: mx.array,
        elemental_strengths: mx.array,
        day_master_strength: mx.array,
        useful_god_id: mx.array,
    ) -> mx.array:
        s = self.stem_emb(stem_ids)
        b = self.branch_emb(branch_ids)
        pos = self.position_emb(pillar_positions)
        elem = self.element_emb(stem_elements)
        pol = self.polarity_emb(stem_polarities)
        pillar_sum = s + b + pos + elem + pol  # (batch, 4, d_emb)
        pillar_tokens = self.pillar_proj(pillar_sum)  # (batch, 4, d_model)

        # Map raw useful_god ids: -1 -> USEFUL_GOD_UNKNOWN_IDX, else identity.
        ug = useful_god_id
        if ug.ndim == 2 and ug.shape[-1] == 1:
            ug = ug.squeeze(-1)
        unknown = mx.full(ug.shape, USEFUL_GOD_UNKNOWN_IDX)
        ug_idx = mx.where(ug >= 0, ug, unknown)
        ug_vec = self.useful_god_emb(ug_idx)  # (batch, n_useful_god)

        iv = interaction_vector.astype(pillar_tokens.dtype)
        es = elemental_strengths.astype(pillar_tokens.dtype)
        dm = day_master_strength.astype(pillar_tokens.dtype)
        if dm.ndim == 1:
            dm = dm[:, None]
        ug_vec = ug_vec.astype(pillar_tokens.dtype)

        chart_summary = mx.concatenate([iv, es, dm, ug_vec], axis=-1)
        chart_token = self.chart_head(chart_summary)[:, None, :]  # (batch, 1, d_model)

        return mx.concatenate([chart_token, pillar_tokens], axis=1)
