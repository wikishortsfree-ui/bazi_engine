"""
BaziLLM: encoders -> per-slot stack -> projection -> prepend virtual tokens -> frozen LLM forward.

Phase 2.2: single MLX runtime for natal/dynamic encoders, projection layer, and
mlx_lm-loaded Qwen. For natal-only use, chart_dynamic_inputs=None uses null
dynamic inputs (all -1, mask 0) so dynamic encoder still returns (batch, 256).

LLaVA-style per-slot projection: the natal encoder emits 5 tokens
(CLS + P_year + P_month + P_day + P_hour); the dynamic and cross-interaction
encoders each contribute one slot (zeros when the cross encoder is absent).
The 7 slots are stacked into ``(batch, 7, 256)`` and fed to ``ProjectionLayer``,
which produces ``(batch, 63, d_llm)`` virtual tokens (9 virtual tokens per slot).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from bazi_model.projection import ProjectionLayer

# Input keys expected by encoders (match train_cross_temporal / encoder APIs).
NATAL_INPUT_KEYS = (
    "stem_ids",
    "branch_ids",
    "stem_elements",
    "stem_polarities",
    "pillar_positions",
)
DYNAMIC_INPUT_KEYS = (
    "dayun_stem_id",
    "dayun_branch_id",
    "liunian_stem_id",
    "liunian_branch_id",
    "liuyue_stem_id",
    "liuyue_branch_id",
    "pillar_presence_mask",
)
NUM_TEMPORAL_SLOTS = 3


def _to_mx(arr: Any) -> mx.array:
    """Convert numpy array or list to mx.array if needed. Prefer numpy/MLX for consistent dtypes."""
    if isinstance(arr, mx.array):
        return arr
    return mx.array(arr)


def _null_dynamic_inputs(batch: int) -> dict[str, mx.array]:
    """Build null dynamic inputs: all -1 ids, mask 0. Dynamic encoder still returns (batch, 256)."""
    return {
        "dayun_stem_id": mx.full((batch,), -1),
        "dayun_branch_id": mx.full((batch,), -1),
        "liunian_stem_id": mx.full((batch,), -1),
        "liunian_branch_id": mx.full((batch,), -1),
        "liuyue_stem_id": mx.full((batch,), -1),
        "liuyue_branch_id": mx.full((batch,), -1),
        "pillar_presence_mask": mx.zeros((batch, NUM_TEMPORAL_SLOTS)),
    }


def _extract_natal_kw(
    inputs: dict[str, Any],
    keys: tuple[str, ...] = NATAL_INPUT_KEYS,
) -> dict[str, mx.array]:
    """Extract natal encoder kwargs; values converted to ``mx.array``.

    ``keys`` defaults to the 5-key encoder contract; pass a wider tuple
    (e.g. ``EmbeddingNatalMLX.natal_input_keys``) when the natal module
    consumes the chart-summary scalars too.
    """
    return {k: _to_mx(inputs[k]) for k in keys if k in inputs}


def _extract_dynamic_kw(inputs: dict[str, Any]) -> dict[str, mx.array]:
    """Extract dynamic encoder kwargs; values converted to mx.array."""
    return {k: _to_mx(inputs[k]) for k in DYNAMIC_INPUT_KEYS if k in inputs}


class BaziLLM(nn.Module):
    """
    Composes natal encoder, dynamic encoder, projection layer, and frozen LLM.
    Forward: encode chart -> concat (natal, dynamic) -> project -> prepend to query emb -> LLM.

    Inherits from mlx.nn.Module so nn.value_and_grad can discover trainable
    parameters.  After construction, call freeze_except_projection() to mark
    encoders and LLM as frozen.
    """

    def __init__(
        self,
        natal_encoder: Any,
        dynamic_encoder: Any,
        projection_layer: ProjectionLayer,
        llm_model: Any,
        tokenizer: Any,
        *,
        cross_interaction_encoder: Any | None = None,
        max_length: int = 1024,
        n_virtual_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.natal_encoder = natal_encoder
        self.dynamic_encoder = dynamic_encoder
        self.cross_interaction_encoder = cross_interaction_encoder
        self.projection_layer = projection_layer
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_virtual_tokens = n_virtual_tokens
        self.gradient_checkpointing = False
        self._checkpointed_llm_fwd = mx.checkpoint(self._llm_forward)
        # Some natal modules (e.g. ``EmbeddingNatalMLX``) consume more chart inputs
        # than the 5-key encoder contract; honor a ``natal_input_keys`` attribute
        # on the natal module if present, otherwise fall back to the default.
        self.natal_input_keys: tuple[str, ...] = tuple(
            getattr(natal_encoder, "natal_input_keys", NATAL_INPUT_KEYS)
        )
        if projection_layer.n_virtual_tokens != n_virtual_tokens:
            raise ValueError(
                f"projection_layer.n_virtual_tokens ({projection_layer.n_virtual_tokens}) "
                f"must equal n_virtual_tokens ({n_virtual_tokens})"
            )

    def freeze_except_projection(self) -> None:
        """Freeze all sub-modules except projection_layer and cross_interaction_encoder."""
        self.natal_encoder.freeze()
        self.dynamic_encoder.freeze()
        self.llm_model.freeze()

    def __call__(
        self,
        chart_natal_inputs: dict[str, Any],
        chart_dynamic_inputs: dict[str, Any] | None,
        query_text: str | list[str],
        cross_features: mx.array | None = None,
    ) -> mx.array:
        """
        Run one forward: encode chart, project to virtual tokens, prepend to query, LLM.

        Args:
            chart_natal_inputs: Dict with natal keys (stem_ids, branch_ids, stem_elements,
                stem_polarities, pillar_positions). Values: mx.array or numpy.
            chart_dynamic_inputs: Dict with dynamic keys, or None for natal-only (null dynamic).
            query_text: Single string (batch=1) or list of strings (batch=len(list)).
                Tokenized and padded to max_length.
            cross_features: (batch, 27) float cross-interaction features, or None.

        Returns:
            logits: (batch, n_virtual + L, vocab_size) where n_virtual = projection output length.
        """
        natal_kw = _extract_natal_kw(chart_natal_inputs, self.natal_input_keys)
        if not natal_kw or set(natal_kw) != set(self.natal_input_keys):
            raise ValueError(
                f"chart_natal_inputs must contain exactly {self.natal_input_keys}; "
                f"got keys {list(natal_kw.keys())}"
            )
        stem_ids = natal_kw["stem_ids"]
        batch = stem_ids.shape[0]
        if batch == 0:
            raise ValueError("batch size must be >= 1")
        if stem_ids.ndim != 2 or stem_ids.shape[1] != 4:
            raise ValueError(
                f"chart_natal_inputs['stem_ids'] must have shape (batch, 4); got {stem_ids.shape}"
            )

        if chart_dynamic_inputs is None:
            dynamic_kw = _null_dynamic_inputs(batch)
        else:
            dynamic_kw = _extract_dynamic_kw(chart_dynamic_inputs)
            if set(dynamic_kw) != set(DYNAMIC_INPUT_KEYS):
                raise ValueError(
                    f"chart_dynamic_inputs must contain exactly {DYNAMIC_INPUT_KEYS}; got keys {list(dynamic_kw.keys())}"
                )

        natal_emb = self.natal_encoder(**natal_kw)  # (batch, 5, d_model)
        dynamic_emb = self.dynamic_encoder(**dynamic_kw)  # (batch, d_model)

        slot_parts = [natal_emb, dynamic_emb[:, None, :]]
        if self.cross_interaction_encoder is not None:
            if cross_features is None:
                cross_features = mx.zeros((batch, 27))
            cross_emb = self.cross_interaction_encoder(cross_features)
            slot_parts.append(cross_emb[:, None, :])
        else:
            # Always reserve slot 6 for the cross-interaction signal so the projection
            # input shape is constant (batch, 7, d_model) whether or not the encoder
            # is present.
            slot_parts.append(mx.zeros((batch, 1, natal_emb.shape[-1])))

        slot_stack = mx.concatenate(slot_parts, axis=1)
        virtual_tokens = self.projection_layer(slot_stack)

        queries = [query_text] if isinstance(query_text, str) else list(query_text)
        if len(queries) != batch:
            raise ValueError(
                f"query_text batch size {len(queries)} does not match chart batch {batch}"
            )

        encoded = self.tokenizer(
            queries,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="np",
        )
        try:
            query_ids = mx.array(encoded["input_ids"])
        except KeyError as e:
            raise TypeError(
                "tokenizer must return a dict with 'input_ids' (e.g. HuggingFace tokenizer with return_tensors='np')"
            ) from e
        L = query_ids.shape[1]

        n_virtual = virtual_tokens.shape[1]
        inner = getattr(self.llm_model, "model", self.llm_model)
        query_emb = inner.embed_tokens(query_ids)
        inputs_embeds = mx.concatenate([virtual_tokens, query_emb], axis=1)
        total_len = n_virtual + L
        placeholder_inputs = mx.zeros((batch, total_len), dtype=mx.int32)

        if self.gradient_checkpointing:
            logits = self._checkpointed_llm_fwd(inputs_embeds, placeholder_inputs)
        else:
            logits = self._llm_forward(inputs_embeds, placeholder_inputs)
        return logits

    def _llm_forward(self, input_embeddings: mx.array, placeholder_inputs: mx.array) -> mx.array:
        return self.llm_model(placeholder_inputs, cache=None, input_embeddings=input_embeddings)
