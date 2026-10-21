"""
Tests for BaziLLM: one forward pass (no error), output shape, no NaNs.

Uses minimal encoders (no loaded weights) and a mock LLM/tokenizer so the test
runs without the real Qwen model or checkpoint files.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx
    from bazi_model.encoders.dynamic_encoder_mlx import DynamicEncoderMLX
    from bazi_model.encoders.embedding_natal_mlx import EmbeddingNatalMLX
    from bazi_model.encoders.natal_encoder_mlx import NatalEncoderMLX
    from bazi_model.model.bazi_llm import BaziLLM
    from bazi_model.projection import D_IN_PER_SLOT, D_LLM, N_SLOTS, ProjectionLayer
    _MLX_AVAILABLE = True
except ImportError:
    mx = None
    BaziLLM = None
    ProjectionLayer = None
    NatalEncoderMLX = None
    EmbeddingNatalMLX = None
    DynamicEncoderMLX = None
    D_IN_PER_SLOT = D_LLM = N_SLOTS = None
    _MLX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _MLX_AVAILABLE, reason="mlx not available")

# Minimal config: 1 layer each, 1 virtual token per slot, to keep tests fast.
# With N_SLOTS = 7 fixed, the smallest n_virtual_tokens is 7 (1 per slot).
NATAL_LAYERS = 1
DYNAMIC_LAYERS = 1
VOCAB_SIZE = 32000
MAX_LEN = 16
N_VIRTUAL_PER_SLOT_TEST = 1
N_VIRTUAL = N_SLOTS * N_VIRTUAL_PER_SLOT_TEST  # 7


def _make_projection(n_virtual_per_slot: int = N_VIRTUAL_PER_SLOT_TEST) -> ProjectionLayer:
    return ProjectionLayer(
        d_in_per_slot=D_IN_PER_SLOT,
        d_llm=D_LLM,
        n_slots=N_SLOTS,
        n_virtual_per_slot=n_virtual_per_slot,
    )


def _make_mock_llm():
    """LLM mock: model.model.embed_tokens, __call__(..., input_embeddings=...) -> logits."""

    class Args:
        hidden_size = D_LLM
        vocab_size = VOCAB_SIZE

    class Inner:
        args = Args()

        def embed_tokens(self, x):
            return mx.ones((x.shape[0], x.shape[1], D_LLM))

    class MockLLM:
        model = Inner()

        def __call__(self, inputs, cache=None, input_embeddings=None):
            if input_embeddings is not None:
                b, s, _ = input_embeddings.shape
                return mx.zeros((b, s, self.model.args.vocab_size))
            raise NotImplementedError("mock expects input_embeddings")

    return MockLLM()


def _make_mock_tokenizer(max_length: int = MAX_LEN):
    """Tokenizer mock: (list of str) -> dict with input_ids (batch, max_length)."""

    class MockTokenizer:
        def __call__(self, queries, padding="max_length", max_length=None, truncation=True, return_tensors="np"):
            batch = len(queries)
            length = max_length if max_length is not None else MAX_LEN
            return {"input_ids": np.zeros((batch, length), dtype=np.int64)}
    return MockTokenizer()


def _make_natal_inputs(batch: int, *, mode: str = "encoder"):
    base = {
        "stem_ids": mx.zeros((batch, 4), dtype=mx.int32),
        "branch_ids": mx.zeros((batch, 4), dtype=mx.int32),
        "stem_elements": mx.zeros((batch, 4), dtype=mx.int32),
        "stem_polarities": mx.zeros((batch, 4), dtype=mx.int32),
        "pillar_positions": mx.broadcast_to(mx.array([[0, 1, 2, 3]]), (batch, 4)),
    }
    if mode == "embedding":
        base["interaction_vector"] = mx.zeros((batch, 64))
        base["elemental_strengths"] = mx.zeros((batch, 5))
        base["day_master_strength"] = mx.zeros((batch, 1))
        base["useful_god_id"] = mx.full((batch,), -1, dtype=mx.int32)
    return base


def _make_natal_module(mode: str):
    if mode == "encoder":
        return NatalEncoderMLX(d_emb=32, d_model=256, num_layers=NATAL_LAYERS)
    if mode == "embedding":
        return EmbeddingNatalMLX(d_emb=32, d_model=256)
    raise ValueError(f"Unknown natal mode: {mode!r}")


_NATAL_MODES = ("encoder", "embedding")


def _make_dynamic_inputs(batch: int):
    return {
        "dayun_stem_id": mx.zeros((batch,), dtype=mx.int32),
        "dayun_branch_id": mx.zeros((batch,), dtype=mx.int32),
        "liunian_stem_id": mx.ones((batch,), dtype=mx.int32),
        "liunian_branch_id": mx.ones((batch,), dtype=mx.int32),
        "liuyue_stem_id": mx.full((batch,), 2, dtype=mx.int32),
        "liuyue_branch_id": mx.full((batch,), 2, dtype=mx.int32),
        "pillar_presence_mask": mx.ones((batch, 3)),
    }


@pytest.mark.parametrize("natal_mode", _NATAL_MODES)
def test_bazi_llm_forward_no_error(natal_mode):
    """One forward pass runs without error."""
    natal_encoder = _make_natal_module(natal_mode)
    dynamic_encoder = DynamicEncoderMLX(d_emb=32, d_model=256, num_layers=DYNAMIC_LAYERS)
    projection = _make_projection()
    llm = _make_mock_llm()
    tokenizer = _make_mock_tokenizer()
    model = BaziLLM(
        natal_encoder,
        dynamic_encoder,
        projection,
        llm,
        tokenizer,
        max_length=MAX_LEN,
        n_virtual_tokens=N_VIRTUAL,
    )
    natal_inputs = _make_natal_inputs(1, mode=natal_mode)
    logits = model(natal_inputs, None, "test query")
    assert logits is not None


@pytest.mark.parametrize("natal_mode", _NATAL_MODES)
def test_bazi_llm_forward_shape(natal_mode):
    """Output shape is (batch, n_virtual + L, vocab_size)."""
    natal_encoder = _make_natal_module(natal_mode)
    dynamic_encoder = DynamicEncoderMLX(d_emb=32, d_model=256, num_layers=DYNAMIC_LAYERS)
    projection = _make_projection()
    llm = _make_mock_llm()
    tokenizer = _make_mock_tokenizer()
    model = BaziLLM(
        natal_encoder,
        dynamic_encoder,
        projection,
        llm,
        tokenizer,
        max_length=MAX_LEN,
        n_virtual_tokens=N_VIRTUAL,
    )
    batch = 2
    natal_inputs = _make_natal_inputs(batch, mode=natal_mode)
    dynamic_inputs = _make_dynamic_inputs(batch)
    logits = model(natal_inputs, dynamic_inputs, ["q1", "q2"])
    expected_seq = N_VIRTUAL + MAX_LEN
    assert logits.shape == (batch, expected_seq, VOCAB_SIZE)


@pytest.mark.parametrize("natal_mode", _NATAL_MODES)
def test_bazi_llm_forward_no_nans(natal_mode):
    """Output contains no NaNs."""
    natal_encoder = _make_natal_module(natal_mode)
    dynamic_encoder = DynamicEncoderMLX(d_emb=32, d_model=256, num_layers=DYNAMIC_LAYERS)
    projection = _make_projection()
    llm = _make_mock_llm()
    tokenizer = _make_mock_tokenizer()
    model = BaziLLM(
        natal_encoder,
        dynamic_encoder,
        projection,
        llm,
        tokenizer,
        max_length=MAX_LEN,
        n_virtual_tokens=N_VIRTUAL,
    )
    natal_inputs = _make_natal_inputs(1, mode=natal_mode)
    logits = model(natal_inputs, None, "hello")
    out_np = np.array(logits)
    assert not np.any(np.isnan(out_np)), "logits should contain no NaNs"
    assert not np.any(np.isinf(out_np)), "logits should contain no Infs"


@pytest.mark.parametrize("natal_mode", _NATAL_MODES)
def test_bazi_llm_natal_only_null_dynamic(natal_mode):
    """When chart_dynamic_inputs is None, forward uses null dynamic and zeros for the cross slot."""
    natal_encoder = _make_natal_module(natal_mode)
    dynamic_encoder = DynamicEncoderMLX(d_emb=32, d_model=256, num_layers=DYNAMIC_LAYERS)
    projection = _make_projection()
    llm = _make_mock_llm()
    tokenizer = _make_mock_tokenizer()
    model = BaziLLM(
        natal_encoder,
        dynamic_encoder,
        projection,
        llm,
        tokenizer,
        max_length=MAX_LEN,
        n_virtual_tokens=N_VIRTUAL,
    )
    natal_inputs = _make_natal_inputs(1, mode=natal_mode)
    logits = model(natal_inputs, None, "natal only")
    assert logits.shape == (1, N_VIRTUAL + MAX_LEN, VOCAB_SIZE)


@pytest.mark.parametrize("natal_mode", _NATAL_MODES)
def test_bazi_llm_forward_shape_full_virtual_tokens(natal_mode):
    """Output shape with the production layout (7 slots * 9 = 63) matches (batch, 63 + L, vocab)."""
    n_virtual_per_slot = 9
    n_virtual = N_SLOTS * n_virtual_per_slot  # 63
    natal_encoder = _make_natal_module(natal_mode)
    dynamic_encoder = DynamicEncoderMLX(d_emb=32, d_model=256, num_layers=DYNAMIC_LAYERS)
    projection = _make_projection(n_virtual_per_slot=n_virtual_per_slot)
    llm = _make_mock_llm()
    tokenizer = _make_mock_tokenizer()
    model = BaziLLM(
        natal_encoder,
        dynamic_encoder,
        projection,
        llm,
        tokenizer,
        max_length=MAX_LEN,
        n_virtual_tokens=n_virtual,
    )
    batch = 1
    natal_inputs = _make_natal_inputs(batch, mode=natal_mode)
    dynamic_inputs = _make_dynamic_inputs(batch)
    logits = model(natal_inputs, dynamic_inputs, "q")
    assert logits.shape == (batch, n_virtual + MAX_LEN, VOCAB_SIZE)
