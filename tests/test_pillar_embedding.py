"""Unit tests for PillarEmbedding: output shape and ten_god padding_idx for -1."""
from __future__ import annotations

import pytest
import torch

from bazi_model.encoders.embeddings import FEATURES_PER_PILLAR, PillarEmbedding


def test_pillar_embedding_output_shape():
    """With d_emb=32 and batch=2, output shape is (2, 4, 192) = (batch, 4, 6*d_emb)."""
    d_emb = 32
    batch = 2
    emb = PillarEmbedding(d_emb=d_emb)
    stem_ids = torch.randint(0, 10, (batch, 4))
    branch_ids = torch.randint(0, 12, (batch, 4))
    ten_god_stem_ids = torch.randint(0, 11, (batch, 4))  # 0-10 valid
    stem_elements = torch.randint(0, 5, (batch, 4))
    stem_polarities = torch.randint(0, 2, (batch, 4))
    pillar_positions = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)

    out = emb(
        stem_ids=stem_ids,
        branch_ids=branch_ids,
        ten_god_stem_ids=ten_god_stem_ids,
        stem_elements=stem_elements,
        stem_polarities=stem_polarities,
        pillar_positions=pillar_positions,
    )
    expected_dim = FEATURES_PER_PILLAR * d_emb  # 6 * 32 = 192
    assert out.shape == (batch, 4, expected_dim), f"expected (2, 4, 192), got {out.shape}"


def test_pillar_embedding_ten_god_padding_idx_handles_minus_one():
    """When ten_god_stem_ids contain -1, forward runs and output shape is (batch, 4, 6*d_emb)."""
    d_emb = 32
    batch = 2
    emb = PillarEmbedding(d_emb=d_emb)
    stem_ids = torch.randint(0, 10, (batch, 4))
    branch_ids = torch.randint(0, 12, (batch, 4))
    # Include -1 in ten_god to exercise padding_idx path
    ten_god_stem_ids = torch.tensor(
        [[0, 10, 5, -1], [-1, 3, 7, 10]],
        dtype=torch.long,
    )
    stem_elements = torch.randint(0, 5, (batch, 4))
    stem_polarities = torch.randint(0, 2, (batch, 4))
    pillar_positions = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)

    out = emb(
        stem_ids=stem_ids,
        branch_ids=branch_ids,
        ten_god_stem_ids=ten_god_stem_ids,
        stem_elements=stem_elements,
        stem_polarities=stem_polarities,
        pillar_positions=pillar_positions,
    )
    assert out.shape == (batch, 4, FEATURES_PER_PILLAR * d_emb)
    assert out.isfinite().all(), "output should have no nan/inf when -1 is present"
    # Backward should not crash (padding index has zero gradient)
    loss = out.sum()
    loss.backward()
