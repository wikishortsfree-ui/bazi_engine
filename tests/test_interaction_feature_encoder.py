"""Unit tests for InteractionFeatureEncoder: output shape and optional useful_god_id."""
from __future__ import annotations

import pytest
import torch

from bazi_model.encoders.interaction_features import InteractionFeatureEncoder


def test_interaction_feature_encoder_output_shape():
    """With batch=2 and d_interaction=96, output shape is (2, 96)."""
    batch = 2
    d_interaction = 96
    enc = InteractionFeatureEncoder(d_interaction=d_interaction)

    interaction_vector = torch.rand(batch, 64)
    elemental_strengths = torch.rand(batch, 5)
    day_master_strength = torch.rand(batch, 1)

    out = enc(
        interaction_vector=interaction_vector,
        elemental_strengths=elemental_strengths,
        day_master_strength=day_master_strength,
    )
    assert out.shape == (batch, d_interaction), f"expected (2, 96), got {out.shape}"


def test_interaction_feature_encoder_with_useful_god_id():
    """Forward with useful_god_id (batch,) or (batch, 1) still yields (batch, d_interaction)."""
    batch = 2
    d_interaction = 96
    enc = InteractionFeatureEncoder(d_interaction=d_interaction)

    interaction_vector = torch.rand(batch, 64)
    elemental_strengths = torch.rand(batch, 5)
    day_master_strength = torch.rand(batch, 1)
    useful_god_id = torch.tensor([0, -1], dtype=torch.long)  # (batch,)

    out = enc(
        interaction_vector=interaction_vector,
        elemental_strengths=elemental_strengths,
        day_master_strength=day_master_strength,
        useful_god_id=useful_god_id,
    )
    assert out.shape == (batch, d_interaction)

    # (batch, 1) shape
    useful_god_id_2d = useful_god_id.unsqueeze(-1)
    out2 = enc(
        interaction_vector=interaction_vector,
        elemental_strengths=elemental_strengths,
        day_master_strength=day_master_strength,
        useful_god_id=useful_god_id_2d,
    )
    assert out2.shape == (batch, d_interaction)
