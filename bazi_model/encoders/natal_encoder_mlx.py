"""
Natal encoder (MLX): same contract as natal_encoder.py but implemented with mlx.nn
for Phase 2 (frozen encoder + projection + LLM on Apple Silicon).

Architecture: PillarEmbedding -> pillar_proj -> [CLS, P0..P3, aux] (6 tokens)
-> transformer stack -> output = (CLS, P_year, P_month, P_day, P_hour) -> (batch, 5, d_model).

The aux slot is dropped per the per-pillar probe (scripts/_probe_natal_encoder_tokens.py):
each pillar token already preserves its pillar's stem and branch identity, so the
downstream LLaVA-style per-slot projection consumes those 5 tokens directly.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from bazi_model.encoders.embeddings_mlx import PillarEmbedding

D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 3  # default for best checkpoint (multi_task_3layer_1_6_1_1.pt)
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1


def _flat_to_nested(flat: dict[str, np.ndarray], key_replace: tuple[str, str] | None = None) -> dict:
    """Build nested dict/list from flat key->numpy for MLX update(). Numeric path segments become list indices."""
    nested: dict = {}
    for key, arr in flat.items():
        if key_replace:
            key = key.replace(key_replace[0], key_replace[1])
        parts = key.split(".")
        if not parts:
            continue
        value = mx.array(arr)
        # Navigate/create path; "layers.0.norm1.weight" -> nested["layers"][0]["norm1"]["weight"]
        current: dict | list = nested
        for i, part in enumerate(parts[:-1]):
            is_digit = part.isdigit()
            idx = int(part) if is_digit else part
            if is_digit:
                if not isinstance(current, list):
                    raise ValueError(f"Expected list at {part!r} in key {key}")
                while len(current) <= idx:
                    current.append({})
                current = current[idx]
            else:
                if not isinstance(current, dict):
                    raise ValueError(f"Expected dict at {part!r} in key {key}")
                next_is_digit = i + 1 < len(parts) and parts[i + 1].isdigit()
                if part not in current:
                    current[part] = [] if next_is_digit else {}
                current = current[part]
        if isinstance(current, list):
            raise ValueError(f"Cannot assign at list in key {key}")
        current[parts[-1]] = value
    # If "layers" was built as dict with int keys (alternate key order), convert to list
    if "layers" in nested and isinstance(nested["layers"], dict):
        d = nested["layers"]
        int_keys = [k for k in d if isinstance(k, int)]
        if int_keys:
            n = max(int_keys) + 1
            nested["layers"] = [d.get(i, {}) for i in range(n)]
    return nested


def _load_encoder_weights_from_npz(
    path: Path,
    module: nn.Module,
    *,
    skip_metadata_keys: frozenset[str] = frozenset(),
) -> None:
    """Load .npz produced by convert_encoders_to_mlx.py into an MLX encoder (natal or dynamic).
    Raises FileNotFoundError if path does not exist. Expects trusted .npz from our conversion script.
    """
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")
    data = dict(np.load(path, allow_pickle=True))
    flat: dict[str, np.ndarray] = {}
    d_model = module.d_model
    for k, v in data.items():
        if k in skip_metadata_keys:
            continue
        arr = np.asarray(v)
        if arr.ndim == 0:
            continue
        key = k.replace("transformer.layers.", "layers.")
        if "self_attn.in_proj_weight" in key:
            for i, name in enumerate(("query_proj", "key_proj", "value_proj")):
                flat[key.replace("self_attn.in_proj_weight", f"self_attn.{name}.weight")] = arr[
                    i * d_model : (i + 1) * d_model
                ]
            continue
        if "self_attn.in_proj_bias" in key:
            for i, name in enumerate(("query_proj", "key_proj", "value_proj")):
                flat[key.replace("self_attn.in_proj_bias", f"self_attn.{name}.bias")] = arr[
                    i * d_model : (i + 1) * d_model
                ]
            continue
        flat[key] = arr
    nested = _flat_to_nested(flat, key_replace=None)
    layer_dicts = nested.pop("layers", None)
    module.update(nested, strict=False)
    if layer_dicts is not None and hasattr(module, "layers"):
        for i, layer in enumerate(module.layers):
            if i < len(layer_dicts):
                layer.update(layer_dicts[i], strict=False)


class TransformerEncoderLayerMLX(nn.Module):
    """
    Single transformer encoder layer matching PyTorch TransformerEncoderLayer
    (norm_first=False): norm1 -> self_attn -> residual -> norm2 -> ffn -> residual.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiHeadAttention(d_model, nhead, bias=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self._activation = activation

    def __call__(self, x: mx.array, *, rng: mx.rng.RNGState | None = None) -> mx.array:
        # Self-attention with post-norm residual
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, x_norm, x_norm)
        attn_out = self.dropout(attn_out, rng=rng) if rng is not None else attn_out
        x = x + attn_out

        # FFN with post-norm residual (PyTorch default activation is relu)
        x_norm = self.norm2(x)
        ffn = self.linear1(x_norm)
        if self._activation == "relu":
            ffn = mx.maximum(0, ffn)
        else:
            ffn = nn.gelu(ffn)
        ffn = self.linear2(ffn)
        ffn = self.dropout(ffn, rng=rng) if rng is not None else ffn
        x = x + ffn
        return x


class NatalEncoderMLX(nn.Module):
    """
    Combines PillarEmbedding and a learned aux token into sequence
    [CLS, P0, P1, P2, P3, aux], runs a transformer stack, returns CLS as (batch, d_model).
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
        self.cls_token = mx.random.normal((1, 1, d_model)) * 0.02
        self.aux_token = mx.random.normal((1, 1, d_model)) * 0.02

        self.layers = [
            TransformerEncoderLayerMLX(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
            )
            for _ in range(num_layers)
        ]

    def __call__(
        self,
        stem_ids: mx.array,
        branch_ids: mx.array,
        stem_elements: mx.array,
        stem_polarities: mx.array,
        pillar_positions: mx.array,
        *,
        rng: mx.rng.RNGState | None = None,
    ) -> mx.array:
        batch = stem_ids.shape[0]
        # ten_god always padded to -1 (no ground-truth) for natal
        ten_god_padding = mx.full((batch, 4), -1)
        pillar_emb = self.pillar_embedding(
            stem_ids=stem_ids,
            branch_ids=branch_ids,
            ten_god_stem_ids=ten_god_padding,
            stem_elements=stem_elements,
            stem_polarities=stem_polarities,
            pillar_positions=pillar_positions,
        )
        pillars_proj = self.pillar_proj(pillar_emb)
        cls = mx.broadcast_to(self.cls_token, (batch, 1, self.d_model))
        aux = mx.broadcast_to(self.aux_token, (batch, 1, self.d_model))
        sequence = mx.concatenate([cls, pillars_proj, aux], axis=1)

        for layer in self.layers:
            sequence = layer(sequence, rng=rng)

        return sequence[:, :5, :]

    def load_weights(self, path: str | Path) -> None:
        """Load weights from .npz produced by convert_encoders_to_mlx.py (PyTorch encoder state_dict keys)."""
        _load_encoder_weights_from_npz(
            Path(path), self, skip_metadata_keys=frozenset({"num_layers", "d_model"})
        )
