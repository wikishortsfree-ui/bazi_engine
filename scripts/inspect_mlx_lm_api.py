#!/usr/bin/env python3
"""
Inspect mlx_lm loaded model API for custom prefix (virtual token) embedding.

- Reports model structure (embed_tokens, layers, forward signature).
- Verifies input_embeddings support via mlx_lm.utils.does_model_support_input_embeddings.
- If --model-dir is provided and the model loads, runs one forward with dummy
  input_embeddings (no real weights needed for shape/API check) and prints output shape.

Usage (from repo root, Phase 2 env):
  python scripts/inspect_mlx_lm_api.py
  python scripts/inspect_mlx_lm_api.py --model-dir ./models/qwen2.5-7b-instruct-mlx

Ref: docs/design_decisions.md §17 (Phase 2: mlx_lm model API for custom prefix).
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect mlx_lm model API for custom prefix embeddings."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional path to MLX-converted model; if set, load and run one forward with input_embeddings.",
    )
    args = parser.parse_args()

    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
        from mlx_lm.utils import does_model_support_input_embeddings
    except ImportError as e:
        print("Error: mlx_lm or mlx not available.", file=sys.stderr)
        print("Install Phase 2 deps: pip install -r requirements_phase2.txt", file=sys.stderr)
        print(f"ImportError: {e}", file=sys.stderr)
        return 1

    # --- Static inspection: Qwen2 model class (same arch as Qwen2.5 in mlx_lm) ---
    try:
        from mlx_lm.models import qwen2
        model_class = qwen2.Model
        inner_class = qwen2.Qwen2Model
    except ImportError:
        print("Warning: Could not import mlx_lm.models.qwen2; skipping class inspection.", file=sys.stderr)
        model_class = None
        inner_class = None

    if model_class is not None:
        sig = inspect.signature(model_class.__call__)
        params = list(sig.parameters.keys())
        print("Qwen2 Model.__call__ parameters:", params)
        has_input_emb = "input_embeddings" in params
        print("  input_embeddings in signature:", has_input_emb)

        if inner_class is not None:
            inner_sig = inspect.signature(inner_class.__call__)
            inner_params = list(inner_sig.parameters.keys())
            print("Qwen2Model (inner).__call__ parameters:", inner_params)
            print("  input_embeddings in inner:", "input_embeddings" in inner_params)

    # --- Load model and check support + optional forward ---
    model_dir = args.model_dir.resolve() if args.model_dir else None
    if model_dir is not None and not model_dir.is_dir():
        print(f"Error: Model directory not found: {model_dir}", file=sys.stderr)
        return 1

    if model_dir is None:
        print("\nNo --model-dir provided; skipping load and forward. Use --model-dir to run a live forward.")
        print("Inspect mlx_lm_api: static checks done.")
        return 0

    print(f"\nLoading model from {model_dir} ...")
    model, tokenizer = load(str(model_dir))
    support = does_model_support_input_embeddings(model)
    print("does_model_support_input_embeddings(model):", support)
    if not support:
        print("Error: Model does not support input_embeddings; BaziLLM cannot use custom prefix.", file=sys.stderr)
        return 1

    # Model structure
    if hasattr(model, "model"):
        inner = model.model
        print("model.model.embed_tokens:", type(inner.embed_tokens).__name__, getattr(inner.embed_tokens, "weight", None) is not None and "shape" in dir(inner.embed_tokens.weight) and inner.embed_tokens.weight.shape)
        print("model.model.layers: len =", len(inner.layers))
        if hasattr(inner, "norm"):
            print("model.model.norm: present")
    else:
        print("model.model: not found (unexpected structure)")

    # One forward with input_embeddings only (placeholder inputs)
    batch, n_virtual, seq_len = 2, 4, 8
    d_llm = model.model.args.hidden_size
    vocab_size = model.model.args.vocab_size
    total_len = n_virtual + seq_len
    dummy_embeds = mx.random.normal((batch, total_len, d_llm)).astype(mx.float16)
    placeholder_inputs = mx.zeros((batch, total_len), dtype=mx.int32)

    out = model(placeholder_inputs, cache=None, input_embeddings=dummy_embeds)
    mx.eval(out)
    print("\nForward with input_embeddings:")
    print("  input_embeddings.shape:", (batch, total_len, d_llm))
    print("  output logits.shape:", out.shape)
    expected = (batch, total_len, vocab_size)
    if out.shape != expected:
        print(f"  Expected shape {expected}; mismatch.", file=sys.stderr)
        return 1
    print("  Output shape OK.")
    print("\ninspect_mlx_lm_api: done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
