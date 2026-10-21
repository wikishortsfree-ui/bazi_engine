#!/usr/bin/env python3
"""
Run one BaziLLM forward with real encoders (from .npz), real projection, and real Qwen.

Loads:
- Natal encoder: checkpoints/natal_encoder_mlx.npz (from convert_encoders_to_mlx.py)
- Dynamic encoder: checkpoints/dynamic_encoder_mlx.npz
- Qwen: models/qwen2.5-7b-instruct-mlx (from mlx_lm.convert)

Prints output logits shape and process memory. No training.

Usage (from repo root, Phase 2 env):
  python scripts/test_bazi_llm_forward.py
  python scripts/test_bazi_llm_forward.py --model-dir ./models/qwen2.5-7b-instruct-mlx --max-length 64
"""

from __future__ import annotations

import argparse
import resource
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resident_mb() -> float:
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One BaziLLM forward with real encoders + Qwen; print shape and memory.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "models" / "qwen2.5-7b-instruct-mlx",
        help="Path to MLX-converted Qwen model directory",
    )
    parser.add_argument(
        "--natal-weights",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "natal_encoder_mlx.npz",
        help="Path to natal encoder MLX weights (.npz)",
    )
    parser.add_argument(
        "--dynamic-weights",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "dynamic_encoder_mlx.npz",
        help="Path to dynamic encoder MLX weights (.npz)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Max token length for query (default: 64)",
    )
    parser.add_argument(
        "--n-virtual",
        type=int,
        default=8,
        help="Number of virtual tokens (default: 8)",
    )
    args = parser.parse_args()

    try:
        import mlx.core as mx
        from mlx_lm import load as mlx_lm_load
        from mlx_lm.utils import does_model_support_input_embeddings

        from bazi_model.encoders.dynamic_encoder_mlx import DynamicEncoderMLX
        from bazi_model.encoders.natal_encoder_mlx import NatalEncoderMLX
        from bazi_model.model.bazi_llm import BaziLLM
        from bazi_model.projection import ProjectionLayer
    except ImportError as e:
        print(
            "Error: mlx, mlx_lm, or bazi_model not available. Install Phase 2 deps: pip install -r requirements_phase2.txt",
            file=sys.stderr,
        )
        print(f"ImportError: {e}", file=sys.stderr)
        return 1

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        print(f"Error: Model directory not found: {model_dir}", file=sys.stderr)
        print(
            "Convert first: python -m mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path ./models/qwen2.5-7b-instruct-mlx",
            file=sys.stderr,
        )
        return 1

    if not args.natal_weights.exists():
        print(f"Error: Natal encoder weights not found: {args.natal_weights}", file=sys.stderr)
        print("Run scripts/convert_encoders_to_mlx.py first.", file=sys.stderr)
        return 1
    if not args.dynamic_weights.exists():
        print(f"Error: Dynamic encoder weights not found: {args.dynamic_weights}", file=sys.stderr)
        print("Run scripts/convert_encoders_to_mlx.py first.", file=sys.stderr)
        return 1

    mem_before = _resident_mb()
    print("Loading encoders and projection ...")
    natal_encoder = NatalEncoderMLX(d_emb=32, d_model=256, num_layers=3)
    natal_encoder.load_weights(str(args.natal_weights))
    dynamic_encoder = DynamicEncoderMLX(d_emb=32, d_model=256, num_layers=2)
    dynamic_encoder.load_weights(str(args.dynamic_weights))
    projection = ProjectionLayer(d_in=512, d_llm=3584, n_virtual_tokens=args.n_virtual)
    print("Loading Qwen from", model_dir, "...")
    llm_model, tokenizer = mlx_lm_load(str(model_dir))
    if not does_model_support_input_embeddings(llm_model):
        print("Error: This Qwen build does not support input_embeddings.", file=sys.stderr)
        return 1
    mem_after_load = _resident_mb()
    print(f"Memory after load: ~{mem_after_load:.0f} MiB (delta ~{mem_after_load - mem_before:.0f} MiB)")

    bazi_llm = BaziLLM(
        natal_encoder,
        dynamic_encoder,
        projection,
        llm_model,
        tokenizer,
        max_length=args.max_length,
        n_virtual_tokens=args.n_virtual,
    )

    # Dummy chart: one sample, valid IDs in range.
    batch = 1
    chart_natal = {
        "stem_ids": mx.array([[0, 1, 2, 3]]),
        "branch_ids": mx.array([[0, 1, 2, 3]]),
        "stem_elements": mx.array([[0, 1, 2, 3]]),
        "stem_polarities": mx.array([[0, 1, 0, 1]]),
        "pillar_positions": mx.array([[0, 1, 2, 3]]),
    }
    query = "请根据八字分析一下运势。"

    print("Running one forward (natal-only, null dynamic) ...")
    logits = bazi_llm.forward(chart_natal, None, query)
    mx.eval(logits)
    mem_after_forward = _resident_mb()
    print(f"Logits shape: {logits.shape}")
    print(f"Memory after forward: ~{mem_after_forward:.0f} MiB")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
