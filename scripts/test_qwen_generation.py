#!/usr/bin/env python3
"""
Test Qwen2.5-7B-Instruct loaded from MLX-converted weights (Phase 2).

Loads the model from models/qwen2.5-7b-instruct-mlx/, runs a short generation
(~50 tokens), prints sample output and process memory.

How to run (from repo root, with Phase 2 venv active):
  python scripts/test_qwen_generation.py
  python scripts/test_qwen_generation.py --model-dir ./models/qwen2.5-7b-instruct-mlx --max-tokens 50

Prerequisites:
  - Convert the HF model first:
    python -m mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path ./models/qwen2.5-7b-instruct-mlx
  - Install Phase 2 deps: pip install -r requirements_phase2.txt

Expected memory (approximate):
  - Unquantized (fp16): ~14 GB GPU/Unified Memory on Apple Silicon.
  - Quantized (e.g. 4-bit): lower; use --quantize when converting if needed.
"""

from __future__ import annotations

import argparse
import resource
import sys
from pathlib import Path


def _resident_mb() -> float:
    """Current process resident set size in MiB (best-effort)."""
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # macOS reports in KB
    except Exception:
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load MLX Qwen model, run short generation, report memory."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models" / "qwen2.5-7b-instruct-mlx",
        help="Path to MLX-converted model directory (default: repo/models/qwen2.5-7b-instruct-mlx)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max new tokens to generate (default: 50)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="你好，请用一句话介绍八字。",
        help="Prompt for generation (default: short Chinese prompt)",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        print(
            f"Error: Model directory not found: {model_dir}",
            file=sys.stderr,
        )
        print(
            "Convert first: python -m mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path ./models/qwen2.5-7b-instruct-mlx",
            file=sys.stderr,
        )
        return 1

    try:
        from mlx_lm import load, generate
    except ImportError as e:
        print("Error: mlx_lm not available. Install Phase 2 deps: pip install -r requirements_phase2.txt", file=sys.stderr)
        print(f"ImportError: {e}", file=sys.stderr)
        return 1

    print(f"Loading model from {model_dir} ...")
    mem_before_mb = _resident_mb()
    model, tokenizer = load(str(model_dir))
    mem_after_load_mb = _resident_mb()
    print(f"Memory after load: ~{mem_after_load_mb:.0f} MiB resident (delta ~{mem_after_load_mb - mem_before_mb:.0f} MiB)")

    # Use chat template for instruct model (single user message)
    messages = [{"role": "user", "content": args.prompt}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = args.prompt

    print(f"Prompt: {args.prompt!r}")
    print(f"Generating up to {args.max_tokens} tokens ...")
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        verbose=False,
    )
    print("Output:")
    print(response)
    print()

    mem_after_gen_mb = _resident_mb()
    print(f"Memory after generation: ~{mem_after_gen_mb:.0f} MiB resident")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
