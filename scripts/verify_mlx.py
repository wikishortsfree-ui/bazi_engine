#!/usr/bin/env python3
"""
Verify MLX environment for Phase 2 (bazi_engine).
- Imports mlx.core (and optionally mlx.nn).
- Checks default device (expect 'gpu' on Apple Silicon; CPU is documented fallback).
- Optional: small matmul benchmark (1024x1024) and prints result/time.
Exits 0 if OK, non-zero with clear message if MLX unavailable or device unexpected.
"""

from __future__ import annotations

import sys
import time


def main() -> int:
    try:
        import mlx.core as mx
    except ImportError as e:
        print("verify_mlx: MLX is not available.", file=sys.stderr)
        print("Install with: pip install mlx (or use requirements_phase2.txt).", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        import mlx.nn  # noqa: F401
    except ImportError:
        pass  # optional

    device = mx.default_device()
    # On Apple Silicon we expect 'gpu'; CPU is a valid fallback (e.g. CI or no Metal).
    if device not in ("gpu", "cpu"):
        print(
            f"verify_mlx: Unexpected default device: {device!r}. Expected 'gpu' or 'cpu'.",
            file=sys.stderr,
        )
        return 1

    print(f"MLX default device: {device}")
    if device == "cpu":
        print("(CPU fallback; 'gpu' expected on Apple Silicon with Metal.)")

    # Optional matmul benchmark: 1024x1024
    n = 1024
    a = mx.random.normal((n, n))
    b = mx.random.normal((n, n))
    mx.eval(a)
    mx.eval(b)
    t0 = time.perf_counter()
    c = a @ b
    mx.eval(c)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000
    sample = float(c[0, 0])
    print(f"Matmul benchmark ({n}x{n}): {elapsed_ms:.2f} ms, sample result: {sample:.4f}")

    print("verify_mlx: OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
