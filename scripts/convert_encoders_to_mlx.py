#!/usr/bin/env python3
"""
Convert Phase 1 PyTorch encoder checkpoints to NumPy format for MLX loading.

Loads:
- Natal encoder: from multi-task checkpoint (key "encoder"), e.g.
  checkpoints/multi_task_3layer_1_6_1_1.pt
- Dynamic encoder: from cross-temporal checkpoint (key "dynamic_encoder"), e.g.
  checkpoints/cross_temporal_best.pt

Saves .npz files with the same state_dict key names; tensors are converted to
numpy (CPU). MLX encoders can load these and assign arrays to their parameters.
Optionally saves metadata (num_layers, d_model) for the natal encoder in the
same .npz so the MLX loader knows architecture.

Run from repo root (e.g. python scripts/convert_encoders_to_mlx.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _to_numpy(obj):
    """Convert a tensor or nested structure to numpy; return as-is if already numpy."""
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
    except ImportError:
        pass
    if isinstance(obj, np.ndarray):
        return obj
    raise TypeError(f"Cannot convert {type(obj)} to numpy")


def convert_state_dict_to_numpy(state_dict: dict) -> dict[str, np.ndarray]:
    out = {}
    for k, v in state_dict.items():
        out[k] = _to_numpy(v)
    return out


def convert_natal_encoder(
    checkpoint_path: Path,
    output_path: Path,
) -> bool:
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "encoder" not in ckpt:
        print(f"Error: checkpoint has no key 'encoder'. Keys: {list(ckpt.keys())}", file=sys.stderr)
        return False
    state_dict = ckpt["encoder"]
    np_dict = convert_state_dict_to_numpy(state_dict)
    # Metadata for MLX natal encoder (num_layers, d_model)
    num_layers = int(ckpt.get("num_layers", 3))
    d_model = int(ckpt.get("d_model", 256))
    np_dict["num_layers"] = np.array(num_layers)
    np_dict["d_model"] = np.array(d_model)
    np.savez_compressed(output_path, **np_dict)
    print(f"Saved natal encoder to {output_path} (num_layers={num_layers}, d_model={d_model}, {len(state_dict)} arrays)")
    return True


def convert_dynamic_encoder(
    checkpoint_path: Path,
    output_path: Path,
) -> bool:
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "dynamic_encoder" not in ckpt:
        print(f"Error: checkpoint has no key 'dynamic_encoder'. Keys: {list(ckpt.keys())}", file=sys.stderr)
        return False
    state_dict = ckpt["dynamic_encoder"]
    np_dict = convert_state_dict_to_numpy(state_dict)
    np.savez_compressed(output_path, **np_dict)
    print(f"Saved dynamic encoder to {output_path} ({len(state_dict)} arrays)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch encoder checkpoints to .npz for MLX encoders.",
    )
    parser.add_argument(
        "--natal_checkpoint",
        type=Path,
        default=Path("checkpoints/multi_task_3layer_1_6_1_1.pt"),
        help="Path to multi-task checkpoint (key: encoder). Default: checkpoints/multi_task_3layer_1_6_1_1.pt",
    )
    parser.add_argument(
        "--dynamic_checkpoint",
        type=Path,
        default=Path("checkpoints/cross_temporal_best.pt"),
        help="Path to cross-temporal checkpoint (key: dynamic_encoder). Default: checkpoints/cross_temporal_best.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for output .npz files. Default: checkpoints",
    )
    parser.add_argument(
        "--natal_only",
        action="store_true",
        help="Convert only the natal encoder.",
    )
    parser.add_argument(
        "--dynamic_only",
        action="store_true",
        help="Convert only the dynamic encoder.",
    )
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("Error: PyTorch is required to load checkpoints. Install with: pip install torch", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ok = True

    if not args.dynamic_only:
        if not args.natal_checkpoint.exists():
            print(f"Error: natal checkpoint not found: {args.natal_checkpoint}", file=sys.stderr)
            ok = False
        else:
            try:
                out_natal = args.output_dir / "natal_encoder_mlx.npz"
                if not convert_natal_encoder(args.natal_checkpoint, out_natal):
                    ok = False
            except Exception as e:
                print(f"Error converting natal encoder: {e}", file=sys.stderr)
                ok = False

    if not args.natal_only:
        if not args.dynamic_checkpoint.exists():
            print(f"Error: dynamic checkpoint not found: {args.dynamic_checkpoint}", file=sys.stderr)
            ok = False
        else:
            try:
                out_dynamic = args.output_dir / "dynamic_encoder_mlx.npz"
                if not convert_dynamic_encoder(args.dynamic_checkpoint, out_dynamic):
                    ok = False
            except Exception as e:
                print(f"Error converting dynamic encoder: {e}", file=sys.stderr)
                ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
