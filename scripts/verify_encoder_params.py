#!/usr/bin/env python3
"""
Verify NatalEncoder and DynamicEncoder parameter counts match the Phase 1 plan
expectation (~2M each). Exits 0 if both are in range [1.5M, 2.5M]; 1 otherwise.

Run from repo root: python scripts/verify_encoder_params.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow importing bazi_model when run as script from repo root
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

MIN_PARAMS = 1_500_000
MAX_PARAMS = 2_500_000


def main() -> int:
    from bazi_model.encoders.natal_encoder import NatalEncoder
    from bazi_model.encoders.dynamic_encoder import DynamicEncoder

    natal = NatalEncoder()
    dynamic = DynamicEncoder()

    natal_params = sum(p.numel() for p in natal.parameters())
    dynamic_params = sum(p.numel() for p in dynamic.parameters())

    print(f"Natal params:   {natal_params:,}")
    print(f"Dynamic params: {dynamic_params:,}")
    print(f"Expected (plan): ~2M each (ballpark [{MIN_PARAMS:,}, {MAX_PARAMS:,}])")

    natal_ok = MIN_PARAMS <= natal_params <= MAX_PARAMS
    dynamic_ok = MIN_PARAMS <= dynamic_params <= MAX_PARAMS

    if natal_ok and dynamic_ok:
        print("OK: both encoders within expected range.")
        return 0
    if not natal_ok:
        print(f"FAIL: NatalEncoder params {natal_params:,} outside [{MIN_PARAMS:,}, {MAX_PARAMS:,}]")
    if not dynamic_ok:
        print(f"FAIL: DynamicEncoder params {dynamic_params:,} outside [{MIN_PARAMS:,}, {MAX_PARAMS:,}]")
    return 1


if __name__ == "__main__":
    sys.exit(main())
