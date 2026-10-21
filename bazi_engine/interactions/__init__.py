"""
Stem/branch interaction detection (合, 冲, etc.).

Public API:
- detect_all_interactions(stems, branches) -> InteractionResult
- InteractionResult
"""
from bazi_engine.interactions.interaction_result import (
    InteractionResult,
    detect_all_interactions,
)

__all__ = ["InteractionResult", "detect_all_interactions"]
