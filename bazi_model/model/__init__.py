"""
BaziLLM: chart encoders + projection + frozen LLM (Phase 2.2).

Composes natal/dynamic encoders (MLX), projection layer, and mlx_lm-loaded
Qwen model; prepends virtual tokens from chart embeddings to query embeddings
and runs one forward through the frozen LLM.
"""

from bazi_model.model.bazi_llm import BaziLLM

__all__ = [
    "BaziLLM",
]
