"""
Per-slot projection layer: maps (batch, n_slots, d_in_per_slot) -> (batch, n_virtual, d_llm)
for BaziLLM. Production layout is 7 slots (5 natal + 1 dynamic + 1 cross) * 9 virtual
tokens per slot = 63 virtual tokens.
"""

from bazi_model.projection.projection_layer import (
    D_IN_PER_SLOT,
    D_LLM,
    N_SLOTS,
    N_VIRTUAL_PER_SLOT,
    ProjectionLayer,
)

__all__ = [
    "D_IN_PER_SLOT",
    "D_LLM",
    "N_SLOTS",
    "N_VIRTUAL_PER_SLOT",
    "ProjectionLayer",
]
