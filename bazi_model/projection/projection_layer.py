"""
Per-slot projection layer (MLX): (batch, n_slots, d_in_per_slot) -> (batch, n_slots * n_virtual_per_slot, d_llm).

LLaVA-style per-slot mapping: each slot owns an independent ``nn.Linear`` from
``d_in_per_slot`` to ``n_virtual_per_slot * d_llm``; the per-slot output is reshaped
to ``(n_virtual_per_slot, d_llm)`` and the slots are concatenated along the token
axis in fixed order (slot_0 first, slot_{n_slots-1} last).

Production layout used by ``BaziLLM``:
- 5 natal slots (CLS + P_year + P_month + P_day + P_hour)
- 1 dynamic slot
- 1 cross-interaction slot
= 7 slots * 9 virtual tokens/slot = 63 virtual tokens.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

D_IN_PER_SLOT = 256
D_LLM = 3584
N_SLOTS = 7
N_VIRTUAL_PER_SLOT = 9


class ProjectionLayer(nn.Module):
    """Per-slot Linear maps producing (batch, n_slots * n_virtual_per_slot, d_llm)."""

    def __init__(
        self,
        d_in_per_slot: int = D_IN_PER_SLOT,
        d_llm: int = D_LLM,
        n_slots: int = N_SLOTS,
        n_virtual_per_slot: int = N_VIRTUAL_PER_SLOT,
    ) -> None:
        super().__init__()
        self.d_in_per_slot = d_in_per_slot
        self.d_llm = d_llm
        self.n_slots = n_slots
        self.n_virtual_per_slot = n_virtual_per_slot
        self.n_virtual_tokens = n_slots * n_virtual_per_slot
        # Slot Linears named slot_0 ... slot_{n_slots-1} so MLX flattens cleanly to
        # keys like ``projection_layer.slot_0.weight`` for save/load round-trips.
        for i in range(n_slots):
            setattr(self, f"slot_{i}", nn.Linear(d_in_per_slot, n_virtual_per_slot * d_llm))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (batch, n_slots, d_in_per_slot)
        Returns:
            (batch, n_slots * n_virtual_per_slot, d_llm)
        """
        if x.ndim != 3 or x.shape[1] != self.n_slots or x.shape[2] != self.d_in_per_slot:
            raise ValueError(
                f"ProjectionLayer expected x shape (batch, {self.n_slots}, "
                f"{self.d_in_per_slot}); got {tuple(x.shape)}"
            )
        batch = x.shape[0]
        outs = []
        for i in range(self.n_slots):
            slot_in = x[:, i, :]
            slot_out = getattr(self, f"slot_{i}")(slot_in)
            outs.append(slot_out.reshape(batch, self.n_virtual_per_slot, self.d_llm))
        return mx.concatenate(outs, axis=1)
