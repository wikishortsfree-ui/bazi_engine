"""
Task heads for Phase 1.3 supervised objectives.

Each head consumes encoder output(s) and produces logits or probabilities for
a specific target (ten-god, element strength, interaction vector, useful god,
cross-temporal dynamic vector). Training loop and loss wiring are in Phase 1.4.
"""
from __future__ import annotations

from .cross_temporal_detector import CrossTemporalHead
from .element_regressor import ElementStrengthHead
from .interaction_detector import InteractionHead
from .ten_god_head import TenGodHead
from .useful_god_classifier import UsefulGodHead

__all__ = [
    "TenGodHead",
    "ElementStrengthHead",
    "InteractionHead",
    "UsefulGodHead",
    "CrossTemporalHead",
    "TASK_HEADS",
]

TASK_HEADS = {
    "ten_god": TenGodHead,
    "element_strength": ElementStrengthHead,
    "interaction": InteractionHead,
    "useful_god": UsefulGodHead,
    "cross_temporal": CrossTemporalHead,
}
