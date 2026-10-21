"""Bāzì (八字) calculator engine: datetime + location → four-pillar chart."""

from .calculator import BaziCalculator
from .enums import Branch, Pillar, Stem, TenGod
from .models import BaziChart
from .solar_time import to_true_solar_time
from .tables import get_hidden_stems
from .temporal import (
    CrossInteractionResult,
    DaYun,
    DaYunSequence,
    LiuNian,
    LiuRi,
    LiuYue,
    TemporalContext,
    check_useful_god_activation,
    compute_liunian,
    compute_liuri,
    compute_liuri_from_date,
    compute_liuyue,
    detect_cross_interactions,
)

__all__ = [
    "BaziCalculator",
    "BaziChart",
    "Branch",
    "check_useful_god_activation",
    "CrossInteractionResult",
    "DaYun",
    "DaYunSequence",
    "detect_cross_interactions",
    "LiuNian",
    "LiuRi",
    "LiuYue",
    "TemporalContext",
    "compute_liunian",
    "compute_liuri",
    "compute_liuri_from_date",
    "compute_liuyue",
    "Pillar",
    "Stem",
    "TenGod",
    "get_hidden_stems",
    "to_true_solar_time",
]
