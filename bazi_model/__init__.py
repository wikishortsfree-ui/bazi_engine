"""
bazi_model: Neural encoder and task heads for Bazi charts.

Consumes feature arrays from bazi_engine.BaziChart.to_feature_arrays()
and produces fixed-size embeddings for downstream tasks (ten-god, element
strength, interactions, useful god). Depends on PyTorch; bazi_engine has
no PyTorch dependency.
"""
from __future__ import annotations

__all__: list[str] = []
