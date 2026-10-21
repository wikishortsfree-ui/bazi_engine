"""
Package-level constants for bāzì (八字) engine.
"""
from __future__ import annotations

# 4 pillars: Year, Month, Day, Hour. Single source of truth for position labels.
PILLAR_COUNT = 4
POSITION_LABELS: tuple[str, str, str, str] = ("year", "month", "day", "hour")

# Sentinel for absent temporal pillars in feature arrays (e.g. 流月 not set → stem/branch ID = -1)
NULL_PILLAR_ID = -1
