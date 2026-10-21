"""
Centralized 五行 (wuxing) relations: 相生, 相克, 生我, 克我.

Single source of truth for element relations used by ten-god logic, useful-god
heuristics, interaction transformation, and temporal useful-god activation.
"""
from __future__ import annotations

# 五行相生 (generates): key 生 value. 木→火→土→金→水→木.
ELEMENT_GENERATES: dict[str, str] = {"木": "火", "火": "土", "土": "金", "金": "水", "水": "木"}

# 我克: day master controls this element. 木克土, 土克水, 水克火, 火克金, 金克木.
ELEMENT_CONTROLS: dict[str, str] = {"木": "土", "土": "水", "水": "火", "火": "金", "金": "木"}

# 克我: element that controls (克) the given element. Used for ten-god and useful-god logic.
ELEMENT_CONTROLS_ME: dict[str, str] = {"木": "金", "金": "火", "火": "水", "水": "土", "土": "木"}

# 生我: element that generates the given element. Inverse of 生 (ELEMENT_GENERATES value→key).
ELEMENT_GENERATES_ME: dict[str, str] = {"火": "木", "土": "火", "金": "土", "水": "金", "木": "水"}
