"""
Encoding tables for stem, branch, element, ten-god, and useful-god IDs.

Used by BaziChart.to_feature_arrays() for encoder input. Do not change IDs after
Phase 1 training starts (frozen contract).
"""
from __future__ import annotations

# Stem encoding: 甲=0, 乙=1, ..., 癸=9 (cycle order)
STEM_TO_ID: dict[str, int] = {
    "甲": 0, "乙": 1, "丙": 2, "丁": 3, "戊": 4,
    "己": 5, "庚": 6, "辛": 7, "壬": 8, "癸": 9,
}

# Branch encoding: 子=0, 丑=1, ..., 亥=11 (cycle order)
BRANCH_TO_ID: dict[str, int] = {
    "子": 0, "丑": 1, "寅": 2, "卯": 3, "辰": 4, "巳": 5,
    "午": 6, "未": 7, "申": 8, "酉": 9, "戌": 10, "亥": 11,
}

# Element encoding: 木=0, 火=1, 土=2, 金=3, 水=4
ELEMENT_TO_ID: dict[str, int] = {
    "木": 0, "火": 1, "土": 2, "金": 3, "水": 4,
}

# Ten God encoding; 日主=10 for day stem
TEN_GOD_TO_ID: dict[str, int] = {
    "比肩": 0, "劫财": 1, "食神": 2, "伤官": 3, "偏财": 4,
    "正财": 5, "七杀": 6, "正官": 7, "偏印": 8, "正印": 9,
    "日主": 10,
}

# Useful God (用神) element-based categories: 比劫=0, 食伤=1, 财=2, 官杀=3, 印=4
USEFUL_GOD_TO_ID: dict[str, int] = {
    "比劫": 0, "食伤": 1, "财": 2, "官杀": 3, "印": 4,
}

# Reverse lookups for round-trip validation
ID_TO_STEM: dict[int, str] = {v: k for k, v in STEM_TO_ID.items()}
ID_TO_BRANCH: dict[int, str] = {v: k for k, v in BRANCH_TO_ID.items()}
ID_TO_ELEMENT: dict[int, str] = {v: k for k, v in ELEMENT_TO_ID.items()}


def stem_polarity(stem: str) -> int:
    """0=yin, 1=yang. Stems: even index=yang, odd=yin."""
    return 1 - (STEM_TO_ID[stem] % 2)  # even index → yang(1), odd → yin(0)


def branch_polarity(branch: str) -> int:
    """0=yin, 1=yang. Branches: even index=yang, odd=yin."""
    return 1 - (BRANCH_TO_ID[branch] % 2)
