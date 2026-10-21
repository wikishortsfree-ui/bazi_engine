"""
Heavenly Stems (天干), Earthly Branches (地支), and Pillar (柱) types.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Stem(str, Enum):
    """Ten heavenly stems (天干) in cycle order."""

    甲 = "甲"
    乙 = "乙"
    丙 = "丙"
    丁 = "丁"
    戊 = "戊"
    己 = "己"
    庚 = "庚"
    辛 = "辛"
    壬 = "壬"
    癸 = "癸"

    @property
    def element(self) -> str:
        """五行: 木/火/土/金/水."""
        return _STEM_ELEMENT[self]

    @property
    def polarity(self) -> str:
        """陰陽：陽 or 陰."""
        return _STEM_POLARITY[self]

    @property
    def index(self) -> int:
        """0–9 in cycle order."""
        return _STEM_INDEX[self]


_STEM_ELEMENT: dict[Stem, str] = {
    Stem.甲: "木",
    Stem.乙: "木",
    Stem.丙: "火",
    Stem.丁: "火",
    Stem.戊: "土",
    Stem.己: "土",
    Stem.庚: "金",
    Stem.辛: "金",
    Stem.壬: "水",
    Stem.癸: "水",
}
_STEM_POLARITY: dict[Stem, str] = {
    Stem.甲: "陽",
    Stem.乙: "陰",
    Stem.丙: "陽",
    Stem.丁: "陰",
    Stem.戊: "陽",
    Stem.己: "陰",
    Stem.庚: "陽",
    Stem.辛: "陰",
    Stem.壬: "陽",
    Stem.癸: "陰",
}


# Lookup by character for Pillar.from_string; index for O(1) .index property
_STEM_BY_CHAR = {s.value: s for s in Stem}
_STEM_INDEX = {s: i for i, s in enumerate(Stem)}


class Branch(str, Enum):
    """Twelve earthly branches (地支) in cycle order."""

    子 = "子"
    丑 = "丑"
    寅 = "寅"
    卯 = "卯"
    辰 = "辰"
    巳 = "巳"
    午 = "午"
    未 = "未"
    申 = "申"
    酉 = "酉"
    戌 = "戌"
    亥 = "亥"

    @property
    def element(self) -> str:
        """五行: 木/火/土/金/水."""
        return _BRANCH_ELEMENT[self]

    @property
    def polarity(self) -> str:
        """陰陽：陽 or 陰."""
        return _BRANCH_POLARITY[self]

    @property
    def index(self) -> int:
        """0–11 in cycle order."""
        return _BRANCH_INDEX[self]

    @property
    def season(self) -> str:
        """季節：春/夏/秋/冬/土 (辰戌丑未=土)."""
        return _BRANCH_SEASON[self]


_BRANCH_ELEMENT: dict[Branch, str] = {
    Branch.子: "水",
    Branch.丑: "土",
    Branch.寅: "木",
    Branch.卯: "木",
    Branch.辰: "土",
    Branch.巳: "火",
    Branch.午: "火",
    Branch.未: "土",
    Branch.申: "金",
    Branch.酉: "金",
    Branch.戌: "土",
    Branch.亥: "水",
}
_BRANCH_POLARITY: dict[Branch, str] = {
    Branch.子: "陽",
    Branch.丑: "陰",
    Branch.寅: "陽",
    Branch.卯: "陰",
    Branch.辰: "陽",
    Branch.巳: "陰",
    Branch.午: "陽",
    Branch.未: "陰",
    Branch.申: "陽",
    Branch.酉: "陰",
    Branch.戌: "陽",
    Branch.亥: "陰",
}
_BRANCH_SEASON: dict[Branch, str] = {
    Branch.子: "冬",
    Branch.丑: "土",
    Branch.寅: "春",
    Branch.卯: "春",
    Branch.辰: "土",
    Branch.巳: "夏",
    Branch.午: "夏",
    Branch.未: "土",
    Branch.申: "秋",
    Branch.酉: "秋",
    Branch.戌: "土",
    Branch.亥: "冬",
}

_BRANCH_BY_CHAR = {b.value: b for b in Branch}
_BRANCH_INDEX = {b: i for i, b in enumerate(Branch)}


class TenGod(str, Enum):
    """Ten gods (十神) relative to the day master, plus 日主 for the day stem itself."""

    比肩 = "比肩"
    劫财 = "劫财"
    食神 = "食神"
    伤官 = "伤官"
    偏财 = "偏财"
    正财 = "正财"
    七杀 = "七杀"
    正官 = "正官"
    偏印 = "偏印"
    正印 = "正印"
    日主 = "日主"


@dataclass(frozen=True)
class Pillar:
    """One pillar (柱): stem + branch at a given position, with ten-god labels."""

    stem: Stem
    branch: Branch
    position: str  # '年','月','日','時'
    ten_god_stem: TenGod
    ten_god_branch: list[TenGod]  # from ShiShenZhi, 主气→余气→杂气 order

    @classmethod
    def from_string(
        cls,
        s: str,
        position: str,
        *,
        ten_god_stem: TenGod,
        ten_god_branch: list[TenGod],
    ) -> Pillar:
        """Parse two-char pillar string (e.g. '癸未') into stem and branch."""
        if len(s) < 2:
            raise ValueError(f"Pillar string must be 2 chars, got {s!r}")
        stem_char = s[0]
        branch_char = s[1]
        stem = _STEM_BY_CHAR.get(stem_char)
        branch = _BRANCH_BY_CHAR.get(branch_char)
        if stem is None:
            raise ValueError(f"Unknown stem character: {stem_char!r}")
        if branch is None:
            raise ValueError(f"Unknown branch character: {branch_char!r}")
        return cls(
            stem=stem,
            branch=branch,
            position=position,
            ten_god_stem=ten_god_stem,
            ten_god_branch=list(ten_god_branch),
        )

    def __str__(self) -> str:
        return self.stem.value + self.branch.value
