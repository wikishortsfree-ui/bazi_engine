"""
Tests for 流年 (Liu Nian) and 流月 (Liu Yue) computation in temporal.py.
"""
import pytest

from bazi_engine.enums import TenGod
from bazi_engine.temporal import LiuNian, LiuYue, compute_liunian, compute_liuyue


def test_liunian_2024_is_jiachen():
    """流年 2024 is 甲辰 (year pillar from 立春)."""
    ln = compute_liunian(2024)
    assert isinstance(ln, LiuNian)
    assert ln.year == 2024
    assert ln.stem == "甲"
    assert ln.branch == "辰"
    assert ln.ganzi == "甲辰"


def test_liuyue_2003_04_is_bingchen():
    """流月 2003-04 is 丙辰 (month pillar for April 2003 by 节)."""
    ly = compute_liuyue(2003, 4)
    assert isinstance(ly, LiuYue)
    assert ly.year == 2003
    assert ly.month == 4
    assert ly.stem == "丙"
    assert ly.branch == "辰"
    assert ly.ganzi == "丙辰"


def test_liuyue_before_lichun_uses_previous_year():
    """Date before 立春 (e.g. Jan 2024) uses previous Bazi year stem 癸 (癸卯)."""
    ly = compute_liuyue(2024, 1)
    assert ly.year_stem == "癸"
    assert ly.year_branch == "卯"


def test_liunian_with_day_master_fills_ten_god():
    """When day_master_stem is provided, ten_god_stem and ten_god_branch are set (流年 甲辰, 日主 丁 → 正印)."""
    ln = compute_liunian(2024, day_master_stem="丁")
    assert ln.ten_god_stem is not None
    assert ln.ten_god_branch is not None
    assert ln.ten_god_stem == TenGod.正印  # 甲木生丁火, 生我, 甲陽丁陰 → 正印
    assert TenGod.七杀 in ln.ten_god_branch  # 辰藏癸, 癸水克丁火 克我 → 七杀


def test_liuyue_with_day_master_fills_ten_god():
    """When day_master_stem is provided, 流月 gets ten_god_stem and ten_god_branch (2003-04 丙辰, 日主 丁 → 劫财)."""
    ly = compute_liuyue(2003, 4, day_master_stem="丁")
    assert ly.ten_god_stem is not None
    assert ly.ten_god_branch is not None
    assert ly.ten_god_stem == TenGod.劫财  # 丙丁同火, 丙陽丁陰 异 → 劫财
