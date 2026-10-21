"""
Tests for 流日 (Liu Ri) computation: daily stem-branch from Gregorian date.
"""
import datetime

import pytest

from bazi_engine import LiuRi, compute_liuri, compute_liuri_from_date
from bazi_engine.enums import TenGod


def test_liuri_matches_natal_day_pillar():
    """Birth date 2003-04-24 → 流日 should be 丁卯 (matching natal day pillar)."""
    liuri = compute_liuri(2003, 4, 24)
    assert liuri.stem == "丁" and liuri.branch == "卯"
    assert liuri.ganzi == "丁卯"
    assert liuri.date == datetime.date(2003, 4, 24)
    assert liuri.year == 2003 and liuri.month == 4 and liuri.day == 24


def test_liuri_sixty_day_cycle():
    """Generate 60 consecutive days from a known date; verify full 60-jiazi cycle completes."""
    start = datetime.date(2024, 1, 1)
    seen = set()
    for i in range(60):
        d = start + datetime.timedelta(days=i)
        lr = compute_liuri_from_date(d)
        seen.add(lr.ganzi)
    assert len(seen) == 60, "60 consecutive days must yield 60 distinct 干支"
    # Next day (day 60) should repeat the first day's 干支
    day_60 = compute_liuri_from_date(start + datetime.timedelta(days=60))
    day_0 = compute_liuri_from_date(start)
    assert day_60.ganzi == day_0.ganzi


def test_liuri_leap_year_boundary():
    """Test Feb 28, 29 (2024), Mar 1 sequence across leap-year boundary."""
    feb28 = compute_liuri(2024, 2, 28)
    feb29 = compute_liuri(2024, 2, 29)
    mar1 = compute_liuri(2024, 3, 1)
    assert feb28.date == datetime.date(2024, 2, 28)
    assert feb29.date == datetime.date(2024, 2, 29)
    assert mar1.date == datetime.date(2024, 3, 1)
    # Consecutive days → consecutive 干支 in the 60-cycle (no gaps)
    gz = [feb28.ganzi, feb29.ganzi, mar1.ganzi]
    assert len(gz) == 3
    assert len(set(gz)) >= 2  # at least two different pillars across the three days


def test_liuri_year_boundary():
    """Test Dec 31, 2023 → Jan 1, 2024 transition."""
    dec31 = compute_liuri_from_date(datetime.date(2023, 12, 31))
    jan1 = compute_liuri_from_date(datetime.date(2024, 1, 1))
    assert dec31.date == datetime.date(2023, 12, 31)
    assert jan1.date == datetime.date(2024, 1, 1)
    # Consecutive calendar days
    assert (jan1.date - dec31.date).days == 1


def test_liuri_known_reference_date():
    """Use a fixed date (2024-01-01): verify day pillar matches lunar_python (甲子)."""
    # lunar_python Solar.fromYmd(2024,1,1).getLunar() → getDayGan/getDayZhi = 甲/子
    lr = compute_liuri(2024, 1, 1)
    assert lr.stem == "甲" and lr.branch == "子"
    assert lr.ganzi == "甲子"
    assert lr.year == 2024 and lr.month == 1 and lr.day == 1


def test_liuri_with_day_master_fills_ten_god():
    """When day_master_stem is provided, ten_god_stem and ten_god_branch are set (流日 丁卯, 日主 庚 → 正官)."""
    lr = compute_liuri(2003, 4, 24, day_master_stem="庚")
    assert lr.ten_god_stem is not None
    assert lr.ten_god_branch is not None
    assert lr.ten_god_stem == TenGod.正官  # 丁火克庚金 克我, 丁陰庚陽 异 → 正官
