"""
Tests for Da Yun (大运) computation and DaYunSequence.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from bazi_engine import BaziCalculator, BaziChart, DaYun, DaYunSequence


def _test_chart() -> BaziChart:
    """Chart: 2003-04-24 21:20:00 America/Los_Angeles, male/female tests."""
    pst = ZoneInfo("America/Los_Angeles")
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=pst)
    lat, lng = 37.354107, -121.955238
    return BaziCalculator.from_datetime(dt, lat, lng)


def test_first_real_dayun_yimao_age_7_year_2009():
    """First real Da Yun for test chart (2003-04-24 21:20, male) is 乙卯 at age 7 (year 2009)."""
    chart = _test_chart()
    calc = BaziCalculator()
    seq = calc.compute_dayun(chart, gender=1, max_periods=10)
    assert len(seq.periods) >= 1
    first = seq.periods[0]
    assert first.stem == "乙" and first.branch == "卯"
    assert first.gan_zhi == "乙卯"
    assert first.start_year == 2009
    assert first.start_age == 7
    assert first.end_year == 2018
    assert first.end_age == 16
    assert first.index == 0


def test_get_period_for_year():
    """get_period_for_year returns the active Da Yun for a given Gregorian year."""
    chart = _test_chart()
    seq = BaziCalculator().compute_dayun(chart, gender=1, max_periods=10)
    # 2009–2018 is first real period 乙卯
    p = seq.get_period_for_year(2009)
    assert p is not None
    assert p.gan_zhi == "乙卯"
    assert seq.get_period_for_year(2010) is p
    assert seq.get_period_for_year(2018) is p
    assert seq.get_period_for_year(2008) is None  # before first period
    # 2019–2028 is next period
    p2 = seq.get_period_for_year(2019)
    assert p2 is not None
    assert p2.gan_zhi != "乙卯"


def test_get_period_for_age():
    """get_period_for_age returns the Da Yun period for a given age."""
    chart = _test_chart()
    seq = BaziCalculator().compute_dayun(chart, gender=1, max_periods=10)
    p = seq.get_period_for_age(7)
    assert p is not None
    assert p.gan_zhi == "乙卯"
    assert seq.get_period_for_age(10) is p
    assert seq.get_period_for_age(16) is p
    assert seq.get_period_for_age(6) is None
    assert seq.get_period_for_age(17) is not None
    assert seq.get_period_for_age(17).gan_zhi != "乙卯"


def test_empty_ganzhi_filtered():
    """Periods with empty ganZhi (pre-start) are not included in the sequence."""
    chart = _test_chart()
    seq = BaziCalculator().compute_dayun(chart, gender=1, max_periods=10)
    for period in seq.periods:
        assert period.stem and period.branch
        assert len(period.gan_zhi) == 2


def test_gender_reversal_direction():
    """Male and female have reversed Da Yun direction (different first period after pre-start)."""
    chart = _test_chart()
    calc = BaziCalculator()
    male = calc.compute_dayun(chart, gender=1, max_periods=3)
    female = calc.compute_dayun(chart, gender=0, max_periods=3)
    # Male: first real period 乙卯 (forward from month pillar 丙辰)
    # Female: first real period should be the opposite direction (e.g. 乙巳 or 甲辰, etc.)
    assert len(male.periods) >= 1 and len(female.periods) >= 1
    # At least the first stem+branch should differ by gender
    assert male.periods[0].gan_zhi != female.periods[0].gan_zhi or male.periods[0].start_year != female.periods[0].start_year


def test_max_periods_cap():
    """max_periods limits the number of periods returned."""
    chart = _test_chart()
    seq = BaziCalculator().compute_dayun(chart, gender=1, max_periods=3)
    assert len(seq.periods) <= 3
