"""
Bāzì (八字) calculator: from datetime + location to four-pillar chart.
"""
from __future__ import annotations

# Allow running as script: python bazi_engine/calculator.py (set parent package before relative imports)
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))
    __package__ = "bazi_engine"

from datetime import datetime
from zoneinfo import ZoneInfo

from lunar_python import Solar

from .enums import Branch, Pillar, TenGod
from .models import BaziChart
from .solar_time import to_true_solar_time
from .tables import get_hidden_stems
from .temporal import DaYun, DaYunSequence

_TEN_GOD_STR_TO_ENUM = {tg.value: tg for tg in TenGod}


def _parse_ten_god(s: str) -> TenGod:
    v = _TEN_GOD_STR_TO_ENUM.get(s)
    if v is None:
        raise ValueError(f"Unknown ten-god string: {s!r}")
    return v


def _pillar_from_ec(
    position: str,
    pillar_str: str,
    ten_god_stem_str: str,
    ten_god_branch_list: list,
) -> Pillar:
    return Pillar.from_string(
        pillar_str,
        position,
        ten_god_stem=_parse_ten_god(ten_god_stem_str),
        ten_god_branch=[_parse_ten_god(x) for x in ten_god_branch_list],
    )


class BaziCalculator:
    """Computes a BaziChart from birth datetime and location."""

    @classmethod
    def from_datetime(cls, dt: datetime, lat: float, lng: float) -> BaziChart:
        """
        Build a BaziChart for the given timezone-aware datetime at (lat, lng).

        Uses true local solar time (longitude + Equation of Time) for the
        four-pillar calculation, then lunar_python with sect=2 (盲派).
        """
        if dt.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")

        solar_dt = to_true_solar_time(dt, lat, lng)
        solar = Solar.fromYmdHms(
            solar_dt.year,
            solar_dt.month,
            solar_dt.day,
            solar_dt.hour,
            solar_dt.minute,
            solar_dt.second,
        )
        lunar = solar.getLunar()
        ec = lunar.getEightChar()
        # Explicit 盲派 (sect=2): ensures consistent day-master and pillar logic.
        ec.setSect(2)

        ec_slots = [
            ("年", ec.getYear, ec.getYearShiShenGan, ec.getYearShiShenZhi),
            ("月", ec.getMonth, ec.getMonthShiShenGan, ec.getMonthShiShenZhi),
            ("日", ec.getDay, ec.getDayShiShenGan, ec.getDayShiShenZhi),
            ("時", ec.getTime, ec.getTimeShiShenGan, ec.getTimeShiShenZhi),
        ]
        pillars = []
        for position, get_str, get_stem_shen, get_branch_shen in ec_slots:
            pillars.append(_pillar_from_ec(
                position, get_str(), get_stem_shen(), get_branch_shen()
            ))
        year_pillar, month_pillar, day_pillar, hour_pillar = pillars

        return BaziChart(
            birth_dt=dt,
            solar_dt=solar_dt,
            lat=lat,
            lng=lng,
            year_pillar=year_pillar,
            month_pillar=month_pillar,
            day_pillar=day_pillar,
            hour_pillar=hour_pillar,
            sect=2,
        )

    def compute_dayun(
        self,
        chart: BaziChart,
        gender: int,
        max_periods: int = 10,
    ) -> DaYunSequence:
        """
        Compute Da Yun (大运) sequence for a chart.

        Uses the chart's true solar birth time and sect so the sequence matches
        the four-pillar calculation. The first period from lunar-python may have
        empty ganZhi (pre-start); such periods are excluded.

        Args:
            chart: BaziChart from from_datetime (provides solar_dt, sect).
            gender: 0 for female, 1 for male (determines forward/backward 大运).
            max_periods: Maximum number of periods to include (default 10 = 100 years).

        Returns:
            DaYunSequence with all computed periods (empty-ganZhi filtered out).
        """
        solar_dt = chart.solar_dt
        solar = Solar.fromYmdHms(
            solar_dt.year,
            solar_dt.month,
            solar_dt.day,
            solar_dt.hour,
            solar_dt.minute,
            solar_dt.second,
        )
        lunar = solar.getLunar()
        ec = lunar.getEightChar()
        ec.setSect(chart.sect)

        yun = ec.getYun(gender)
        raw_da_yun = yun.getDaYun()

        periods: list[DaYun] = []
        for i, dy in enumerate(raw_da_yun):
            if i >= max_periods:
                break
            gan_zhi = dy.getGanZhi()
            if not gan_zhi or len(gan_zhi) < 2:
                continue
            stem, branch = gan_zhi[0], gan_zhi[1]
            periods.append(
                DaYun(
                    stem=stem,
                    branch=branch,
                    start_year=dy.getStartYear(),
                    end_year=dy.getEndYear(),
                    start_age=dy.getStartAge(),
                    end_age=dy.getEndAge(),
                    index=len(periods),
                )
            )

        return DaYunSequence(periods=periods, gender=gender)


if __name__ == "__main__":
    pst = ZoneInfo("America/Los_Angeles")
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=pst)
    lat, lng = 37.354107, -121.955238
    chart = BaziCalculator.from_datetime(dt, lat, lng)

    expected = {"year": "癸未", "month": "丙辰", "day": "丁卯", "hour": "庚戌"}
    actual = {}
    for key in expected:
        val = getattr(chart, f"{key}_str")
        actual[key] = val
        print(f"{key.capitalize()}:", val)
    ok = all(actual[key] == expected[key] for key in expected)
    if ok:
        print("PASS")
    else:
        print("FAIL")
        for key in expected:
            if actual[key] != expected[key]:
                print(f"  {key}: expected {expected[key]!r}, got {actual[key]!r}")
    assert ok, f"expected {expected}, got {actual}"

    print("year_pillar.stem.element:", chart.year_pillar.stem.element)
    print("day_pillar.stem.polarity:", chart.day_pillar.stem.polarity)
    print("hour_pillar.branch.season:", chart.hour_pillar.branch.season)
    assert chart.year_pillar.stem.element == "水", "癸 is Water"
    assert chart.day_pillar.stem.polarity == "陰", "丁 is yin Fire"
    assert chart.hour_pillar.branch.season == "土", "this chart: 庚戌 → 戌=土"
    assert Branch.亥.season == "冬", "亥 is winter"
    print("(亥.season:", Branch.亥.season, "→ 冬)")
    print("Pillar enum checks PASS")

    assert [s.value for s, _ in get_hidden_stems(Branch.未)] == ["己", "丁", "乙"]

    assert chart.year_pillar.ten_god_stem == TenGod.七杀
    assert chart.month_pillar.ten_god_stem == TenGod.劫财
    assert chart.hour_pillar.ten_god_stem == TenGod.正财
    assert chart.day_pillar.ten_god_branch == [TenGod.偏印]
    print("Ten-god checks PASS")
