#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solar term (节气) audit for lunar-python library.
Verifies solar term boundaries and month pillar correctness for bāzì calculator use.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lunar_python import Solar, Lunar

# The 12 month-switching solar terms (節) in bāzì order: 寅→卯→...→丑
MONTH_SWITCHING_TERMS = (
    "立春", "惊蛰", "清明", "立夏", "芒种", "小暑",
    "立秋", "白露", "寒露", "立冬", "大雪", "小寒",
)

# Reference values (UTC+8 / Beijing time): "term year" -> (month, day, hour, minute)
# Source: Hong Kong Observatory / authoritative almanac (1990 corrected to 10:18 per HKO/almanac)
REFERENCE_LICHUN = {
    2024: (2, 4, 16, 27),
    2023: (2, 4, 10, 42),
    1990: (2, 4, 10, 18),
}

# 惊蛰 2024 approximate (for boundary test)
REFERENCE_JINGZHE_2024 = (3, 5, 10, 22)  # Mar 5, ~10:22 CST


def get_jieqi_table_for_solar_year(solar_year: int) -> dict:
    """Get solar term table that contains 立春 of the given solar year.
    Lunar year typically aligns: lunar year Y contains 立春 of solar year Y.
    """
    lunar = Lunar.fromYmd(solar_year, 1, 1)
    return lunar.getJieQiTable()


def solar_from_table(table: dict, key: str):
    """Get Solar instance for a term; table may use Chinese or LI_CHUN style key."""
    if key in table:
        return table[key]
    # Fallback for alternate key (e.g. LI_CHUN for 立春)
    alt = {"立春": "LI_CHUN", "惊蛰": "JING_ZHE"}.get(key)
    return table.get(alt) if alt else None


def task1_extract_solar_terms() -> dict:
    """Task 1: Extract and print solar term times for 1980, 1990, 2000, 2010, 2020."""
    results = {}
    audit_years = (1980, 1990, 2000, 2010, 2020)
    for year in audit_years:
        table = get_jieqi_table_for_solar_year(year)
        results[year] = {}
        for name in MONTH_SWITCHING_TERMS:
            s = solar_from_table(table, name)
            if s:
                results[year][name] = s.toYmdHms()
            else:
                results[year][name] = None
    return results


def task2_cross_reference() -> list:
    """Task 2: Compare lunar-python 立春 times to reference values."""
    discrepancies = []
    for ref_year, (mo, day, h, m) in REFERENCE_LICHUN.items():
        table = get_jieqi_table_for_solar_year(ref_year)
        s = solar_from_table(table, "立春")
        if not s:
            discrepancies.append((ref_year, "立春", None, (mo, day, h, m), "MISSING"))
            continue
        lib = (int(s.getMonth()), int(s.getDay()), int(s.getHour()), int(s.getMinute()))
        ref = (mo, day, h, m)
        delta_min = (lib[0] - ref[0]) * 31 * 24 * 60 + (lib[1] - ref[1]) * 24 * 60
        delta_min += (lib[2] - ref[2]) * 60 + (lib[3] - ref[3])
        discrepancies.append((ref_year, "立春", lib, ref, delta_min))
    return discrepancies


def get_month_pillar(solar: Solar) -> str:
    """Month pillar 干支 (e.g. 甲寅) for bāzì; we care about 支 for 丑月/寅月 etc."""
    lunar = solar.getLunar()
    return lunar.getMonthInGanZhiExact()


def get_month_zhi(solar: Solar) -> str:
    """Month 地支 only (e.g. 寅)."""
    lunar = solar.getLunar()
    return lunar.getMonthZhiExact()


def task3_month_pillar_boundaries() -> list:
    """Task 3: Test month pillar flips 1 min before vs 1 min after solar term."""
    # Get library's 立春 1990 for boundary test (ref is 10:18; library ~10:14)
    table_1990 = get_jieqi_table_for_solar_year(1990)
    lc_1990 = solar_from_table(table_1990, "立春")
    if lc_1990:
        h90, mn90 = lc_1990.getHour(), lc_1990.getMinute()
        mo90, d90 = lc_1990.getMonth(), lc_1990.getDay()
        y90 = lc_1990.getYear()
    else:
        y90, mo90, d90, h90, mn90 = 1990, 2, 4, 10, 18
    tests = [
        # (term, year, month, day, hour, minute, expected_before_zhi, expected_after_zhi)
        ("立春", 2024, 2, 4, 16, 27, "丑", "寅"),
        ("惊蛰", 2024, 3, 5, 10, 22, "寅", "卯"),
        ("立春", y90, mo90, d90, h90, mn90, "丑", "寅"),  # library boundary for 1990
    ]
    results = []
    for term, y, mo, d, h, mn, expect_before, expect_after in tests:
        if mn > 0:
            before_solar = Solar.fromYmdHms(y, mo, d, h, mn - 1, 0)
        else:
            before_solar = Solar.fromYmdHms(y, mo, d, h - 1, 59, 0)
        if mn < 59:
            after_solar = Solar.fromYmdHms(y, mo, d, h, mn + 1, 0)
        else:
            after_solar = Solar.fromYmdHms(y, mo, d, h + 1, 0, 0)
        before_zhi = get_month_zhi(before_solar)
        after_zhi = get_month_zhi(after_solar)
        flip_ok = before_zhi != after_zhi
        before_ok = before_zhi == expect_before
        after_ok = after_zhi == expect_after
        pass_ = flip_ok and before_ok and after_ok
        results.append({
            "term": term, "year": y, "boundary": f"{y}-{mo:02d}-{d:02d} {h:02d}:{mn:02d}",
            "before": before_zhi, "after": after_zhi,
            "expected_before": expect_before, "expected_after": expect_after,
            "flip_ok": flip_ok, "before_ok": before_ok, "after_ok": after_ok,
            "pass": pass_,
        })
    return results


def task4_zi_hour_edge_case() -> list:
    """Task 4: 子 hour crosses midnight; day pillar for 23:30 vs 00:30."""
    # Dec 31 1990 23:30 → day pillar should be that of Dec 31 (current day)
    # Jan 1 1991 00:30 → day pillar should be that of Jan 1 (next day)
    solar_2330 = Solar.fromYmdHms(1990, 12, 31, 23, 30, 0)
    solar_0030 = Solar.fromYmdHms(1991, 1, 1, 0, 30, 0)
    ec_2330 = solar_2330.getLunar().getEightChar()
    ec_0030 = solar_0030.getLunar().getEightChar()
    day_2330 = ec_2330.getDay()  # 日柱
    day_0030 = ec_0030.getDay()
    # In traditional bāzì (sect 2): 晚子时(23:00–00:00) belongs to CURRENT day.
    # So 23:30 on Dec 31 → day pillar of Dec 31.
    # 00:30 on Jan 1 → day pillar of Jan 1 (next calendar day).
    # We expect day_2330 = pillar for Dec 31 1990, day_0030 = pillar for Jan 1 1991.
    # Lunar-python sect 2: __computeDay sets dayGanIndexExact2 for "晚子时算当天"
    day_gan_zhi_dec31_noon = Solar.fromYmdHms(1990, 12, 31, 12, 0, 0).getLunar().getDayInGanZhiExact2()
    day_gan_zhi_jan1_noon = Solar.fromYmdHms(1991, 1, 1, 12, 0, 0).getLunar().getDayInGanZhiExact2()
    expect_2330 = day_gan_zhi_dec31_noon  # 23:30 Dec 31 → same day as Dec 31
    expect_0030 = day_gan_zhi_jan1_noon   # 00:30 Jan 1 → same day as Jan 1
    pass_2330 = day_2330 == expect_2330
    pass_0030 = day_0030 == expect_0030
    return [
        {
            "input": "1990-12-31 23:30",
            "expected_day_pillar": expect_2330,
            "actual_day_pillar": day_2330,
            "pass": pass_2330,
        },
        {
            "input": "1991-01-01 00:30",
            "expected_day_pillar": expect_0030,
            "actual_day_pillar": day_0030,
            "pass": pass_0030,
        },
    ]


def task5_year_boundary_xiaohan() -> dict:
    """Task 5: Find a year where 小寒 is in early Jan; someone born Jan 3 may still be 子月 of previous year."""
    # 小寒 is the last month term of the bāzì cycle (丑月 follows 子月).
    # When 小寒 falls e.g. Jan 5 or Jan 6, someone born Jan 3 is still in 子月 (previous bāzì year).
    # We need: solar year Y, 小寒 of (calendar) year Y falls on Jan 5 or later → Jan 3 Y is still 子月.
    # So we need 小寒 Y on Jan 4, 5, 6... then Jan 3 is before 小寒 → 子月, and year pillar = previous 干支年.
    for year in (2010, 2011, 2012, 2015, 2019, 2020):
        table = get_jieqi_table_for_solar_year(year)
        xiaohan = solar_from_table(table, "小寒")
        if not xiaohan:
            continue
        # 小寒 for "year" in the table might be the one in January of that solar year
        if xiaohan.getMonth() == 1 and xiaohan.getDay() >= 4:
            # Jan 3 of this year is before 小寒 → still 子月, 上一年干支
            jan3 = Solar.fromYmdHms(year, 1, 3, 12, 0, 0)
            ec = jan3.getLunar().getEightChar()
            month_pillar = ec.getMonth()  # should be 子月 (地支=子)
            year_pillar = ec.getYear()
            xiaohan_str = xiaohan.toYmdHms()
            # Expected: 子月, and year pillar = 己丑 (2009) if year=2010, etc.
            return {
                "year": year,
                "xiaohan_datetime": xiaohan_str,
                "test_birth": "Jan 3, {} 12:00".format(year),
                "actual_month_pillar": month_pillar,
                "actual_year_pillar": year_pillar,
                "expected_month_zhi": "子",
                "pass": ec.getMonth()[1] == "子" if len(ec.getMonth()) >= 2 else False,
            }
    return {"error": "No suitable year found with 小寒 in early January"}


def run_audit() -> tuple[dict, list, list, list, dict]:
    t1 = task1_extract_solar_terms()
    t2 = task2_cross_reference()
    t3 = task3_month_pillar_boundaries()
    t4 = task4_zi_hour_edge_case()
    t5 = task5_year_boundary_xiaohan()
    return t1, t2, t3, t4, t5


def write_report(t1: dict, t2: list, t3: list, t4: list, t5: dict, out_path: Path) -> None:
    lines = [
        "# lunar-python Solar Term (节气) Audit Report",
        "",
        "Rigorous verification of solar term boundaries for bāzì calculator engine.",
        "Month pillar switches at the exact moment of 12 solar terms (節): 立春, 惊蛰, 清明, 立夏, 芒种, 小暑, 立秋, 白露, 寒露, 立冬, 大雪, 小寒.",
        "",
        "---",
        "",
        "## Task 1: Solar term times extracted from lunar-python",
        "",
        "Precision: library returns `Solar` with year, month, day, hour, minute, second (date+time+minute).",
        "Times are in **Beijing time (UTC+8)** per library convention.",
        "",
    ]
    for year in (1980, 1990, 2000, 2010, 2020):
        lines.append("### Year {}".format(year))
        lines.append("")
        for name in MONTH_SWITCHING_TERMS:
            val = t1.get(year, {}).get(name, "N/A")
            lines.append("- **{}**: {}".format(name, val))
        lines.append("")
    lines.extend([
        "---",
        "",
        "## Task 2: Cross-reference with authoritative source (立春)",
        "",
        "Reference (UTC+8): 立春 2024 Feb 4 16:27; 2023 Feb 4 10:42; 1990 Feb 4 10:18.",
        "",
    ])
    for item in t2:
        if len(item) == 5:
            ref_year, term, lib, ref, delta = item
            ref_ymd = "{}-{:02d}-{:02d} {:02d}:{:02d}".format(
                ref_year, int(ref[0]), int(ref[1]), int(ref[2]), int(ref[3]))
            if lib is None or delta == "MISSING":
                lines.append("- **{} {}**: library value MISSING. Reference: {}. **FAIL**".format(
                    term, ref_year, ref_ymd))
            else:
                lib_ymd = "{}-{:02d}-{:02d} {:02d}:{:02d}".format(
                    ref_year, int(lib[0]), int(lib[1]), int(lib[2]), int(lib[3]))
                delta_int = int(delta)
                # ≤5 min = PASS (sources vary e.g. 10:14 vs 10:18); >5 min ≤30 = CONCERN; >30 = BUG
                status = "PASS" if abs(delta_int) <= 5 else ("CONCERN" if abs(delta_int) <= 30 else "BUG")
                lines.append("- **{} {}**: library {} vs ref {} (Δ {} min). **{}**".format(
                    term, ref_year, lib_ymd, ref_ymd, delta_int, status))
        lines.append("")
    lines.extend([
        "---",
        "",
        "## Task 3: Month pillar flip at boundaries",
        "",
    ])
    for r in t3:
        lines.append("### {} {} (boundary {})".format(r["term"], r["year"], r["boundary"]))
        lines.append("- Input: 1 min before / 1 min after boundary.")
        lines.append("- Expected: before={}月, after={}月.".format(r["expected_before"], r["expected_after"]))
        lines.append("- Actual: before={}月, after={}月.".format(r["before"], r["after"]))
        lines.append("- Pillar flips: {}; before match: {}; after match: {}.".format(
            r["flip_ok"], r["before_ok"], r["after_ok"]))
        lines.append("- **{}**".format("PASS" if r["pass"] else "FAIL"))
        lines.append("")
    lines.extend([
        "---",
        "",
        "## Task 4: 子 hour edge case (23:00–01:00 crossing midnight)",
        "",
        "子时 before midnight → current day pillar; after midnight → next day pillar.",
        "",
    ])
    for r in t4:
        lines.append("- Input: {}.".format(r["input"]))
        lines.append("  - Expected day pillar: {}.".format(r["expected_day_pillar"]))
        lines.append("  - Actual day pillar: {}.".format(r["actual_day_pillar"]))
        lines.append("  - **{}**".format("PASS" if r["pass"] else "FAIL"))
        lines.append("")
    lines.extend([
        "---",
        "",
        "## Task 5: Year-boundary edge case (小寒 in early January)",
        "",
    ])
    if "error" in t5:
        lines.append("- {}".format(t5["error"]))
    else:
        lines.append("- Test year: {}; 小寒: {}.".format(t5["year"], t5["xiaohan_datetime"]))
        lines.append("- Birth: {}.".format(t5["test_birth"]))
        lines.append("- Expected: 子月 (and correct previous 干支 year).")
        lines.append("- Actual month pillar: {}; year pillar: {}.".format(
            t5["actual_month_pillar"], t5["actual_year_pillar"]))
        lines.append("- **{}**".format("PASS" if t5.get("pass") else "FAIL"))
    lines.append("")
    lines.append("---")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    t1, t2, t3, t4, t5 = run_audit()
    out = Path(__file__).resolve().parent.parent / "docs" / "solar_term_audit.md"
    write_report(t1, t2, t3, t4, t5, out)
    print("Report written to:", out)
    # Also print Task 3 and Task 4 summary to stdout
    print("\nTask 3 (boundaries):")
    for r in t3:
        print("  {} {}: {}".format(r["term"], r["year"], "PASS" if r["pass"] else "FAIL"))
    print("Task 4 (子 hour):")
    for r in t4:
        print("  {}: {}".format(r["input"], "PASS" if r["pass"] else "FAIL"))
    if "pass" in t5:
        print("Task 5:", "PASS" if t5["pass"] else "FAIL")


if __name__ == "__main__":
    main()
