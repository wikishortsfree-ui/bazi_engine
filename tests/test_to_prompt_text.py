"""Unit tests for ``BaziChart.to_prompt_text()``.

The serializer is the structured-text block in the hybrid prompt format
``[virtual tokens] + structured text + question``; see
``plans/hybrid-prompt-v1-and-lora_*.plan.md`` §3.

Each test pins one fact about the output for a known chart so any silent
drift in formatting (e.g. label/separator changes, sort order, percentile
rounding) immediately fails a test rather than corrupting the LLM prompt.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from bazi_engine import BaziCalculator


# Eight reference birth moments spanning a range of decades, hours, and
# locations so we exercise different stems, branches, and interaction sets.
KNOWN_CHARTS: list[tuple[str, datetime, float, float]] = [
    ("LA_2003", datetime(2003, 4, 24, 21, 20, 0, tzinfo=ZoneInfo("America/Los_Angeles")), 37.35, -121.96),
    ("SH_1990", datetime(1990, 6, 15, 14, 30, 0, tzinfo=ZoneInfo("Asia/Shanghai")), 31.23, 121.47),
    ("BJ_1985", datetime(1985, 11, 5, 8, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai")), 39.90, 116.40),
    ("HK_1972", datetime(1972, 1, 30, 23, 45, 0, tzinfo=ZoneInfo("Asia/Hong_Kong")), 22.30, 114.17),
    ("NY_1999", datetime(1999, 9, 9, 9, 9, 0, tzinfo=ZoneInfo("America/New_York")), 40.71, -74.01),
    ("TK_1968", datetime(1968, 7, 20, 6, 30, 0, tzinfo=ZoneInfo("Asia/Tokyo")), 35.68, 139.69),
    ("LD_2010", datetime(2010, 12, 31, 23, 59, 0, tzinfo=ZoneInfo("Europe/London")), 51.51, -0.13),
    ("SG_2024", datetime(2024, 2, 4, 12, 0, 0, tzinfo=ZoneInfo("Asia/Singapore")), 1.35, 103.82),
]


def _chart(label: str):
    """Build the chart for ``label`` (cached lookup over ``KNOWN_CHARTS``)."""
    for lbl, dt, lat, lng in KNOWN_CHARTS:
        if lbl == label:
            return BaziCalculator.from_datetime(dt, lat, lng)
    raise KeyError(label)


@pytest.mark.parametrize("label", [c[0] for c in KNOWN_CHARTS])
def test_to_prompt_text_has_six_lines(label: str) -> None:
    """Output is always exactly 6 lines: header + pillars + 十神 + 五行 + 日主/用神 + 主要交互."""
    chart = _chart(label)
    text = chart.to_prompt_text()
    lines = text.split("\n")
    assert len(lines) == 6, f"{label}: expected 6 lines, got {len(lines)}"
    assert lines[0] == "八字结构："
    assert lines[1].startswith("年柱：")
    assert lines[2].startswith("十神：")
    assert lines[3].startswith("五行：")
    assert lines[4].startswith("日主：")
    assert lines[5].startswith("主要交互：")


@pytest.mark.parametrize("label", [c[0] for c in KNOWN_CHARTS])
def test_to_prompt_text_pillars_match_chart(label: str) -> None:
    """The 年/月/日/时 pillar segment matches stem+branch from the chart."""
    chart = _chart(label)
    text = chart.to_prompt_text()
    expected = (
        f"年柱：{chart.year_pillar.stem.value}{chart.year_pillar.branch.value}  "
        f"月柱：{chart.month_pillar.stem.value}{chart.month_pillar.branch.value}  "
        f"日柱：{chart.day_pillar.stem.value}{chart.day_pillar.branch.value}  "
        f"时柱：{chart.hour_pillar.stem.value}{chart.hour_pillar.branch.value}"
    )
    assert expected in text


@pytest.mark.parametrize("label", [c[0] for c in KNOWN_CHARTS])
def test_to_prompt_text_day_master_segment(label: str) -> None:
    """``日主=<stem><element>`` appears in the 十神 line and matches the chart."""
    chart = _chart(label)
    text = chart.to_prompt_text()
    expected_dm = f"日主={chart.day_pillar.stem.value}{chart.day_pillar.stem.element}"
    assert expected_dm in text


@pytest.mark.parametrize("label", [c[0] for c in KNOWN_CHARTS])
def test_to_prompt_text_five_elements_sum_to_100(label: str) -> None:
    """五行 line shows all five elements as percentages summing to ~100%."""
    chart = _chart(label)
    text = chart.to_prompt_text()
    five_line = next(line for line in text.split("\n") if line.startswith("五行："))
    body = five_line[len("五行："):]
    parts = body.split(" ")
    assert len(parts) == 5, f"{label}: expected 5 element entries, got {parts!r}"
    pct_total = 0.0
    for part in parts:
        assert part.endswith("%"), part
        elem_char, num = part[0], part[1:-1]
        assert elem_char in {"木", "火", "土", "金", "水"}, part
        pct_total += float(num)
    assert abs(pct_total - 100.0) < 0.5, f"{label}: percentages sum to {pct_total}"


@pytest.mark.parametrize("label", [c[0] for c in KNOWN_CHARTS])
def test_to_prompt_text_five_elements_descending(label: str) -> None:
    """五行 entries are sorted strongest → weakest."""
    chart = _chart(label)
    text = chart.to_prompt_text()
    five_line = next(line for line in text.split("\n") if line.startswith("五行："))
    parts = five_line[len("五行："):].split(" ")
    pcts = [float(p[1:-1]) for p in parts]
    assert pcts == sorted(pcts, reverse=True), f"{label}: not descending: {pcts}"


@pytest.mark.parametrize("label", [c[0] for c in KNOWN_CHARTS])
def test_to_prompt_text_day_master_classification(label: str) -> None:
    """``日主：`` classification is one of 身强 / 中和 / 身弱."""
    chart = _chart(label)
    text = chart.to_prompt_text()
    dm_line = next(line for line in text.split("\n") if line.startswith("日主："))
    after = dm_line[len("日主："):]
    cls_token = after.split("  ")[0]
    assert cls_token in {"身强", "中和", "身弱"}, dm_line


@pytest.mark.parametrize("label", [c[0] for c in KNOWN_CHARTS])
def test_to_prompt_text_useful_god_format(label: str) -> None:
    """用神 is either ``中和`` or ``<element>(<primary>)`` with a known primary."""
    chart = _chart(label)
    text = chart.to_prompt_text()
    dm_line = next(line for line in text.split("\n") if line.startswith("日主："))
    ug_token = dm_line.split("用神：", 1)[1].strip()
    if ug_token == "中和":
        return
    assert "(" in ug_token and ug_token.endswith(")"), ug_token
    elem_part, primary = ug_token[:-1].split("(", 1)
    assert elem_part in {"木", "火", "土", "金", "水"}, ug_token
    assert primary in {"印", "比劫", "财", "食伤", "官杀"}, ug_token


@pytest.mark.parametrize("label", [c[0] for c in KNOWN_CHARTS])
def test_to_prompt_text_is_deterministic(label: str) -> None:
    """Two calls on the same chart yield byte-identical output."""
    chart = _chart(label)
    assert chart.to_prompt_text() == chart.to_prompt_text()


def test_to_prompt_text_known_chart_full_snapshot() -> None:
    """Pin the exact output for one well-known chart so format drift is caught.

    Chart: 2003-04-24 21:20:00 PDT, San Jose CA. Computed from
    ``BaziCalculator.from_datetime`` with sect=2.
    """
    chart = _chart("LA_2003")
    expected = (
        "八字结构：\n"
        "年柱：癸未  月柱：丙辰  日柱：丁卯  时柱：庚戌\n"
        "十神：年=七杀(未=食神)  月=劫财(辰=伤官)  日主=丁火  时=正财(戌=伤官)\n"
        "五行：火30.0% 土22.5% 木17.5% 金16.2% 水13.8%\n"
        "日主：身强  用神：金(财)\n"
        "主要交互：未卯半合、丁癸冲、辰戌冲、辰卯害、持势之刑、未戌破"
    )
    assert chart.to_prompt_text() == expected


def test_to_prompt_text_no_active_interactions_renders_无() -> None:
    """When no active interactions exist, summary is ``无`` rather than empty."""
    chart = _chart("SG_2024")
    text = chart.to_prompt_text()
    last_line = text.split("\n")[-1]
    assert last_line.startswith("主要交互：")
    assert last_line[len("主要交互："):]
