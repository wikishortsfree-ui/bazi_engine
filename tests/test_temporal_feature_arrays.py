"""
Tests for TemporalContext.get_presence_mask() and BaziChart.to_feature_arrays() temporal arrays.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from bazi_engine import BaziCalculator, BaziChart, LiuNian, TemporalContext
from bazi_engine.encodings import BRANCH_TO_ID, STEM_TO_ID


def _test_chart() -> BaziChart:
    """Chart: 2003-04-24 21:20:00 America/Los_Angeles."""
    pst = ZoneInfo("America/Los_Angeles")
    dt = datetime(2003, 4, 24, 21, 20, 0, tzinfo=pst)
    return BaziCalculator.from_datetime(dt, 37.354107, -121.955238)


def test_presence_mask_partial_temporal():
    """Only 流年 present → correct mask (dayun/liuyue/liuri False, liunian True)."""
    ctx = TemporalContext(liunian=LiuNian(year=2024, stem="甲", branch="辰"))
    mask = ctx.get_presence_mask()
    assert mask["dayun"] is False
    assert mask["liunian"] is True
    assert mask["liuyue"] is False
    assert mask["liuri"] is False


def test_feature_arrays_uses_sentinel_for_absent():
    """Absent 流月 → liuyue_stem_id and liuyue_branch_id are -1; pillar_presence_mask[2]==0."""
    chart = _test_chart()
    chart.temporal_context = TemporalContext(
        liunian=LiuNian(year=2024, stem="甲", branch="辰"),
    )
    arr = chart.to_feature_arrays()
    assert "liuyue_stem_id" in arr and "liuyue_branch_id" in arr
    assert arr["liuyue_stem_id"][0] == -1
    assert arr["liuyue_branch_id"][0] == -1
    assert arr["pillar_presence_mask"][2] == 0  # liuyue absent


def test_temporal_mask_matches_presence():
    """pillar_presence_mask (dayun, liunian, liuyue) matches get_presence_mask()."""
    chart = _test_chart()
    ctx = TemporalContext(liunian=LiuNian(year=2024, stem="甲", branch="辰"))
    chart.temporal_context = ctx
    presence = ctx.get_presence_mask()
    arr = chart.to_feature_arrays()
    assert arr["pillar_presence_mask"][0] == (1 if presence["dayun"] else 0)
    assert arr["pillar_presence_mask"][1] == (1 if presence["liunian"] else 0)
    assert arr["pillar_presence_mask"][2] == (1 if presence["liuyue"] else 0)
    assert arr["liunian_stem_id"][0] == STEM_TO_ID["甲"]
    assert arr["liunian_branch_id"][0] == BRANCH_TO_ID["辰"]
