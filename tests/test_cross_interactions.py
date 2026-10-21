"""
Tests for cross-context interactions: natal vs temporal (dayun/liunian/liuyue/liuri) and dayun vs liunian.
"""
import pytest

from bazi_engine.temporal import (
    DaYun,
    LiuNian,
    TemporalContext,
    CrossInteractionResult,
    detect_cross_interactions,
    check_useful_god_activation,
)


def test_cross_stem_clash():
    """Natal 丁 + 流年 癸 → natal_liunian_clashes has one entry (丁癸 天干相冲)."""
    natal_stems = ["甲", "乙", "丁", "丙"]  # 丁 at day (pos 2)
    natal_branches = ["子", "丑", "寅", "卯"]
    liunian = LiuNian(year=2024, stem="癸", branch="卯")
    ctx = TemporalContext(liunian=liunian)
    result = detect_cross_interactions(natal_stems, natal_branches, ctx)
    assert isinstance(result, CrossInteractionResult)
    assert len(result.natal_liunian_stem_clashes) == 1
    entry = result.natal_liunian_stem_clashes[0]
    assert entry["natal_pos"] == 2
    assert set([entry["item_a"], entry["item_b"]]) == {"丁", "癸"}
    assert result.has_natal_liunian_interactions is True


def test_cross_branch_combination():
    """Natal 卯 + Da Yun 戌 → natal_dayun_combinations has one entry (卯戌 六合)."""
    natal_stems = ["甲", "乙", "丙", "丁"]
    natal_branches = ["子", "丑", "寅", "卯"]  # 卯 at hour (pos 3)
    dayun = DaYun(
        stem="甲", branch="戌",
        start_year=2020, end_year=2030, start_age=30, end_age=40, index=0,
    )
    ctx = TemporalContext(dayun=dayun)
    result = detect_cross_interactions(natal_stems, natal_branches, ctx)
    assert len(result.natal_dayun_branch_combinations) == 1
    entry = result.natal_dayun_branch_combinations[0]
    assert entry["natal_pos"] == 3
    assert set([entry["branch_a"], entry["branch_b"]]) == {"卯", "戌"}
    assert entry["result_element"] == "火"
    assert result.has_natal_dayun_interactions is True


def test_no_temporal_context_empty_lists():
    """Empty TemporalContext → all interaction lists empty and flags False."""
    natal_stems = ["甲", "乙", "丙", "丁"]
    natal_branches = ["子", "丑", "寅", "卯"]
    ctx = TemporalContext()
    result = detect_cross_interactions(natal_stems, natal_branches, ctx)
    assert len(result.natal_dayun_stem_combinations) == 0
    assert len(result.natal_dayun_stem_clashes) == 0
    assert len(result.natal_dayun_branch_combinations) == 0
    assert len(result.natal_dayun_branch_conflicts) == 0
    assert len(result.natal_liunian_stem_clashes) == 0
    assert len(result.natal_liuyue_stem_combinations) == 0
    assert len(result.natal_liuri_stem_clashes) == 0
    assert len(result.dayun_liunian_stem_clashes) == 0
    assert len(result.dayun_liunian_branch_clashes) == 0
    assert result.has_natal_dayun_interactions is False
    assert result.has_natal_liunian_interactions is False
    assert result.has_natal_liuyue_interactions is False
    assert result.has_natal_liuri_interactions is False


def test_dayun_liunian_interaction():
    """Da Yun 寅 + 流年 申 → dayun_liunian_clashes has entry (寅申 六冲)."""
    natal_stems = ["甲", "乙", "丙", "丁"]
    natal_branches = ["子", "丑", "寅", "卯"]
    dayun = DaYun(
        stem="甲", branch="寅",
        start_year=2020, end_year=2030, start_age=30, end_age=40, index=0,
    )
    liunian = LiuNian(year=2024, stem="庚", branch="申")
    ctx = TemporalContext(dayun=dayun, liunian=liunian)
    result = detect_cross_interactions(natal_stems, natal_branches, ctx)
    assert len(result.dayun_liunian_branch_clashes) == 1
    entry = result.dayun_liunian_branch_clashes[0]
    assert set([entry["item_a"], entry["item_b"]]) == {"寅", "申"}


def test_elemental_delta_fire_liunian_increases_fire():
    """Natal with weak 火, 流年 火 → 火 delta > 0."""
    # Natal: 甲甲乙壬 + 子丑寅亥 → little fire (only from 寅 藏干 and possibly 亥)
    natal_stems = ["甲", "甲", "乙", "壬"]
    natal_branches = ["子", "丑", "寅", "亥"]
    liunian = LiuNian(year=2024, stem="丙", branch="午")  # 丙丁→火, 午→fire
    ctx = TemporalContext(liunian=liunian)
    result = detect_cross_interactions(natal_stems, natal_branches, ctx)
    assert result.elemental_delta["火"] > 0


def test_elemental_delta_sum_near_zero():
    """Sum of all 5 element deltas should be ≈ 0 (both natal and extended normalize to 1.0)."""
    natal_stems = ["癸", "丙", "丁", "辛"]
    natal_branches = ["未", "辰", "卯", "亥"]
    liunian = LiuNian(year=2024, stem="甲", branch="辰")
    ctx = TemporalContext(liunian=liunian)
    result = detect_cross_interactions(natal_stems, natal_branches, ctx)
    delta_sum = sum(result.elemental_delta[e] for e in ("木", "火", "土", "金", "水"))
    assert delta_sum == pytest.approx(0.0)


def test_total_shift_magnitude():
    """total_shift_magnitude is the sum of absolute elemental deltas."""
    natal_stems = ["甲", "乙", "丙", "丁"]
    natal_branches = ["子", "丑", "寅", "卯"]
    liunian = LiuNian(year=2024, stem="庚", branch="申")
    ctx = TemporalContext(liunian=liunian)
    result = detect_cross_interactions(natal_stems, natal_branches, ctx)
    expected = sum(abs(result.elemental_delta[e]) for e in ("木", "火", "土", "金", "水"))
    assert result.total_shift_magnitude == pytest.approx(expected)


def test_useful_god_active_when_element_increased():
    """用神木, 流年木, delta['木'] > 0.05 → 'active'."""
    # Natal with little 木; 流年 甲寅 adds strong 木 so delta 木 > 0.05
    natal_stems = ["癸", "丙", "丁", "辛"]
    natal_branches = ["未", "辰", "卯", "亥"]
    liunian = LiuNian(year=2024, stem="甲", branch="寅")  # 甲=木, 寅=木
    ctx = TemporalContext(liunian=liunian)
    result = detect_cross_interactions(natal_stems, natal_branches, ctx, useful_god="木")
    assert result.elemental_delta["木"] > 0.05
    assert result.useful_god_activation == "active"
    assert "得令" in (result.useful_god_analysis or "")


def test_useful_god_suppressed_when_clashed():
    """用神木, 流年金克木 or 寅申冲 → 'suppressed'."""
    # 大运 寅 + 流年 申 → 寅申冲 (木-金), 用神木 suppressed
    natal_stems = ["甲", "乙", "丙", "丁"]
    natal_branches = ["子", "丑", "寅", "卯"]
    dayun = DaYun(stem="甲", branch="寅", start_year=2020, end_year=2030, start_age=30, end_age=40, index=0)
    liunian = LiuNian(year=2024, stem="庚", branch="申")
    ctx = TemporalContext(dayun=dayun, liunian=liunian)
    result = detect_cross_interactions(natal_stems, natal_branches, ctx, useful_god="木")
    assert result.useful_god_activation == "suppressed"
    assert "六冲" in (result.useful_god_analysis or "") or "克" in (result.useful_god_analysis or "")


def test_useful_god_neutral_small_delta():
    """delta['木'] = 0.02 (in [-0.05, 0.05]) → 'neutral'."""
    # Call check_useful_god_activation directly with small delta and no clashes
    empty = []
    no_clash_result = CrossInteractionResult(
        natal_dayun_stem_combinations=empty,
        natal_dayun_stem_clashes=empty,
        natal_dayun_branch_combinations=empty,
        natal_dayun_branch_conflicts=empty,
        natal_liunian_stem_combinations=empty,
        natal_liunian_stem_clashes=empty,
        natal_liunian_branch_combinations=empty,
        natal_liunian_branch_conflicts=empty,
        natal_liuyue_stem_combinations=empty,
        natal_liuyue_stem_clashes=empty,
        natal_liuyue_branch_combinations=empty,
        natal_liuyue_branch_conflicts=empty,
        natal_liuri_stem_combinations=empty,
        natal_liuri_stem_clashes=empty,
        natal_liuri_branch_combinations=empty,
        natal_liuri_branch_conflicts=empty,
        dayun_liunian_stem_combinations=empty,
        dayun_liunian_stem_clashes=empty,
        dayun_liunian_branch_combinations=empty,
        dayun_liunian_branch_clashes=empty,
        dayun_liuyue_stem_combinations=empty,
        dayun_liuyue_stem_clashes=empty,
        dayun_liuyue_branch_combinations=empty,
        dayun_liuyue_branch_clashes=empty,
        liunian_liuyue_stem_combinations=empty,
        liunian_liuyue_stem_clashes=empty,
        liunian_liuyue_branch_combinations=empty,
        liunian_liuyue_branch_clashes=empty,
        has_natal_dayun_interactions=False,
        has_natal_liunian_interactions=False,
        has_natal_liuyue_interactions=False,
        has_natal_liuri_interactions=False,
        elemental_delta={"木": 0.02, "火": 0.0, "土": 0.0, "金": 0.0, "水": 0.0},
        total_shift_magnitude=0.02,
    )
    status, explanation = check_useful_god_activation("木", no_clash_result.elemental_delta, no_clash_result)
    assert status == "neutral"
    assert "0.020" in explanation or "0.02" in explanation
