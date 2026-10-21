"""
Integration tests for InteractionResult and detect_all_interactions.

Test chart: 癸未 丙辰 丁卯 辛亥
  Pillar:    Year  Month Day  Hour
  Stems:     癸    丙    丁    辛
  Branches:  未    辰    卯    亥
"""
import pytest

from bazi_engine.interactions import InteractionResult, detect_all_interactions
from bazi_engine.interactions._constants import TRANSFORMATION_STRENGTH_THRESHOLD
from bazi_engine.interactions.interaction_result import (
    _check_transformation,
    _month_supports_element,
)


# Known test chart: 癸未 丙辰 丁卯 辛亥
STEMS = ["癸", "丙", "丁", "辛"]
BRANCHES = ["未", "辰", "卯", "亥"]


def test_detect_all_interactions_returns_interaction_result():
    """detect_all_interactions returns an InteractionResult with all 8 list fields."""
    result = detect_all_interactions(STEMS, BRANCHES)
    assert isinstance(result, InteractionResult)
    assert hasattr(result, "stem_combinations")
    assert hasattr(result, "branch_six_combinations")
    assert hasattr(result, "three_harmony")
    assert hasattr(result, "stem_clashes")
    assert hasattr(result, "branch_clashes")
    assert hasattr(result, "harms")
    assert hasattr(result, "punishments")
    assert hasattr(result, "destructions")
    assert isinstance(result.stem_combinations, list)
    assert isinstance(result.branch_six_combinations, list)
    assert isinstance(result.three_harmony, list)
    assert isinstance(result.stem_clashes, list)
    assert isinstance(result.branch_clashes, list)
    assert isinstance(result.harms, list)
    assert isinstance(result.punishments, list)
    assert isinstance(result.destructions, list)


def test_chart_stem_combinations():
    """癸丙丁辛: 丙辛合→水 at Month(1), Hour(3); no stem in clash for 丙/辛 so is_pure=True."""
    result = detect_all_interactions(STEMS, BRANCHES)
    assert len(result.stem_combinations) == 1
    c = result.stem_combinations[0]
    assert c["stem_a"] == "丙" and c["stem_b"] == "辛"
    assert c["pos_a"] == 1 and c["pos_b"] == 3
    assert c["result_element"] == "水"
    assert c["is_pure"] is True


def test_chart_stem_clashes():
    """癸丙丁辛: 丁癸冲 at Year(0), Day(2); pos_a/pos_b follow canonical item order (丁 at 2, 癸 at 0)."""
    result = detect_all_interactions(STEMS, BRANCHES)
    assert len(result.stem_clashes) == 1
    r = result.stem_clashes[0]
    assert r["item_a"] == "丁" and r["item_b"] == "癸"
    assert {r["pos_a"], r["pos_b"]} == {0, 2}
    assert (r["pos_a_label"], r["pos_b_label"]) in [("year", "day"), ("day", "year")]


def test_chart_branch_six_combinations():
    """未辰卯亥: no 六合 pair (e.g. 寅亥 needs 寅, 午未 needs 午)."""
    result = detect_all_interactions(STEMS, BRANCHES)
    assert result.branch_six_combinations == []


def test_chart_branch_clashes():
    """未辰卯亥: no 六冲 pair (each 冲 needs both members)."""
    result = detect_all_interactions(STEMS, BRANCHES)
    assert result.branch_clashes == []


def test_chart_three_harmony():
    """未辰卯亥: Wood 三合 full — 亥卯未 all present (central 卯); no 六冲 so not blocked by clash."""
    result = detect_all_interactions(STEMS, BRANCHES)
    assert len(result.three_harmony) == 1
    h = result.three_harmony[0]
    assert h["element"] == "Wood"
    assert h["combo_type"] == "full"
    assert h["has_central"] is True
    assert set(h["present_branches"]) == {"亥", "卯", "未"}
    assert set(h["positions"]) == {0, 2, 3}
    assert h["blocked_by_clash"] is False


def test_clash_breaks_sanhe_full():
    """冲 breaks 三合: 申子辰 + 午 → full Water 三合 with 子午冲 → blocked_by_clash True."""
    stems = ["甲", "丙", "戊", "庚"]
    branches = ["申", "子", "辰", "午"]  # Water 三合 申子辰; 子午 六冲
    result = detect_all_interactions(stems, branches)
    assert len(result.three_harmony) == 1
    h = result.three_harmony[0]
    assert h["element"] == "Water"
    assert h["combo_type"] == "full"
    assert set(h["present_branches"]) == {"申", "子", "辰"}
    assert h["blocked_by_clash"] is True


def test_clash_breaks_sanhe_half():
    """冲 breaks 半合: half 三合 with one branch in 六冲 has blocked_by_clash True."""
    # Water 半合 申子 (no 辰); 子午 冲. So 申子 at 0,1 and 午 at 3.
    stems = ["甲", "乙", "丙", "丁"]
    branches = ["申", "子", "寅", "午"]
    result = detect_all_interactions(stems, branches)
    # One half: Water 申子; 子 is in 子午冲
    water = next((d for d in result.three_harmony if d["element"] == "Water"), None)
    assert water is not None
    assert water["combo_type"] == "half"
    assert set(water["present_branches"]) == {"申", "子"}
    assert water["blocked_by_clash"] is True


def test_chart_harms():
    """未辰卯亥: 六害 卯辰 at Day(2), Month(1); pos_a < pos_b so (1,2) → 辰, 卯."""
    result = detect_all_interactions(STEMS, BRANCHES)
    assert len(result.harms) == 1
    r = result.harms[0]
    assert {r["branch_a"], r["branch_b"]} == {"卯", "辰"}
    assert r["pos_a"] == 1 and r["pos_b"] == 2


def test_chart_punishments():
    """未辰卯亥: no 无恩/持势/无礼/自刑 (no 子卯, no triplets, no duplicate 辰午酉亥)."""
    result = detect_all_interactions(STEMS, BRANCHES)
    assert result.punishments == []


def test_chart_destructions():
    """未辰卯亥: no 相破 pair (e.g. 寅亥 needs 寅, 午卯 needs 午)."""
    result = detect_all_interactions(STEMS, BRANCHES)
    assert result.destructions == []


def test_to_binary_vector_shape_and_dtype():
    """to_binary_vector returns flat float64 array of 64 dimensions."""
    result = detect_all_interactions(STEMS, BRANCHES)
    vec = result.to_binary_vector()
    assert vec.shape == (64,)
    assert vec.dtype == float
    assert vec.min() >= 0 and vec.max() <= 1


def test_to_binary_vector_chart_specific_bits():
    """For 癸未丙辰丁卯辛亥, stem combo 丙辛 and stem clash 丁癸 and three_harmony Wood full and harm 卯辰 are set."""
    result = detect_all_interactions(STEMS, BRANCHES)
    vec = result.to_binary_vector()
    # [0:5] stem_combinations present — 丙辛 is index 2 (丙辛 in STEM_COMBINATION_PAIRS order: 甲己,乙庚,丙辛,丁壬,戊癸)
    assert vec[2] == 1.0
    # [5:10] is_pure for same — 丙辛 is pure
    assert vec[5 + 2] == 1.0
    # [30:36] stem_clashes — 丁癸; sorted(STEM_CLASH_PAIRS) is lexicographic, 丁癸 comes first
    assert vec[30] == 1.0
    # [22:26] three_harmony full — Wood is index 3 (Water, Fire, Metal, Wood)
    assert vec[22 + 3] == 1.0
    # [42:48] harms — 卯辰 is index 3 (子未,丑午,寅巳,卯辰,申亥,酉戌)
    assert vec[42 + 3] == 1.0


def test_input_validation():
    """detect_all_interactions raises if stems or branches length != 4."""
    with pytest.raises(ValueError, match="must each have 4 items"):
        detect_all_interactions(["甲", "乙"], BRANCHES)
    with pytest.raises(ValueError, match="must each have 4 items"):
        detect_all_interactions(STEMS, ["子", "丑"])
    with pytest.raises(ValueError, match="must each have 4 items"):
        detect_all_interactions(["甲", "乙", "丙", "丁", "戊"], BRANCHES)


def test_blocking_six_combo_blocked_by_clash():
    """If a branch is in 冲, 六合 involving that branch has is_blocked=True."""
    # Chart with 子午 冲 and 寅亥 六合: 寅 not in clash, 亥 not in clash → 六合 not blocked.
    # Chart with 子午 冲 and 子丑 六合: 子 in clash → 六合 is_blocked=True.
    stems = ["甲", "庚", "丙", "丁"]  # 甲庚 at 0,1
    branches = ["子", "午", "丑", "寅"]  # 子午 冲 at 0,1; 子丑 六合 at 0,2
    result = detect_all_interactions(stems, branches)
    assert len(result.branch_six_combinations) == 1
    assert result.branch_six_combinations[0]["branch_a"] == "子" and result.branch_six_combinations[0]["branch_b"] == "丑"
    assert result.branch_six_combinations[0]["is_blocked"] is True


def test_weakening_destruction_by_six_combo():
    """If same branch pair is in 六合, 破 entry has is_weakened=True."""
    # 寅亥: 六合→木 and 相破. Chart 寅亥子丑 has 寅亥 六合 + 寅亥 破 (and 子丑 六合).
    branches = ["寅", "亥", "子", "丑"]
    stems = ["甲", "乙", "丙", "丁"]
    result = detect_all_interactions(stems, branches)
    assert len(result.destructions) == 1
    assert (result.destructions[0]["branch_a"], result.destructions[0]["branch_b"]) in (
        ("寅", "亥"),
        ("亥", "寅"),
    )
    assert result.destructions[0]["is_weakened"] is True


# --- 合化 transformation_active (strength-helper + seasonal-transformation) ---


# --- Seasonal support (月令): equal or 生 (generates) ---


def test_month_supports_element_fire_month_supports_earth():
    """巳月 (火) 生 土 → supports 甲己合土 (火生土)."""
    assert _month_supports_element("巳", "土") is True


def test_month_supports_element_noon_fire_supports_earth():
    """午月 (火) 生 土 → supports 土."""
    assert _month_supports_element("午", "土") is True


def test_month_supports_element_earth_month_equals_earth():
    """辰月 (土) equals 土 → supports 土."""
    assert _month_supports_element("辰", "土") is True


def test_month_supports_element_wood_month_does_not_support_earth():
    """卯月 (木) does not equal or 生 土 → no support."""
    assert _month_supports_element("卯", "土") is False


def test_month_supports_element_earth_month_supports_metal():
    """辰月 (土) 生 金 → supports 乙庚合金."""
    assert _month_supports_element("辰", "金") is True


def test_month_supports_element_water_month_supports_wood():
    """亥月 (水) 生 木 → supports 丁壬合木."""
    assert _month_supports_element("亥", "木") is True


def test_stem_combo_transformation_active_in_fire_month_via_sheng():
    """甲己→土 in 巳月 (Fire): 火生土 gives seasonal support; 土≥30%, no clash → transformation_active True."""
    # 甲己 at 0,1; 土 from 己+己+辰+戌 (no 戊 in stems to avoid 戊甲 clash); month 巳 (火) 生 土
    stems = ["甲", "己", "己", "丁"]
    branches = ["辰", "巳", "戌", "子"]  # month = 巳 (火)
    result = detect_all_interactions(stems, branches)
    assert len(result.stem_combinations) == 1
    assert result.stem_combinations[0]["result_element"] == "土"
    assert result.stem_combinations[0]["transformation_active"] is True


def test_stem_combo_transformation_active_when_season_strength_no_clash():
    """甲己→土: 月令土, 土≥30%, no stem clash → transformation_active True."""
    # Stems: 甲 己 丙 丁 (甲己 at 0,1). Branches: 辰 戌 丑 子 → 土土土水, month=戌(土). 土 = 己+辰+戌+丑 = 4/8 = 50%.
    stems = ["甲", "己", "丙", "丁"]
    branches = ["辰", "戌", "丑", "子"]
    result = detect_all_interactions(stems, branches)
    assert len(result.stem_combinations) == 1
    assert result.stem_combinations[0]["stem_a"] == "甲" and result.stem_combinations[0]["stem_b"] == "己"
    assert result.stem_combinations[0]["result_element"] == "土"
    assert result.stem_combinations[0]["is_pure"] is True
    assert result.stem_combinations[0]["transformation_active"] is True


def test_stem_combo_transformation_inactive_when_strength_below_threshold():
    """甲己→土: 土<30% (only 己 in 8 slots) → transformation_active False."""
    stems = ["甲", "己", "丙", "丁"]  # 木土火火
    branches = ["子", "卯", "午", "亥"]  # 水木火水 → 土 = 1/8 = 12.5%
    result = detect_all_interactions(stems, branches)
    assert len(result.stem_combinations) == 1
    assert result.stem_combinations[0]["result_element"] == "土"
    assert result.stem_combinations[0]["transformation_active"] is False


def test_stem_combo_transformation_inactive_when_month_not_supporting():
    """甲己→土: 月令木 (卯) → no seasonal support → transformation_active False."""
    stems = ["甲", "己", "戊", "己"]  # 甲己 at 0,1; 土 = 己戊己 + 辰戌 = 4 (stems 己戊己 + branches 辰戌)
    branches = ["寅", "卯", "辰", "戌"]  # month = 卯 (木)
    result = detect_all_interactions(stems, branches)
    assert len(result.stem_combinations) == 1
    assert result.stem_combinations[0]["result_element"] == "土"
    assert result.stem_combinations[0]["transformation_active"] is False


def test_stem_combo_transformation_inactive_when_stem_clash():
    """甲己→土 but 甲 or 己 in stem clash → is_pure False → transformation_active False."""
    # 甲庚冲: 甲 at 0, 庚 at 1. 甲己: 己 at 2, 甲 at 0. So 甲 is clashed → is_pure False.
    stems = ["甲", "庚", "己", "丁"]
    branches = ["辰", "戌", "丑", "子"]  # 土≥30%, month=戌(土)
    result = detect_all_interactions(stems, branches)
    assert len(result.stem_combinations) == 1
    assert result.stem_combinations[0]["is_pure"] is False
    assert result.stem_combinations[0]["transformation_active"] is False


def test_branch_six_transformation_active_when_season_strength_not_blocked():
    """六合 子丑→土: 月令土, 土≥30%, not is_blocked → transformation_active True."""
    stems = ["戊", "己", "戊", "己"]  # 土 4/8 = 50%
    branches = ["子", "丑", "辰", "戌"]  # 子丑 六合 at 0,1; month=丑(土); no 冲
    result = detect_all_interactions(stems, branches)
    assert len(result.branch_six_combinations) >= 1
    zi_chou = next(
        (d for d in result.branch_six_combinations if {d["branch_a"], d["branch_b"]} == {"子", "丑"}),
        None,
    )
    assert zi_chou is not None
    assert zi_chou["result_element"] == "土"
    assert zi_chou["is_blocked"] is False
    assert zi_chou["transformation_active"] is True


def test_branch_six_transformation_inactive_when_blocked_by_clash():
    """六合 子丑 but 子午冲 → is_blocked True → transformation_active False."""
    stems = ["戊", "己", "戊", "己"]
    branches = ["子", "午", "丑", "寅"]  # 子午 冲; 子丑 六合 at 0,2
    result = detect_all_interactions(stems, branches)
    zi_chou = next(
        (d for d in result.branch_six_combinations if {d["branch_a"], d["branch_b"]} == {"子", "丑"}),
        None,
    )
    assert zi_chou is not None
    assert zi_chou["is_blocked"] is True
    assert zi_chou["transformation_active"] is False


def test_transformation_threshold_at_exactly_30_percent_activates():
    """At exactly 30.0% strength (>= threshold), transformation activates. Boundary: >= not >."""
    element_strengths = {"木": 0.0, "火": 0.0, "土": 0.30, "金": 0.35, "水": 0.35}
    assert TRANSFORMATION_STRENGTH_THRESHOLD == 0.30
    # 土 exactly at threshold; month 丑 supports 土; no clash
    assert _check_transformation(
        "土",
        "丑",
        element_strengths,
        TRANSFORMATION_STRENGTH_THRESHOLD,
        no_clash_flag=True,
    ) is True


def test_transformation_threshold_below_30_percent_inactive():
    """Just below 30% strength → transformation does not activate."""
    element_strengths = {"木": 0.0, "火": 0.0, "土": 0.29, "金": 0.35, "水": 0.36}
    assert _check_transformation(
        "土",
        "丑",
        element_strengths,
        TRANSFORMATION_STRENGTH_THRESHOLD,
        no_clash_flag=True,
    ) is False


def test_sanhe_absorbs_liuhe():
    """When a full 三合 is present and not blocked by 冲, 六合 that shares a branch with it has absorbed=True."""
    # 申子辰 = full Water 三合; 子丑 = 六合. 子 is in both → 子丑 六合 should be absorbed.
    stems = ["甲", "乙", "丙", "丁"]
    branches = ["申", "子", "辰", "丑"]  # full 三合 申子辰; 六合 子丑 (pos 1,3)
    result = detect_all_interactions(stems, branches)
    # One full Water 三合, not blocked (no 冲)
    water = next((h for h in result.three_harmony if h["element"] == "Water"), None)
    assert water is not None and water["combo_type"] == "full" and not water.get("blocked_by_clash", False)
    # One 六合: 子丑
    assert len(result.branch_six_combinations) == 1
    liuhe = result.branch_six_combinations[0]
    assert {liuhe["branch_a"], liuhe["branch_b"]} == {"子", "丑"}
    assert liuhe["absorbed"] is True


def test_complex_chart_three_harmony_clash_six_combo():
    """Complex chart: 三合 (half) + 六冲 + 六合; assert all hierarchy flags in one go."""
    # 申子午丑: 子午 六冲, 子丑 六合, 申子 半合 (Water). 冲 breaks 半合; 冲 blocks 六合.
    stems = ["甲", "乙", "丙", "丁"]
    branches = ["申", "子", "午", "丑"]
    result = detect_all_interactions(stems, branches)
    # 六冲: 子午
    assert len(result.branch_clashes) == 1
    assert {result.branch_clashes[0]["item_a"], result.branch_clashes[0]["item_b"]} == {"子", "午"}
    # 三合: Water half 申子; 子 in 冲 → blocked_by_clash True
    water = next((h for h in result.three_harmony if h["element"] == "Water"), None)
    assert water is not None
    assert water["combo_type"] == "half"
    assert set(water["present_branches"]) == {"申", "子"}
    assert water["blocked_by_clash"] is True
    # 六合: 子丑; 子 in 冲 → is_blocked True; no non-blocked full 三合 → absorbed False
    zi_chou = next(
        (d for d in result.branch_six_combinations if {d["branch_a"], d["branch_b"]} == {"子", "丑"}),
        None,
    )
    assert zi_chou is not None
    assert zi_chou["is_blocked"] is True
    assert zi_chou["transformation_active"] is False
    assert zi_chou["absorbed"] is False
