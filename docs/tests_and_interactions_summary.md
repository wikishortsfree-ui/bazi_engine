# Tests and Interactions Module Summary

Summary of all tests and the `bazi_engine.interactions` module for planning: use this to ask whether any function or behavior is missing test coverage or implementation.

---

## 1. Interactions module: overview

**Public API** (from `bazi_engine.interactions`):

- **`detect_all_interactions(stems, branches)`** → `InteractionResult`
- **`InteractionResult`** – dataclass with 8 list fields and `to_binary_vector()`
- **`compute_element_strengths(stems, branches)`** → `dict[str, float]` (also used internally; tested via `test_element_strength.py`)

All inputs are 4-pillar: `stems` and `branches` each have 4 items (Year, Month, Day, Hour). Positions are 0–3.

---

## 2. General flow of `detect_all_interactions`

1. **Validate** – `stems` and `branches` must have length 4; else `ValueError`.
2. **Raw detection (stems first, then branches)**  
   - `detect_stem_clashes(stems)` → stem clash positions  
   - `detect_stem_combinations(stems, blocked_positions=stem_clash_positions)` → `is_pure` False when a stem is in clash  
   - `detect_branch_clashes(branches)`  
   - `detect_branch_six_combinations(branches)`  
   - `detect_three_harmony(branches)`  
   - `detect_branch_harms(branches)`  
   - `detect_punishments(branches)`  
   - `detect_branch_destructions(branches)`
3. **Blocking / interference**  
   - **冲 blocks 六合**: any branch in a 六冲 → 六合 entries using that branch get `is_blocked=True`.  
   - **合化 (transformation_active)**: `compute_element_strengths(stems, branches)`; for stem combos and 六合, set `transformation_active` True only when: result element has strength ≥ 30%, month branch supports that element, and (for stem) `is_pure` True / (for 六合) not `is_blocked`.  
   - **冲 breaks 三合**: 三合 (full or half) with any branch in 六冲 → `blocked_by_clash=True`.  
   - **三合 absorbs 六合**: 六合 that shares a branch with a non-blocked full 三合 → `absorbed=True`.  
   - **六合 weakens 破**: 破 (destruction) for a branch pair that is also 六合 → `is_weakened=True`.
4. **Return** – `InteractionResult` with the above 8 lists (with blocking/transformation/absorbed/weakened flags applied).

---

## 3. Functions and constants by file

### `_constants.py`

| Symbol | Role |
|--------|------|
| `PILLAR_COUNT` | 4 |
| `TRANSFORMATION_STRENGTH_THRESHOLD` | 0.30 |
| `POSITION_LABELS` | `("year", "month", "day", "hour")` |
| `iter_position_pairs()` | Yields (i, j) for i < j over 0..3 |
| `require_four_pillars(items, name)` | Raises if len != 4 |
| `build_pair_lookup(items, key_a, key_b)` | dict by frozenset pair for `to_binary_vector` |

### `interaction_result.py`

| Symbol | Role |
|--------|------|
| `compute_element_strengths(stems, branches)` | Per-element fraction over 8 slots (4 stems + 4 branches), keys 木/火/土/金/水, sum 1.0 |
| `_month_supports_element(month_branch, result_element)` | Whether month branch’s element equals result or 生 (generates) result (for 合化) |
| `_check_transformation(...)` | True if 化: seasonal support, strength ≥ threshold, no clash |
| `InteractionResult` | Dataclass: stem_combinations, branch_six_combinations, three_harmony, stem_clashes, branch_clashes, harms, punishments, destructions |
| `InteractionResult.to_binary_vector()` | 64-dim float64 vector (fixed layout; spec: binary_vector_and_positions_spec.md) |
| `detect_all_interactions(stems, branches)` | Runs all detectors and blocking/transformation logic |

### `stem_combinations.py`

| Symbol | Role |
|--------|------|
| `STEM_COMBINATION_PAIRS` | (甲己→土, 乙庚→金, 丙辛→水, 丁壬→木, 戊癸→火) |
| `detect_stem_combinations(stems, *, blocked_positions=None)` | 天干合; one record per pair present; `is_pure` from blocked_positions |

### `clashes.py`

| Symbol | Role |
|--------|------|
| `STEM_CLASH_PAIRS` | 甲庚, 乙辛, 丙壬, 丁癸, 戊甲, 己乙 |
| `BRANCH_CLASH_PAIRS` | 子午, 丑未, 寅申, 卯酉, 辰戌, 巳亥 |
| `detect_stem_clashes(stems)` | 天干相冲; item_a/item_b canonical order, pos_a/pos_b, labels |
| `detect_branch_clashes(branches)` | 地支六冲; same shape |

### `branch_six_combinations.py`

| Symbol | Role |
|--------|------|
| `BRANCH_SIX_PAIRS` | 子丑→土, 寅亥→木, 卯戌→火, 辰酉→金, 巳申→水, 午未→土 |
| `detect_branch_six_combinations(branches)` | 地支六合; branch_a, branch_b, pos_a, pos_b, result_element |

### `three_harmony.py`

| Symbol | Role |
|--------|------|
| `THREE_HARMONY_GROUPS` | Water(申子辰), Fire(寅午戌), Metal(巳酉丑), Wood(亥卯未) with central branch |
| `detect_three_harmony(branches)` | 三合 full / 半合 half; element, combo_type, has_central, present_branches, positions |

### `harms.py`

| Symbol | Role |
|--------|------|
| `BRANCH_SIX_HARMS` | 子未, 丑午, 寅巳, 卯辰, 申亥, 酉戌 |
| `detect_branch_harms(branches)` | 地支六害; branch_a, branch_b, pos_a, pos_b |

### `punishments.py`

| Symbol | Role |
|--------|------|
| `UNGRATEFUL_TRIPLET` | 寅巳申 (无恩之刑) |
| `BULLYING_TRIPLET` | 丑戌未 (持势之刑) |
| `DISRESPECTFUL_PAIR` | 子卯 (无礼之刑) |
| `SELF_PUNISHMENT_BRANCHES` | 辰午酉亥 (自刑, need 2+ same) |
| `detect_punishments(branches)` | 地支相刑; punishment_type, members/positions, is_full or branch_a/branch_b or branch/positions |

### `destructions.py`

| Symbol | Role |
|--------|------|
| `BRANCH_DESTRUCTION_PAIRS` | 子酉, 午卯, 寅亥, 巳申, 丑辰, 未戌 |
| `detect_branch_destructions(branches)` | 地支相破; branch_a, branch_b, pos_a, pos_b |

---

## 4. Test files and what they cover

### `test_element_strength.py` (element strengths for 合化)

- **test_strength_all_wood_sums_to_one** – All 8 slots 木 → 木=1.0, others 0, sum 1.0.
- **test_strength_half_wood_half_fire** – 4 木 + 4 火 → 0.5 each.
- **test_strength_known_fractions** – Specific chart → 土/金/水 0.25, 木/火 0.125.
- **test_strength_always_sums_to_one** – Arbitrary chart sums to 1.0, keys are 5 elements.
- **test_strength_invalid_length_raises** – Non-4 stems or branches → ValueError.

**Function under test:** `compute_element_strengths` (from `interaction_result`).

---

### `test_clashes.py` (天干相冲, 地支六冲)

- **test_stems_must_have_four_items** – detect_stem_clashes rejects len ≠ 4.
- **test_branches_must_have_four_items** – detect_branch_clashes rejects len ≠ 4.
- **test_no_stem_clash** – No 相冲 pair → empty.
- **test_stem_clash_one** – One pair (e.g. 甲庚) → one dict with positions/labels.
- **test_stem_clash_canonical_order** – item_a/item_b in canonical order (e.g. 甲 before 庚).
- **test_no_branch_clash** – No 六冲 → empty.
- **test_branch_clash_one** – One 六冲 (e.g. 子午) → one dict.
- **test_branch_clash_multiple** – Two 六冲 pairs → two dicts.
- **test_branch_clash_all_six_types** – Each of 6 六冲 pair types detected.

**Functions:** `detect_stem_clashes`, `detect_branch_clashes`.

---

### `test_branch_six_combinations.py` (地支六合)

- **test_branches_must_have_four_items** – Rejects len ≠ 4.
- **test_no_combination** – No 六合 pair → empty.
- **test_one_combination** – One pair (e.g. 子丑) → one dict with result_element.
- **test_one_combination_reversed_positions** – pos_a < pos_b with reversed order.
- **test_two_simultaneous_combinations** – Two 六合 pairs → two dicts.
- **test_all_six_pair_types** – All 6 pair types and result elements.

**Function:** `detect_branch_six_combinations`.

---

### `test_three_harmony.py` (三合, 半合)

- **test_branches_must_have_four_items** – Rejects len ≠ 4.
- **test_no_combo** – No 三合/半合 → empty.
- **test_full_three_harmony** – All 3 of a group → combo_type=full, has_central=True.
- **test_half_combo_with_central** – 2 members including central → half, has_central=True.
- **test_half_combo_without_central** – 2 members, no central → half, has_central=False.

**Function:** `detect_three_harmony`.

---

### `test_harms.py` (地支六害)

- **test_branches_must_have_four_items** – Rejects len ≠ 4.
- **test_no_harm** – No 六害 pair → empty.
- **test_one_harm** – One pair → one dict.
- **test_one_harm_reversed_positions** – pos_a/pos_b with reversed order.
- **test_two_simultaneous_harms** – Two 六害 pairs.
- **test_all_six_harm_pair_types** – All 6 harm types.
- **test_harm_at_non_adjacent_positions** – Pair at (0, 3).

**Function:** `detect_branch_harms`.

---

### `test_punishments.py` (地支相刑)

- **test_branches_must_have_four_items** – Rejects len ≠ 4.
- **test_no_punishments** – No 刑 → empty.
- **test_ungrateful_partial_two_of_three** – 无恩: 2 of 寅巳申 → is_full False.
- **test_ungrateful_full_three_present** – 无恩: all 3 → is_full True.
- **test_ungrateful_positions_out_of_order** – members_present/positions in cycle order.
- **test_bullying_partial/full** – 持势: 2 or 3 of 丑戌未.
- **test_disrespectful_both_present** – 无礼: 子卯 → branch_a/branch_b, pos_a/pos_b.
- **test_disrespectful_positions_swapped** – 卯 before 子 still canonical 子, 卯.
- **test_self_punishment_same_branch_twice** – 自刑: same branch (e.g. 辰) twice.
- **test_self_punishment_four_positions_same_branch** – 3 positions same branch → one dict, all positions.
- **test_self_punishment_single_appearance_not_flagged** – Single 辰/午/酉/亥 → not 自刑.
- **test_partial_triplet_one_member_no_ungrateful/no_bullying** – Only 1 of triplet → no that 刑.
- **test_no_disrespectful_with_only_zi** – Only 子 → no 无礼.
- **test_chart_with_multiple_punishment_types** – 无恩 + 无礼 → two dicts.
- **test_chart_three_types** – 自刑 + 无礼 (two types in one chart).
- **test_self_punishment_two_different_branches_duplicated** – Two different 自刑 branches → two dicts.

**Function:** `detect_punishments`.

---

### `test_destructions.py` (地支相破)

- **test_branches_must_have_four_items** – Rejects len ≠ 4.
- **test_no_destruction** – No 相破 → empty.
- **test_one_destruction** – One pair → branch_a, branch_b, pos_a, pos_b.
- **test_one_destruction_other_positions** – Pair at (0, 2).
- **test_two_destructions** – Two pairs.
- **test_all_six_destruction_pair_types** – All 6 相破 types.
- **test_destruction_overlap_with_six_combination** – 寅亥 and 巳申 (破 + 六合) still reported as destruction.

**Function:** `detect_branch_destructions`.

---

### `test_interaction_result.py` (integration: detect_all_interactions + InteractionResult)

**Chart used:** 癸未 丙辰 丁卯 辛亥 (STEMS/BRANCHES constants).

- **test_detect_all_interactions_returns_interaction_result** – Result type and 8 list attributes.
- **test_chart_stem_combinations** – 丙辛合→水, is_pure True, positions.
- **test_chart_stem_clashes** – 丁癸冲, positions, labels.
- **test_chart_branch_six_combinations** – This chart has no 六合 → empty.
- **test_chart_branch_clashes** – No 六冲 → empty.
- **test_chart_three_harmony** – Wood 三合 full (亥卯未), not blocked.
- **test_clash_breaks_sanhe_full** – 申子辰 + 午 → 子午冲 → blocked_by_clash True.
- **test_clash_breaks_sanhe_half** – Half 三合 with branch in 冲 → blocked_by_clash True.
- **test_chart_harms** – 卯辰 六害, positions.
- **test_chart_punishments** – No 刑 → empty.
- **test_chart_destructions** – No 破 → empty.
- **test_to_binary_vector_shape_and_dtype** – 64-dim float, values in [0,1].
- **test_to_binary_vector_chart_specific_bits** – Bits for 丙辛, 丁癸, Wood full, 卯辰 harm.
- **test_input_validation** – detect_all_interactions rejects len ≠ 4.
- **test_blocking_six_combo_blocked_by_clash** – 子午冲 + 子丑六合 → 六合 is_blocked True.
- **test_weakening_destruction_by_six_combo** – 寅亥 六合 + 寅亥 破 → is_weakened True.
- **test_stem_combo_transformation_active_when_season_strength_no_clash** – 甲己→土, 月令土, 土≥30%, no clash → transformation_active True.
- **test_stem_combo_transformation_inactive_when_strength_below_threshold** – 土 < 30% → transformation_active False.
- **test_stem_combo_transformation_inactive_when_month_not_supporting** – 月令木 → transformation_active False.
- **test_stem_combo_transformation_inactive_when_stem_clash** – 甲 in 冲 → is_pure False → transformation_active False.
- **test_branch_six_transformation_active_when_season_strength_not_blocked** – 子丑→土, 月令土, 土≥30%, not blocked → transformation_active True.
- **test_branch_six_transformation_inactive_when_blocked_by_clash** – 子午冲 + 子丑六合 → transformation_active False.
- **test_sanhe_absorbs_liuhe** – Full 三合 (not blocked) + 六合 sharing branch → 六合 absorbed True.
- **test_complex_chart_three_harmony_clash_six_combo** – 申子午丑: 六冲, 半合 blocked, 六合 blocked, transformation_active False, absorbed False.

**Functions/behavior:** `detect_all_interactions`, `InteractionResult`, `to_binary_vector`, blocking/transformation/absorbed/weakened logic.

---

## 5. Coverage matrix (for planning)

| Area | Unit tests | Integration (detect_all + result) |
|------|------------|------------------------------------|
| **Element strengths** | test_element_strength.py | Used in transformation_active |
| **Stem combinations** | — | test_interaction_result (chart + transformation) |
| **Stem clashes** | test_clashes.py | test_interaction_result |
| **Branch clashes** | test_clashes.py | test_interaction_result, blocking 三合/六合 |
| **六合** | test_branch_six_combinations.py | test_interaction_result, blocked, transformation, absorbed |
| **三合** | test_three_harmony.py | test_interaction_result, blocked_by_clash |
| **六害** | test_harms.py | test_interaction_result |
| **刑** | test_punishments.py | test_interaction_result (empty for chart) |
| **破** | test_destructions.py | test_interaction_result, is_weakened |
| **to_binary_vector** | — | test_interaction_result (shape + chart-specific bits) |
| **_constants** | — | Used implicitly (require_four_pillars, build_pair_lookup) |

**Not directly unit-tested as standalone:**  
- `detect_stem_combinations` (only via detect_all_interactions).  
- `_month_supports_element`, `_check_transformation` (only via integration).  
- `iter_position_pairs`, `require_four_pillars`, `build_pair_lookup` (used by detectors / to_binary_vector).

---

## 6. Questions for a planning agent

1. **Missing functions** – Is any public or important internal function in `bazi_engine/interactions/` without test coverage (unit or integration)?
2. **Missing behaviors** – Are there blocking/transformation rules or edge cases (戊甲/己乙 stem clash are implemented; 害 vs 六合 mutual exclusion, etc.) that are specified elsewhere but not implemented or not tested?
3. **Stem combinations** – Should `detect_stem_combinations` have its own unit tests (e.g. all five pairs, is_pure with blocked_positions)?
4. **Constants / helpers** – Should `_constants` (e.g. `build_pair_lookup`, `require_four_pillars`) or the 64-dim vector layout be tested in isolation?
5. **Binary vector** – Is the fixed 64-dim layout documented and stable enough, or should there be tests for every index for at least one chart?
6. **Invalid inputs** – Besides len ≠ 4, are there other invalid inputs (e.g. unknown stem/branch characters) that should be validated and tested?

Use this doc plus the code in `bazi_engine/interactions/` and `tests/` to decide what to add or change.
