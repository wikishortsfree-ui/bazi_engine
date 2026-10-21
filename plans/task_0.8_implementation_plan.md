# Task 0.8: Output Serialization & API - Implementation Plan

**Estimated Duration:** 2.5 days  
**Dependencies:** Tasks 0.1-0.7 (all core calculator components)  
**Purpose:** Create clean interfaces for data generation pipeline (Phase 2) and inference preprocessing

---

## Overview

This task builds the final output layer for the bāzì calculator engine. It provides three critical interfaces:

1. **JSON serialization** – For storing 10k+ generated charts to disk during Phase 2 data generation
2. **NumPy array export** – For feeding raw feature arrays directly into the encoder during training/inference
3. **Batch processing utility** – For efficient bulk chart generation from CSV inputs

---

## Sub-Task Breakdown

### 0.8.1 - JSON Serialization (0.5 days)

**Goal:** Implement `BaziChart.to_json()` that serializes all fields to well-structured JSON.

#### Implementation Requirements

**File:** `bazi_engine/core/bazi_chart.py`

**Method signature:**
```python
def to_json(self, indent: int = 2) -> str:
    """
    Serialize the complete BaziChart to JSON string.
    
    Args:
        indent: JSON indentation level (default 2, use None for compact)
    
    Returns:
        JSON string with all chart data
    """
```

**JSON Schema Structure:**

```json
{
  "metadata": {
    "birth_datetime": "2003-04-24T21:20:00+08:00",
    "timezone_offset": 8,
    "calculation_timestamp": "2026-03-03T10:30:00Z",
    "calculator_version": "0.1.0"
  },
  
  "natal_pillars": {
    "year": {"stem": "癸", "branch": "未", "element_stem": "水", "element_branch": "土"},
    "month": {"stem": "丙", "branch": "辰", "element_stem": "火", "element_branch": "土"},
    "day": {"stem": "丁", "branch": "卯", "element_stem": "火", "element_branch": "木"},
    "hour": {"stem": "辛", "branch": "亥", "element_stem": "金", "element_branch": "水"}
  },
  
  "day_master": {
    "stem": "丁",
    "element": "火",
    "strength_percentile": 0.42,
    "classification": "身弱"
  },
  
  "ten_gods": {
    "year_stem": "七杀",
    "month_stem": "劫财",
    "day_stem": "日主",
    "hour_stem": "偏财",
    "year_branch": ["食神", "比肩", "偏印"],
    "month_branch": ["伤官", "偏印", "七杀"],
    "day_branch": ["偏印"],
    "hour_branch": ["正官", "正印"]
  },
  
  "hidden_stems": {
    "year_branch": [
      {"stem": "己", "element": "土", "weight": 0.6, "ten_god": "食神"},
      {"stem": "丁", "element": "火", "weight": 0.3, "ten_god": "比肩"},
      {"stem": "乙", "element": "木", "weight": 0.1, "ten_god": "偏印"}
    ],
    "month_branch": [...],
    "day_branch": [...],
    "hour_branch": [...]
  },
  
  "elemental_strengths": {
    "wood": 0.125,
    "fire": 0.125,
    "earth": 0.25,
    "metal": 0.25,
    "water": 0.25
  },
  
  "useful_god": {
    "primary": "印",
    "primary_element": "木",
    "favorable_elements": ["木", "水"],
    "unfavorable_elements": ["金", "土"],
    "methodology": "身弱用印",
    "is_ambiguous": false
  },
  
  "interactions": {
    "stem_combinations": [
      {
        "positions": [1, 3],
        "position_labels": ["month", "hour"],
        "stems": ["丙", "辛"],
        "result_element": "水",
        "is_pure": true,
        "transformation_active": false
      }
    ],
    
    "stem_clashes": [
      {
        "positions": [2, 0],
        "position_labels": ["day", "year"],
        "stems": ["丁", "癸"]
      }
    ],
    
    "branch_six_combinations": [],
    
    "branch_clashes": [],
    
    "three_harmony": [
      {
        "element": "木",
        "combo_type": "full",
        "has_central": true,
        "present_branches": ["亥", "卯", "未"],
        "positions": [3, 2, 0],
        "blocked_by_clash": false
      }
    ],
    
    "harms": [
      {
        "positions": [2, 1],
        "position_labels": ["day", "month"],
        "branches": ["卯", "辰"]
      }
    ],
    
    "punishments": [],
    
    "destructions": []
  },
  
  "dynamic": null
}
```

**Dynamic (temporal) structure when present:**
```json
{
  "dynamic": {
    "da_yun": {
      "stem": "乙",
      "branch": "卯",
      "start_year": 2009,
      "start_age": 7,
      "end_year": 2018,
      "end_age": 16
    },
    
    "liu_nian": {
      "year": 2024,
      "stem": "甲",
      "branch": "辰"
    },
    
    "liu_yue": {
      "year": 2024,
      "month": 3,
      "stem": "戊",
      "branch": "辰"
    },
    
    "cross_natal_temporal_interactions": {
      "natal_vs_dayun": {
        "clashes": [...],
        "combinations": [...],
        "harms": [...],
        "punishments": [...]
      },
      "natal_vs_liunian": {...},
      "natal_vs_liuyue": {...},
      "dayun_vs_liunian": {...}
    },
    
    "elemental_delta": {
      "wood": 0.05,
      "fire": -0.02,
      "earth": 0.08,
      "metal": -0.03,
      "water": -0.08
    },
    
    "useful_god_activation": {
      "is_active": true,
      "mechanism": "大运乙卯木助身, 流年甲辰半合水局生身"
    }
  }
}
```

#### TODO List

- [ ] **Add imports** to `BaziChart`: `import json`, `from datetime import datetime, timezone`
- [ ] **Add metadata fields** to `BaziChart` dataclass:
  - `birth_datetime: datetime`
  - `timezone_offset: int`
  - `calculation_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))`
  - `calculator_version: str = "0.1.0"`
- [ ] **Implement `to_json()` method**:
  - Build nested dict following schema above
  - Handle `None` values for optional fields (dynamic, useful_god when ambiguous)
  - Use `json.dumps()` with `ensure_ascii=False` for proper Chinese character encoding
  - Add `default=str` handler for datetime serialization
- [ ] **Add helper method `_serialize_pillar(pillar: Pillar) -> dict`** for DRY
- [ ] **Add helper method `_serialize_interactions(interactions: InteractionResult) -> dict`**
- [ ] **Write unit test `test_to_json_natal_only.py`**:
  - Verify JSON is valid (can be parsed)
  - Check all required keys present
  - Verify Chinese characters preserved
  - Verify `dynamic` is `null` when not provided
- [ ] **Write unit test `test_to_json_with_dynamic.py`**:
  - Verify dynamic section present and complete
  - Verify cross-natal-temporal interactions serialized
- [ ] **Test round-trip** (not required for v1, but useful):
  - `BaziChart.from_json(chart.to_json()) == chart`

---

### 0.8.2 - NumPy Array Export (1 day)

**Goal:** Implement `BaziChart.to_feature_arrays()` that returns raw NumPy arrays for encoder input.

#### Implementation Requirements

**File:** `bazi_engine/core/bazi_chart.py`

**Method signature:**
```python
def to_feature_arrays(self) -> dict[str, np.ndarray]:
    """
    Export chart as raw feature arrays for encoder input.
    
    Returns:
        Dictionary with the following keys:
        - 'stem_ids': int64[4] - stem indices 0-9
        - 'branch_ids': int64[4] - branch indices 0-11
        - 'pillar_positions': int64[4] - always [0,1,2,3]
        - 'stem_elements': int64[4] - element indices 0-4
        - 'branch_elements': int64[4] - element indices 0-4
        - 'stem_polarities': int64[4] - 0=yin, 1=yang
        - 'branch_polarities': int64[4] - 0=yin, 1=yang
        - 'ten_god_stem_ids': int64[4] - ten god category 0-9 (day=日主 special)
        - 'ten_god_branch_ids': int64[12] - flattened; 3 per branch (padded -1)
        - 'interaction_vector': float64[64] - from InteractionResult.to_binary_vector()
        - 'elemental_strengths': float64[5] - [wood, fire, earth, metal, water]
        - 'day_master_strength': float64[1] - percentile 0.0-1.0
        - 'useful_god_id': int64[1] - categorical 0-4 (element-based: 比劫/食伤/财/官杀/印 per encodings.USEFUL_GOD_TO_ID) or -1 if ambiguous
        
        If dynamic context present, also includes:
        - 'dayun_stem_id', 'dayun_branch_id': int64[1]
        - 'liunian_stem_id', 'liunian_branch_id': int64[1]
        - 'liuyue_stem_id', 'liuyue_branch_id': int64[1] (or -1 if absent)
        - 'dynamic_interaction_vector': float64[N] - cross-natal-temporal interactions
        - 'elemental_delta': float64[5]
        - 'useful_god_active': int64[1] - 0/1 binary flag
        - 'pillar_presence_mask': int64[3] - [has_dayun, has_liunian, has_liuyue]
    """
```

#### Encoding Tables Required

**File:** `bazi_engine/core/encodings.py` (new file)

```python
# Stem encoding: 甲=0, 乙=1, ..., 癸=9
STEM_TO_ID = {
    "甲": 0, "乙": 1, "丙": 2, "丁": 3, "戊": 4,
    "己": 5, "庚": 6, "辛": 7, "壬": 8, "癸": 9
}

# Branch encoding: 子=0, 丑=1, ..., 亥=11
BRANCH_TO_ID = {
    "子": 0, "丑": 1, "寅": 2, "卯": 3, "辰": 4, "巳": 5,
    "午": 6, "未": 7, "申": 8, "酉": 9, "戌": 10, "亥": 11
}

# Element encoding: 木=0, 火=1, 土=2, 金=3, 水=4
ELEMENT_TO_ID = {
    "木": 0, "火": 1, "土": 2, "金": 3, "水": 4
}

# Ten God encoding
TEN_GOD_TO_ID = {
    "比肩": 0, "劫财": 1, "食神": 2, "伤官": 3, "偏财": 4,
    "正财": 5, "七杀": 6, "正官": 7, "偏印": 8, "正印": 9,
    "日主": 10  # Special case for day stem
}

# Useful God encoding (element-based)
USEFUL_GOD_TO_ID = {
    "比劫": 0, "食伤": 1, "财": 2, "官杀": 3, "印": 4,
    # More granular if needed:
    # "比肩": 0, "劫财": 1, "食神": 2, "伤官": 3, ...
}

# Polarity: Yin/Yang
def stem_polarity(stem: str) -> int:
    """0=yin, 1=yang"""
    return STEM_TO_ID[stem] % 2  # Even index=yang, odd=yin

def branch_polarity(branch: str) -> int:
    """0=yin, 1=yang"""
    return BRANCH_TO_ID[branch] % 2
```

#### TODO List

- [ ] **Create `bazi_engine/core/encodings.py`** with all encoding tables above
- [ ] **Add `import numpy as np`** to `BaziChart`
- [ ] **Implement `to_feature_arrays()` method**:
  - For each array, ensure correct dtype (int64 vs float64)
  - Use `-1` as sentinel for missing/padding values (e.g. absent 藏干, absent 流月)
  - Flatten ten_god_branch_ids: 4 branches × 3 hidden stems = 12 slots
  - Call `self.interactions.to_binary_vector()` for interaction_vector
  - Map useful god to category ID; use `-1` if `is_ambiguous=True`
- [ ] **Handle dynamic arrays**:
  - Only add dynamic keys if `self.dynamic is not None`
  - `pillar_presence_mask`: 3-element array indicating which temporal pillars exist
  - Cross-natal-temporal interaction vector: flatten all 4 cross-matrices from dynamic
- [ ] **Write unit test `test_to_feature_arrays_natal.py`**:
  - Verify all required keys present
  - Check array shapes and dtypes
  - Verify stem_ids[0] == STEM_TO_ID["癸"] for test chart
  - Verify interaction_vector.shape == (64,)
  - Verify elemental_strengths.sum() ≈ 1.0
- [ ] **Write unit test `test_to_feature_arrays_dynamic.py`**:
  - Verify dynamic keys present when dynamic != None
  - Verify pillar_presence_mask correctness
  - Test with partial dynamic (only 大运, no 流月)
- [ ] **Write unit test `test_encoding_consistency.py`**:
  - Round-trip: stem ID → stem character via reverse lookup
  - Verify polarity functions: 甲(yang)=1, 乙(yin)=0

---

### 0.8.3 - Batch Processing Utility (0.5 days)

**Goal:** Script that takes CSV of birth datetimes and outputs JSONL file of BaziChart objects.

#### Implementation Requirements

**File:** `scripts/batch_generate_charts.py`

**Input CSV format:**
```csv
birth_datetime,timezone_offset,gender
1990-05-15 14:30:00,8,male
2003-04-24 21:20:00,8,male
1985-11-03 08:45:00,-5,female
```

**Output JSONL format:**
Each line is a complete `BaziChart.to_json()` result (compact, no indentation):
```jsonl
{"metadata": {...}, "natal_pillars": {...}, ...}
{"metadata": {...}, "natal_pillars": {...}, ...}
```

**CLI interface:**
```bash
python scripts/batch_generate_charts.py \
  --input birth_dates.csv \
  --output charts.jsonl \
  --include-dynamic \
  --current-date 2024-03-15
```

**Arguments:**
- `--input` (required): Path to input CSV
- `--output` (required): Path to output JSONL
- `--include-dynamic` (optional): Compute 大运/流年 for --current-date
- `--current-date` (optional): Date for dynamic calculation (default: today)
- `--verbose` (optional): Print progress every N charts

#### TODO List

- [ ] **Create `scripts/batch_generate_charts.py`**
- [ ] **Add argparse CLI** with arguments above
- [ ] **Implement CSV parsing**:
  - Use `pandas.read_csv()` or `csv.DictReader`
  - Validate required columns present
  - Parse datetime strings with `datetime.fromisoformat()` or `dateutil.parser`
- [ ] **Implement batch loop**:
  - For each row: call `BaziCalculator.from_datetime()`
  - If `--include-dynamic`: also compute dynamic for `--current-date`
  - Call `chart.to_json(indent=None)` for compact output
  - Write to JSONL (one JSON object per line)
- [ ] **Add error handling**:
  - Catch `ValueError` for invalid dates
  - Log errors with row number
  - Option: `--skip-errors` vs `--fail-fast`
- [ ] **Add progress reporting** with `tqdm` if available
- [ ] **Write integration test `test_batch_script.py`**:
  - Create temp CSV with 5 rows
  - Run script
  - Verify output JSONL has 5 lines
  - Verify each line is valid JSON
  - Verify charts have correct birth_datetime from input

---

### 0.8.4 - CLI Tool (Optional, 0.5 days)

**Goal:** Manual spot-checking tool for single charts.

#### Implementation Requirements

**File:** `bazi_engine/__main__.py`

**CLI interface:**
```bash
# Pretty-print single chart
python -m bazi_engine --date "1990-05-15 14:30:00" --tz 8 --format pretty

# JSON output
python -m bazi_engine --date "1990-05-15 14:30:00" --tz 8 --format json

# With dynamic
python -m bazi_engine --date "1990-05-15 14:30:00" --tz 8 --current-date 2024-03-15 --format pretty
```

**Pretty format example:**
```
八字命盘
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
出生时间: 1990-05-15 14:30:00 (UTC+8)

四柱:
  年柱: 庚午  (金火)  七杀
  月柱: 辛巳  (金火)  正官
  日柱: 癸未  (水土)  日主
  时柱: 己未  (土土)  伤官

日主: 癸水  身弱 (强度: 0.23)
用神: 印 (金)

五行强度:
  木: ███░░░░░░░ 15.0%
  火: █████░░░░░ 25.0%
  土: ██████░░░░ 30.0%
  金: ████░░░░░░ 20.0%
  水: ██░░░░░░░░ 10.0%

相互关系:
  ✓ 天干合: 无
  ✓ 天干冲: 无
  ✓ 地支六合: 午未 (月-时)
  ✓ 地支六冲: 无
  ✓ 三合局: 无
  ✓ 六害: 无
  ✓ 相刑: 未未 自刑 (日-时)
  ✓ 相破: 无
```

#### TODO List

- [ ] **Create `bazi_engine/__main__.py`**
- [ ] **Add argparse CLI** with arguments above
- [ ] **Implement pretty formatter**:
  - Use Unicode box drawing characters
  - Color output with `colorama` (optional)
  - Show bar charts for elemental strengths
- [ ] **Implement JSON formatter**: call `chart.to_json(indent=2)`
- [ ] **Add `--format arrays` option**: print `to_feature_arrays()` keys and shapes
- [ ] **Manual testing**: Run on 3-5 known charts, verify output matches expectations

---

## Integration Points

### With Previous Tasks

**Task 0.2 (BaziCalculator):**
- `BaziCalculator.from_datetime()` must now populate `birth_datetime` and `timezone_offset` fields in returned `BaziChart`

**Task 0.3-0.7 (Interactions, Elemental Strength, Useful God):**
- All computed fields must be stored in `BaziChart` dataclass
- `to_json()` serializes these as-is
- `to_feature_arrays()` converts them to NumPy arrays

**Task 0.6 (Dynamic Calculator):**
- If dynamic context provided, `BaziChart.dynamic` field is populated
- `to_json()` and `to_feature_arrays()` handle `dynamic != None` case

### With Phase 2 (Data Generation)

**Usage in synthetic data pipeline:**
1. Generate 10k random birth datetimes
2. Use `batch_generate_charts.py` to produce `charts.jsonl`
3. Feed charts to GPT-4o via system prompt for interpretation generation
4. Store (chart JSON, interpretation) pairs for training

**JSONL format is critical:**
- One chart per line = easy parallel processing
- Can stream-read during training without loading full file into memory
- Compatible with HuggingFace `datasets` library: `load_dataset("json", data_files="charts.jsonl")`

### With Encoder Pre-training (Phase 1)

**Usage in encoder training:**
```python
from bazi_engine import BaziCalculator

chart = BaziCalculator.from_datetime(...)
arrays = chart.to_feature_arrays()

# Feed to encoder
stem_ids = torch.from_numpy(arrays['stem_ids'])
branch_ids = torch.from_numpy(arrays['branch_ids'])
# ... etc
```

---

## Testing Strategy

### Unit Tests (per sub-task)

- **0.8.1**: `test_to_json_natal_only.py`, `test_to_json_with_dynamic.py`
- **0.8.2**: `test_to_feature_arrays_natal.py`, `test_to_feature_arrays_dynamic.py`, `test_encoding_consistency.py`
- **0.8.3**: `test_batch_script.py`
- **0.8.4**: Manual testing only (CLI tools typically don't need unit tests)

### Integration Tests

**File:** `tests/integration/test_full_pipeline.py`

- [ ] **Test end-to-end natal chart**:
  - `from_datetime()` → `to_json()` → parse JSON → verify all fields
- [ ] **Test end-to-end with dynamic**:
  - `from_datetime()` with dynamic → `to_json()` → verify dynamic section
- [ ] **Test feature array export**:
  - `from_datetime()` → `to_feature_arrays()` → verify shapes/dtypes
  - Round-trip with encoder mock: arrays → encoder input → verify no crashes

### Edge Cases to Test

- [ ] **Solar term boundary**: Birth datetime exactly at 立春 → verify month pillar
- [ ] **Timezone handling**: Birth in UTC-5 vs UTC+8 → verify pillars differ
- [ ] **Missing dynamic**: Chart with `dynamic=None` → JSON has `"dynamic": null`
- [ ] **Partial dynamic**: Only 大运, no 流年 → pillar_presence_mask = [1, 0, 0]
- [ ] **Ambiguous useful god**: Chart where useful god unclear → `useful_god_id = -1`

---

## File Structure After Task 0.8

```
bazi_engine/
├── core/
│   ├── bazi_chart.py          # BaziChart dataclass with to_json(), to_feature_arrays()
│   ├── encodings.py           # NEW: STEM_TO_ID, BRANCH_TO_ID, etc.
│   └── calculator.py          # BaziCalculator (from Task 0.2)
├── interactions/
│   └── ...                    # Interaction detection (Tasks 0.3)
├── __main__.py                # NEW: CLI tool (optional)

scripts/
├── batch_generate_charts.py  # NEW: CSV → JSONL batch processor

tests/
├── test_to_json_natal_only.py           # NEW
├── test_to_json_with_dynamic.py         # NEW
├── test_to_feature_arrays_natal.py      # NEW
├── test_to_feature_arrays_dynamic.py    # NEW
├── test_encoding_consistency.py         # NEW
├── test_batch_script.py                 # NEW
└── integration/
    └── test_full_pipeline.py            # NEW
```

---

## Definition of Done

### 0.8.1 - JSON Serialization
- [ ] `BaziChart.to_json()` implemented and tested
- [ ] Natal-only chart serializes correctly
- [ ] Chart with dynamic context serializes correctly
- [ ] Chinese characters preserved in JSON
- [ ] All unit tests pass

### 0.8.2 - NumPy Array Export
- [ ] `bazi_engine/core/encodings.py` created with all encoding tables
- [ ] `BaziChart.to_feature_arrays()` implemented and tested
- [ ] Array shapes and dtypes verified for natal chart
- [ ] Dynamic arrays verified when present
- [ ] All unit tests pass

### 0.8.3 - Batch Processing
- [ ] `scripts/batch_generate_charts.py` created
- [ ] Can process 100+ charts from CSV → JSONL
- [ ] Error handling for invalid dates
- [ ] Integration test passes

### 0.8.4 - CLI Tool (Optional)
- [ ] `bazi_engine/__main__.py` created
- [ ] Pretty format displays correctly for manual testing
- [ ] JSON format works
- [ ] Manually verified on 3+ test charts

### Overall Integration
- [ ] All 0.8 unit tests pass (pytest coverage > 90% for new code)
- [ ] Integration test `test_full_pipeline.py` passes
- [ ] Can successfully generate 100-chart JSONL file from random dates
- [ ] Feature arrays can be loaded into PyTorch tensors without errors
- [ ] Documentation updated with usage examples

---

## Implementation Notes

### Performance Considerations

**JSON serialization:**
- For 10k charts: use `indent=None` (compact) during batch generation to save disk space
- Full dataset will be ~500MB-1GB depending on chart complexity

**NumPy arrays:**
- Use `dtype=np.int64` for categorical features (not `np.int32`) for compatibility with PyTorch default long tensor
- Use `dtype=np.float64` for continuous features (better precision than float32 during aggregation)

**Batch processing:**
- Use `tqdm` for progress bar on large batches
- Consider multiprocessing for >10k charts (parallelize calculator, not I/O)

### Data Validation

**JSON schema validation (optional but recommended):**
- Create JSON Schema file `schemas/bazi_chart.schema.json`
- Validate all generated charts against schema before training
- Catch serialization bugs early

**Array shape validation:**
- Add assertions in `to_feature_arrays()` to verify shapes before return
- Example: `assert stem_ids.shape == (4,), f"Expected shape (4,), got {stem_ids.shape}"`

### Encoding Design Choices

**Stem/Branch ID assignment:**
- Use cycle order (甲=0, 乙=1, ..., 癸=9) for stems
- Use cycle order (子=0, 丑=1, ..., 亥=11) for branches
- This preserves natural ordering → encoder can learn positional patterns

**Ten God encoding:**
- Use separate ID for 日主 (10) vs other ten gods (0-9)
- Alternative: omit day stem from ten_god array entirely (it's always 日主)
- Current approach: keep it explicit for clarity

**Useful God encoding:**
- Two options:
  1. Element-based (比劫/食伤/财/官杀/印) → 5 categories
  2. Full ten-god mapping → 10 categories
- **Recommendation:** Start with element-based (simpler, matches 子平 methodology)

---

## Dependencies

**Python packages:**
- `numpy` - array export
- `pandas` (optional) - CSV parsing (can use stdlib `csv` instead)
- `tqdm` (optional) - progress bars
- `colorama` (optional) - CLI pretty printing

**Internal dependencies:**
- All of Tasks 0.1-0.7 must be complete
- `BaziChart` dataclass must have all fields from calculator
- `InteractionResult.to_binary_vector()` must return 64-dim array

---

## Timeline (2.5 days)

**Day 1 Morning (0.5d):**
- Create `encodings.py`
- Implement `to_json()` for natal charts
- Write unit tests for JSON serialization

**Day 1 Afternoon (0.5d):**
- Add dynamic serialization to `to_json()`
- Test with dynamic context
- Handle edge cases (None values, ambiguous useful god)

**Day 2 Morning (0.5d):**
- Implement `to_feature_arrays()` for natal charts
- Write unit tests for array export

**Day 2 Afternoon (0.5d):**
- Add dynamic arrays to `to_feature_arrays()`
- Test encoding consistency
- Integration testing

**Day 3 (0.5d):**
- Create batch processing script
- CLI tool (if time permits)
- Final integration test with Phase 2 mock pipeline

---

## Success Metrics

After Task 0.8, you should be able to:

1. ✅ Generate 10,000 charts from random birth dates in <5 minutes
2. ✅ Load JSONL into Python: `charts = [json.loads(line) for line in open("charts.jsonl")]`
3. ✅ Convert any chart to NumPy arrays ready for PyTorch encoder
4. ✅ Manually verify chart accuracy with CLI tool: `python -m bazi_engine --date "..." --format pretty`
5. ✅ All tests pass: `pytest tests/ -v --cov=bazi_engine --cov-report=term-missing`

---

## Risk Mitigation

**Risk: JSON schema drift**
- Mitigation: Lock schema version in metadata; add schema validator before Phase 2

**Risk: Encoding changes break trained models**
- Mitigation: Never change `encodings.py` after Phase 1 starts; document as frozen

**Risk: Dynamic arrays shape mismatch**
- Mitigation: Use sentinel values (-1) and pillar_presence_mask; encoder must handle variable shapes

**Risk: Batch script crashes on large files**
- Mitigation: Process in chunks; add `--resume-from-line N` flag for crash recovery

---

## Next Steps After 0.8

With serialization complete, Phase 0 is done. You can now:

1. **Validate full pipeline** with 100 charts from `batch_generate_charts.py`
2. **Create reference dataset** of 10-15 hand-verified charts for encoder pre-training validation
3. **Begin Phase 1** (Encoder Pre-training) using `to_feature_arrays()` outputs
4. **Begin Phase 2** (Data Generation) using `to_json()` outputs as GPT-4o input

The calculator engine is now production-ready for the ML pipeline.
