# Scripts Guide: Data Generation to Training

This document describes all scripts in `scripts/` and the recommended flow from generating training data through training the bazi model tasks.

---

## 1. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  DATA GENERATION                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  generate_training_charts.py  →  dates.csv (birth datetimes + optional dynamic)  │
│           │                                                                      │
│           ▼                                                                      │
│  compute_charts.py  OR  batch_generate_charts.py  →  charts.jsonl                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  VALIDATION (optional but recommended)                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│  validate_dataset.py charts.jsonl     →  Day master / month branch balance      │
│  verify_encoder_params.py             →  Encoder param counts (~2M each)          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  TRAINING                                                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Single-task:  train_interaction | train_useful_god | train_element_strength |     │
│                train_ten_god | train_cross_temporal                               │
│  Multi-task:   train_multi_task.py (shared encoder, 4 natal tasks)                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Artifacts:**

| Artifact        | Description |
|-----------------|-------------|
| `dates.csv`     | Rows: birth_datetime, timezone_offset, lat, lng; optional: gender, flowing_year, flowing_month. Input to chart computation. |
| `charts.jsonl`  | One JSON line per chart (BaziChart.to_json). Consumed by ChartDataset and all training scripts. Typically under `data/charts.jsonl`. |
| `checkpoints/`  | Saved model states (e.g. `best.pt`, `final.pt`) from training scripts. |

---

## 2. Data Generation Scripts

### 2.1 `generate_training_charts.py`

**Purpose:** Generate a CSV of random birth date/time and location rows for later chart computation. No chart computation is done here.

**Output:** CSV with columns: `birth_datetime`, `timezone_offset`, `lat`, `lng`, `gender`, `flowing_year`, `flowing_month`.

**When to use:** Start of the pipeline when you need synthetic training data. Use `--stratified` for better (month, hour) coverage; use `--with-dynamic-fraction` so a fraction of rows have temporal fields for the cross-temporal task.

**Usage examples:**

```bash
# Default: 1000 rows → dates.csv
python scripts/generate_training_charts.py

# Larger dataset, stratified by (month, hour), with 20% having temporal fields
python scripts/generate_training_charts.py --num-samples 50000 --output dates.csv \
  --stratified --with-dynamic-fraction 0.2 --seed 42

# Validate day-master distribution after writing
python scripts/generate_training_charts.py --num-samples 10000 --output dates.csv --validate

# Validate an existing CSV only
python scripts/generate_training_charts.py --validate-only dates.csv
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--num-samples` | 1000 | Number of rows to generate |
| `--output` | dates.csv | Output CSV path |
| `--seed` | None | Random seed for reproducibility |
| `--with-dynamic-fraction` | 0.0 | Fraction of rows to add gender, flowing_year, flowing_month (e.g. 0.2 for cross-temporal) |
| `--stratified` | False | Stratify by (month, hour) bins (288 bins), then shuffle |
| `--validate` | False | After writing, run day-master distribution check |
| `--validate-only PATH` | — | Only run day-master validation on existing CSV; then exit |

---

### 2.2 `compute_charts.py`

**Purpose:** Read a CSV of birth dates/locations, compute one BaziChart per row (with optional temporal context when `flowing_year` and `gender` are present), and write one JSON line per chart. Uses **multiprocessing** for speed.

**Input:** CSV from `generate_training_charts.py` or compatible (columns: birth_datetime, timezone_offset, lat, lng; optional: gender, flowing_year, flowing_month).

**Output:** JSONL file (e.g. `data/charts.jsonl`).

**When to use:** Preferred for large-scale data generation; use this when you have a dates CSV and want charts written in row order with optional parallelism.

**Usage examples:**

```bash
python scripts/compute_charts.py --input dates.csv --output data/charts.jsonl

# Parallel with 8 workers, skip failing rows
python scripts/compute_charts.py --input dates.csv --output data/charts.jsonl --workers 8 --skip-errors
```

**Options:**

| Option | Required | Description |
|--------|----------|-------------|
| `--input` | Yes | Input CSV path (e.g. dates.csv) |
| `--output` | Yes | Output JSONL path (e.g. data/charts.jsonl) |
| `--workers` | No | Worker processes (default: CPU count - 1) |
| `--skip-errors` | No | Skip rows that fail; default is fail-fast |

**Note:** Rows with both `flowing_year` and `gender` get temporal context (大运/流年/流月 and cross-interactions). Ensure your CSV has these columns for the fraction of rows you want temporal data (e.g. from `--with-dynamic-fraction`).

---

### 2.3 `batch_generate_charts.py`

**Purpose:** Same idea as `compute_charts.py` but single-process and with an explicit `--current-date` for dynamic charts. Reads CSV, writes one JSON line per chart.

**Input:** CSV with birth_datetime, timezone_offset; optional lat, lng, gender.

**Output:** JSONL file.

**When to use:** When you need a fixed reference date for 大运/流年/流月 (e.g. `--current-date 2024-03-15`) or when you prefer a simple single-process script.

**Usage examples:**

```bash
python scripts/batch_generate_charts.py --input dates.csv --output data/charts.jsonl

# With temporal context at a specific date
python scripts/batch_generate_charts.py --input dates.csv --output data/charts.jsonl \
  --include-dynamic --current-date 2024-03-15
```

**Options:**

| Option | Required | Description |
|--------|----------|-------------|
| `--input` | Yes | Input CSV path |
| `--output` | Yes | Output JSONL path |
| `--include-dynamic` | No | Compute 大运/流年/流月 for --current-date |
| `--current-date` | No | Date for dynamic (YYYY-MM-DD); default: today |
| `--skip-errors` | No | Skip rows that fail |

---

### 2.4 Astrodatabank (Phase 2 data curation)

**Purpose:** Ingest real birth data from Astrodatabank, convert to BaziChart JSONL, then extract life events from biographies via Qwen (Dashscope) for 流年/流月 grounding.

**Flow (two scales):**

- **Large scrape (tens of thousands of rows):** `astrodatabank_scrape.py` (Playwright + MediaWiki `allpages` + batched wikitext) → `data/astrodatabank_scraped.jsonl` → then the same `astrodatabank_to_bazi.py` step with `--input` pointing at that file. This is how you get **full** coverage beyond the small public export.
- **Smaller / API / export:** `astrodatabank_api.py` (category-limited), or `astrodatabank_export_to_jsonl.py` (public **c_sample.zip**, ~6k names) → raw JSONL → `astrodatabank_to_bazi.py` → Bazi JSONL → `astrodatabank_extract_events.py` → event-annotated JSONL (and optional 流年/流月 example records).

- **astrodatabank_scrape.py:** Headless Chromium passes astro.com’s browser check, then calls the wiki API to list all pages and fetch wikitext in batches. Writes `data/astrodatabank_scraped.jsonl` (same schema as raw JSONL for `astrodatabank_to_bazi.py`). Use `--resume` to continue after interruption; optional `--skip-existing data/astrodatabank_raw.jsonl` to skip names already collected from another source. Requires `pip install playwright` and `python -m playwright install chromium`. **Runtime:** hours (rate limits + many pages); run on a stable network (VPN/firewall can break TLS or close connections).
- **astrodatabank_api.py:** Fetches AA/A-rated entries from Astrodatabank MediaWiki API via HTTP (often blocked by bot checks); category caps limit how many rows you get per run.
- **astrodatabank_to_bazi.py:** Converts raw JSONL to BaziChart JSONL using true solar time; output: `data/astrodatabank_bazi.jsonl`.
- **astrodatabank_extract_events.py:** For each record, calls Qwen-Plus (Dashscope) to extract life events (year, type, description; optional month) from biography. Writes `data/astrodatabank_events.jsonl` with an `events` array per line. Optional `--liunian-output` produces 流年/流月 example records (chart with `dynamic` for each event year/month) for Phase 2 temporal training.

**Requirements:** `pip install dashscope python-dotenv`; set `DASHSCOPE_API_KEY` when processing rows with non-empty biography.

**API key (Alibaba Cloud / Dashscope):** The script reads `DASHSCOPE_API_KEY` from the environment (or from a **`.env`** at the project root). For **US Virginia** region keys (not CN or SG), set `DASHSCOPE_BASE_URL=https://dashscope-us.aliyuncs.com/api/v1` in `.env` as well; otherwise the SDK may use the default China endpoint. Get the key from [Alibaba Cloud Console](https://dashscope.console.aliyun.com/) → Dashscope → API-KEY.

**Usage:**

```bash
# Bulk: scrape wiki (Playwright) — prefer this when you need many thousands of rows
PYTHONUNBUFFERED=1 python scripts/astrodatabank_scrape.py --output data/astrodatabank_scraped.jsonl
python scripts/astrodatabank_to_bazi.py --input data/astrodatabank_scraped.jsonl --output data/astrodatabank_bazi.jsonl
python scripts/select_charts.py --input data/astrodatabank_bazi.jsonl --output data/selected_charts.jsonl

# Alternative: API (limited) or adbexport c_sample (~6k rows) → astrodatabank_raw.jsonl
python scripts/astrodatabank_api.py --target 2500 --output data/astrodatabank_raw.jsonl
python scripts/astrodatabank_to_bazi.py --input data/astrodatabank_raw.jsonl --output data/astrodatabank_bazi.jsonl
python scripts/select_charts.py --input data/astrodatabank_bazi.jsonl --output data/selected_charts.jsonl

python scripts/astrodatabank_extract_events.py --input data/selected_charts.jsonl --output data/astrodatabank_events.jsonl
# Optional: also write 流年/流月 examples (chart + dynamic per event)
python scripts/astrodatabank_extract_events.py --input data/selected_charts.jsonl --output data/astrodatabank_events.jsonl --liunian-output data/astrodatabank_liunian_examples.jsonl --default-gender 1
```

---

### 2.5 `generate_stratified_charts.py` (Phase 2 synthetic charts)

**Purpose:** Generate stratified synthetic Bazi charts for Phase 2 data curation. Uses acceptance sampling so day master and month branch are ~uniform, day-master strength extremes (身强/身弱) and charts with 刑/破/害 are oversampled, and useful-god diversity is balanced. A fraction of charts include 大运/流年/流月 (temporal_context + cross_interactions) for 流年/流月 training.

**Output:** JSONL with one object per line: `{"birth_datetime": "...", "chart": {...}}` where `chart` is full BaziChart JSON (including optional `dynamic` for the temporal subset). Default path: `data/synthetic_charts.jsonl`.

**When to use:** As Source 2 in the Phase 2 curation pipeline (see Phase 2 Data Curation Plan). Output feeds into prompt builder + Qwen bulk generation, then QC and curation.

**Usage:**

```bash
# Default: 10,000 charts, 20% with dynamic, → data/synthetic_charts.jsonl
python scripts/generate_stratified_charts.py

# Custom count and output path
python scripts/generate_stratified_charts.py --num-charts 5000 --output data/synthetic_charts.jsonl --seed 42

# Higher fraction with 流年/流月 context
python scripts/generate_stratified_charts.py --num-charts 10000 --dynamic-fraction 0.3
```

| Option | Default | Description |
|--------|---------|-------------|
| `--num-charts` | 10000 | Target number of charts (may write slightly more to meet diversity targets) |
| `--output` | data/synthetic_charts.jsonl | Output JSONL path |
| `--dynamic-fraction` | 0.2 | Fraction of charts with temporal_context + cross_interactions |
| `--seed` | None | Random seed for reproducibility |
| `--max-attempts-ratio` | 30 | Max sampling attempts = num_charts × this value |

---

## 3. Validation & Verification Scripts

### 3.1 `validate_dataset.py`

**Purpose:** Check chart dataset diversity and balance. Reads `charts.jsonl` and reports total count, share with temporal data, day-master (天干) distribution, month-branch (地支) distribution, and balance warnings.

**Usage:**

```bash
python scripts/validate_dataset.py data/charts.jsonl
```

**Output:** Printed report (total charts, temporal %, per-stem and per-branch counts, balance ✅/⚠️).

---

### 3.2 `verify_encoder_params.py`

**Purpose:** Ensure NatalEncoder and DynamicEncoder parameter counts are in the expected range (~1.5M–2.5M each). Exits 0 if both are in range, 1 otherwise. No inputs; uses default encoder configs.

**Usage:**

```bash
python scripts/verify_encoder_params.py
```

**When to use:** After changing encoder architecture or before committing Phase 1–related changes.

---

### 3.3 `solar_term_audit.py`

**Purpose:** Audit solar term (节气) boundaries and month-pillar correctness for the bāzì calculator (uses `lunar_python`). No chart data required; standalone verification of the calendar/term logic.

**When to use:** When debugging month-pillar or solar-term edge cases, or after upgrading the lunar library.

---

## 4. Training Scripts

All training scripts expect **charts.jsonl** (path typically `data/charts.jsonl`). They use `ChartDataset` and the shared `Trainer` in `bazi_model.training`. Checkpoints are written under `checkpoints/` by default.

### 4.1 `train_interaction.py`

**Purpose:** Train interaction-detection head: NatalEncoder + InteractionHead. Predicts the 64-dim binary interaction vector from natal chart embedding.

**Usage:**

```bash
python scripts/train_interaction.py --data data/charts.jsonl

# Custom batch size, epochs, checkpoint dir
python scripts/train_interaction.py --data data/charts.jsonl --batch_size 64 --epochs 30 --checkpoint_dir checkpoints/interaction
```

**Key options:** `--data` (default: data/charts.jsonl), `--batch_size`, `--epochs`, `--lr`, `--checkpoint_dir`, `--log_dir`, `--seed`.

---

### 4.2 `train_useful_god.py`

**Purpose:** Train 用神 (useful god) classifier: NatalEncoder + UsefulGodHead. Uses `useful_god_id` from charts (0–4 or -1).

**Usage:**

```bash
python scripts/train_useful_god.py --data data/charts.jsonl
```

**Key options:** Same pattern as above (`--data`, `--batch_size`, `--epochs`, `--checkpoint_dir`, etc.).

---

### 4.3 `train_element_strength.py`

**Purpose:** Train element strength regressor: NatalEncoder + ElementStrengthHead. Target: `elemental_strengths` (5-dim), MSE loss. Uses full dataset (no temporal_only filter).

**Usage:**

```bash
python scripts/train_element_strength.py --data data/charts.jsonl
# or
python -m scripts.train_element_strength --data data/charts.jsonl
```

---

### 4.4 `train_ten_god.py`

**Purpose:** Train ten-god classification head: NatalEncoder + TenGodHead on charts.jsonl.

**Usage:**

```bash
python scripts/train_ten_god.py --dataset data/charts.jsonl --epochs 10
python scripts/train_ten_god.py --dataset data/charts.jsonl --batch-size 32 --lr 1e-3 --checkpoint-dir checkpoints --log-dir runs/ten_god
```

**Note:** This script uses `--dataset` (not `--data`) for the JSONL path.

---

### 4.5 `train_cross_temporal.py`

**Purpose:** Train cross-temporal head: NatalEncoder + DynamicEncoder + CrossTemporalHead. Uses **only charts that have temporal (dynamic) data** (`ChartDataset(..., temporal_only=True)`).

**Requirement:** Your `charts.jsonl` must contain a non-empty `dynamic` section for a subset of rows (e.g. generated with `--with-dynamic-fraction` and computed via `compute_charts.py` or `batch_generate_charts.py` with dynamic context).

**Usage:**

```bash
python scripts/train_cross_temporal.py --data data/charts.jsonl
# or
python -m scripts.train_cross_temporal --data data/charts.jsonl --epochs 20
```

**Key options:** `--data`, `--batch_size`, `--epochs`, `--checkpoint_dir`, `--log_dir`, `--seed`, `--num_workers`. Optional `--segment_weights` (7 comma-separated floats) and `--pos_weight_from_data` to improve macro F1 and weak segments (see `docs/design_decisions.md` §11).

**Tuning:** Use `scripts/tune_cross_temporal.py` for short-horizon trials over lr, segment_weights, and pos_weight_from_data, then optional full run with the best config (similar to `tune_phase1.py` for multi-task).

---

### 4.6 `train_multi_task.py`

**Purpose:** Train multiple natal tasks with one shared NatalEncoder: ten-god, element strength, interaction detection, useful god. Cross-temporal is **not** included (use `train_cross_temporal.py` for that).

**Usage:**

```bash
python scripts/train_multi_task.py --data data/charts.jsonl
python scripts/train_multi_task.py --data data/charts.jsonl --epochs 30 --log-dir runs/multi_task
```

**Key options:** `--data`, `--batch_size`, `--epochs`, `--lr`, `--checkpoint_dir`, `--log_dir`, `--seed`. Optional task weights can be added in code if needed.

---

## 5. End-to-End Flow: What to Run and When

### Minimal path (natal-only tasks)

1. **Generate dates CSV**
   ```bash
   python scripts/generate_training_charts.py --num-samples 20000 --output dates.csv --stratified --seed 42
   ```

2. **Compute charts**
   ```bash
   python scripts/compute_charts.py --input dates.csv --output data/charts.jsonl [--workers 8] [--skip-errors]
   ```
   (If your CSV has no `gender`/`flowing_year`, charts will have no temporal block; that’s fine for natal-only training.)

3. **Validate dataset (optional)**
   ```bash
   python scripts/validate_dataset.py data/charts.jsonl
   ```

4. **Train (pick one or more)**
   ```bash
   python scripts/train_interaction.py --data data/charts.jsonl
   python scripts/train_useful_god.py --data data/charts.jsonl
   python scripts/train_element_strength.py --data data/charts.jsonl
   python scripts/train_ten_god.py --dataset data/charts.jsonl
   # Or multi-task:
   python scripts/train_multi_task.py --data data/charts.jsonl
   ```

### Full path (including cross-temporal)

1. **Generate dates with a temporal fraction**
   ```bash
   python scripts/generate_training_charts.py --num-samples 50000 --output dates.csv \
     --stratified --with-dynamic-fraction 0.2 --seed 42
   ```

2. **Compute charts** (rows with gender + flowing_year get temporal context)
   ```bash
   python scripts/compute_charts.py --input dates.csv --output data/charts.jsonl --workers 8 --skip-errors
   ```

3. **Validate**
   ```bash
   python scripts/validate_dataset.py data/charts.jsonl
   ```
   Confirm that “Temporal charts” is non-zero (e.g. ~20% if you used 0.2).

4. **Train natal tasks** (as above).

5. **Train cross-temporal** (uses only temporal charts)
   ```bash
   python scripts/train_cross_temporal.py --data data/charts.jsonl
   ```

### Verification only (no new data)

- **Encoder params:** `python scripts/verify_encoder_params.py`
- **Solar terms:** `python scripts/solar_term_audit.py`
- **Existing charts:** `python scripts/validate_dataset.py data/charts.jsonl`
- **Existing dates CSV:** `python scripts/generate_training_charts.py --validate-only dates.csv`

---

## 6. Quick Reference

| Script | Input | Output | Purpose |
|--------|--------|--------|---------|
| `generate_training_charts.py` | — (or `--validate-only` CSV) | dates.csv | Generate/validate birth datetime CSV |
| `compute_charts.py` | dates.csv | charts.jsonl | Compute charts (multiprocessing) |
| `batch_generate_charts.py` | CSV | charts.jsonl | Compute charts (single process, optional current-date) |
| `validate_dataset.py` | charts.jsonl | stdout report | Dataset diversity/balance |
| `verify_encoder_params.py` | — | exit 0/1 | Encoder param range check |
| `solar_term_audit.py` | — | stdout | Solar term / month-pillar audit |
| `train_interaction.py` | charts.jsonl | checkpoints/ | Interaction 64-dim head |
| `train_useful_god.py` | charts.jsonl | checkpoints/ | Useful god classifier |
| `train_element_strength.py` | charts.jsonl | checkpoints/ | Element strength regressor |
| `train_ten_god.py` | charts.jsonl | checkpoints/ | Ten-god classifier |
| `train_cross_temporal.py` | charts.jsonl (with dynamic) | checkpoints/ | Cross-temporal head (temporal_only) |
| `train_multi_task.py` | charts.jsonl | checkpoints/ | Multi-task natal (4 tasks, 1 encoder) |

**Conventions:**

- Run all commands from the **repository root**.
- Default data path: `data/charts.jsonl`; default checkpoints: `checkpoints/`.
- For cross-temporal training, generate charts with `gender` and `flowing_year` (and optionally `flowing_month`) for a subset of rows, then use `compute_charts.py` or `batch_generate_charts.py` so those rows get a `dynamic` block in the JSONL.
