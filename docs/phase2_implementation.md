# Phase 2 Implementation (Detailed Spec)

Step-by-step implementation spec for Phase 2 (projection layer + frozen LLM). Source: `docs/phase2_detailed_plan.docx` and the Phase 2 implementation plan.

**Related:** [docs/phase2_model_and_env.md](phase2_model_and_env.md) (environment, LLM, encoders), [docs/phase1_best_checkpoint.md](phase1_best_checkpoint.md), [docs/design_decisions.md](design_decisions.md).

---

## Task 2.3 — Dataset and task balancing

### 2.3.1 Format and preprocessing

#### Curated input format (expected)

Input is a JSONL file (e.g. `data/curated_synthetic.jsonl`) where each line is one example. Required fields:

| Field        | Type             | Description |
| ------------ | ---------------- | ----------- |
| `chart`      | object or string | Full BaziChart JSON (as from `BaziChart.to_json()`) or a reference. Must be parseable by `BaziChart.from_json(json.dumps(chart))`. |
| `task_type`  | string           | One of: `本命分析`, `流年`, `流月`, `八字问答`. |
| `query`      | string           | User question or prompt (Chinese). |
| `reasoning`  | string           | Model reasoning (chain-of-thought) before the final answer. |
| `output`     | string           | Final answer to be generated. |

**Design choice:** Chart is stored as **full BaziChart JSON** so the prepare script and training can call `BaziChart.from_json(...)` and then:

- **Natal-only tasks** (本命分析, 八字问答): `to_input_features("multi")` for encoder input.
- **Temporal tasks** (流年, 流月): chart must have non-null `dynamic` and `cross_interactions`; use `to_input_features("cross_temporal")` for dynamic encoder input.

No precomputed chart_features in the minimal implementation; optional later for speed.

#### Prepared output format (Phase 2 train/val JSONL)

After running `scripts/prepare_phase2_dataset.py`, each line has the same fields plus:

| Field             | Type   | Description |
| ----------------- | ------ | ----------- |
| `task_token`      | string | Prepend to model input, e.g. `[TASK:本命分析]`. |
| `input_sequence`  | string | (Optional) Full text input: task_token + query. |
| `target_sequence`| string | (Optional) Text on which loss is computed: reasoning + output. |

**Target sequence and loss masking (align with Task 2.4):**

- **Model input:** Virtual tokens (from chart) + tokenized `task_token + query`.
- **Target (labels):** Tokenized `reasoning + output`.
- **Loss mask:** Positions for `task_token + query` → mask 0 (no loss); positions for `reasoning + output` → mask 1 (loss computed).
- So **query is in input only**, not in target. **Loss is only on reasoning + output.** Documented in the prepare script docstring and in the training loss (Task 2.4).

#### Prepare script contract

- **Script:** `scripts/prepare_phase2_dataset.py`
- **Input:** Curated JSONL path (e.g. `data/curated_synthetic.jsonl`).
- **Output:** `data/phase2_train.jsonl`, `data/phase2_val.jsonl` (or paths from args); optional **`data/phase2_test.jsonl`** when `--test_ratio` is positive (stratified by `task_type`, same rules as val).
- **Behaviour:** Validate each line (chart, task_type, query, reasoning, output); for 流年/流月 validate chart has `temporal_context` and `cross_interactions`; add `task_token`; train/val/(test) split; write JSONL. Log counts per task_type and skipped rows. Requires `val_ratio + test_ratio < 1`.

---

### 2.3.2 Task balancing

**Target distribution:**

| task_type | Target fraction |
| --------- | ---------------- |
| 本命分析   | 40%             |
| 流年      | 30%             |
| 流月      | 10%             |
| 八字问答   | 20%             |

So 流年/流月 together are 40%.

**Implementation:** Task balancing is implemented in **Task 2.4** (training script `scripts/train_phase2.py`), not in the prepare script. Options:

- **Weighted random sampler:** Assign each example a weight = target_fraction / actual_fraction for its `task_type` (with small epsilon to avoid div-by-zero). Use in the Phase 2 DataLoader.
- **Oversampling:** Duplicate examples from underrepresented task types so the dataset approximates the target distribution; then use a standard random sampler.

**Contract for Task 2.4:** The Phase 2 dataset class must expose `task_type` per sample. The training script uses either a `WeightedRandomSampler` (or custom sampler) built from these task_type values and the target fractions above, or an oversampled dataset. Script args: e.g. `--balance_tasks`, `--target_distribution` or `--task_weights`; document in the train script.

**Constants (single source of truth):** Define the exact task_type list and target fractions in one place, e.g. in the prepare script as `TASK_TYPES` and `TARGET_DISTRIBUTION`, and reuse in `train_phase2.py` for the sampler.

---

## Other tasks

- **Task 2.1:** Environment and model preparation — see [docs/phase2_model_and_env.md](phase2_model_and_env.md).
- **Task 2.2:** Projection layer and BaziLLM — see [bazi_model/projection/projection_layer.py](../bazi_model/projection/projection_layer.py), [bazi_model/model/bazi_llm.py](../bazi_model/model/bazi_llm.py).
- **Task 2.4:** Training infrastructure (loss, optimizer, loop, validation, early stopping) — **detailed plan:** [docs/phase2_task_2_4_plan.md](phase2_task_2_4_plan.md).
- **Task 2.5:** Training execution and tuning.
- **Task 2.6:** Evaluation and attention analysis.
