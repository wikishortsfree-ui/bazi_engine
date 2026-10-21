# State of the Project — Review for Next Steps

**Date:** 2026-04-29
**Author/Compiler:** consolidated from `docs/`, `plans/`, `tasks/`, `reports/`, `logs/`, `tasks/lessons.md`, and recent agent transcripts.
**Audience:** project owner deciding what to do next; future agents needing one-page orientation.

> This is a **review document**, not a plan. It centralises everything that has been tried and everything that is currently known about why each thing worked or failed, plus the live constraints. It is intentionally opinionated about what the failure modes are and what to consider next, but does not commit to a path.

---

## 0. TL;DR

**Goal of the project:** turn a Chinese 八字 (bazi) chart into an LLM-driven interpretation. The system has three layers: a deterministic **bazi engine** (rule-based pillars, interactions, strengths) → a small trained **chart encoder + projection** that turns the chart into virtual prefix tokens → a frozen **Qwen 2.5-7B-Instruct** decoder that reads those virtual tokens plus a question and writes the answer.

**Where we are:**

- The bazi engine and Phase 1 multi-task encoder are **production-quality** (four natal tasks at 100 % / 99 %+, ordinal at 81 % capped by label noise).
- Phase 2 (prose interpretation training) and Phase 3 prep (QA-objective retraining) have produced **four projection-stack architectures** in sequence. None has cleared the factual probe gate so far. Failures cluster around two specific holes (per-pillar identity and missing natal-internal interaction signal) that a fifth architecture (`v1.2 / v3` embedding-only natal head) was designed to fix.
- The `v3` embedding pilot was **paused at step 950 / ~2350** (40 % of epoch 1) on 2026-04-29. Best partial val_loss = 0.9242 and still improving. Pilot probe (the gate) has **not** been run yet.
- **Hard constraint surfacing on this run:** local Apple Silicon machine is RAM-pressured (≥ 98 % during training); throughput became erratic mid-pilot. This is the immediate blocker on completing v3 the way the plan was written.

**Three plausible next directions, ranked by expected information value per hour of compute:**

1. **Probe the partial v3 best checkpoint now** (cheap, ~minutes). Even at 40 % of epoch 1, the probe will say whether the embedding architecture moved the per-pillar / ten-god categories at all. Cheapest decision-quality action available.
2. **Either resume v3 to completion on a less constrained machine, or shrink the pilot** (smaller subset, fewer steps) so the architecture decision is isolated from the wall-clock blocker.
3. **Stop iterating on the projection stack and add LoRA / partial unfreeze on the LLM.** The Phase 2 v2 handoff already concluded "projection-only training has hit its ceiling for factual accuracy"; v1.1 and v1.2 of the QA retrain are testing whether that conclusion was premature, but if v3 also misses the gate, the next move is up the stack, not deeper into projection variants.

Detail in §10. Everything before that is the evidence base for choosing among them.

---

## 1. System architecture (current, as of v3 pilot)

```
                                                           ┌─────────────────────────────────┐
                                                           │  Frozen Qwen 2.5-7B-Instruct    │
                                                           │  (4-bit MLX, 3584-d hidden)     │
BaziChart (date, sect)                                     │                                 │
        │                                                  │   ┌───────────────────────┐     │
        ▼                                                  │   │ logits over vocab     │     │
┌──────────────────────────┐                               │   └───────────────────────┘     │
│   bazi_engine (rule)     │                               │            ▲                    │
│  • 4 pillars (Y M D H)   │                               │            │                    │
│  • interactions (合冲刑害破)│                              │   ┌───────────────────────┐     │
│  • element strengths     │      virtual tokens           │   │ transformer stack     │     │
│  • day-master strength   │      (b, 63, 3584)            │   └───────────────────────┘     │
│  • useful god (用神)      │   ────────────────►           │            ▲                    │
│  • interaction_vector(64)│                               │   ┌────────┴──────────────┐     │
└─────────┬────────────────┘                               │   │ input embeddings:     │     │
          │ to_input_features("multi")                     │   │ [virtual | query]     │     │
          ▼                                                │   └───────────────────────┘     │
┌────────────────────────────────────────────────┐         └─────────────────────────────────┘
│         Chart encoder stack (trainable)        │                       ▲
│                                                │       per-slot Linear │
│ slot 0: chart-summary (interactions+strengths) │      (7 × 9 × 3584)   │
│ slot 1: pillar 0 (year)                        │                       │
│ slot 2: pillar 1 (month)                       │       ┌───────────────┴────────────────┐
│ slot 3: pillar 2 (day)                         │       │ ProjectionLayer                 │
│ slot 4: pillar 3 (hour)                        │       │  per-slot Linear(256→9·3584)    │
│ slot 5: dynamic encoder (大运 / 流年)            │       │  n_slots=7, n_virtual=63        │
│ slot 6: cross-interaction encoder (27→256)     │       └─────────────────────────────────┘
│        d_model = 256 per slot                  │
└────────────────────────────────────────────────┘
```

**Shapes that are now contractual:**

- `n_slots = 7`, `n_virtual_tokens = 63` (`= 7 × 9`).
- `d_in_per_slot = 256`, `d_llm = 3584` (Qwen2.5-7B hidden size).
- Loss is causal-LM, masked to the answer span only (`compute_lm_loss` in `bazi_model/training/phase2_loss.py`).

**Frozen vs trainable today:**

- Frozen: Qwen LLM (always); `NatalEncoderMLX` and `DynamicEncoderMLX` (in v1.1 and earlier QA runs).
- Trainable: `ProjectionLayer`, `CrossInteractionEncoderMLX`. In v1.2 (current pilot) the natal module is replaced by `EmbeddingNatalMLX` and is **also** trained jointly.

---

## 2. Phase 1 — natal encoder (production-quality)

`docs/phase1_best_checkpoint.md` and `docs/phase1_final_architecture.md` are the source of truth. Summary:

| Aspect | Final value | Why |
| --- | --- | --- |
| Encoder | 3-layer Transformer, d_model = 256, d_ff = 1024, 4 heads | 3 layers chosen over 2 for ordinal (81.45 % vs 77.70 %). |
| Training mode | Multi-task only (one shared encoder, four heads) | Sequential ten-god capped at ~89 %; sequential useful-god collapsed at 23 %. Multi-task gets 100 % / 99.94 %. |
| Task weights | 1, 6, 1, 1 (ten-god, element, interaction, useful-god) | Element loss magnitude is dominated by CE losses; weight 6 brought ordinal up. |
| Aux losses | MSE + 0.15·pairwise ranking on element | Pure MSE is misaligned with ordinal accuracy. |
| Save-by | Composite (mean of four metrics) | Lowest val loss biases toward high-weighted task. |
| Cross-temporal | **Trained separately** (`train_cross_temporal.py`) | Different distribution; mixing in multi-task batch is messy. |

**Final checkpoint:** `checkpoints/multi_task_3layer_1_6_1_1.pt`. **Targets met:** ten-god 100 %, element MSE 0.00148, interaction F1 ≥ 99.99 %, useful-god 99.94 %. **Not met:** element ordinal 81.45 % vs > 90 % target, but ordinal-error analysis shows ~80 % of mistakes are pairs with true strength diff < 0.02 (label noise floor); we shipped this checkpoint.

**Key design decisions** (each in `docs/design_decisions.md`):

- **§1, §3, §12** Loss choices per head (MSE for element, focal+CE+label smoothing for ten-god, BCE for interaction).
- **§3.1** No data leakage — `to_input_features` returns only minimal inputs; all targets live in `to_ground_truth`. Encoders accept only `stem_ids, branch_ids, stem_elements, stem_polarities, pillar_positions` plus a learned aux token.
- **§13** Useful-god head pretrained on element-strength encoder (Option A).
- **§16** Pretrain-then-multitask (`train_element_strength.py` → `train_multi_task.py --encoder_checkpoint`) is supported.

**MLX parity:** `NatalEncoderMLX` and `DynamicEncoderMLX` re-implement the PyTorch encoders in MLX with weight-loading parity tests (`tests/test_encoder_mlx_parity.py`, `tests/test_embedding_natal_mlx.py`).

---

## 3. Phase 2 — prose interpretation training

### 3.1 v1 baseline (2-encoder, 8 virtual tokens)

| Metric | Value |
| --- | --- |
| Val loss (best) | 2.0210 |
| Perplexity | 7.56 |
| BLEU | 0.09 |
| ROUGE-L | 0.21 |
| **Factual consistency** | **44 %** |

Training: frozen Qwen + frozen encoders + trainable `ProjectionLayer (d_in=512, n_virtual=8)`. Curated 2–3 k chart+interpretation prose dataset. CE masked over the long-form interpretation.

### 3.2 v2 (3-encoder + cross-interaction encoder, 8 virtual tokens, 3 epochs)

`tasks/phase2_v2_handoff.md` is the source.

| Metric | v1 | v2 | Δ |
| --- | --- | --- | --- |
| Val loss (best) | 2.0210 | **1.8254** | −9.7 % |
| Perplexity | 7.56 | **6.22** | −17.7 % |
| BLEU | 0.09 | **0.11** | +25.8 % |
| ROUGE-L | 0.21 | **0.23** | +11.7 % |
| **Factual consistency** | 44 % | **28 %** | **−16 pp (worse)** |

Same checkpoint pattern, but: (a) added `CrossInteractionEncoderMLX` (27→128→256, trainable); (b) projection input dim 512 → 768 (concat natal+dynamic+cross); (c) trained 3 full epochs with `--resume`.

**v2 conclusion (from the handoff doc):** "Cross-interaction features + 3 epochs improved all language metrics but factual grounding degraded. The frozen LLM generates more fluent hallucinations. **Projection-only training has hit its ceiling for factual accuracy. LoRA fine-tuning is the necessary next step.**"

This conclusion is what the Phase 3 prep workstream is trying either to confirm or to disprove cheaply before committing to LoRA.

### 3.3 Phase 2 lessons captured in code/docs

- §17 `mlx_lm` Qwen model API for `input_embeddings` (custom virtual prefix); `mlx_lm.utils.does_model_support_input_embeddings(model)` must return True; pass placeholder `inputs` array of zeros for shape; positions 0..n_virtual−1 used for virtual tokens, n_virtual..L−1 for query.
- §18 Projection input is `concat(natal=256, dynamic=256, cross=256) = 768`; **single Linear** (Option A) `Linear(768, n_virtual·d_llm)` chosen over MLP.
- §19 Loss masking convention: full sequence is `input_sequence + "\n" + target_sequence`; mask 0 on prefix (task+query) and padding, mask 1 on (reasoning+output); shifted labels.
- §19.2 Optimizer: AdamW(weight_decay=0.01), LR with 10 % linear warmup → cosine to 10 %; checkpoint format `.npz` with `projection_layer.*` keys.
- §21 Virtual token expansion 8 → 64. Old 8-token checkpoints not shape-compatible.
- `--resume` flag in `train_phase2.py` was added to enable multi-day pause/resume. **This is essential** on Apple Silicon and matters again for the v3 pilot (see §6, §10).
- **`generate_step` foot-gun (`tasks/phase2_v2_handoff.md`):** `mlx_lm` v0.31.0 does not export `generate_step` at top level; access via `mlx_lm.stream_generate.__globals__["generate_step"]`. Yielded tokens may be `int` (not `mx.array`); `eos_token_ids` may be `int` not list.

---

## 4. Phase 3 prep — workstream A (factual probe gate)

The probe (`scripts/probe_factual_64token.py`) asks 20 single-fact questions over `data/phase2_test.jsonl` charts: 4 in each of `pillar_identity`, `ten_gods`, `five_elements`, `day_master_strength`, `interactions`. 19 are categorical, 1 is numeric. Same 20 questions / same charts across all probe runs, so per-category numbers are **directly comparable across rows**.

### 4.1 Probe matrix (all runs to date)

| Run / checkpoint | Train objective | Init | n_virtual | Categorical | pillar_id | ten_gods | 5_elements | dm_strength | interactions | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase 2 v1 / 8 token / prose | prose CE | fresh | 8 | ~30 % | 0 % | 25 % | n/a | n/a | **75 %** | "0 % / 75 % split by answer geometry" finding (`docs/design_decisions.md` §22). |
| `phase3_prep/qa_pilot_fresh` (5 k, 1 ep, fresh, lr 5e-5, single-Linear projection) | Stage 1 QA | fresh | 64 | **47.4 %** (9/19) | 0 % | 25 % | **100 %** | 50 % | 50 % | Best per-category 5_elements; pillar still 0 %. `reports/probe_qa_pilot_fresh.json`. |
| `phase3_prep/qa_pilot_warm` (5 k, 1 ep, warm from prose, lr 2e-5) | Stage 1 QA | warm | 64 | 42.1 % (8/19) | 0 % | 50 % | 50 % | 25 % | 75 % | Warm preserves interactions; loses 5_elements vs fresh. `reports/probe_qa_pilot_warm.json`. |
| `phase3_prep/qa_full` (20 k, 3 ep, fresh, lr 5e-5, val_loss = 0.7398 best) | Stage 1 QA | fresh | 64 | **26.3 %** (5/19) | **0 %** | 25 % | 50 % | **0 %** | 50 % | Worse than the 5 k pilot — more training did **not** help; in fact regressed. Trigger for `v1.2` re-architecture. `reports/probe_factual_64token_qa_results.json`. |
| `phase3_prep/qa_arch_v2_pilot` (5 k, 1 ep, fresh, **per-pillar 7-slot projection**, val_loss = 1.18) | Stage 1 QA | fresh | 63 | **21.1 %** (4/19) | 0 % | 0 % | 50 % | 25 % | 25 % | Per-pillar architecture (5 natal slots + dynamic + cross) introduced — but *all* the gain came from `5_elements`/`interactions`; pillar_id and ten_gods still 0/4. **This led to v3 plan.** `reports/probe_qa_arch_v2_pilot.json`. |
| `phase3_prep/qa_arch_v3_pilot` (20 k, 1 ep, fresh, **EmbeddingNatalMLX**) | Stage 1 QA | fresh | 63 | **NOT YET PROBED** | — | — | — | — | — | Paused at step 950 / 2350 (val_loss 0.9242 best at step 800, monotonically improving). |

### 4.2 Linear probe on the natal encoder (independent investigation)

`scripts/_probe_natal_encoder_tokens.py` — `logs/probe_natal_encoder.log`. Capture all 6 encoder output tokens for 750 charts, train a per-token linear classifier on each of `(year_stem, year_branch, …, hour_branch)`. **Result:**

```
P_year  -> year_stem:    test_acc = 1.000
P_year  -> year_branch:  test_acc = 1.000
P_month -> month_stem:   test_acc = 1.000
P_month -> month_branch: test_acc = 1.000
P_day   -> day_stem:     test_acc = 1.000
P_day   -> day_branch:   test_acc = 1.000
P_hour  -> hour_stem:    test_acc = 1.000
P_hour  -> hour_branch:  test_acc = 1.000
CLS     -> day_stem:     test_acc = 1.000
CLS     -> day_branch:   test_acc = 0.993  (other pillars: 0.49–0.79)
```

**Implication:** the per-pillar tokens already contain perfect identity; the production setup that flowed `CLS` only into projection was **discarding 5/6 of the encoder output tokens**, including the only ones that can name the year / month / hour pillars. This is the linear-probe finding.

### 4.3 Cross-features finding

`bazi_model/data/phase2_dataset.py::_cross_interaction_features` populates only `natal_liunian_*` / `natal_dayun_*` flags inside the 27-d `cross_features` vector. **For natal-only Stage 1 QA, this is all-zero.** Slot 6 of every QA training example fed an all-zero vector through a randomly initialised cross encoder. Meanwhile the natal-internal 64-d `interaction_vector` (`InteractionResult.to_binary_vector()`) is already produced by `chart.to_input_features("multi")` — it just was not plumbed to the model.

### 4.4 The pattern across the probe matrix

Three things are now visible side-by-side:

1. **The 0 % wall on `pillar_identity` is structural, not a training-time problem.** It is 0 % for prose, for QA-fresh-pilot, for QA-full (20 k × 3 ep), and for the v2 per-pillar pilot. More training did not help. A linear probe shows the bits are *available* in the encoder; the projection (`d_in=768` aggregate or per-slot pre-v3) is the suspect.
2. **`ten_gods` correlates with `pillar_identity`** in the probe. Both are 0 % across most runs. Ten-god reasoning needs the relationship between the day stem and other stems; if pillar identity isn't reaching the LLM, ten-god can't be either.
3. **The QA objective improved interactions slightly and never regressed it below 25 %** — but that means the prose run already had interactions at 75 %; the QA *full* run dropped it to 50 %. Stratified sampling under-trained interactions despite the design intent.

---

## 5. Confirmed dead-ends / what we know does **not** work

In rough order of confidence:

1. **More projection-only training does not fix factual grounding.** v1→v2 prose: factual 44 % → 28 % despite better val loss / BLEU / ROUGE. v1.0 QA-pilot (5 k) → v1.0 QA-full (20 k × 3 ep): categorical 47 % → 26 %. *The training improves what it is asked to (val loss); it does not improve probe accuracy.* (`tasks/phase2_v2_handoff.md`, `docs/design_decisions.md` §22.)
2. **8 virtual tokens is too few.** Roughly ~100 loosely independent chart facts; 8 slots force extreme compression. (`docs/design_decisions.md` §21.)
3. **CLS-only projection wastes the per-pillar tokens.** Linear probe shows the per-pillar tokens carry exact identity; CLS does not. (`logs/probe_natal_encoder.log`, §4.2.)
4. **Sequential single-task training of the natal encoder.** Useful-god collapses (23 %); ten-god caps at ~89 %. Multi-task gets 100 % / 99.94 %. (`docs/phase1_best_checkpoint.md` §2.1.)
5. **Pure MSE on element strength gives ~72 % ordinal.** Adding pairwise ranking aux loss (0.15) gets it to ~81 %. Beyond that hits a label-ambiguity ceiling (80 % of errors have true diff < 0.02). (`docs/design_decisions.md` §12, `docs/phase1_best_checkpoint.md` §6.2.)
6. **Saving by lowest val loss with unequal task weights biases the checkpoint.** Switched to save-by-composite. (`docs/design_decisions.md` §15.)
7. **Loss on the long-form interpretation smears per-fact gradient.** Single-fact QA loss-mask geometry concentrates gradient on the answer tokens. Reading from `tasks/lessons.md` (2026-04-27): "When a probe metric splits cleanly along answer-length lines, suspect the loss-mask geometry, not model capacity." (`docs/design_decisions.md` §22.) *This was clearly true vs 8-token prose; whether it is sufficient at 64 tokens is what v1.0/v1.1/v1.2 are testing — and v1.0 says no.*
8. **Stratified sampling alone did not protect interactions in the QA-full run.** Interactions 75 % (prose) → 50 % (qa_full). The 10 % interactions weight in the target distribution may have been too low, or weights aren't being realised in the actual training mix (this is one of the v1.1-status follow-ups in §22 of design_decisions but has not been investigated yet).

---

## 6. Resource constraints (current blocker, surfacing on this run)

- **Hardware:** Apple Silicon Mac, MLX backend, single GPU.
- **Memory pressure:** During the v3 pilot, `vm_stat` showed ~98 % page utilisation (~5 k free pages out of ~2 M). Throughput became erratic — first 50 steps took ~27 minutes (~32 s/step), then step 50 → 100 took ~20 minutes (slower per step), and steps 100 → 950 took ~7 hours total (highly variable per-step time consistent with swap/thrash).
- **Implication:** A single 3-epoch run on 20 k QA pairs at the current per-step cost is on the order of 1–3 days of wall-clock and is fragile to other resident apps competing for RAM. The training script has `--init_mode warm` (warm-starts projection only) but **does not yet have full optimizer-state resume** for `train_projection_qa.py` (the Phase 2 prose script does, via `--resume`).
- **Disk:** 12 GiB free on data volume; each checkpoint is ~232 MB (model) + ~465 MB (optimizer) → ~700 MB per pair. Plenty of room for many checkpoints.
- **Runs that completed under this constraint:** the QA-full 20 k × 3 ep prose-objective run (`qa_full`, val_loss 0.7398) finished. The v2 per-pillar pilot 5 k × 1 ep finished. The v3 pilot did not.
- **Ports / dev servers:** macOS port 5000 is taken by AirPlay (`tasks/lessons.md`). Use 5050 / 8080.

---

## 7. Inventory of what exists in the repo (quick reference)

### Engine (`bazi_engine/`)

- `models.py` — `BaziChart`, `to_input_features("multi"|"cross_temporal")`, `to_ground_truth(...)`, JSON I/O.
- `interactions/` — `detect_all_interactions(stems, branches)` → `InteractionResult` with blocking/transformation rules (`docs/tests_and_interactions_summary.md`). 64-d binary vector via `InteractionResult.to_binary_vector()`.
- `strength/` — element strengths, day-master strength, useful god (sect-aware: 0/1/2 = 子平/明派/盲派 default).
- `temporal.py` — 大运 / 流年 / 流月.

### Encoders (`bazi_model/encoders/`)

- `natal_encoder.py` (PyTorch) and `natal_encoder_mlx.py` (MLX) — per-pillar transformer, returns `(b, 5, 256)` (CLS + P_year + P_month + P_day + P_hour) since v2-arch.
- `dynamic_encoder.py` / `dynamic_encoder_mlx.py` — temporal encoder, `(b, 256)`.
- `cross_interaction_encoder_mlx.py` — `(b, 27) → (b, 256)` MLP.
- `embedding_natal_mlx.py` (**new, v1.2**) — embedding-only natal head, returns `(b, 5, 256)`. Slot 0 is `ChartSummaryHead(interaction_vector || elemental_strengths || day_master_strength || useful_god_one_hot)`; slots 1–4 are per-pillar embedding sums + Linear.

### Projection (`bazi_model/projection/`)

- `projection_layer.py` — per-slot Linears (`slot_0 .. slot_6`), `n_slots = 7`, `n_virtual_per_slot = 9`. Total 63 virtual tokens × 3584-d.

### Training (`bazi_model/training/`, `scripts/`)

- `phase2_loop.py` — shared helpers (`create_optimizer`, `clip_grads`, `_run_validation`, `_save_checkpoint`, `_load_projection_from_checkpoint`).
- `phase2_loss.py` — `compute_lm_loss(logits, labels, loss_mask)`.
- `train_phase2.py` — prose training (with `--resume` full state).
- `train_projection_qa.py` — Stage 1 QA training (with `--init_mode fresh|warm` and `--natal_mode encoder|embedding`). Note: **no full `--resume`** here yet (only `--init_mode warm` for projection weights).
- `probe_factual_64token.py` — 20-question probe with `--natal_mode {encoder, embedding}` and bucketing (categorical / numeric).
- `_probe_natal_encoder_tokens.py` — linear-probe on encoder output tokens (the §4.2 finding).
- `evaluate_phase2.py` — perplexity / BLEU / ROUGE / factual consistency on the prose checkpoint.

### Datasets

- `data/charts.jsonl` — all charts (Phase 1 training).
- `data/phase2_train|val|test.jsonl` — 6000 / 750 / 750 charts (regenerated 2026-04-28); used for prose training and the probe.
- `data/stage1_qa.jsonl` (~1 M QA pairs) + `data/qa_charts.jsonl` (chart_id → chart) — Stage 1 single-fact QA.
- `data/stage2_qa.jsonl` — Stage 2 multi-fact (built but not used for training yet).
- `data/curated_synthetic.jsonl`, `data/synthetic_final.jsonl` — Phase 2 prose curation outputs.
- `data/classical_texts/` — optional 命理 corpus for RAG (`scripts/build_classical_rag_index.py` → BAAI/bge-small-zh-v1.5 + FAISS).

### Checkpoints (live)

- `checkpoints/multi_task_3layer_1_6_1_1.pt` — Phase 1 best.
- `checkpoints/natal_encoder_mlx.npz`, `dynamic_encoder_mlx.npz` — MLX-converted encoders.
- `checkpoints/phase3_prep/checkpoint_best.npz` — 64-token projection from prose objective (largest, ~700 MB; 768→64×3584).
- `checkpoints/phase3_prep/qa_full/checkpoint_best.npz` — QA-objective 20 k × 3 ep (val_loss 0.7398).
- `checkpoints/phase3_prep/qa_arch_v2_pilot/checkpoint_best.npz` — per-pillar 5 k × 1 ep pilot (val_loss 1.18).
- `checkpoints/phase3_prep/qa_arch_v3_pilot/checkpoint_best.npz` — embedding-only **partial** pilot (step 800 / 2350, val_loss 0.9242, monotonically improving).

### Tests

53+ `tests/test_*.py`; pertinent ones for the current path:

- `test_natal_encoder.py`, `test_encoder_mlx_parity.py`, `test_dynamic_encoder.py`.
- `test_embedding_natal_mlx.py` (new, 4/4 passing) — shape contract, identity preservation in slot 3, sensitivity in slot 0 to interaction vector, `useful_god_id = -1` handled.
- `test_projection_layer.py`, `test_bazi_llm.py`, `test_input_target_api.py`.
- `test_phase2_dataset.py`, `test_stage1_qa_dataset.py`, `test_phase2_loss.py`, `test_qa_templates.py`.

### Plans

- `plans/phase1_implementation_plan.md`, `plans/phase_2_data_curation_44602ed5.plan.md`, `plans/task_0.8_implementation_plan.md`.
- (User-level, in `~/.cursor/plans/`) `projection_qa_retrain_88f7bf91.plan.md`, `per-pillar-token-projection_a1391d32.plan.md`, `embedding-natal-arch-v3_e240dad3.plan.md`, `virtual_token_expansion_3fc8ea5b.plan.md`, `phase_2.2_projection_and_bazillm_*.plan.md`, ~12 more bazi-related.

### Docs (`docs/`)

- `design_decisions.md` — single source of truth for architectural / loss / API choices (22 sections, with §22.v1.2 covering the embedding head rationale).
- `phase1_best_checkpoint.md`, `phase1_final_architecture.md`, `phase1_training_strategy_clarification.md` — Phase 1 reference.
- `phase2_implementation.md`, `phase2_model_and_env.md`, `phase2_task_2_4_plan.md` — Phase 2 reference.
- `pre_phase3_preparation_plan.docx` (binary) — Phase 3 prep workstream plan.
- `binary_vector_and_positions_spec.md` — 64-d interaction-vector layout spec (do not change without versioning).
- `tests_and_interactions_summary.md` — test coverage and interaction blocking rules.
- `code_review_issues_and_simplifications.md` — open Fix/Simplify items (cross-context `"punishments": []`, DaYun `index` in JSON, `BaziChart.pillars` property, splitting `to_feature_arrays`).
- `agent_optimizations.md`, `tasks_audit.md`, `interactions_audit.md`, `solar_term_audit.md`, `lunar_python_audit.md`, `interaction_refinements_assessment.md`, `phase1_extra_experiments.md`, `phase1_integration_checklist.md`, `phase2_data_detailed_plan_FINAL.docx`, `phase2_detailed_plan.docx`, `astrodatabank_api_notes.md`, `fix_data_leakage.md`, `scripts_guide.md`, `strength_explanation.md`, `training_process_explained.md`.

---

## 8. Findings — what we know is true

(Numbered for reference in §10.)

- **F1.** The bazi engine is correct (production-quality test coverage; 53+ test files; rules for blocking and transformation in `docs/tests_and_interactions_summary.md`).
- **F2.** Phase 1 multi-task encoder hits all natal targets except element ordinal (81 %, capped by label noise). `multi_task_3layer_1_6_1_1.pt` is the production encoder.
- **F3.** Each per-pillar encoder token preserves its pillar's stem and branch identity at **100 %** test accuracy under a linear probe (§4.2).
- **F4.** `cross_features` is **all-zero** for natal-only Stage 1 QA training. The natal-internal 64-d `interaction_vector` exists in `to_input_features` but is not plumbed to the model in v1.0 / v1.1 (§4.3).
- **F5.** Phase 2 v2 prose training improved language metrics (val loss, perplexity, BLEU, ROUGE) but **regressed factual consistency** (44 % → 28 %).
- **F6.** Phase 3 prep v1.0 (QA objective, single Linear projection, 64 virtual tokens): pilot 47 % cat → full 26 % cat. **More training and a 64-token projection together did not solve the per-pillar / ten-god problem.**
- **F7.** Phase 3 prep v1.1 (QA + per-pillar 7-slot projection): pilot 21 % cat with `pillar_identity = 0/4` and `ten_gods = 0/4`. The per-pillar projection is necessary but **not sufficient**: the cross slot was still feeding zero, and the per-pillar tokens were going through a frozen transformer that the linear-probe shows is at best a no-op.
- **F8.** The probe categories partition cleanly by *answer geometry*: questions whose answer is one chart fact (pillar_identity, ten_gods) failed in every run; questions whose answer is concentrated in the binary interaction vector (interactions) varied 25 %–75 %.
- **F9.** Loss-mask geometry matters. Long-form prose CE smears gradient over hundreds of tokens; short factual answers concentrate gradient. This is documented in `tasks/lessons.md` and §22 of `design_decisions.md`.
- **F10.** `ProjectionLayer` checkpoint format is **shape-coupled** to projection architecture. Old 8-token checkpoints, prose 64-token aggregate Linear checkpoints, and per-pillar 63-token slot Linear checkpoints are mutually incompatible. Warm-starting across these is silently a no-op (random init).
- **F11.** Apple Silicon RAM pressure causes severe per-step slowdown and run-fragility on the embedding pilot. Single-machine assumption may not be tenable for 3-epoch / 20 k runs.
- **F12.** `--resume` (full optimizer-state restore) has been essential on the prose training script for multi-day pause/resume; the QA training script does **not** have it yet — `--init_mode warm` only restores projection weights.

---

## 9. Open questions and unknowns

- **Q1.** Does the `EmbeddingNatalMLX` architecture (v1.2 / v3) actually move pillar_identity and ten_gods off the 0 % floor? Probing the partial v3 best checkpoint will give a lower-bound answer. Probing a fully-trained v3 (3 ep) is the gate the plan is currently written against.
- **Q2.** Does the QA stratified sampler actually realise its 25/25/15/15/10/10 distribution in the training mix? `qa_full` regressed interactions 75 % → 50 % despite the 10 % weight; the §22.v1.1 follow-ups list "verify Stage 1 answers are being faithfully passed through `phase2_collate`" and "check that the stratified weights are actually realized in the training mix" as both unverified.
- **Q3.** Is the `pillar_identity = 0 %` a *projection* failure, an *encoder slot pickup* failure, or a *frozen-LLM cannot be conditioned this way* failure? The linear probe (§4.2) and v3 architecture decision narrow it to (b)/(c). Probing v3 distinguishes (b) from (c).
- **Q4.** Where is the actual ceiling on a frozen 7B Qwen with virtual tokens for short factual recall? `tasks/phase2_v2_handoff.md` claims projection-only "has hit its ceiling"; v3 is the last cheap experiment that can falsify that. After v3, LoRA / partial unfreeze becomes the natural next axis.
- **Q5.** What is the failure mode of the *numeric* probe questions (day-master percentile, element percentages)? All runs are 0 % numeric. Likely tokenizer/decoding plus genuine regression difficulty under a frozen LM. Marked report-only and not gated on, but if categorical works the next probe extension is to confirm numeric isn't structurally impossible.
- **Q6.** How will the temporal pipeline (大运 / 流年 / 流月 in slot 5) interact with the v1.2 embedding-only natal head? The plan claims slots 5 and 6 keep their semantics so temporal training drops in cleanly, but this has not been tested under v1.2.
- **Q7.** Open Fix/Simplify items in `docs/code_review_issues_and_simplifications.md` (§3.3 etc.) — cross-context `"punishments": []`, DaYun `index` in JSON, `BaziChart.pillars` property, splitting `to_feature_arrays`. None of these block training, but they accumulate engine-side debt.

---

## 10. Candidate next steps (ranked)

Each option lists **what it answers**, **rough cost**, and **risk if it fails**. Pick a path; do not run multiple in parallel without a separate compute budget.

### Option A — Probe the partial v3 pilot **first** (recommended next action)

- **What it answers:** Does `EmbeddingNatalMLX` move `pillar_identity` / `ten_gods` off the 0 % floor at all?
- **Cost:** ~5 min wall-clock. No new training.
- **How:** `python scripts/probe_factual_64token.py --natal_mode embedding --projection_checkpoint checkpoints/phase3_prep/qa_arch_v3_pilot/checkpoint_best.npz --output_json reports/probe_qa_arch_v3_pilot_partial.json`.
- **Decision tree:**
  - If categorical ≥ 30 % **and** pillar_identity ≥ 25 % **and** ten_gods ≥ 25 % at this very partial training depth → architecture is working. Either resume to completion (B) or run a smaller fresh pilot (C) to confirm.
  - If pillar_identity = 0 % and ten_gods = 0 % even with the embedding architecture → strong evidence the projection-stack ceiling is real. Move to Option E or F.
  - In between → ambiguous, consider C.
- **Risk if it fails:** None. Worst case is "data point is inconclusive because training was incomplete".

### Option B — Resume v3 to completion on a less constrained machine

- **What it answers:** Whether v1.2 clears the v1.2 pilot gate (categorical ≥ 40 %, pillar_id > 30 %, ten_gods > 30 %).
- **Cost:** ~6–12 h on the same machine assuming RAM pressure is resolved (close other apps), or ~hours on a beefier rig. Need a `--resume` capability to avoid retraining from step 0.
- **Prerequisite:** add full optimizer-state resume to `train_projection_qa.py` (mirror what `train_phase2.py --resume` does).
- **Risk if it fails:** wasted training time. Mitigation: do A first to filter.

### Option C — Shrink the v3 pilot so the architecture decision is isolated from wall-clock

- **What it answers:** same as B but cheaper.
- **How:** rerun `train_projection_qa.py --natal_mode embedding --subset_size 5000 --epochs 1` (mirror the v1.1 per-pillar pilot config). 5 k × 1 ep at the v1.1 pace was ~25 min.
- **Caveat:** the v3 plan moved from 5 k → 20 k *because* v1.1's 2000-train / 3000-val split was a confound. Keep that in mind and optionally use `--qa_val_per_category 100` to keep more train.
- **Risk if it fails:** small; ~hour of compute.

### Option D — Investigate the §22.v1.1 follow-ups (data / sampler audit)

- **What it answers:** Is the stratified sampler realising its target distribution? Is the loss-mask covering exactly the answer span? Is interactions getting under-trained vs the 75 % prose baseline because of low weight?
- **Cost:** ~few hours of audit code + a small validation run.
- **Why it matters even if v3 works:** if interactions can recover to 75 %+ purely by fixing the sampler/mask, that is an independent ~50pp swing in the probe that compounds with whatever architecture wins.
- **Risk if it fails:** discover a silent bug; that's a feature, not a bug.

### Option E — LoRA / partial unfreeze on Qwen

- **What it answers:** Whether the projection-only ceiling claim is right. If A/B/C show that even v1.2 stalls, this is the next architectural axis.
- **Cost:** medium. Requires PEFT-compatible BaziLLM wrapper (out-of-scope flagged in v1.2 plan), LoRA hyperparameter tuning, and noticeably more memory than projection-only.
- **Risk if it fails:** large compute spend. Mitigation: do not start until v1.2 is settled (A → B/C); if v1.2 clears the gate, defer LoRA further.

### Option F — Hybrid prompt approach (inject chart facts as text)

- **What it answers:** Whether *direct text injection* of the per-pillar identity bypasses the projection-decoding ceiling. From the v2 handoff: "inject structured chart facts (ten-gods, interactions, elemental strengths) as explicit text in the prompt alongside virtual tokens".
- **Cost:** small to author the prompt builder; medium to refit the probe / training loops to support hybrid.
- **Risk if it fails:** not so much "fails" as "becomes the production path" — it solves the factual probe but doesn't progress the research goal of teaching a frozen LLM to *read* virtual tokens. Acceptable as a fallback for a usable product before LoRA matures.

### Option G — Stage 2 multi-fact QA / Astrodatabank Stage 3a

- **What it answers:** None of the current blockers. These are downstream workstreams (Stage 2 and Stage 3a in the Phase 3 prep plan) that depend on a working factual probe gate. **Defer until the gate is cleared.**

### Recommended sequence

1. **Now: Option A** (probe partial v3). Fast, free.
2. Branch on the result:
   - Success: **Option D** (audit sampler/mask) → **Option B/C** (full v3) → re-probe with acceptance criteria.
   - Failure (pillar_id and ten_gods both 0 %): **Option F** (hybrid prompt) for a near-term usable system, then **Option E** (LoRA) for the research goal.

---

## 11. Glossary / quick references

- **Pillars (柱):** four time-derived columns — Year, Month, Day, Hour (0..3 indexed). Each is one stem (天干, 10 options) and one branch (地支, 12 options).
- **Day stem (日主):** the stem of the Day pillar; the reference point for all ten-god relationships.
- **Ten gods (十神):** 10 relationships between any other stem and the day-master, derived from element + polarity. Plus 11th 日主 / 比肩 self-label.
- **Useful god (用神):** the corrective element/category for the chart's strength imbalance; ambiguous in many charts (mapped to class 5 / -1, see §5 of `design_decisions.md`).
- **Sect (流派):** divination school. Default = sect 2 (盲派) for day-master logic.
- **大运 / 流年 / 流月:** decade / year / month overlays in the temporal pipeline.
- **64-d interaction vector:** fixed-size binary representation of natal interactions (合冲刑害破), spec in `docs/binary_vector_and_positions_spec.md`. **Versioned**; do not reorder.
- **Probe gate:** the 20-question categorical / numeric probe in `scripts/probe_factual_64token.py`. Acceptance criteria for "Workstream A done": categorical ≥ 55 %, per-category floors `pillar_identity / ten_gods / five_elements ≥ 50 %`, `interactions ≥ 60 %`. Reported buckets are `categorical` (gated) and `numeric` (report-only).

---

*If next-step decisions change after this review, append a dated revision below; do not silently rewrite the body. The body is the standing record of "what we tried and what we know"; the revision log is "how we changed our minds".*

## Revision log

- 2026-04-29 — Initial centralisation. Compiled by an agent across `docs/`, `plans/`, `tasks/`, `reports/`, `logs/`, and recent agent transcripts. Pilot v3 paused at step 950; partial probe not yet run.
