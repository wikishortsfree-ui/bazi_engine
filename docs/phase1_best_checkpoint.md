# Phase 1 Best Checkpoint: Process and Final Results

This document records how we arrived at the Phase 1 “best” checkpoint and the data backing the final choice. It is the single source of truth for **what to use for Phase 2** and **how we got there**.

**Date:** 2026-03-07  
**Recommended checkpoint:** `checkpoints/multi_task_3layer_1_6_1_1.pt` — 3 layers, default d_model=256, task_weights 1,6,1,1 (not d_model=512).

---

## 1. Objective and success criteria

Phase 1 trains a shared **NatalEncoder** with four task heads (ten-god, element strength, interaction, useful god). Success is defined by these targets (from `plans/phase1_implementation_plan.md` and `docs/phase1_training_strategy_clarification.md`):

| Task            | Metric(s)                    | Target        |
|-----------------|------------------------------|---------------|
| Ten-god         | Accuracy                     | >95%          |
| Element strength| MSE, ordinal accuracy        | MSE <0.05, ordinal >90% |
| Interaction     | F1 (common / rare bits)      | >95% / >80%   |
| Useful god      | Accuracy (unambiguous)       | >70%          |
| Cross-temporal  | F1                            | >90%          |

Cross-temporal is trained separately (`train_cross_temporal.py`); the “best” Phase 1 checkpoint refers to the **natal multi-task** model only.

---

## 2. Process: what we tried

### 2.1 Sequential vs multi-task (decided early)

- **Sequential:** One encoder per task (`train_ten_god.py`, `train_element_strength.py`, etc.). Result: ten-god 89%, useful-god **23%** (class collapse). Not usable as a single encoder for Phase 2.
- **Multi-task:** One shared encoder, four heads, combined loss. Result: ten-god **100%**, useful-god **99.93%**, interaction and element much better.
- **Decision:** Multi-task only for production. Sequential kept only for debugging. See `docs/phase1_training_strategy_clarification.md`.

### 2.2 Baseline multi-task (default config)

- **Config:** 2 layers, `d_model=256`, `task_weights 1,1,1,1`, no ordinal auxiliary loss, save by validation loss.
- **Result:** Ten-god, useful-god, and interaction met or exceeded targets. **Element ordinal was the bottleneck** (~63–72% on test, target >90%). MSE was always well below 0.05.

### 2.3 Hyperparameter tuning (`tune_phase1.py`)

- **Method:** Short-horizon trials (5 epochs per trial) over a grid of `lr` × element task weight; validation metric recorded; optional full 30-epoch run with best config.
- **Search space:** `lr ∈ {0.0005, 0.001, 0.002}`, element weight ∈ {1, 2, 3, 4} (task_weights `1,{w},1,1`).
- **Result (from `reports/tune_phase1_results.json`):** Best **validation** metric at trial 3: `lr=0.0005`, `task_weights 1,4,1,1`, best_val_metric **0.9178**. Higher lr (0.002) hurt. So we fixed **lr=0.0005** and pushed element weight **above 4** in later runs.

### 2.4 Task weighting and “save by composite”

- With unequal task weights, saving the checkpoint with **lowest validation loss** can over-represent the high-weight task. We switched to **`--save_by composite`** so the saved checkpoint is the one with **highest** mean of (ten_god_acc, element ordinal, interaction acc, useful_god_acc). See `docs/design_decisions.md` §15.

### 2.5 Ordinal auxiliary loss

- Element strength is evaluated with **ordinal accuracy** (correct ranking of the five elements). MSE alone does not directly optimize ranking.
- We added a **pairwise ranking auxiliary loss** in `ElementStrengthHead` and exposed **`--ordinal_loss_weight`** (default **0.15**). Total element loss = MSE + 0.15 × pairwise_ranking_loss. See `docs/design_decisions.md` §12.

### 2.6 Extra experiments (capacity and element weight)

From `docs/phase1_extra_experiments.md` we ran:

1. **Higher element weight:** `task_weights 1,5,1,1` and **1,6,1,1** to stress element/ordinal.
2. **More capacity:**  
   - **3 layers** (same `d_model=256`), and  
   - **d_model=512** (2 layers then 3 layers with 512).

Combined best: **3 layers**, **default d_model=256**, **task_weights 1,6,1,1**, **ordinal_loss_weight=0.15**, **save_by composite**. (We also tried d_model=512; the conventionally named best is 3-layer 1,6,1,1 without d512.)

---

## 3. Final configuration (recommended)

The recommended Phase 1 training command and checkpoint are:

**Training command:**

```bash
python scripts/train_multi_task.py \
  --data data/charts.jsonl \
  --epochs 30 --batch_size 64 \
  --lr 0.0005 --task_weights 1,6,1,1 \
  --num_layers 3 \
  --val_ratio 0.1 --test_ratio 0.1 --seed 42 \
  --checkpoint_dir checkpoints --log_dir runs/multi_task \
  --save_by composite --ordinal_loss_weight 0.15 \
  --save_name multi_task_3layer_1_6_1_1.pt
```

**Checkpoint file:** `checkpoints/multi_task_3layer_1_6_1_1.pt`

Config: **3 layers**, **default d_model=256** (no `--d_model 512`). This run has the **highest ordinal and lowest MSE** in our reports.

**Key hyperparameters:**

| Parameter               | Value    | Note                                      |
|------------------------|----------|-------------------------------------------|
| `lr`                   | 0.0005   | From tune_phase1; 0.001/0.002 worse       |
| `task_weights`         | 1,6,1,1  | Element weight 6 for ordinal focus        |
| `num_layers`           | 3        | Extra capacity vs baseline 2             |
| `d_model`              | 256      | Default; best is 3-layer 1,6,1,1 not d512 |
| `ordinal_loss_weight`  | 0.15     | Pairwise ranking auxiliary               |
| `save_by`              | composite| Best checkpoint by balanced metrics     |
| `batch_size`           | 64       | Stable gradients                         |
| `epochs`               | 30       | With early stopping (patience 5)          |

---

## 4. Final results (backed by data)

All metrics below are on the **test** split (val_ratio=0.1, test_ratio=0.1, seed=42). Source: `scripts/evaluate_phase1.py` outputs in `reports/`.

### 4.1 Comparison across checkpoints

| Checkpoint / run        | Ten-god acc | Element MSE | Element ordinal | Interaction F1 (common/rare) | Useful-god acc (unamb.) |
|-------------------------|-------------|-------------|------------------|------------------------------|--------------------------|
| multi_task (baseline)   | 100%        | 0.00164    | **72.06%**       | 99.50% / 99.98%              | 99.90%                   |
| multi_task_after_tune   | 100%        | 0.00163    | **73.60%**       | 99.66% / 99.98%              | 99.88%                   |
| 1_5_1_1                 | 100%        | 0.00152    | **77.11%**       | 99.66% / 100%                | 99.94%                   |
| 1_6_1_1 (2-layer)       | 100%        | 0.00151    | **77.70%**       | 99.73% / 99.98%              | 99.91%                   |
| **3layer_1_6_1_1 (best)** | 100%     | **0.00148**| **81.45%**       | 99.99% / 100%                | 99.94%                   |
| 3layer_d512 (optional)   | 100%        | 0.00166    | 80.95%       | 99.99% / 100%                | 99.93%                   |

- **Best (conventional):** 3-layer, task_weights 1,6,1,1, default d_model=256 (`phase1_metrics_3layer_1_6_1_1.json`): ordinal **81.45%**, MSE **0.00148**.
- **Recommended for Phase 2:** `multi_task_3layer_1_6_1_1.pt`. The 3-layer d512 run was tried but the named best is 3-layer 1,6,1,1 (no d512).

### 4.2 Target vs achieved (best run)

| Task             | Target       | Best achieved (3-layer 1,6,1,1) | Met? |
|------------------|-------------|--------------------------------------|------|
| Ten-god          | >95%        | **100%**                             | Yes  |
| Element MSE      | <0.05       | **0.00148**                          | Yes  |
| Element ordinal  | >90%        | **81.45%**                           | No*  |
| Interaction F1   | >95% / >80% | **99.99% / 100%**                    | Yes  |
| Useful god       | >70%        | **99.94%**                           | Yes  |

\* Ordinal remains below 90%. It was the main focus of tuning; 81.45% is the best we have in the reports. Further gains might require more data, different ordinal loss, or curriculum; for Phase 2 we proceed with this checkpoint.

### 4.3 Exact numbers from report files

- **`reports/phase1_metrics_3layer_1_6_1_1.json`** (best ordinal/MSE):  
  - `element_strength.mse`: 0.00148, `ordinal_accuracy`: 0.8145  
  - `ten_god.accuracy`: 1.0, `useful_god.accuracy_unambiguous`: 0.9994  
  - `interaction.index.f1_common_bits`: 0.99994, `f1_rare_bits`: 1.0  

- **`reports/phase1_metrics_3layer_d512.json`**:  
  - `element_strength.mse`: 0.00166, `ordinal_accuracy`: 0.8095  
  - Same ten-god/useful-god/interaction level as above.

- **`reports/tune_phase1_results.json`**:  
  - Best trial: 3, lr 0.0005, task_weights "1,4.0,1,1", best_val_metric 0.9178.

---

## 5. How to reproduce

1. **Train the best checkpoint (3-layer, 1,6,1,1, default d_model=256):**
   ```bash
   python scripts/train_multi_task.py \
     --data data/charts.jsonl \
     --epochs 30 --batch_size 64 --lr 0.0005 \
     --task_weights 1,6,1,1 --num_layers 3 \
     --val_ratio 0.1 --test_ratio 0.1 --seed 42 \
     --checkpoint_dir checkpoints --log_dir runs/multi_task \
     --save_by composite --ordinal_loss_weight 0.15 \
     --save_name multi_task_3layer_1_6_1_1.pt
   ```

2. **Evaluate and write metrics:**
   ```bash
   python scripts/evaluate_phase1.py --data data/charts.jsonl --use_multi_task \
     --multi_task_checkpoint checkpoints/multi_task_3layer_1_6_1_1.pt \
     --val_ratio 0.1 --test_ratio 0.1 --seed 42 \
     --output reports/phase1_metrics_3layer_1_6_1_1.json
   ```

3. **Use in Phase 2:** Load `checkpoint['encoder']` from `checkpoints/multi_task_3layer_1_6_1_1.pt` for the projection-layer / LLM integration. See `docs/phase1_training_strategy_clarification.md` § Phase Transition.

---

## 6. Element strength: why it’s harder and optional next steps

### 6.1 Does the training setup make it harder?

Partly. The setup has a few inherent tensions:

- **MSE vs ordinal:** We use MSE (match exact strengths) + pairwise ranking (match order). MSE gradients can conflict with ordinal: e.g. pushing one element up to fix magnitude can flip the order of another pair. So the objective is not purely aligned with the metric we care about (ordinal accuracy).
- **Multi-task:** The encoder is shared; ten-god, interaction, and useful-god reach 99%+ and may dominate the representation. We already counter that with higher element task weight (1,6,1,1) and ordinal auxiliary loss.
- **Regression sensitivity:** Ordinal is binary (right order vs wrong) but we train on continuous values. Small prediction errors can flip order when two elements have similar true strength, so the task is inherently sensitive.

So the setup is a plausible contributor, but it’s not the only one.

### 6.2 Label ambiguity: ordinal error analysis

We ran `scripts/analyze_ordinal_errors.py` on the baseline checkpoint (`reports/ordinal_error_analysis.json` and `ordinal_error_analysis_1_5_1_1.json`). Results:

- **Baseline (multi_task_best.pt):** 72.06% ordinal correct. Of the **wrong** samples, **80%** are “borderline”: the two elements whose order we got wrong have **true strength difference &lt; 0.02**.
- **1_5_1_1:** 77.11% correct. Of wrong samples, **87%** borderline (true diff &lt; 0.02).

So most ordinal errors are cases where the “correct” order is barely determined by the engine (e.g. 0.201 vs 0.199). That suggests a **ceiling from label ambiguity**: the 90% ordinal target may be unreachable without either redefining “correct” (e.g. ignore pairs with true margin &lt; 0.02) or changing how we generate labels.

**Why ordinal doesn’t get higher even with pure element training:** (1) **Same label ceiling:** The engine’s formula (stems + 藏干 weights, then normalize) often produces two elements with almost identical strength. For those pairs the “correct” order is effectively arbitrary; the model is asked to match noise. (2) **Single task = single signal:** In element-only training the encoder only gets gradients from the element head. It still has to learn the full mapping (stems, branches) → 5 strengths from that one loss. Multi-task can do better because the other tasks (ten-god, interaction, useful-god) give the encoder a richer representation, which the element head then uses. (3) **Loss doesn’t ignore ambiguous pairs:** We penalize every wrong-ordered pair equally. Pairs with true strength diff &lt; 0.02 are hard to get “right” and can confuse training; margin-weighted ordinal loss (downweight or skip such pairs) would focus the model on clear orderings. So the ceiling is partly in the **labels** (many near-ties) and partly in the **task setup** (one head, same loss for all pairs); pure element training doesn’t remove either.

### 6.3 Have we tried enough?

We have tried: task weights (1,2–1,6,1,1), ordinal auxiliary loss (0.15), save-by-composite, more capacity (3 layers), and tuning lr. That’s a solid set of levers. The combination (3-layer, 1,6,1,1, ordinal loss) gives **81.45%** ordinal and **0.00148** MSE.

**Recommendation:** For Phase 2, **we’ve tried enough** to ship: the best checkpoint is strong on the other three tasks and best-in-class on element. If ordinal becomes critical later, the next levers would be:

1. **Run ordinal analysis on the best checkpoint** (3layer_1_6_1_1) to confirm the same borderline-vs-clear split.
2. **Margin-weighted ordinal loss (implemented):** Use **`--ordinal_margin_min 0.02`** so the pairwise loss only penalizes pairs with true strength diff >= 0.02. Train multi-task or element-only with this flag; eval scripts now report **ordinal_accuracy_relaxed** (correct on clear pairs only) alongside strict ordinal accuracy.
3. **Ordinal-only or listwise loss:** Replace or heavily downweight MSE and train mainly for ranking (e.g. listnet/listwise or stronger pairwise). Risk: MSE might get worse; only do this if ordinal is the only thing that matters for element.
4. **Pretrain-then-multitask:** Train element-only first (`train_element_strength.py`), then load that encoder into multi-task (`--encoder_checkpoint`). Gives the encoder a strong element signal before other tasks. **Commands:** see §6.4 below.

**Commands to try for higher ordinal accuracy:**

- **Margin-weighted loss:** Train with `--ordinal_margin_min 0.02` so the ordinal loss only penalizes pairs with true strength diff >= 0.02. Eval reports **ordinal_accuracy_relaxed** (clear pairs only).
  ```bash
  python scripts/train_multi_task.py --data data/charts.jsonl --epochs 30 --batch_size 64 --lr 0.0005 \
    --task_weights 1,6,1,1 --num_layers 3 --ordinal_loss_weight 0.15 --ordinal_margin_min 0.02 \
    --val_ratio 0.1 --test_ratio 0.1 --seed 42 --checkpoint_dir checkpoints --save_by composite \
    --save_name multi_task_ordinal_margin.pt
  ```
- **Stronger ordinal weight:** Try `--ordinal_loss_weight 0.25` (with or without `--ordinal_margin_min 0.02`).

No need to block Phase 2 on these; document them as optional follow-ups if ordinal is revisited.

### 6.4 Pretrain-then-multitask: exact commands

Encoder must match: the element script uses **2-layer** `NatalEncoder` (default). Multi-task must use **2 layers** when loading that checkpoint (omit `--num_layers 3`), or the state_dict keys won’t match.

**Step 1 — Train element-only (encoder + element head):**

Use **lr 1e-3** (not 0.0005) and the script’s **CosineAnnealingLR** and **encoder_lr_ratio** so loss doesn’t stall at ~0.09. Defaults: lr=1e-3, encoder_lr_ratio=2.0, scheduler=CosineAnnealingLR.

```bash
python scripts/train_element_strength.py \
  --data data/charts.jsonl \
  --epochs 20 --batch_size 64 --lr 0.001 \
  --val_ratio 0.1 --test_ratio 0.1 --seed 42 \
  --checkpoint_dir checkpoints \
  --log_dir runs/element_strength
```

This writes `checkpoints/element_strength_best.pt` (and `checkpoints/best.pt` during training). If loss still plateaus early, try `--encoder_lr_ratio 3.0` or `--lr 0.002`.

**Step 2 — Multi-task from that encoder (2-layer, same as pretrain):**

```bash
python scripts/train_multi_task.py \
  --data data/charts.jsonl \
  --encoder_checkpoint checkpoints/element_strength_best.pt \
  --epochs 30 --batch_size 64 --lr 0.0005 \
  --task_weights 1,6,1,1 \
  --val_ratio 0.1 --test_ratio 0.1 --seed 42 \
  --checkpoint_dir checkpoints --log_dir runs/multi_task \
  --save_by composite --ordinal_loss_weight 0.15 \
  --save_name multi_task_after_element_pretrain.pt
```

Do **not** pass `--num_layers 3` here (encoder is 2-layer from step 1). Optionally use `--no_load_element_head` to init the element head from scratch in step 2.

**Step 3 — Evaluate:**

```bash
python scripts/evaluate_phase1.py --data data/charts.jsonl --use_multi_task \
  --multi_task_checkpoint checkpoints/multi_task_after_element_pretrain.pt \
  --val_ratio 0.1 --test_ratio 0.1 --seed 42 \
  --output reports/phase1_metrics_after_element_pretrain.json
```

To use a **3-layer** encoder with pretrain-then-multitask, `train_element_strength.py` would need a `--num_layers` argument so step 1 produces a 3-layer checkpoint; currently it only builds a 2-layer encoder.

**Why pretraining loss was flat (~0.09 from epoch 1):** (1) **0.09 is a deceptively low baseline:** with random init the head outputs near-uniform probabilities; MSE(uniform, target) is small because both sum to 1 and L2 error is low, and the ordinal term is moderate. So the loss starts low and has little visible headroom—the metric that matters is **ordinal accuracy**, not raw loss. (2) With a single task, the encoder gets weak gradients; low LR + no scheduler made the encoder barely move. Fixes: higher LR (1e-3), CosineAnnealingLR, encoder LR = head LR × 2; and **log val ordinal_accuracy** during element training so you see real progress (the script now logs it).

---

## 7. References

- **Strategy and multi-task vs sequential:** `docs/phase1_training_strategy_clarification.md`
- **Tuning and parameters:** `docs/phase1_training_strategy_clarification.md` § “Are these parameters optimal?”
- **Extra experiments (weights, capacity):** `docs/phase1_extra_experiments.md`
- **Save by composite, ordinal loss:** `docs/design_decisions.md` §12, §15
- **Evaluation script and metrics:** `scripts/evaluate_phase1.py`, `reports/phase1_metrics_*.json`
- **Tuning script:** `scripts/tune_phase1.py`, `reports/tune_phase1_results.json`
- **Ordinal error analysis:** `scripts/analyze_ordinal_errors.py`, `reports/ordinal_error_analysis.json`
