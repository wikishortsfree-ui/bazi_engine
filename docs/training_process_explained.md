# How Our Training Works — Explained Simply

This report explains the whole training process: what we put in, what the model gives back, and what we check the answers against. Think of it like learning with flashcards and a teacher's answer key.

---

## The Big Picture

1. **We create lots of "birth charts"** (like making many practice problems).
2. **A rule-based engine solves each chart** and writes down the "correct" answers (ten-god, interactions, strengths, useful god, etc.). Those are our **ground truth** — the answer key.
3. **We feed minimal chart structure into a neural network** (the **input**) — only raw stems, branches, elements, and polarities. The model never sees the answers in its input.
4. **The network tries to guess those same answers** (the **output**).
5. **We compare the network's guesses to the engine's answers** and adjust the network so next time it guesses better. That's **validation**: we're checking against the ground truth.

So: **input** = minimal chart structure we show the model. **Output** = what the model predicts. **Ground truth** = what the engine says is correct; we train and evaluate by matching output to ground truth.

---

## Where Does the Data Come From?

- We **don't** use real people's birth data or human-labeled answers.
- We use **synthetic charts**: random birth date/time and location → `generate_training_charts.py` → `dates.csv` → `compute_charts.py` → `charts.jsonl`.
- For each chart, the **bazi_engine** (rules and formulas) computes everything: stems, branches, ten-god, interactions, strengths, useful god, and (if temporal context is present) 大运/流年/流月 and cross-temporal interactions.
- Those engine results are serialized and saved in **charts.jsonl**. When we train, we read that file. So **all "correct" answers come from the engine**; we're teaching the network to imitate the engine.

---

## No Data Leakage: Inputs vs Targets

A critical design decision: **the model's input never contains the answers it's trying to predict.** This prevents the model from simply copying inputs to outputs instead of learning the underlying patterns.

The separation is enforced at the data layer:

- **`BaziChart.to_input_features(task_name)`** returns only minimal inputs the encoder must learn from (raw chart structure — stems, branches, elements, polarities, positions).
- **`BaziChart.to_ground_truth(task_name)`** returns only the task's target labels/values.
- **`ChartDataset`** takes a `task_name` and returns `(inputs_dict, targets_dict)` for each sample — inputs and targets are always separate dicts with **no overlapping keys**.
- **Encoders** only accept the minimal input keys. Ground-truth features (ten_god_ids, interaction_vector, elemental_strengths, useful_god_id, dynamic_interaction_vector) are **never** fed to any encoder.
- **Training scripts** pass only `inputs` to the encoder(s) and compute loss from `targets`.

---

## One Chart = One Row of Numbers

For every chart we have:

- **Natal part**: the four pillars (year, month, day, hour), each with a stem and branch, plus everything the engine derives from them (ten-god, interactions, elemental strengths, day-master strength, useful god).
- **Optional "dynamic" part**: 大运, 流年, 流月 and cross-interactions (only for charts where temporal context was computed).

The engine writes all of this into a single chart. We then split that chart into **input features** and **ground truth** via `to_input_features(task_name)` and `to_ground_truth(task_name)`. So the same chart gives us both the **input** and the **ground truth**, but they are always disjoint.

---

## Input: What We Feed Into the Model

The model never sees the raw chart text or the engine's answers. It only sees **minimal structural numbers** that describe the chart.

### Natal input (used by all natal tasks)

| What we feed (name) | In simple terms | Shape (per chart) |
|---------------------|------------------|--------------------|
| `stem_ids` | Which of the 10 stems is in each pillar (year/month/day/hour) | 4 numbers |
| `branch_ids` | Which of the 12 branches is in each pillar | 4 numbers |
| `stem_elements` | Element of each stem (Wood, Fire, Earth, Metal, Water) | 4 numbers |
| `stem_polarities` | Yin or Yang of each stem | 4 numbers |
| `pillar_positions` | Which pillar is which (0=year, 1=month, 2=day, 3=hour) | 4 numbers |

That's it — **only 5 feature arrays** (20 numbers total). No ten-god labels, no interaction vector, no elemental strengths, no useful god. The encoder must learn to derive all of that from raw chart structure alone.

### Extra input for the "cross-temporal" task only

When we train the model to predict **dynamic** (time-based) interactions, we also feed:

| What we feed | In simple terms | Shape (per chart) |
|--------------|------------------|--------------------|
| `dayun_stem_id`, `dayun_branch_id` | 大运 pillar | 1 stem + 1 branch |
| `liunian_stem_id`, `liunian_branch_id` | 流年 | 1 stem + 1 branch |
| `liuyue_stem_id`, `liuyue_branch_id` | 流月 | 1 stem + 1 branch |
| `pillar_presence_mask` | Which of 大运/流年/流月 we actually have | 3 numbers (0 or 1) |

So for cross-temporal, **input** = natal features (5 arrays) **plus** these dynamic pillar IDs and presence mask. No dynamic_interaction_vector is fed as input.

---

## Encoder Architecture

### NatalEncoder

The NatalEncoder turns the 5 minimal input arrays into a single **natal_emb** vector of size 256. It works like this:

1. Each pillar's (stem, branch, element, polarity, position) is embedded via `PillarEmbedding` into a dense vector. Ten-god is always padded to -1 (never seen).
2. The 4 pillar vectors are projected to d_model (256).
3. A learned **CLS token** is prepended and a learned **aux token** is appended, giving a sequence of 6 tokens: `[CLS, P0, P1, P2, P3, aux]`.
4. This sequence goes through a 2-layer transformer encoder.
5. The **CLS token output** is the natal_emb (batch, 256) — the representation used by all task heads.

### DynamicEncoder

The DynamicEncoder turns temporal pillar IDs into a single **dynamic_emb** vector of size 256. It works like this:

1. Each temporal slot (大运, 流年, 流月) is embedded via stem + branch embeddings. Absent slots (ID = -1) are replaced with a learned **null_dynamic_emb** vector.
2. The 3 slot vectors are projected to d_model (256).
3. A learned **aux token** is appended, giving a sequence of 4 tokens: `[slot0, slot1, slot2, aux]`.
4. This sequence goes through a 2-layer transformer encoder.
5. All 4 token outputs are concatenated and projected down to d_model (256) — the dynamic_emb.

---

## Output: What the Model Predicts

The model has a shared encoder that turns the input into an internal representation (embedding), then **one "head" per task** that predicts one kind of answer. So:

- **Output** = whatever that head predicts (one type of answer per task).

| Task | What the model outputs | Shape (per chart) |
|------|------------------------|--------------------|
| **Ten-god** | For each of the 4 pillars, which of 11 ten-god classes (logits) | 4 × 11 numbers |
| **Element strength** | 5 numbers (probabilities) for the 5 elements, summing to 1 | 5 numbers |
| **Interaction** | 64 numbers (logits) → after sigmoid: chance each interaction type is present | 64 numbers |
| **Useful god** | 6 numbers (logits) → after argmax: one of 6 classes (5 用神 types + "ambiguous") | 6 numbers → 1 class |
| **Cross-temporal** | 66 numbers: 61 binary logits + 5 continuous predictions | 66 numbers |

So **output** = the model's guess for that task (ten-god labels, element strengths, interaction bits, useful god class, or dynamic interaction predictions).

---

## Ground Truth: What We Validate Against

**Ground truth** = the "answer key." We don't have human labels; we use **exactly what the engine computed** for that chart when we built the dataset. The `to_ground_truth(task_name)` method provides these target fields, completely separate from the input features.

| Task | Ground truth we validate against | Where it comes from |
|------|----------------------------------|----------------------|
| **Ten-god** | `ten_god_ids` — the true ten-god class (0–10) for each of the 4 pillars | Engine from stems/branches and day master |
| **Element strength** | `elemental_strengths` — the 5 strengths that sum to 1 | Engine: `compute_element_strengths(stems, branches)` |
| **Interaction** | `interaction_vector` — 64 bits (0 or 1) | Engine: `detect_all_interactions(...).to_binary_vector()` |
| **Useful god** | `useful_god_id` — 0–4 or 5 for ambiguous (engine's -1 → class 5) | Engine: `_compute_useful_god(...)` |
| **Cross-temporal** | `dynamic_interaction_vector` — 66-dim (61 binary + 5 continuous) | Engine: cross_interactions → 66-dim vector |

So **ground truth** = the engine's answer for that chart. We train by making the **output** match this **ground truth** (using the right loss for each task), and we **evaluate** by comparing predictions to the same ground truth.

---

## Loss Functions

Each task head uses a loss function matched to its target data type:

| Head | Output | Target | Loss |
|------|--------|--------|------|
| **TenGodHead** | (batch, 4, 11) logits | ten_god_ids (batch, 4), 0..10 | CrossEntropy |
| **ElementStrengthHead** | (batch, 5) probs (softmax) | elemental_strengths (batch, 5), sum 1 | MSE |
| **UsefulGodHead** | (batch, 6) logits | useful_god_id 0..4 or 5 | CrossEntropy |
| **InteractionHead** | (batch, 64) logits | interaction_vector 0/1 | BCEWithLogits (optional pos_weight) |
| **CrossTemporalHead** | (batch, 66) mixed | dynamic_interaction_vector (mixed) | Hybrid: BCE + MSE |

**Why hybrid loss for cross-temporal?** The 66-dim vector mixes binary flags (dims 0–59, 65) and continuous elemental-delta values (dims 60–64). BCE is correct for binary dims; MSE is correct for continuous dims. Applying BCE to unbounded continuous values causes the loss to diverge.

All heads follow a unified return convention: `forward_loss(...)` returns `(loss, prediction)` — loss first, then prediction tensor.

---

## How We Check If the Model Is Doing Well

- **Training:** We compute a **loss** (how wrong the output is vs ground truth), then backprop and update the model so the loss goes down.
- **Validation:** On charts we didn't use for updating, we compare **model output** to **ground truth** and report metrics:
  - **Ten-god / Useful god:** accuracy (right class vs engine's class).
  - **Element strength:** MAE (mean absolute error between predicted 5-dim vector and engine's 5-dim vector).
  - **Interaction:** binary accuracy (after thresholding logits at 0.5, how many of the 64 bits match the engine's 0/1).
  - **Cross-temporal:** binary accuracy on binary dims + delta MSE on continuous dims.

So in one sentence: **we validate by comparing the model's output to the engine's output (ground truth) on the same chart.**

---

## The Dataset and DataLoader

**`ChartDataset`** reads `charts.jsonl` and yields `(inputs_dict, targets_dict)` per sample:

- Takes a `task_name` (one of: `ten_god`, `element_strength`, `interaction`, `useful_god`, `cross_temporal`, `multi`).
- For each chart: calls `to_input_features(task_name)` for inputs and `to_ground_truth(task_name)` for targets.
- The `multi` task name combines all four natal task targets into one targets dict for multi-task training.
- When `temporal_only=True`, only charts with a `dynamic` section (and for cross_temporal, with `cross_interactions`) are included.
- **`chart_collate_fn_inputs_targets`** stacks `(inputs, targets)` into batched dicts for the DataLoader.

---

## The Trainer

The task-agnostic `Trainer` handles the train/val loop, device moves, early stopping, checkpointing, and TensorBoard logging. It works with any task through a **step function**:

- **`step_fn(batch, model, device) → (loss, metrics_dict | None)`**
- The trainer moves the batch to device (supports both single-dict and `(inputs, targets)` tuple) and sets `model.train()` / `model.eval()`.
- For training: the step_fn must zero_grad, forward, compute loss, backward, and optimizer.step().
- For validation: the step_fn must only forward and return loss + optional metrics.
- The optimizer is typically captured in the step_fn closure.
- Validation metrics are averaged over batches and logged to TensorBoard per epoch.
- Early stopping halts training when validation loss doesn't improve for N consecutive epochs.
- Best checkpoint is saved automatically when validation improves.

---

## Training Strategies

### Single-task (one script per task)

Each script trains one NatalEncoder + one head from scratch:

| Script | Encoder | Head | Dataset filter |
|--------|---------|------|----------------|
| `train_ten_god.py` | NatalEncoder | TenGodHead | All charts |
| `train_element_strength.py` | NatalEncoder | ElementStrengthHead | All charts |
| `train_interaction.py` | NatalEncoder | InteractionHead | All charts |
| `train_useful_god.py` | NatalEncoder | UsefulGodHead | All charts |
| `train_cross_temporal.py` | NatalEncoder + DynamicEncoder | CrossTemporalHead | temporal_only=True |

Use for maximum per-task performance and independent tuning.

### Multi-task (shared encoder)

`train_multi_task.py` trains one shared NatalEncoder with four heads (ten-god, element strength, interaction, useful god). Each step:
1. Runs the shared encoder once to get natal_emb.
2. Computes per-task losses from the four heads.
3. Sums losses (with optional weights) and backprops once.
4. Logs per-task loss breakdown to TensorBoard.

Supports `--task_weights` (comma-separated: ten_god,element,interaction,useful_god) for tuning relative importance.

Cross-temporal is **not** included in multi-task (it requires DynamicEncoder and a different dataset filter); train it separately.

---

## Summary Table (Input → Model → Output vs Ground Truth)

| Task | Input (what we feed) | Output (what model predicts) | Ground truth (what we compare to) |
|------|----------------------|------------------------------|-----------------------------------|
| **Ten-god** | 5 minimal natal features (stems, branches, elements, polarities, positions) | 4×11 logits → 4 ten-god classes | Engine's `ten_god_ids` (4 labels) |
| **Element strength** | Same 5 natal features | 5 probs (sum 1) | Engine's `elemental_strengths` (5 numbers) |
| **Interaction** | Same 5 natal features | 64 logits → 64 bits | Engine's `interaction_vector` (64 bits) |
| **Useful god** | Same 5 natal features | 6 logits → 1 class | Engine's `useful_god_id` (0–4 or 5) |
| **Cross-temporal** | 5 natal features + temporal slot IDs + presence mask | 66 numbers (61 binary logits + 5 continuous) | Engine's `dynamic_interaction_vector` (66-dim) |

---

## The Whole Pipeline in One Paragraph

We **generate many synthetic birth charts** and run them through the **bazi_engine** to get **ground truth** (ten-god, interactions, strengths, useful god, and optionally dynamic interactions). We split each chart into **minimal input features** (stem IDs, branch IDs, elements, polarities, positions) and **ground truth targets** (task-specific labels/values) — inputs and targets are always separate with no overlap, preventing data leakage. We **feed the input** into a neural network encoder (NatalEncoder for natal tasks; NatalEncoder + DynamicEncoder for cross-temporal); each **task head** produces an **output** (ten-god labels, element strengths, 64 or 66 interaction predictions, or useful god class). We **train** by minimizing the loss between **output** and **ground truth**, and **validate** by comparing **output** to **ground truth** on held-out charts. The encoder must learn everything from raw chart structure — it never sees the answers it's trying to predict.
