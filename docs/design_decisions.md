# Design Decisions

Record of important design choices in the bazi_engine and bazi_model codebase. Update this doc when making or revisiting architectural or loss/API decisions.

---

## 1. Elemental Strength Head: MSE vs KL divergence

**Context:** The element strength head (`bazi_model/tasks/element_regressor.py`) predicts a 5-d distribution over elements (木, 火, 土, 金, 水) from the natal embedding. Targets are rule-based elemental strengths from `compute_element_strengths(stems, branches)`, normalized to sum to 1.0.

**Decision:** Use **MSE** between predicted probabilities (softmax output) and target strengths:

```python
loss = nn.functional.mse_loss(pred_probs, elemental_strengths)
```

**Rationale:**

- **Zero targets:** Rule-based strengths can be exactly 0.0 for absent elements (e.g. no Metal → 金 = 0.0). MSE handles zeros without special handling. KL would require clamping `pred_probs` (e.g. `min=1e-8`) to avoid log(0) and careful handling when `target_k = 0`; gradients can explode when `pred_k` is small and `target_k > 0`.
- **Target semantics:** Targets are continuous normalized scores, not categorical labels or empirical counts. Treating the task as regression to a fixed vector makes MSE a natural fit; KL is more appropriate when the target is explicitly a reference distribution and we care about information-theoretic matching.
- **Stability and tuning:** MSE gives linear-in-error gradients and plays well in multi-task setups; KL can require scaling or clipping when combined with other losses.

**Alternative considered:** KL divergence, e.g. `kl_div(pred_probs.log(), target_strengths, reduction='batchmean')`. Reasonable if we want distribution-matching semantics and heavier penalty for putting mass where the target has none; then we must clamp predictions and optionally smooth targets (e.g. add ε and renormalize) to avoid zeros.

**References:** `bazi_engine/strength/elemental_strength.py` (targets); `bazi_model/tasks/element_regressor.py` (head and loss).

---

## 2. Binary vector and position encoding (64-dim layout)

**Context:** Natal interactions (合, 冲, 刑, 害, 破, etc.) are variable-length (e.g. multiple clashes, different punishment types). The model needs a fixed-size representation for training and inference.

**Decision:**

- **Positions:** All pillar indices are **0-indexed**: 0 = year, 1 = month, 2 = day, 3 = hour. Constants: `PILLAR_COUNT = 4`, `POSITION_LABELS = ("year", "month", "day", "hour")`. Stems and branches are always ordered `[Year, Month, Day, Hour]`.
- **64-dim vector:** Single source of truth is `docs/binary_vector_and_positions_spec.md`. The vector encodes **presence (and some flags) per interaction type/slot**; it does **not** encode which positions are involved or how many instances. Variable-length lists map to a fixed set of bits (e.g. one bit per clash pair type: present or not).

**Rationale:**

- **Stability:** Changing the 64-dim layout breaks any trained model that consumes it. Version and document any change in the spec.
- **Fixed size:** Enables dense batch training and consistent encoder outputs; position/count details remain in the raw `InteractionResult` when needed.

**References:** `docs/binary_vector_and_positions_spec.md`; `bazi_engine/_constants.py`; `bazi_engine/interactions`.

---

## 3. Task head loss choices (summary)

| Head | Output | Target | Loss |
|------|--------|--------|------|
| **TenGodHead** | (batch, 4, 11) logits | ten_god_stem_ids (batch, 4), 0..10 | **CE or Focal** + class weights + label smoothing (see §3.2) |
| **ElementStrengthHead** | (batch, 5) probs (softmax) | elemental_strengths (batch, 5), sum 1 | **MSE + ordinal** (see §1, §12) |
| **UsefulGodHead** | (batch, 6) logits | useful_god_id 0..4 or -1 → class 5 | CrossEntropy |
| **InteractionHead** | (batch, 64) logits | interaction_vector 0/1 | BCEWithLogits (optional pos_weight) |
| **CrossTemporalHead** | (batch, 66) mixed | dynamic_interaction_vector (mixed) | **Hybrid: BCE + MSE** (see §11) |

**Rationale:**

- **Classification (TenGod, UsefulGod):** Single or per-position class labels → CrossEntropy.
- **Regression to distribution (ElementStrength):** Continuous normalized vector, possible zeros → MSE (see §1).
- **Multi-label binary (Interaction):** Each dimension is 0/1 (or soft); independence across dims → BCEWithLogits. `InteractionHead` supports `pos_weight` to upweight rare positive bits from dataset stats.
- **Hybrid (CrossTemporal):** The 66-dim vector mixes binary flags (dims 0–59, 65) and continuous elemental-delta (dims 60–64). See §11.

**References:** `bazi_model/tasks/*.py`; `docs/tasks_audit.md`.

---

## 3.2 Ten-god head: class weights, Focal loss, label smoothing

**Context:** Ten-god accuracy was ~89% with per-class F1 low for 比肩(2), 食神(4), 七杀(6); confusion with adjacent/similar types. Plan: improve with class balancing and hard-example focus.

**Decision (Phase 1 evaluation improvements):**

- **Class weights:** Compute **sqrt-inverse class frequency** from the **train set** only (same rationale as UsefulGod: avoids saddle at uniform predictions while up-weighting rarer classes). Passed to `TenGodHead(class_weight=...)` and used in both CE and Focal loss.
- **Focal loss:** Optional Focal loss (default on in `train_ten_god.py`) with **gamma=2** to down-weight easy examples and focus gradient on hard/minority classes. Formula: `(1 - p_t)^gamma * CE`; with optional per-class weight as alpha_t.
- **Label smoothing:** **0.1** in CrossEntropy (and in the underlying CE used by Focal) to reduce overconfident wrong predictions and soften boundaries between adjacent ten-god types.

**Rationale:** Sqrt-inverse avoids the uniform-prediction saddle; Focal targets hard classes; label smoothing is cheap and reduces overconfidence. All three are used together by default; script flags: `--no-class-weights`, `--no-focal-loss`, `--focal-gamma`, `--label-smoothing`.

**References:** `bazi_model/tasks/ten_god_head.py`; `scripts/train_ten_god.py`; `plans/phase1_evaluation_improvements_*.plan.md`.

**Note (sequential vs multi-task ten-god):** Sequential ten-god (2-layer encoder, class weights + Focal + label smoothing) reached 89.13% test accuracy; multi-task (shared encoder, plain CE head) reached 100%. The gap is **not** due to 3-layer vs 2-layer: 2-layer multi-task also gets 100% ten-god. The gain is attributed to **multi-task learning** (shared representation + auxiliary tasks). Task weighting (1,6,1,1) was tuned for element ordinal; ten-god has weight 1 and does not require upweighting to reach 100%. See `reports/phase1_evaluation.md` § "Investigation: Why sequential ten-god 89% vs multi-task 100%?".

---

## 3.1 Training data: no leakage (inputs vs targets)

**Context:** Feeding ground-truth targets (ten-gods, interactions, strengths, useful god, cross-temporal vector) as encoder input causes catastrophic data leakage: the model can copy inputs to outputs and initial loss is artificially low. The encoder must learn from raw chart structure only.

**Decision:** Strict separation at the **data layer**:

- **BaziChart:** `to_input_features(task_name)` returns only minimal inputs (stem_ids, branch_ids, stem_elements, stem_polarities, pillar_positions; for cross_temporal also temporal slot IDs and pillar_presence_mask). `to_ground_truth(task_name)` returns only the task's targets. No target key ever appears in inputs.
- **ChartDataset:** Takes `task_name`; `__getitem__` returns `(inputs_dict, targets_dict)` from `to_input_features` and `to_ground_truth`. Use `chart_collate_fn_inputs_targets` to batch.
- **Encoders:** NatalEncoder accepts only the minimal input keys (no ten_god_stem_ids, interaction_vector, elemental_strengths, day_master_strength, useful_god_id). The 6th token is a learned `aux_token`. DynamicEncoder accepts only temporal IDs and pillar_presence_mask (no dynamic_interaction_vector); 4th token is a learned `aux_token`.
- **Training scripts:** Pass only `inputs` to the encoder(s); compute loss from `targets`. Batch is `(inputs, targets)`; Trainer's `move_batch_to_device` supports this tuple.

**Rationale:** Leakage is impossible when targets never appear in the batch's input dict. Encoders learn from raw structure; task heads predict from encoder output only. Matches inference (no ground truth available).

**References:** `docs/fix_data_leakage.md`; `bazi_engine/models.py` (`to_input_features`, `to_ground_truth`); `bazi_model/data/chart_dataset.py`; `bazi_model/encoders/natal_encoder.py`, `dynamic_encoder.py`; all `scripts/train_*.py`.

---

## 4. Unified `forward_loss` return convention

**Context:** A generic trainer that supports multiple heads should not branch on head type for return shape.

**Decision:** All task heads return `(loss, pred)` from `forward_loss`: **loss first**, then prediction (logits or probs). The trainer can always do `loss, pred = head.forward_loss(...)`, call `loss.backward()`, and use `pred` for metrics.

**Rationale:** Consistent API reduces trainer complexity and avoids per-head branching.

**References:** `docs/tasks_audit.md` §2; each head in `bazi_model/tasks/`.

---

## 5. Useful god ambiguous class (class 5)

**Context:** 用神 (useful god) can be ambiguous; the engine may output -1 for “ambiguous” instead of one of 比劫, 食伤, 财, 官杀, 印.

**Decision:** Map -1 to **class index 5** (ambiguous). All samples are used for training; evaluation can report “unambiguous accuracy” by filtering to `target != 5` (or `useful_god_id != -1`).

**Rationale:** Keeps a single 6-class head and avoids dropping ambiguous samples from the loss; downstream can still measure accuracy on unambiguous-only if needed.

**References:** `bazi_model/tasks/useful_god_classifier.py`; `USEFUL_GOD_TO_ID` and `AMBIGUOUS_CLASS_ID`.

---

## 6. Interaction head: optional pos_weight for rare bits

**Context:** The 64-dim interaction vector has many bits that are 0 in most charts (e.g. rare 刑 or 破). Unweighted BCE can underfit rare positives.

**Decision:** `InteractionHead` accepts an optional `pos_weight` (scalar or (64,) tensor). When provided (e.g. from dataset positive rates), BCEWithLogits uses it to upweight the positive class per dimension. Use `(1 - p) / p` or similar from dataset stats for rare-bit upweighting.

**Rationale:** Improves learning on imbalanced binary dimensions without changing the 64-dim spec.

**References:** `bazi_model/tasks/interaction_detector.py`.

---

## 7. Sect (流派) and day-master logic

**Context:** Different sects (e.g. 阴历 vs 盲派) can use different rules for day-master strength and 用神.

**Decision:** Sect is an input to the calculator (e.g. `sect=2` for 盲派). Day-master strength and useful-god logic in `bazi_engine` respect sect where implemented; default is sect 2. Model heads are trained on charts generated with a given sect; changing sect in production should match training data or be documented.

**Rationale:** Keeps engine and dataset consistent; avoids mixing sects without notice.

**References:** `bazi_engine/models.py`; `docs/lunar_python_audit.md`; project rules (`.cursor/rules/project.mdc`).

---

## 8. Stems and branches input order

**Context:** All code that takes four pillars assumes a fixed order.

**Decision:** Stems and branches are always **ordered [Year, Month, Day, Hour]** (i.e. `stems[0]` = year stem, `branches[3]` = hour branch). Position indices are 0 = year, 1 = month, 2 = day, 3 = hour.

**Rationale:** Single convention avoids bugs from inconsistent ordering across calculator, encoders, and datasets.

**References:** `docs/binary_vector_and_positions_spec.md`; `bazi_engine/_constants.py`; `.cursor/rules/project.mdc`.

---

## 9. Trainer step function and batch float32

**Context:** The task-agnostic `Trainer` in `bazi_model/training/trainer.py` takes a `step_fn(batch, model, device) -> (loss, metrics_dict | None)`. Scripts build encoder + head and pass a closure that captures the optimizer.

**Decision:**

- **Step contract:** The trainer moves the batch to device and sets `model.train()` / `model.eval()` before each call. For train, step_fn must perform zero_grad, forward, loss, backward, optimizer.step(); for val, forward and return loss only. step_fn can branch on `model.training`. Returned metrics_dict is averaged over batches and logged to TensorBoard per epoch.
- **Float32:** ChartDataset and chart_collate_fn yield int64 and float64 tensors. Models/losses often expect float32. Cast float tensors to float32 either in the step_fn before the forward pass or via a dataset transform when constructing the dataset.

**Rationale:** Keeps the trainer task-agnostic; task-specific logic (which keys, which loss, casting) stays in the training scripts or step_fn closure.

**References:** `bazi_model/training/trainer.py` (module docstring); `scripts/train_*.py`.

---

## 10. Training strategy: sequential vs multi-task

**Context:** Phase 1 trains the natal encoder on four supervised tasks (ten-god, element strength, interaction, useful god). Two strategies are supported.

**Decision:**

- **Sequential:** One script per task (`train_ten_god.py`, `train_element_strength.py`, `train_interaction.py`, `train_useful_god.py`). Each run trains one encoder + one head; checkpoints are per task. Use for maximum per-task performance and independent tuning.
- **Multi-task:** One script `scripts/train_multi_task.py` trains one shared `NatalEncoder` and one head per task. Each step sums (optionally weighted) per-task losses and backprops once. Per-task metrics are logged (e.g. `train/loss_ten_god`, `val/loss_element`) for comparison. Use for a single shared encoder and faster iteration; may require task-weight tuning.

**Rationale:** Multi-task is the production path (single shared encoder for Phase 2); sequential is for debugging/ablation only. Cross-temporal is excluded from the multi-task script (different data: `temporal_only=True`, and requires `DynamicEncoder`). See `docs/phase1_training_strategy_clarification.md`.

**References:** `scripts/train_multi_task.py`; `reports/phase1_evaluation.md` § Ablation: Sequential vs multi-task training.

---

## 11. CrossTemporalHead: hybrid loss (BCE + MSE)

**Context:** The 66-dim `dynamic_interaction_vector` encodes three types of data: 60 binary interaction flags (dims 0–59), 5 continuous elemental-delta values (dims 60–64), and 1 binary useful-god-activation flag (dim 65). The original implementation applied `BCEWithLogits` across all 66 dims.

**Problem:** `BCEWithLogits` requires targets in [0, 1]. The elemental-delta values are unbounded real numbers (can be negative or > 1). This caused the loss to go massively negative during training (observed: −115,000 by epoch 22), because the BCE formula `max(x,0) − x·y + log(1+exp(−|x|))` is driven to −∞ when `y` is outside [0, 1].

**Decision:** Split the loss by dimension type:

- **Binary dims** [0:60] + [65] (61 dims): `BCEWithLogits`, unchanged.
- **Continuous dims** [60:65] (5 dims): `MSELoss` on raw predictions vs target deltas.
- Combined: `loss = bce + mse_weight * mse` (default `mse_weight=1.0`).

Index constants (`BINARY_INDICES`, `CONTINUOUS_INDICES`) are defined in the module and registered as buffers for device consistency.

**Rationale:** Matching the loss to the target's data type is a correctness requirement, not an optimization choice. BCE on binary, MSE on continuous. The `mse_weight` parameter allows tuning the relative importance if the two loss scales differ significantly.

**Optional segment weighting and pos_weight:** To improve macro F1 and weak segments (natal×大运, useful_god_active), `CrossTemporalHead` supports optional `segment_weights` (length 7, one per segment: natal×大运, natal×流年, natal×流月, 大运×流年, 大运×流月, 流年×流月, useful_god_active) and optional `pos_weight` (length 61 for the binary dims). When `segment_weights` is set, BCE is computed per segment and combined as a size-weighted average so that all 1s matches the default single-BCE mean. When `pos_weight` is set, it is passed to BCE for the 61 binary dims (same API as `InteractionHead`). Both default to `None`; no buffers are registered when unused, so existing checkpoints and call sites using `CrossTemporalHead()` are unchanged. The evaluator loads the head with `strict=False` so checkpoints trained with these buffers still load for inference. Re-training with `--segment_weights "2,1,1,1,1,1,4"` and/or `--pos_weight_from_data` in `train_cross_temporal.py` is an option to raise macro F1 and per-segment F1 for natal×大运 and useful_god_active without changing the 66-dim layout or inference API.

**References:** `bazi_model/tasks/cross_temporal_detector.py`; `bazi_engine/models.py` (`_dynamic_interaction_vector` layout); `scripts/train_cross_temporal.py`.

---

## 12. Element strength: pairwise ranking (ordinal) auxiliary loss

**Context:** Element strength is evaluated with MSE and **ordinal accuracy** (fraction of samples where argsort(pred) equals argsort(true)). MSE alone does not directly optimize ranking; ordinal accuracy was ~82% in phase 1, below the >90% target.

**Decision:** Add an optional **pairwise ranking auxiliary loss** to `ElementStrengthHead`. For each pair of elements (i, j) where target[i] > target[j], we want pred[i] > pred[j]. Contribution per such pair: `softplus(pred[j] - pred[i])`, so wrong-order pairs are penalized. The ordinal loss is averaged over all valid ordered pairs in the batch and added to MSE: `total_loss = mse + ordinal_loss_weight * pairwise_ranking_loss`. Default `ordinal_loss_weight=0.15` (in the 0.1–0.2 range so MSE still dominates). Set `ordinal_loss_weight=0` to disable and recover MSE-only behaviour.

**Rationale:** Directly optimizes the ordinal metric (ranking of the five elements). No curriculum or two-phase training; one hyperparameter. Softplus is smooth and differentiable; it penalizes any violation of the desired order.

**Margin-weighted ordinal (ordinal_margin_min):** Set `ordinal_margin_min=0.02` so the pairwise loss only includes pairs with true strength diff >= 0.02 (skip borderline pairs). Use `--ordinal_margin_min 0.02` in train_multi_task.py or train_element_strength.py. Eval scripts report `ordinal_accuracy_relaxed` for clear pairs only.

**Loss vs metric (plain language):** We *evaluate* with **ordinal accuracy** (fraction of samples where the predicted order of the 5 elements matches the true order). We *cannot* use that number directly as the loss, because “right order” is computed with **argsort**, which is not differentiable—the training algorithm needs gradients. So we use a **differentiable surrogate**: the pairwise ranking loss. For every pair (i, j) where the true strength of i is higher than j, we penalize if the model predicts the opposite; that pushes the model toward the correct order and is differentiable. So the loss already has an “ordinal” part (pairwise ranking) plus MSE (to keep predicted strengths in a sensible range). We do not replace the loss with ordinal accuracy; we approximate “get the order right” with the pairwise term and keep MSE so predictions stay reasonable.

---

## 13. Useful god: pretrained encoder from element strength (Option A)

**Context:** Useful-god (用神) is a deterministic function of day-master strength percentile. When training the useful-god head in isolation, the encoder does not receive strength; it must infer it from stems/branches. This contributed to class collapse (model predicting a single class). The element-strength head already learns strength-related structure (ordinal accuracy ~82%).

**Decision:** In `scripts/train_useful_god.py`, support loading the encoder from an **element_strength** checkpoint (`--encoder_checkpoint`). Train only the UsefulGodHead on top of this encoder (optionally with `--freeze_encoder`). Do **not** require multi-task training (Option B) as the default; multi-task can be tried later as an ablation.

**Rationale:** Simpler than multi-task: no loss weighting or shared-encoder interference. Useful god is a function of strength; reusing an encoder that already predicts strength gives the head a direct signal. Multi-task can be evaluated separately.

**References:** `scripts/train_useful_god.py`; Phase 1 evaluation improvements plan (useful-god collapse).

---

## 14. Class balancing: sqrt-inverse frequency and Focal loss (useful god, ten-god)

**Context:** Useful-god and ten-god tasks have class imbalance. Useful-god collapsed to a single class; ten-god had low per-class F1 for 比肩, 食神, 七杀. Full inverse-frequency weighting can create a saddle at uniform predictions.

**Decision:**

- **Useful god:** Keep **sqrt-inverse class frequency** for CrossEntropy weights; add **Focal loss** (e.g. `gamma=2`) in the useful-god classifier to down-weight easy (majority) examples and focus gradient on hard/minority classes.
- **Ten-god:** Use **sqrt-inverse class frequency** for per-class weights, **Focal loss** (`gamma=2`), and **label smoothing** (e.g. 0.1) to reduce overconfident wrong predictions and soften boundaries between adjacent types.

**Rationale:** Sqrt-inverse avoids the uniform-prediction saddle; Focal targets hard/minority classes; label smoothing is cheap and reduces overconfidence. One hyperparameter (gamma) per loss; no data-pipeline change.

**References:** `bazi_model/tasks/useful_god_classifier.py`; `bazi_model/tasks/ten_god_head.py`; training scripts; Phase 1 evaluation improvements plan.

---

## 15. Evaluation: train/val/test split and test metrics

**Context:** Phase 1 evaluation originally used only train/val; reporting only on the validation set can overstate performance (e.g. interaction F1 100% on val). A held-out test set gives an honest estimate.

**Decision:**

- **Split:** Support **train/val/test** (e.g. 80/10/10) with `--val_ratio` and `--test_ratio` in training scripts and in `scripts/evaluate_phase1.py`. Same seed and ratios in training and evaluation so splits are reproducible.
- **Reporting:** When `test_ratio > 0`, `evaluate_phase1.py` evaluates on the **test** set and adds `eval_split: "test"` to the output; when `test_ratio == 0`, evaluation uses the validation set (legacy). Report test metrics in `reports/phase1_evaluation.md` when using 80/10/10.

**Rationale:** Test split gives an unbiased estimate; documenting both val-only and test evaluation avoids confusion.

**References:** `scripts/evaluate_phase1.py`; all `scripts/train_*.py`; `reports/phase1_evaluation.md`.

---

## 16. Interaction evaluation: index-based and positive-rate-based F1

**Context:** Interaction F1 is reported for “common” vs “rare” bits. Index-based (common = 0–29, rare = 30–63) is fixed by spec; positive-rate-based defines rare as bits with low positive rate on the evaluation set.

**Decision:** `evaluate_phase1.py` always reports **both** index-based and positive-rate-based common/rare F1 for the interaction task. Output JSON contains `interaction.index` and `interaction.positive_rate`, each with `f1_common_bits`, `f1_rare_bits`, and `common_rare_mode`. No CLI flag to choose one; both are computed so “100%” is not the only number and any easy-val bias is visible.

**Rationale:** Positive-rate-based surfaces bits that are rare on the eval set; index-based matches the binary vector spec. Documenting both avoids overclaiming when val is easy.

**References:** `scripts/evaluate_phase1.py`; `reports/phase1_evaluation.md` § Interaction: common vs rare bit groups.

---

## 14. Phase 1 multi-task: hyperparameter tuning

**Context:** Multi-task training uses fixed defaults (lr 1e-3, batch 32/64, epochs 30, task_weights 1,1,1,1). These are not the result of a full search; they are sufficient to meet most Phase 1 targets. Element ordinal accuracy often remains below the >90% target.

**Decision:** Treat current values as **reasonable defaults**. To find better parameters: (1) Prefer tuning **task_weights** first (e.g. `1,2,1,1` or `1,3,1,1` for element) to improve ordinal accuracy. (2) Then consider lr, batch_size, patience. (3) For rigorous search, use a small manual grid, random search, or Optuna over lr, batch_size, and task_weights; evaluate each run with `evaluate_phase1.py` and compare per-task metrics or a composite score.

**Rationale:** Full hyperparameter search is expensive; task weighting is the highest-impact lever for the one underperforming task. Documenting the search strategy in one place avoids ad-hoc tuning.

**References:** `docs/phase1_training_strategy_clarification.md` § “Are these parameters optimal? How to find the best parameters”; `scripts/train_multi_task.py`; `scripts/evaluate_phase1.py`.

---

## 15. Multi-task: best checkpoint by loss vs by composite metric

**Context:** The Trainer saves the checkpoint with **lowest validation (combined) loss**. With unequal task weights, that can favor the task(s) with higher weight and may not align with "best" on the actual Phase 1 metrics (ten-god accuracy, element ordinal, interaction F1, useful-god accuracy).

**Decision:** Acknowledge that **lowest val loss ≠ best for all metrics**. Offer two modes: (1) **Save by loss** (default): current behavior. (2) **Save by composite metric**: save the checkpoint with **highest** composite (mean of ten_god_acc, element_ordinal, interaction_binary_acc, useful_god_acc). Use `--save_by composite` in `train_multi_task.py`. Composite is computed from the same batch outputs used for loss (no extra forward pass).

**Rationale:** With e.g. `task_weights 1,3,1,1`, minimizing loss can over-improve element at the cost of others; saving by composite balances all four tasks by the metrics we care about.

**References:** `scripts/train_multi_task.py`; `bazi_model/training/trainer.py`.

---

## 16. Phase 1: pretrain on element then multi-task

**Context:** Element strength (especially ordinal accuracy) is the hardest of the four natal tasks. Multi-task from scratch may underfit element; pretraining the encoder (and optionally the element head) on element-only then fine-tuning with all four tasks can give the encoder a strong element signal before other tasks are added.

**Decision:** Support **pretrain-then-multitask** via `train_multi_task.py --encoder_checkpoint <path>`. (1) Run `train_element_strength.py` to produce an element-only checkpoint (encoder + element head). (2) Run `train_multi_task.py --encoder_checkpoint checkpoints/element_strength_best.pt`; the script loads the encoder and (by default) the element head from that checkpoint, initializes the other heads randomly, and trains all four tasks. Use `--no_load_element_head` to load only the encoder and train the element head from scratch in the multi-task phase. Checkpoint format: either multi-task dict (`encoder`, `element_head`) or element model state_dict (`encoder.*`, `head.*`).

**Rationale:** Same encoder architecture; element checkpoint is compatible. Fine-tuning with task_weights (e.g. 1,3,1,1) helps retain element performance while learning ten-god, interaction, useful-god. Optional element-head load allows ablating “encoder only” vs “encoder + element head” init.

**References:** `scripts/train_multi_task.py` (`_load_encoder_and_element_from_checkpoint`); `scripts/train_element_strength.py`.

---

## 17. Phase 2: mlx_lm model API for custom prefix (virtual tokens)

**Context:** BaziLLM prepends virtual tokens (from a projection of natal + dynamic embeddings) to the query before the frozen LLM. We need to run the mlx_lm-loaded model on **combined embeddings** (virtual prefix + query token embeddings) rather than token IDs only.

**Inspection (mlx_lm, Qwen2 / Qwen2.5):**

- **Load:** `model, tokenizer = mlx_lm.load(path)` returns the top-level `Model` (e.g. `qwen2.Model`). For Qwen2.5-7B the config uses `model_type` that maps to the same Qwen2 architecture in mlx_lm.
- **Structure:** The returned `model` has:
  - `model.model` — the inner transformer (`Qwen2Model`): `embed_tokens`, `layers` (list of blocks), `norm`.
  - `model.model.embed_tokens` — `nn.Embedding(vocab_size, hidden_size)`; use for query token embeddings.
  - `model.model.layers` — transformer blocks; no need to call directly when using the top-level forward.
- **Forward API:** The model’s `__call__` accepts:
  - `inputs: mx.array` — token IDs, shape `(batch, seq_len)`.
  - `cache=None` — KV cache for generation; use `None` for a single forward (e.g. teacher-forcing).
  - `input_embeddings: Optional[mx.array] = None` — if provided, **replaces** the embedding lookup: the model uses `input_embeddings` as the hidden states and does **not** use `inputs` for embedding. Sequence length and positions are inferred from the **embedding sequence length** (positions 0 to L−1).
- **Custom prefix usage:** To run with virtual tokens prepended:
  1. Build `virtual_tokens`: shape `(batch, n_virtual, d_llm)` from the projection layer.
  2. Get `query_emb = model.model.embed_tokens(query_ids)` → `(batch, L, d_llm)`.
  3. `inputs_embeds = mx.concatenate([virtual_tokens, query_emb], axis=1)` → `(batch, n_virtual + L, d_llm)`.
  4. Call `model(inputs, cache=None, input_embeddings=inputs_embeds)`. The `inputs` argument is unused when `input_embeddings` is set; pass a placeholder of shape `(batch, n_virtual + L)` (e.g. zeros) for API compatibility.
  5. Output logits have shape `(batch, n_virtual + L, vocab_size)`. Apply loss on the query positions (e.g. last position or all query positions) as needed.
- **Position IDs:** Not passed explicitly. RoPE in the attention layers uses the sequence index (0 to seq_len−1); no separate `position_ids` argument. So virtual tokens get positions 0..n_virtual−1 and query tokens get n_virtual..n_virtual+L−1.
- **Attention mask:** Built inside the model from `create_attention_mask(h, cache[0])` using `h.shape[1]`. With `cache=None` and full-sequence forward, a causal mask is used. For training with teacher-forcing, this is correct (causal over the combined sequence).
- **Check support:** `mlx_lm.utils.does_model_support_input_embeddings(model)` returns `True` for Qwen2/Qwen2.5 and other models that accept `input_embeddings`.

**Decision:** BaziLLM will use the **existing** `input_embeddings` path: no custom forward or wrapper. Build combined embeddings (projection output + query embeddings from `model.model.embed_tokens`), pass them as `input_embeddings`, and pass a placeholder `inputs` array. Use `cache=None` for single forward; use the standard KV cache when implementing generation.

**References:** `mlx_lm.models.qwen2` (Qwen2Model, Model); `mlx_lm.models.base` (create_attention_mask); `mlx_lm.generate` (generate_step with input_embeddings); `mlx_lm.utils` (load, does_model_support_input_embeddings). Verification script: `scripts/inspect_mlx_lm_api.py`.

---

## 18. Phase 2.2 / v2: Projection and BaziLLM (768-d input, Option A projection, attention/position)

**Context:** BaziLLM maps encoder outputs to virtual tokens prepended to the query before the frozen LLM. The v2 stack concatenates **natal (256) + dynamic (256) + cross-interaction (256)** → **768-d** before projection.

**Decisions:**

- **Projection input dimension — 768 (v2):** The projection layer accepts **768-dimensional** input: `concat(natal_emb, dynamic_emb, cross_emb)` each 256-d. `CrossInteractionEncoderMLX` consumes structured interaction features from the chart; for charts without full temporal context, dynamic/cross inputs still use the same tensor shapes (placeholders where applicable) so `d_in` stays fixed.

- **Projection architecture — Option A (simple):** **Single linear** plus reshape: `Linear(768, n_virtual_tokens * d_llm)` → `(batch, n_virtual_tokens, d_llm)`. **Option B (two-layer MLP)** via a bottleneck remains a fallback if memory or optimization warrants it; not the default.

- **Constants:** `d_llm = 3584` (Qwen2.5-7B hidden size). `n_virtual_tokens` is **configurable in the range 4–64** (unit tests may use smaller values). Default training/eval for Phase 3 prep uses **64** virtual tokens; legacy Phase 2 baselines used **8**. See §21 for the 8→64 rationale.

- **Positions:** Virtual tokens occupy **0 … n_virtual−1**; text tokens follow. Same `input_embeddings` / causal-mask behaviour as §17.

- **Attention behaviour:** Unchanged from the earlier spec: causal LM over `[virtual | query]`; loss on text positions per §19.

**Rationale:** One `d_in` and Option A keep the checkpoint contract simple; 768-d matches the three frozen encoders used in production Phase 2 v2.

**References:** `bazi_model/projection/projection_layer.py`; `bazi_model/model/bazi_llm.py`; `bazi_model/encoders/cross_interaction_encoder_mlx.py`; §17; §21.

---

## 19. Phase 2 dataset format and loss masking

**Context:** Phase 2 training uses a curated JSONL of (chart, task_type, query, reasoning, output). The model receives virtual tokens (from chart) + tokenized task_token + query; we need a clear contract for what is in the input vs what is in the target so loss is not applied on the query.

**Decision:**

- **Dataset format:** Curated input has `chart` (full BaziChart JSON), `task_type`, `query`, `reasoning`, `output`. Prepared output adds `task_token` (e.g. `[TASK:本命分析]`). Chart is stored as full JSON so training can call `BaziChart.from_json` and `to_input_features("multi")` or `to_input_features("cross_temporal")` as needed. See `docs/phase2_implementation.md` § Task 2.3.
- **Loss masking:** **Input to the model** = virtual tokens + tokenized `task_token + query`. **Target (labels)** = tokenized `reasoning + output`. **Loss mask:** positions for `task_token + query` get mask 0 (no loss); positions for `reasoning + output` get mask 1. So the query is in **input only**; loss is computed **only on reasoning + output**.

**Rationale:** Prevents the model from being trained to copy the query; gradients flow only on the reasoning and final answer. Prepare script and training loss must align on this convention.

**References:** `docs/phase2_implementation.md` (Task 2.3); `scripts/prepare_phase2_dataset.py`; Phase 2 training loss (Task 2.4).

---

### 19.1 BaziLLM training convention (full sequence, labels, loss_mask)

**Context:** For teacher-forcing training, BaziLLM must receive the full sequence (task + query + reasoning + output) so that logits at each position align with labels for the next token. Labels and loss_mask must be built consistently with this single tokenization.

**Decision:**

- **Full sequence as BaziLLM input:** Training passes the **full** string `input_sequence + "\\n" + target_sequence` (i.e. task_token + query + reasoning + output) to `BaziLLM.forward(..., query_text=full_sequence_per_sample)`. The model tokenizes this once; virtual tokens are prepended by BaziLLM, so the effective sequence is `[virtual_1..virtual_N, token_1..token_L]` with length `n_virtual + L`.
- **Labels (Option A):** Same length as the tokenized text sequence. `labels[i] = token_id at position i+1` (shifted). The last position has no next token, so `labels[seq_len-1]` is set to `ignore_index` (e.g. -100) or pad; those positions are excluded from the loss via the mask.
- **Loss mask:** `loss_mask[i] = 1` where we want to predict `labels[i]` (i.e. where the next token is part of reasoning or output). So we need the **prefix length in tokens** = length of `tokenize(task_token + "\\n" + query)`. Then: mask 0 for indices `0 .. prefix_len-2` (predicting into the query), mask 1 for `prefix_len-1 .. seq_len-2` (predicting reasoning and output), and 0 at `seq_len-1`. Any position where `labels[i]` is pad/ignore also gets mask 0.
- **Data pipeline:** `Phase2ChartDataset` loads phase2_*.jsonl; `phase2_collate(batch, tokenizer, max_length, ...)` produces `input_ids`, `labels`, `loss_mask`, plus `chart_natal_inputs` and `chart_dynamic_inputs`. The training loop uses the same tokenization for the model forward (full sequence) and for the precomputed labels/mask from the collate, so indices align.

**Rationale:** One tokenization of the full sequence avoids misalignment between model logits and labels; building the mask from prefix length keeps the contract that loss is only on reasoning + output.

**References:** `scripts/train_phase2.py` (docstring § BaziLLM training convention); `bazi_model/data/phase2_dataset.py` (`phase2_collate`); `bazi_model/training/phase2_loss.py` (`compute_lm_loss`).

---

### 19.2 Phase 2 training: optimizer, schedule, checkpoint format

**Context:** Phase 2 trains only the projection layer (encoders and LLM frozen) on MLX. We need a single place for optimizer/schedule choices and checkpoint layout so eval and resume are unambiguous.

**Decisions:**

- **Loss convention (Option A):** Labels are the **shifted** target, same length as the tokenized sequence: `labels[i] = token at position i+1`. Positions to ignore (virtual, task+query prefix, padding) use `ignore_index` (e.g. -100) and `loss_mask[i]=0`; we compute loss only where `loss_mask[i]=1` (reasoning + output). Mean over valid positions. Implemented in `compute_lm_loss(logits, labels, loss_mask)` in `bazi_model/training/phase2_loss.py`.
- **Mask construction:** `phase2_collate` builds `loss_mask`: 0 for (task+query) prefix and padding, 1 for (reasoning+output). Prefix length in tokens = `len(tokenize(input_sequence))`; mask 1 from `prefix_len-1` to `seq_len-2` (and 0 where labels are pad/ignore). Collate also returns `full_texts` (list of full strings) for BaziLLM forward.
- **Optimizer:** `mlx.optim.AdamW(learning_rate=schedule, weight_decay=0.01)`. Only projection parameters are updated; encoder/LLM grads are zeroed before `optimizer.update`.
- **Schedule:** 10% linear warmup (0 → peak LR), then cosine decay to 10% of peak over the remaining steps. Total steps = epochs × steps_per_epoch; schedule is a single callable passed as `learning_rate` to AdamW.
- **Checkpoint format (phase2_baseline):** Saved to `checkpoints/phase2_baseline/` as `.npz` (MLX/NumPy compatible). **Keys:** `step` (int64 scalar), `best_val_loss` (float32 scalar), `projection_layer.linear.weight`, `projection_layer.linear.bias` (and any other projection parameters under the `projection_layer` prefix). Optional: `checkpoint_*_optimizer.npz` for optimizer state (if supported). For eval: load only the projection weights (keys starting with `projection_layer.`) into `BaziLLM.projection_layer`; strip the prefix and use nested update. For resume: load projection and optionally optimizer state and step.

**Rationale:** Same loss/mask as §19.1; single schedule and checkpoint layout avoids drift between train and eval.

**References:** `scripts/train_phase2.py` (make_lr_schedule, _save_checkpoint, _load_projection_from_checkpoint); `docs/phase2_task_2_4_plan.md`.

---

## 20. Classical text RAG (optional, Phase 2 curation)

**Context:** The Phase 2 data plan includes an optional corpus of classical 命理 excerpts for retrieval at prompt-build or inference time (e.g. 《滴天髓》《子平真诠》《穷通宝鉴》《三命通会》 excerpts).

**Decision:**

- **Corpus location:** Place UTF-8 `.txt` files under `data/classical_texts/` (see `data/classical_texts/README.md`). The repo does not ship full copyrighted texts; add your own licensed or public-domain excerpts locally.
- **Index build:** `scripts/build_classical_rag_index.py` reads those files, splits text into overlapping token windows (default ~300 tokens, 50 overlap) using the embedding model’s tokenizer, embeds with **BAAI/bge-small-zh-v1.5** via `sentence-transformers`, L2-normalizes vectors, and writes **FAISS** `IndexFlatIP` plus `data/classical_metadata.json` (chunk id, source path, text).
- **Dependencies:** `faiss-cpu` and `sentence-transformers` (see `requirements.txt`). First run downloads the embedding weights from Hugging Face.

**Rationale:** Keeps retrieval optional and reproducible; small Chinese embedding model balances quality and size; inner product on normalized vectors matches cosine similarity for k-NN.

**References:** `scripts/build_classical_rag_index.py`; `data/classical_texts/README.md`.

---

## 21. Virtual token expansion (8 → 64)

**Context:** Phase 2 experiments showed **8 virtual tokens** are too few for a **frozen** LLM to recover fine-grained chart facts from the encoder embedding alone. The chart encodes on the order of ~100 loosely independent facts; eight slots force extreme compression and hurt factual decoding in probes.

**Decision:** Raise the default virtual token count to **64** for projection retraining (Phase 3 prep). Encoder weights stay frozen and unchanged; **only the projection layer (and, in current training, the cross-interaction encoder trained jointly)** adapt to the new token width. Old **8-token projection checkpoints are not shape-compatible** with 64-token projection—train projection from scratch or resume a 64-token checkpoint.

**Rationale:**

- **Information density:** ~64 slots gives roughly **1–2 facts per token** as a mental model—still compressed, but closer to what the frozen decoder can unpack than eight slots.
- **Context budget:** Qwen2.5-7B is used with a **1024** token budget for this pipeline. Training uses **`max_length` 960** for text so that **64 virtual + 960 text ≤ 1024**, preserving the same total sequence length as the earlier **8 + 1016** setup.
- **Why not >64:** Each extra virtual token consumes prefix length and KV cache; beyond 64, marginal gains for this chart cardinality are uncertain while text budget shrinks. Revisit if factual probes plateau.

**Parameter count:** `Linear(768, 64 × 3584)` is on the order of **~176M** parameters (weights + bias)—much larger than an 8-token projection (~22M weights for the linear alone). Training remains tractable because the LLM and natal/dynamic encoders are frozen, the curated set is small (2–3k examples), and only projection (+ cross encoder when trainable) receive gradients.

**Acceptance (engineering):** Retrained checkpoint targets validation loss **≤ 1.85** on the curated train/val split; factual probe scripts target **≥ ~50%** single-fact accuracy vs ~15% with 8 tokens (see project evaluation docs).

**References:** `scripts/train_phase2.py`, `scripts/evaluate_phase2.py`; `tasks/phase2_v2_handoff.md`.

---

## 22. Projection retraining objective: factual QA, not prose

**Context:** The 64-token projection (§21) was originally retrained on the Phase 2 curated 2–3k chart+interpretation prose dataset, with CE loss masked over the long-form interpretation. The factual probe (`scripts/probe_factual_64token.py`) reported ~30% overall accuracy, with a stark per-category split: 0% on `pillar_identity`, 25% on `ten_gods`, but ~75% on `interactions`. The categories that failed are exactly the ones whose answer is one chart fact (one stem, one branch, one ten-god label); the categories that succeeded already have wide signal in the binary interaction vector.

**Decision:** Retrain the 64-token projection (and cross-interaction encoder) on **Stage 1 single-fact QA** instead of prose. Concretely:

- **Dataset:** `data/stage1_qa.jsonl` (≈1M factual QA pairs) joined with `data/qa_charts.jsonl` (`chart_id → chart` lookup), wrapped by `Stage1QADataset` so each item exposes `input_sequence=question`, `target_sequence=answer`, and the same `natal_inputs` / `dynamic_inputs` / `cross_features` shape as `Phase2ChartDataset`.
- **Loss:** unchanged — `phase2_collate` + `compute_lm_loss` masks the loss to the answer-token span only. Because the answer is short (often one or two CJK characters), the gradient on each retrievable fact is undiluted instead of being averaged over hundreds of prose tokens.
- **Stratification:** 20k subsample with weights tuned to oversample weak categories (pillar_identity 25%, ten_gods 25%, five_elements 15%, day_master_strength 15%, interactions 10%, other 10%). Uniform sampling from 1M would smother the weak categories under interaction questions that already train well.
- **Probe split (§A.2.3):** report two buckets separately — categorical (16 questions, gated at ≥55%) and numeric (≈4 questions, reported but not gated). The previous single-target ≥50% mixed easy categorical questions with hard numeric regressions and obscured the signal.

**Rationale:**

- **Per-fact gradient signal:** A 30% overall ceiling under the prose objective is not a model-capacity problem; it is a loss-mask geometry problem. Every prose token shares gradient capacity, so a fact that appears in one position out of, say, 400 contributes ≈1/400 of the per-example signal. Short factual answers concentrate that signal on the tokens that actually contain the fact.
- **Category-specific evidence:** The original probe broke cleanly by answer geometry — 0% on questions whose answer is in the prose only as a passing reference (day stem, year pillar) vs ~75% on questions whose answer is highly constrained by the binary vector (interaction pairs). This is what motivated switching the objective rather than scaling prose data or unfreezing more parameters.
- **Drop-in compatibility:** The new dataset emits the same per-item shape `Phase2ChartDataset` does, so `phase2_collate`, `compute_lm_loss`, the optimizer, the cosine-warmup schedule, and the checkpoint format (`projection_layer.*`, `cross_interaction_encoder.*` keys) are reused unchanged. The probe (`--projection_checkpoint`) is also drop-in compatible.

**Status (v1.1):** The full 20k stratified run (`checkpoints/phase3_prep/qa_full/checkpoint_best.npz`, val_loss=0.7398) **did not** clear the acceptance criteria on the 20-question probe: categorical 26.3%, pillar_identity 0%, ten_gods 25%, day_master_strength 0%, interactions 50% (regression from the 75% baseline observed under the prose objective). See `reports/probe_factual_64token_qa_results.json`. Likely contributors to follow up on, in order: (a) data-side — verify Stage 1 answers are being faithfully passed through `phase2_collate` and that `loss_mask` covers exactly the answer span; (b) sampler-side — check that the stratified weights are actually realized in the training mix and that interactions did not get over-suppressed; (c) capacity-side — frozen LLM + projection alone may simply lack the capacity for some categorical answers, which is precisely what LoRA (E.1 / E.4) is meant to unlock. Do not declare Workstream A complete on the basis of this checkpoint.

### v1.2: Embedding-only natal head (architecture v3)

**Context:** The v1.1 per-slot pilot (`qa_arch_v2_pilot`) probe confirmed two specific failure modes that point at the natal-side architecture, not at the loss/data/sampler:

1. **The per-pillar encoder tokens already preserve identity by construction.** A linear probe over the 4 pillar tokens emitted by `NatalEncoderMLX` (see `scripts/_probe_natal_encoder_tokens.py`) reaches **100% test accuracy** on `(stem_id, branch_id)` for every pillar. There is no representation learning the transformer can usefully add at the per-pillar level — a direct embedding lookup over `(stem, branch, position, element, polarity)` provides the same identity guarantee with no compute, no training, and no opportunity for the encoder to scramble the signal en route to the projection.
2. **`cross_features` is all-zero for natal-only QA charts.** `bazi_model/data/phase2_dataset.py` (the `phase2_collate` builder used by Stage 1 QA) only populates the `natal_liunian_*` / `natal_dayun_*` flags inside the 27-d `cross_features` vector; natal-internal combinations / clashes / harms / punishments / destructions are not in it. So Stage 1 QA training feeds an all-zero 27-d vector through a randomly-initialized cross encoder for slot 6 of every example. Meanwhile the natal-internal 64-d interaction binary vector (`InteractionResult.to_binary_vector()`) is **already** computed by `chart.to_input_features()` and exposed as `interaction_vector` — it is simply not plumbed to the model.

The v1.1 per-slot probe scored `pillar_identity = 0/4` and `ten_gods = 0/4` — exactly the categories that depend on per-pillar identity flowing cleanly to the LLM, and exactly the categories that an embedding-only natal head fixes by construction.

**Decision:** Replace `NatalEncoderMLX` (transformer over per-pillar tokens) with a parameter-light **`EmbeddingNatalMLX`** for QA training. Same 5-slot output contract `(batch, 5, d_model)`, but:

- **slot 0 = `ChartSummaryHead`** — small MLP over `interaction_vector (64) || elemental_strengths (5) || day_master_strength (1) || useful_god_one_hot (11)`. Plumbs the natal-internal interaction signal that v1.1 was missing entirely, and folds in the strength/useful-god scalars so the LLM sees them through the projection.
- **slots 1–4 = per-pillar embedding sums** — `pillar_token_i = stem_emb + branch_emb + pos_emb + elem_emb + polarity_emb`, then `Linear(d_emb → d_model)`. No transformer, no self-attention. Identity is preserved by construction.

The new head is small (~58M trainable params with projection + cross encoder; the natal head itself is ~30k) and trained jointly with projection and cross encoder. Slots 5 (dynamic) and 6 (cross_features) keep their v1.1 semantics, so future temporal training drops in cleanly.

**Rationale:**

- **Linear-probe evidence forces the choice:** if the per-pillar identity is already linearly recoverable from the encoder output, the encoder transformer is at best a no-op and at worst a learnable bottleneck that the limited Stage 1 QA gradient can degrade. Going embedding-only removes the bottleneck.
- **Plumbing the missing signal is free:** `interaction_vector`, `elemental_strengths`, `day_master_strength`, and `useful_god_id` are already produced by `chart.to_input_features("multi")`. The only change is adding them to the dataset items and the natal kwargs — no engine changes.
- **Slot 0 as a single chart-summary slot (rather than splitting strength to its own slot)** keeps `n_virtual_tokens = 63` and `n_slots = 7`, so projection shape and downstream dataset/temporal compatibility are unchanged. Promoting strength to its own slot later is a one-line refactor if a future probe shows slot 0 is overloaded.

**Pilot gate (v1.2):** `--subset_size 20000 --epochs 1 --qa_val_per_category 200 --natal_mode embedding`. Gate on the probe is stricter than the v1.1 aggregate: **categorical ≥ 40% AND pillar_identity > 30% AND ten_gods > 30%**. The category floors are required because aggregate improvement alone would not show whether the embedding architecture actually fixed the per-pillar identity hole — and that is the entire point of the change. Failing the gate triggers reinvestigation rather than scaling up.

**Acceptance (full v1.2 run, 3 epochs / 20k):** same as v1.1 — categorical ≥ 55%, per-category floors `pillar_identity / ten_gods / five_elements ≥ 50%`, `interactions ≥ 60%`.

**Risks and mitigations:**

- *Embedding-only might underfit five_elements / day_master_strength* (which the encoder pretraining included as auxiliary tasks). Mitigation: slot 0 carries `elemental_strengths` and `day_master_strength` as exact float scalars; if probe still underfits, expand slot 0 with the per-element hidden-stem weights already in features.
- *Joint training of natal + projection + cross might destabilize.* Same lr (5e-5) and grad clip (1.0) as projection-only training; grad-sanity sweep extends to the new module's parameters.
- *Future temporal training* must accept the new natal kwargs. Encoder path remains the default (`--natal_mode encoder`) so temporal training can stay on the encoder until embedding-only is validated for that regime too.

**References:** `bazi_model/encoders/embedding_natal_mlx.py`; `bazi_model/data/stage1_qa_dataset.py`; `bazi_model/data/phase2_dataset.py` (`phase2_collate` plumbing of new keys); `bazi_model/model/bazi_llm.py` (`natal_input_keys`); `scripts/train_projection_qa.py` (`--natal_mode`); `scripts/probe_factual_64token.py` (`--natal_mode`); `scripts/_probe_natal_encoder_tokens.py` (linear-probe finding); `tests/test_embedding_natal_mlx.py`; `plans/embedding-natal-arch-v3_*.plan.md`.

**References:** `bazi_model/data/stage1_qa_dataset.py`; `scripts/train_projection_qa.py`; `scripts/probe_factual_64token.py` (with `bucket` field on each `ProbeQuestion`); `reports/probe_factual_64token_qa_results.json`; `docs/pre_phase3_preparation_plan.docx` (Steps A.2.2 and A.2.3).

---

- **Dimension constants:** Consider centralizing `D_NATAL`, `D_DYNAMIC`, etc. in one module to avoid drift (`docs/tasks_audit.md` §4).
- **Binary vector versioning:** If the 64-dim layout is ever changed, add a version field and document in `docs/binary_vector_and_positions_spec.md`.
