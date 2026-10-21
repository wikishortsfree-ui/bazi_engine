# Task 2.4 — Training Infrastructure (Detailed Plan)

Detailed implementation plan for **Phase 2 Training Infrastructure**: loss, optimizer/schedule, training loop, validation, and early stopping. Implements the projection-layer-only training with frozen encoders and frozen LLM on MLX.

**Sources:** [docs/phase2_detailed_plan.docx](phase2_detailed_plan.docx) (v1.0), [.cursor/plans/phase_2_implementation_document_75382377.plan.md](../.cursor/plans/phase_2_implementation_document_75382377.plan.md) §2.5.

**Prerequisites:** Task 2.1 (environment, Qwen2.5-7B MLX, encoders), Task 2.2 (projection layer, BaziLLM), Task 2.3 (dataset format, prepare script, task balancing).

**Related:** [docs/phase2_implementation.md](phase2_implementation.md) (Task 2.3 target sequence and loss masking), [docs/phase2_model_and_env.md](phase2_model_and_env.md) (encoder freezing), [docs/design_decisions.md](design_decisions.md) §17–18.

---

## Todos (implementation order)

- [ ] **2.4.1 Loss module** — Add `bazi_model/training/phase2_loss.py` (or `loss.py`) with `compute_lm_loss(logits, labels, loss_mask)` (MLX arrays; masked causal CE, mean over valid positions).
- [ ] **2.4.1 Loss test** — Add `tests/test_phase2_loss.py`: mask correctness (loss unchanged when masked positions change), gradients zero on masked positions.
- [ ] **2.4.2 Optimizer/schedule** — In `train_phase2.py`: AdamW (projection_lr 5e-5, encoder_lr 0 or 1e-5), optional warmup+cosine schedule, gradient clipping (max_norm=1.0), gradient accumulation (default 2).
- [ ] **2.4.3 Data loading** — Phase 2 dataset/collate from `phase2_train.jsonl` / `phase2_val.jsonl`: chart inputs + tokenized sequence, labels (shifted), loss_mask; task balancing (weighted sampler or oversampling).
- [ ] **2.4.3 BaziLLM training convention** — Decide and implement: full sequence (task+query+reasoning+output) as BaziLLM input; build labels and loss_mask from query/target lengths; document in script and design_decisions.
- [ ] **2.4.3 Training loop** — `scripts/train_phase2.py`: epoch loop, batch → forward BaziLLM → compute_lm_loss → value_and_grad → clip → optimizer update (every accumulation_steps), log loss.
- [ ] **2.4.3 Checkpointing** — Save latest + best (by val loss) to `checkpoints/phase2_baseline/`; include projection params, optimizer state, step, best_val_loss; document format/keys.
- [ ] **2.4.4 Validation** — Every N steps (and each epoch): run val set, compute val loss/perplexity; optionally log a few sample generations.
- [ ] **2.4.4 Early stopping** — If val loss does not improve for 5 consecutive checks, stop; keep best checkpoint as final.
- [ ] **Docs** — Record loss convention, mask construction, and MLX optimizer/schedule in `docs/design_decisions.md` if not already present.

---

## 1. Objectives and scope

- **Goal:** Train only the projection layer (and optionally encoders at low LR) so that chart embeddings map to virtual tokens the LLM can condition on. Encoders and LLM are frozen (or encoders at 1e-5).
- **Stack:** MLX (no PyTorch in the training loop). Data from Task 2.3: `(chart_features, query, reasoning, output)` with task balancing.
- **Deliverables:**
  - `bazi_model/training/loss.py` — causal LM loss with loss mask (query positions zeroed).
  - `scripts/train_phase2.py` — full training script: optimizer, schedule, loop, gradient accumulation, checkpointing, validation, early stopping.
  - Unit test for loss (mask correctness, no gradient on query positions).
  - Checkpoint layout and default paths documented.

---

## 2. Task 2.4.1 — Loss

### 2.1 Contract

- **File:** `bazi_model/training/loss.py` (or `bazi_model/training/phase2_loss.py` to avoid clashing with existing PyTorch trainer).
- **Function:** `compute_lm_loss(logits, labels, loss_mask)`:
  - **logits:** `(batch, seq_len, vocab_size)` from `BaziLLM.forward` (sequence = virtual_tokens + query_tokens; loss is applied only on the query/target part — see below).
  - **labels:** `(batch, seq_len)` int32, token IDs of the **full** sequence (virtual + query). Positions where we do **not** compute loss (e.g. virtual token positions and optionally query prefix) must have a sentinel (e.g. `-100` or a pad id) so they are excluded, **or** a separate mask is used.
  - **loss_mask:** `(batch, seq_len)` float 0/1. Position with mask 0 → no loss; mask 1 → loss. So: mask = 0 for virtual token positions and for “query” (user prompt) positions; mask = 1 for “reasoning + output” (target) positions.

**Design (align with Task 2.3):** Target sequence for next-token prediction is **reasoning + output**. The model input is virtual_tokens + tokenized(task_token + query). So the **logits** at position `i` predict the token at position `i+1`. For causal LM we need:

- Input sequence: `[virtual_1, ..., virtual_N, q_1, ..., q_L, r_1, ..., r_M, o_1, ..., o_K]`  
  where `q_*` = query tokens, `r_*` = reasoning, `o_*` = output.
- We compute loss only on positions that correspond to “predict the next token” when the next token is part of reasoning or output. So **loss_mask** is 1 at positions where the **next** token is in reasoning or output (i.e. we predict r_1 from q_L, r_2 from r_1, …, o_K from o_{K-1}), and 0 elsewhere (virtual tokens, query tokens, and optionally padding).

Concrete convention to document in the plan and in code:

- **Option A (recommended):** Labels are the **shifted** target: labels[i] = token at position i+1. Length = seq_len. Use a mask of shape (batch, seq_len): mask[i]=1 where we want to predict labels[i] (i.e. where labels[i] is not padding/sentinel). Then loss = cross_entropy(logits[:, i, :], labels[:, i]) only where mask[i]=1, then mean over masked positions.
- **Option B:** Labels same length as “target only” (reasoning+output). Then we need to slice logits at the corresponding positions (first target position = first token after query). Option A is simpler and matches standard causal LM masking.

**Implementation (Option A):**

- Input sequence to the model: `[virtual_1..virtual_n_virtual, task_token, query..., reasoning..., output...]` (or task_token+query then reasoning+output). So total length L_total = n_virtual + L_query + L_reasoning + L_output.
- Labels: same length L_total; labels[j] = token_id at position j+1 (shifted). For j in [0, n_virtual+L_query - 1] we do not compute loss → mask[j]=0. For j in [n_virtual+L_query, L_total-1] we compute loss → mask[j]=1 (and labels[j] must not be padding).
- Padding: if we pad to max_length, labels at pad positions should have a special value (e.g. -100 or pad_token_id) and mask=0 there.

**API:**

```python
def compute_lm_loss(
    logits: mx.array,   # (batch, seq_len, vocab_size)
    labels: mx.array,   # (batch, seq_len) int32, shifted target; use -100 or pad_id for ignore
    loss_mask: mx.array # (batch, seq_len) float 0/1; 1 = compute loss at this position
) -> mx.array:
    """Causal LM cross-entropy; only positions with loss_mask=1 contribute. Returns scalar."""
```

- Use standard cross-entropy: for each (b, i) with mask[b,i]=1, CE(logits[b,i], labels[b,i]). Then sum and divide by (mask.sum() + eps) for mean over valid positions. Or use a single flattened view with mask applied (mask out logits/labels where mask=0 so they don’t contribute).
- **Unit test:** Build a tiny example where mask=0 on first half and mask=1 on second half; verify that gradients w.r.t. logits are zero where mask=0 (e.g. by checking that changing logits in masked-out positions does not change the loss value).

### 2.2 Reference

- Task 2.3 in phase2_implementation.md: “Loss is only on reasoning + output”; “Loss mask: positions for task_token+query → 0, reasoning+output → 1”.
- design_decisions.md §17: BaziLLM output logits shape `(batch, n_virtual + L, vocab_size)`.

---

## 3. Task 2.4.2 — Optimizer and schedule

### 3.1 Optimizer (MLX)

- **Optimizer:** `mx.optim.AdamW` (e.g. `mlx.optim.AdamW(learning_rate=5e-5, weight_decay=0.01)`). See [MLX optimizers](https://ml-explore.github.io/mlx/build/html/python/optimizers.html).
- **Parameter groups:**
  - **Projection layer:** learning rate **5e-5** (default from plan).
  - **Encoders:** LR **0** (frozen) or **1e-5** (optional low LR). In MLX, only include in the optimizer the parameters that are trained; frozen parameters are not passed to the optimizer, so they never get updates. So either:
    - One optimizer over projection parameters only (encoders frozen), or
    - Two groups: projection params with lr 5e-5, encoder params with lr 1e-5 (if we support “encoder low LR” mode). Document both in the script (e.g. `--encoder_lr 0` vs `--encoder_lr 1e-5`).

### 3.2 Schedule

- **Warmup + cosine decay:** Warmup over first 10% of steps, then cosine decay to 10% of peak LR. Total steps = (num_epochs * num_batches_per_epoch) or (num_training_steps if using a fixed step budget).
- **MLX:** Learning rate can be a callable (step -> lr). Implement a helper that returns a schedule function; pass it as `learning_rate=schedule_fn` to AdamW. Or use a constant LR for the first baseline and add schedule in a follow-up (document as “Phase 2.4.2 schedule”).
- **Gradient accumulation:** Effective batch size = batch_size * accumulation_steps. Plan suggests accumulation 2–4 for effective batch 8–16. So e.g. batch_size=4, accumulation_steps=2 → effective 8. Accumulation: run `accumulation_steps` forward/backward passes, sum or average gradients, then call optimizer update once.

### 3.3 Gradient clipping

- Plan (docx) mitigation for NaN: “clip gradients to max norm 1.0”. In MLX, after computing gradients, clip by global norm before `optimizer.update`. Document default `max_grad_norm=1.0` in the train script.

### 3.4 Summary table (for implementation doc)

| Item              | Default / choice |
| ----------------- | ----------------- |
| Optimizer         | AdamW             |
| Weight decay      | 0.01              |
| Projection LR     | 5e-5              |
| Encoder LR        | 0 (frozen) or 1e-5 |
| Schedule          | 10% warmup + cosine to 10% peak |
| Gradient accumulation | 2–4 (effective batch 8–16) |
| Max grad norm     | 1.0               |

---

## 4. Task 2.4.3 — Training loop

### 4.1 Script and entrypoint

- **Script:** `scripts/train_phase2.py`.
- **Runs from:** Repo root. Uses Phase 2 env (MLX, mlx_lm, etc.); see `requirements_phase2.txt` and `docs/phase2_model_and_env.md`.

### 4.2 Data loading

- **Input:** Prepared JSONL from Task 2.3: `data/phase2_train.jsonl`, `data/phase2_val.jsonl` (or paths via args).
- **Batch content:** Each batch must provide:
  - Chart features for BaziLLM: natal encoder inputs (and optionally dynamic encoder inputs for temporal tasks). These can be precomputed in the dataset or computed on the fly from chart JSON. Precomputation avoids repeated encoder forward in the loop; on-the-fly is simpler and uses the same encoder as in the model.
  - Text side: either (a) full input sequence (task_token + query) and target sequence (reasoning + output) with lengths, or (b) tokenized input_ids, labels, loss_mask already aligned to the model’s sequence (virtual + query + reasoning + output). Prefer (b) for clarity: dataset or collate returns `input_ids`, `labels`, `loss_mask`, plus chart inputs.
- **Task balancing:** Use weighted sampler or oversampling so that task_type distribution matches target (本命分析 40%, 流年 30%, 流月 10%, 八字问答 20%). Expose script args, e.g. `--balance_tasks`, `--target_distribution` (or read from Task 2.3 constants).

### 4.3 Model and forward

- Load BaziLLM (natal encoder, dynamic encoder, projection, LLM, tokenizer). Encoders and LLM loaded from Phase 1 checkpoints (converted to MLX) and MLX Qwen model; projection is the only randomly initialized (or optionally load from a previous Phase 2 run).
- **Forward:** For each batch, call `BaziLLM.forward(chart_natal_inputs, chart_dynamic_inputs, query_text)`. The current `BaziLLM.forward` takes `query_text` and tokenizes internally with `max_length`. For training we need to pass the **full** input sequence (task_token + query + reasoning + output) so that logits cover the positions where we have labels. So either:
  - Extend BaziLLM (or the train script) to accept pre-tokenized input for training: e.g. `input_ids` for the “query” part (which is actually task + query + reasoning + output for teacher-forcing), and we build labels/loss_mask from the same tokenization; or
  - Keep “query_text” as the full string “task_token + query + reasoning + output” and tokenize once; then labels = shifted input_ids, and loss_mask is set from lengths (query length vs target length). Document the exact convention in the script.

Recommended: **Training uses a single sequence** = virtual_tokens + tokenized(task_token + query + reasoning + output). Labels = shift right by 1; loss_mask = 0 for (virtual + task + query) positions, 1 for (reasoning + output) positions. So the “query_text” passed to BaziLLM for training is the full prompt+target string, and we need to know the length of (task_token + query) to build the mask. The prepare script (Task 2.3) should output either lengths or a precomputed mask; the train script builds labels and mask accordingly.

### 4.4 Step (one batch)

1. Get batch: chart inputs, input_ids (or query text), labels, loss_mask.
2. Forward: `logits = model.forward(chart_natal, chart_dynamic, ...)` (with the training sequence convention above).
3. Loss: `loss = compute_lm_loss(logits, labels, loss_mask)`.
4. Backward: In MLX, use `nn.value_and_grad(model, loss_fn)` where `loss_fn` takes (model, batch) and returns the masked LM loss. Then `loss, grads = value_and_grad(model, batch)`, then clip grads, then `optimizer.update(model, grads)`, then `mx.eval(model.parameters(), optimizer.state)`.
5. If using gradient accumulation: accumulate grads over N micro-batches, then update once.

### 4.5 Checkpointing

- **Frequency:** Every N steps (e.g. 500) and at the end of each epoch; and **always** when validation improves (best checkpoint).
- **Content:** Projection state_dict (or MLX equivalent: projection parameters), optimizer state, current step/epoch, best_val_loss. Optionally encoder state if we trained them.
- **Path:** `checkpoints/phase2_baseline/` (or `--checkpoint_dir`). Files: e.g. `checkpoint_latest.npz` (or latest step), `checkpoint_best.npz` (best by val loss).
- **Format:** MLX uses numpy-style saving (e.g. `mx.savez` or save only trainable params). Document the exact keys so evaluation and inference can load the projection (and optionally encoders) correctly.

### 4.6 Logging

- Log train loss (and optionally perplexity) every step or every N steps.
- Log learning rate if using a schedule.
- Optionally log a few sample generations every 100–500 steps (same as docx “Sample outputs every 100 steps”).

---

## 5. Task 2.4.4 — Validation and early stopping

### 5.1 Validation

- **Frequency:** Every N steps (e.g. 200 or 500) and/or every epoch. Run on the full validation set (or a fixed subset if large).
- **Metric:** Validation loss (mean of `compute_lm_loss` over val batches). Optionally validation perplexity = exp(val_loss).
- **Optional:** Generate a few samples (e.g. 5) from the val set and log them (text) for manual inspection.

### 5.2 Early stopping

- **Rule:** If validation loss does not improve for **5 consecutive** validation checks, stop training.
- **Best checkpoint:** Always save when val loss improves. On early stop, **restore** the best checkpoint (or document that the script already keeps `checkpoint_best.npz` so no explicit “restore” is needed for downstream; evaluation just loads best).

### 5.3 Implementation note

- The existing `bazi_model/training/trainer.py` is PyTorch (epoch loop, early stopping by val loss, checkpoint on improve). For Phase 2 we do **not** reuse it; we implement an MLX-native loop in `scripts/train_phase2.py` that mirrors the same logic: epoch → batches → val every N steps → compare val loss → save best → early stop after 5 consecutive non-improvements.

---

## 6. File and module map (Task 2.4)

| Deliverable       | Path |
| ----------------- | ---- |
| LM loss (masked)  | `bazi_model/training/loss.py` (or `phase2_loss.py`) |
| Train script      | `scripts/train_phase2.py` |
| Unit test (loss)  | `tests/test_phase2_loss.py` (or under `tests/test_*.py` for loss mask) |
| Checkpoint dir    | `checkpoints/phase2_baseline/` (default) |

---

## 7. Default hyperparameters (baseline run)

Document in both this plan and in `scripts/train_phase2.py`:

| Parameter           | Default   |
| ------------------- | --------- |
| batch_size          | 4         |
| accumulation_steps  | 2         |
| projection_lr       | 5e-5      |
| encoder_lr          | 0         |
| epochs              | 3–5       |
| n_virtual_tokens    | 8         |
| max_length          | 1024      |
| weight_decay        | 0.01      |
| max_grad_norm       | 1.0       |
| warmup_fraction     | 0.1       |
| cosine_final_fraction | 0.1    |
| early_stopping_patience | 5 (checks) |

---

## 8. Dependencies on other tasks

- **Task 2.3:** Dataset must supply (chart inputs, input_ids or full text, labels, loss_mask) in the convention above. Prepare script must output train/val JSONL with task_token and clear boundary between “query” and “reasoning+output” so the train script can build labels and mask.
- **Task 2.2:** BaziLLM must accept the training-time sequence (full prompt+target) and return logits of shape `(batch, seq_len, vocab_size)` so that indices align with labels and loss_mask.
- **Task 2.1:** Encoders and LLM loaded and frozen; projection layer is the only (or main) trainable component.

---

## 9. Risks and mitigations (from docx, relevant to 2.4)

- **NaN:** Lower LR (2e-5), gradient clipping (max norm 1.0), inspect batch that caused divergence.
- **OOM:** Reduce batch size (e.g. 2), increase accumulation to 4; reduce max_length to 768; consider gradient checkpointing if supported in MLX for the LLM forward.
- **Encoder load mismatch:** Already addressed in Task 2.1 (conversion and parity tests). Training script assumes encoder outputs match Phase 1.

---

## 10. Completion criteria for Task 2.4

- [ ] `compute_lm_loss(logits, labels, loss_mask)` implemented and unit-tested (mask correctness, no gradient on masked positions).
- [ ] `scripts/train_phase2.py` implements: data load with task balancing, BaziLLM forward, masked loss, AdamW (projection ± encoder), optional schedule, gradient accumulation, gradient clipping, checkpointing (latest + best), validation every N steps, early stopping (5 checks).
- [ ] Defaults and checkpoint path documented; one full training run completes without error and saves a best checkpoint to `checkpoints/phase2_baseline/`.
- [ ] Design decisions (loss convention, mask construction, MLX optimizer/schedule) recorded in `docs/design_decisions.md` if not already there.

Once Task 2.4 is complete, proceed to **Task 2.5** (baseline run, hyperparameter tuning) and **Task 2.6** (evaluation and attention analysis).
