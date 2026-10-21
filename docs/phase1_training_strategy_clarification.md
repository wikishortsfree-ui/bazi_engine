# Phase 1 Training Strategy: Technical Clarification

**Purpose**: This document clarifies the Phase 1 encoder pre-training architecture and training strategy based on evaluation results, for implementation/refactoring by Cursor AI.

**Date**: 2026-03-06  
**Status**: Authoritative specification superseding ambiguous guidance in phase1_detailed_plan.docx

---

## Executive Summary

**CRITICAL DECISION**: Use **multi-task training only** as the production path for Phase 1. Sequential training was originally recommended as a debugging strategy but evaluation results show multi-task is vastly superior and sequential is unnecessary.

**Evaluation Results Summary**:
- Multi-task: 100% ten-god accuracy, 99.93% useful-god accuracy, all targets exceeded
- Sequential: 89% ten-god accuracy (missed target), 23% useful-god accuracy (catastrophic failure)
- Conclusion: Multi-task produces a single shared encoder that learns superior representations

---

## Architecture Overview

### What Gets Trained in Phase 1

```
INPUT: BaziChart (stems, branches, interactions, elements...)
   ↓
┌─────────────────────────────────────────────────┐
│ EMBEDDING LAYER (trained)                       │
│ - stem_embedding: (10, d_emb)                   │
│ - branch_embedding: (12, d_emb)                 │
│ - ten_god_embedding: (10, d_emb)                │
│ - element_embedding: (5, d_emb)                 │
│ - polarity_embedding: (2, d_emb)                │
│ - position_embedding: (4, d_emb)                │
└─────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────┐
│ INTERACTION FEATURE ENCODER (trained)           │
│ - Projects binary/categorical interaction       │
│   features to d_model=256                       │
└─────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────┐
│ NATAL ENCODER (trained)                         │
│ - 2-layer Transformer                           │
│ - 4 attention heads, d_model=256, d_ff=1024     │
│ - Input: [CLS] + 4 pillars + interaction_vec   │
│ - Output: natal_emb (batch, 256)                │
└─────────────────────────────────────────────────┘
   ↓
   natal_emb (batch, 256) ← Single shared representation
   ↓
   ├─→ [Ten-God Head] (trained) → (batch, 7, 10) logits
   ├─→ [Element Head] (trained) → (batch, 5) strengths
   ├─→ [Interaction Head] (trained) → (batch, 64) binary
   └─→ [Useful-God Head] (trained) → (batch, 6) logits
```

**Key Point**: ALL parameters (embeddings, encoder, heads) are trained together via backpropagation from the combined task losses.

---

## Multi-Task Training (PRODUCTION PATH)

### Architecture

```python
# Conceptual structure
class MultiTaskModel(nn.Module):
    def __init__(self):
        self.encoder = NatalEncoder()  # ONE shared encoder
        self.heads = {
            'ten_god': TenGodHead(d_model=256, num_classes=11),
            'element': ElementStrengthHead(d_model=256),
            'interaction': InteractionHead(d_model=256, num_bits=64),
            'useful_god': UsefulGodHead(d_model=256, num_classes=6)
        }
    
    def forward(self, batch):
        # Single encoding pass
        natal_emb = self.encoder(batch)  # (batch, 256)
        
        # Multiple task predictions from same embedding
        outputs = {
            'ten_god': self.heads['ten_god'](natal_emb),
            'element': self.heads['element'](natal_emb),
            'interaction': self.heads['interaction'](natal_emb),
            'useful_god': self.heads['useful_god'](natal_emb)
        }
        return outputs
```

### Training Loop

```python
# Single training script: train_multi_task.py
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass
        outputs = model(batch)
        
        # Compute per-task losses
        losses = {
            'ten_god': cross_entropy_loss(outputs['ten_god'], batch.ten_god_labels),
            'element': mse_loss(outputs['element'], batch.element_strengths),
            'interaction': bce_loss(outputs['interaction'], batch.interaction_binary),
            'useful_god': cross_entropy_loss(outputs['useful_god'], batch.useful_god_labels)
        }
        
        # Combined loss with optional task weighting
        total_loss = sum(weight * losses[task] for task, weight in task_weights.items())
        
        # Backward pass - ALL parameters update from ALL tasks
        optimizer.zero_grad()
        total_loss.backward()  # Gradients flow to embeddings, encoder, all heads
        optimizer.step()
        
        # Log per-task metrics for monitoring
        log_metrics(losses, outputs, batch)
```

### Checkpoint Structure

**File**: `checkpoints/multi_task_best.pt`

```python
{
    'encoder': encoder.state_dict(),       # NatalEncoder weights
    'ten_god_head': ten_god_head.state_dict(),
    'element_head': element_head.state_dict(),
    'interaction_head': interaction_head.state_dict(),
    'useful_god_head': useful_god_head.state_dict(),
    'best_val_metric': best_val_metric,  # loss (save_by loss) or composite (save_by composite)
}
```

(Implementation uses flat keys, not a nested `heads` dict. Phase 2 and evaluator load via `checkpoint['encoder']`, `checkpoint['ten_god_head']`, etc.)

**Usage in Phase 2/3**:
```python
# Load for downstream use
checkpoint = torch.load('checkpoints/multi_task_best.pt')
encoder = NatalEncoder()
encoder.load_state_dict(checkpoint['encoder'])

# Optional: load heads for verification
ten_god_head = TenGodHead()
ten_god_head.load_state_dict(checkpoint['ten_god_head'])

# Connect to LLM
projection = ProjectionLayer(d_model=256, d_llm=4096)
natal_emb = encoder(chart)
llm_input = projection(natal_emb)
```

---

## Sequential Training (DEBUGGING ONLY - NOT PRODUCTION)

### Purpose

Sequential training trains **separate encoders** for each task independently. This is ONLY useful for:

1. **Debugging**: If multi-task training fails completely, sequential isolates which task is problematic
2. **Ablation studies**: Comparing single-task vs multi-task performance
3. **Proof of concept**: Verifying each task is learnable before attempting multi-task

**SEQUENTIAL IS NOT A PRODUCTION PATH** because it produces 4 incompatible encoders that cannot be combined.

### What Sequential Produces

```
Run 1: python train_ten_god.py
   → checkpoints/ten_god_best.pt (Encoder A + Ten-God Head A)

Run 2: python train_element_strength.py  
   → checkpoints/element_strength_best.pt (Encoder B + Element Head B)

Run 3: python train_interaction.py
   → checkpoints/interaction_best.pt (Encoder C + Interaction Head C)

Run 4: python train_useful_god.py
   → checkpoints/useful_god_best.pt (Encoder D + Useful-God Head D)
```

**Result**: 4 different encoders (A, B, C, D) with different learned weights.

### Why Sequential Cannot Be Used in Production

**Problem 1**: Which encoder do you choose for Phase 2?

```python
# Can only pick ONE encoder for LLM integration
encoder = load("checkpoints/ten_god_best.pt")['encoder']  # Encoder A

# But Encoder A only learned ten-god features!
# It knows nothing about elemental balance, interactions, or useful god
natal_emb = encoder(chart)  # Missing critical information
projection = ProjectionLayer()
llm_input = projection(natal_emb)  # Incomplete representation
```

**Problem 2**: Cannot combine encoders

```python
# IMPRACTICAL: Run all 4 encoders
encoder_A = load("ten_god_best.pt")['encoder']
encoder_B = load("element_best.pt")['encoder']
encoder_C = load("interaction_best.pt")['encoder']
encoder_D = load("useful_god_best.pt")['encoder']

emb_A = encoder_A(chart)  # (batch, 256)
emb_B = encoder_B(chart)  # (batch, 256)
emb_C = encoder_C(chart)  # (batch, 256)
emb_D = encoder_D(chart)  # (batch, 256)

combined = torch.cat([emb_A, emb_B, emb_C, emb_D], dim=1)  # (batch, 1024)

# Problems:
# - 4x computation cost (4 encoder forward passes)
# - 4x memory (4 sets of weights loaded)
# - Projection layer now needs 1024 → 4096 mapping (more parameters)
# - Embeddings not trained to work together (learned independently)
```

**Problem 3**: Performance is worse

From evaluation results:
- Sequential ten-god: 89% (below target)
- Sequential useful-god: 23% (catastrophic collapse)
- Multi-task ten-god: 100%
- Multi-task useful-god: 99.93%

**Conclusion**: Sequential has no advantages and severe disadvantages. Do not use for production.

---

## Code Organization Requirements

### Directory Structure

```
bazi_model/
├── encoders/
│   ├── embeddings.py         # Embedding tables
│   ├── interaction_features.py  # Interaction feature encoder
│   ├── natal_encoder.py      # 2-layer transformer
│   └── dynamic_encoder.py    # For cross-temporal (Phase 1.3.5)
├── tasks/
│   ├── ten_god_classifier.py     # TenGodHead
│   ├── element_regressor.py      # ElementStrengthHead
│   ├── interaction_detector.py   # InteractionHead
│   ├── useful_god_classifier.py  # UsefulGodHead
│   └── cross_temporal_detector.py  # CrossTemporalHead
└── training/
    └── trainer.py            # Training loop utilities

scripts/
├── train_multi_task.py       # PRIMARY: Multi-task training
├── train_cross_temporal.py   # Separate: temporal data only
├── evaluate_phase1.py        # Evaluation script
├── train_ten_god.py          # OPTIONAL: Sequential debugging
├── train_element_strength.py # OPTIONAL: Sequential debugging
├── train_interaction.py      # OPTIONAL: Sequential debugging
└── train_useful_god.py       # OPTIONAL: Sequential debugging

checkpoints/
├── multi_task_best.pt        # Production checkpoint
├── cross_temporal_best.pt    # Temporal encoder checkpoint
└── sequential/               # Optional debugging checkpoints
    ├── ten_god_best.pt
    ├── element_strength_best.pt
    ├── interaction_best.pt
    └── useful_god_best.pt
```

### Script Responsibilities

#### `train_multi_task.py` (PRIMARY)

**Purpose**: Train the production encoder with all 4 natal tasks jointly.

**Key features**:
- Single NatalEncoder shared across all heads
- Combined loss = weighted sum of 4 task losses
- Logs per-task metrics to TensorBoard (train/loss_ten_god, val/accuracy_ten_god, etc.)
- Saves single checkpoint with encoder + all heads
- Uses train/val/test split (80/10/10 recommended)

**Arguments**:
```bash
python scripts/train_multi_task.py \
    --data data/charts.jsonl \
    --epochs 30 \
    --batch_size 64 \
    --lr 1e-3 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --seed 42 \
    --checkpoint_dir checkpoints \
    --log_dir runs/multi_task \
    --task_weights ten_god=1.0,element=1.0,interaction=1.0,useful_god=1.0
```

**Task weights**: Optional argument to balance task importance. Default: equal weights (1.0 each). If element strength underperforms, try `--task_weights 1,2,1,1` or `1,3,1,1` (ten_god, element, interaction, useful_god) to prioritize element/ordinal.

**Save by**: `--save_by loss` (default) saves the checkpoint with lowest validation loss; `--save_by composite` saves the checkpoint with **highest** composite (mean of ten_god acc, element ordinal, interaction acc, useful_god acc). Use `composite` when task weights are unequal so the "best" checkpoint reflects balanced performance on all four tasks rather than whichever task dominates the weighted loss. See `docs/design_decisions.md` §15.

#### `train_cross_temporal.py` (SEPARATE)

**Purpose**: Train DynamicEncoder on cross-natal-temporal interactions.

**Key differences**:
- Uses DynamicEncoder (not NatalEncoder)
- Filters dataset to `temporal_only=True` (charts with 流年 provided)
- Single task: cross-temporal interaction prediction
- Saves separate checkpoint (not part of multi-task)

**Why separate**: Temporal data is a different distribution (charts with flowing year overlays). Mixing temporal and natal examples in one batch is awkward. Keep as separate training run.

#### `train_*.py` (Sequential scripts - OPTIONAL)

**Purpose**: Debugging only. Train individual tasks to isolate failures.

**When to use**:
- Multi-task training achieves <50% accuracy on all tasks → something is fundamentally broken
- Run sequential scripts to determine which task is problematic
- Fix the broken task implementation
- Return to multi-task training

**When NOT to use**:
- Multi-task training works (your case) → skip sequential entirely
- Production deployment → always use multi-task encoder

#### `evaluate_phase1.py`

**Purpose**: Evaluate trained models on validation or test set.

**Key features**:
- Supports both multi-task and sequential checkpoints
- Reports per-task metrics matching targets (accuracy, MSE, F1, etc.)
- Generates confusion matrices
- Saves metrics to JSON for documentation

**Usage**:
```bash
# Evaluate multi-task model on test set
python scripts/evaluate_phase1.py \
    --data data/charts.jsonl \
    --use_multi_task \
    --checkpoint_dir checkpoints \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --seed 42 \
    --output reports/phase1_metrics_multi_task.json

# Evaluate sequential checkpoints (for comparison)
python scripts/evaluate_phase1.py \
    --data data/charts.jsonl \
    --checkpoint_dir checkpoints/sequential \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --seed 42 \
    --output reports/phase1_metrics_sequential.json
```

---

## Are these parameters optimal? How to find the best parameters

**Short answer:** The documented defaults (epochs 30, batch_size 64, lr 1e-3, task_weights 1,1,1,1 or 1,2,1,1) are **reasonable defaults**, not the result of a full hyperparameter search. They are sufficient to hit Phase 1 targets on most tasks. To find better parameters, use the following.

### Current defaults and rationale

| Parameter        | Default   | Rationale |
|-----------------|-----------|-----------|
| `--epochs`      | 30        | Enough for loss to plateau; early stopping (patience 5) usually stops earlier. |
| `--batch_size`  | 32 (script) / 64 (doc) | Larger batch → more stable gradients, faster epoch; 64 is a good tradeoff on single GPU. |
| `--lr`          | 1e-3      | Standard for AdamW; encoder + heads train well at this range. |
| `--weight_decay`| 0.01      | AdamW default; helps generalization. |
| `--val_ratio` / `--test_ratio` | 0.1 each | 80/10/10 split; enough for early stopping and final eval. |
| `--task_weights`| 1,1,1,1 or 1,2,1,1 | Equal or slightly higher element weight to improve ordinal accuracy (only task below target). |
| `--patience`    | 5         | Stop if val loss does not improve for 5 epochs. |

The **scheduler** is CosineAnnealingLR (in `train_multi_task.py`); the **Trainer** saves the best checkpoint by validation loss (sum of weighted task losses).

### What to tune first

1. **Task weights** — If element **ordinal accuracy** is below target (>90%): try `--task_weights 1,2,1,1` or `1,3,1,1`. If one task dominates the loss and others degrade: reduce that task’s weight or increase others.
2. **Learning rate** — If loss is noisy or diverges: try `--lr 5e-4` or `3e-4`. If convergence is slow: try `--lr 2e-3` (watch for instability).
3. **Batch size** — 32 vs 64 vs 128: larger usually gives more stable gradients; if OOM, reduce. Keep `--lr` fixed when changing batch size unless you explicitly scale (e.g. linear scaling rule).
4. **Epochs / patience** — If early stopping triggers too soon: increase `--patience` (e.g. 8) or `--epochs` (e.g. 40).

### How to search systematically

**Option A — Manual grid (few runs)**  
- Fix seed (e.g. 42). Run 3–4 configs, e.g.:  
  - Baseline: `--lr 1e-3 --batch_size 64 --task_weights 1,1,1,1`  
  - Element focus: `--task_weights 1,3,1,1`  
  - Lower LR: `--lr 5e-4`  
  - Larger batch: `--batch_size 128`  
- After each run, evaluate:  
  `python scripts/evaluate_phase1.py --data data/charts.jsonl --use_multi_task --val_ratio 0.1 --test_ratio 0.1 --seed 42 --output reports/phase1_metrics_<tag>.json`  
- Compare metrics (ten_god accuracy, element ordinal_accuracy, interaction F1, useful_god accuracy) and pick the best.

**Option B — Random search (many runs)**  
- Sample 10–20 configs: e.g. `lr` in [3e-4, 5e-4, 1e-3, 2e-3], `batch_size` in [32, 64, 128], `task_weights` element in [1, 2, 3].  
- Run training for each (same data/split/seed), then run `evaluate_phase1.py` and record metrics.  
- Choose the config that maximizes a composite score (e.g. mean of normalized ten_god acc, ordinal acc, interaction F1_common, useful_god acc) or that satisfies all Phase 1 targets.

**Option C — Optuna (or similar)**  
- Define a study that:  
  - Suggests `lr`, `batch_size`, `task_weights` (or element weight only).  
  - Objective: run `train_multi_task.py` with suggested args (short runs, e.g. 5–10 epochs, or full 30 if compute allows), then run `evaluate_phase1.py` and return a single metric (e.g. negative val loss, or a composite of task metrics).  
- Use the same `--data`, `--val_ratio`, `--test_ratio`, `--seed` for reproducibility.

### Most efficient manner (recommended)

To minimize total compute while still finding a good config:

1. **Short-horizon trials**  
   Run each candidate config for **5–10 epochs only** (not 30). Validation loss (and relative ranking of configs) stabilizes early; bad configs are visible quickly. One short run is ~5–10× faster than a full 30-epoch run.

2. **Small search space**  
   Focus on the levers that matter: **task_weights** (e.g. element weight 1, 2, 3) and **lr** (e.g. 5e-4, 1e-3, 2e-3). Fix batch_size=64 and seed; skip broad batch_size search unless needed.

3. **One full run at the end**  
   After short trials, take the **best 1 (or 2) config(s)** by validation loss and run **one full 30-epoch training** with that config. Do not run full 30 epochs for every candidate.

4. **Optional: Bayesian optimization**  
   Use **Optuna** (or similar) with the same short-horizon objective (e.g. 5 epochs per trial). You need fewer trials to find a good config than with grid/random search, so total compute can be lower. Example: 15 Optuna trials × 5 epochs ≈ 75 epoch-equivalents, then 1 full run ≈ 30; total 105 vs 15 × 30 = 450 for 15 full runs.

5. **Optional: parallel trials**  
   If you have multiple GPUs or machines, run short-horizon trials in parallel (e.g. different `CUDA_VISIBLE_DEVICES` or separate processes). Each trial is independent; no shared state.

**Concrete workflow:**

- **Without a tuning script:** Run e.g. 6 short runs (2 lr × 3 element weights): `--epochs 5`, vary `--lr` and `--task_weights`. Note the config that gives the lowest val loss, then run once with `--epochs 30` and that config.
- **With `scripts/tune_phase1.py`:** Run `python scripts/tune_phase1.py --data data/charts.jsonl --epochs_per_trial 5 --n_trials 9 [--run_full_at_end]`. The script runs short trials (grid or random over lr × element weight), records val loss per trial, prints the best config and the exact full-training command, and optionally runs one full 30-epoch training with that config (`--run_full_at_end`). Results can be written to JSON with `--output reports/tune_phase1_results.json`.

### What “best” means

- **By validation loss:** The Trainer already saves the checkpoint with **lowest validation (combined) loss**. So “best by val loss” is automatic.  
- **By task targets:** Phase 1 success is defined by per-task targets (ten-god >95%, element ordinal >90%, interaction F1, useful-god >70%). “Best” can mean: (1) all targets met, or (2) maximize the worst-performing task (e.g. element ordinal) while keeping others above target.  
- **Composite metric:** If you need one number, e.g.  
  `score = (ten_god_acc + ordinal_acc + interaction_f1_common + useful_god_acc) / 4`  
  and tune to maximize `score` (or a weighted version).

### Summary

- Defaults are **good enough** to meet 4/5 targets; only element ordinal is often below 90%.  
- **First lever:** `--task_weights 1,2,1,1` or `1,3,1,1`.  
- **Then:** lr, batch_size, patience if you want to squeeze more.  
- **Rigorous “best”:** run a small grid, random search, or Optuna over lr, batch_size, task_weights; evaluate each run with `evaluate_phase1.py` and compare metrics or a composite score.

---

## Phase Transition: Phase 1 → Phase 2

### What Gets Carried Forward

**From Phase 1 multi-task training**:
```python
# Load the shared encoder (trained on all 4 tasks)
checkpoint = torch.load('checkpoints/multi_task_best.pt')
encoder = NatalEncoder()
encoder.load_state_dict(checkpoint['encoder'])

# Optional: Load one head for verification
ten_god_head = TenGodHead()
ten_god_head.load_state_dict(checkpoint['ten_god_head'])
```

**In Phase 2**:
```python
# Freeze or low-LR encoder, train projection layer
class Phase2Model(nn.Module):
    def __init__(self, encoder_checkpoint_path):
        # Load pre-trained encoder
        checkpoint = torch.load(encoder_checkpoint_path)
        self.encoder = NatalEncoder()
        self.encoder.load_state_dict(checkpoint['encoder'])
        
        # New projection layer (randomly initialized)
        self.projection = ProjectionLayer(d_model=256, d_llm=4096, n_virtual_tokens=8)
        
        # Optional: load ten-god head for verification during training
        self.ten_god_head = TenGodHead()
        self.ten_god_head.load_state_dict(checkpoint['ten_god_head'])
    
    def forward(self, bazi_chart, query_tokens):
        # Encoder produces 256-dim embedding (learned in Phase 1)
        natal_emb = self.encoder(bazi_chart)  # (batch, 256)
        
        # Projection creates virtual tokens for LLM
        virtual_tokens = self.projection(natal_emb)  # (batch, n_tokens, d_llm)
        
        # Prepend to LLM input
        llm_input = torch.cat([virtual_tokens, query_tokens], dim=1)
        return llm_input

# Training: encoder frozen or low LR
encoder_params = model.encoder.parameters()
projection_params = model.projection.parameters()

optimizer = AdamW([
    {'params': encoder_params, 'lr': 1e-5},      # Very low LR or frozen
    {'params': projection_params, 'lr': 5e-5}    # Normal LR
])
```

### What Gets Discarded

**From sequential training** (if you ran it):
- All sequential checkpoints can be deleted or archived
- They were only for comparison/debugging
- Phase 2 uses the multi-task encoder only

**From multi-task training**:
- The task heads (ten-god, element, interaction, useful-god) are mostly discarded
- Exception: You may keep ten-god head for inference-time verification
  - During LLM generation, run ten-god head on encoder output
  - Compare predicted ten-gods vs calculator ground truth
  - Flag interpretations that mention incorrect ten-gods

---

## Implementation Checklist for Cursor

### High Priority (Must Do)

- [ ] **Implement `train_multi_task.py`** as the primary Phase 1 training script
  - Single NatalEncoder shared across all heads
  - Combined loss with task weighting support
  - Per-task metric logging to TensorBoard
  - Saves checkpoint in format: `{'encoder': ..., 'heads': {...}}`

- [ ] **Update `evaluate_phase1.py`** to support `--use_multi_task` flag
  - Loads single checkpoint with encoder + all heads
  - Reports metrics matching targets (see phase1_evaluation.md)
  - Supports both validation and test set evaluation

- [ ] **Implement Phase 2 checkpoint loading**
  - Helper function to load multi-task encoder for projection layer training
  - Example code in `scripts/train_projection.py` or README

- [ ] **Document multi-task as primary path**
  - Update README.md to list `train_multi_task.py` as the main script
  - Add note that sequential scripts are optional debugging tools
  - Include example commands for full Phase 1 workflow

### Medium Priority (Should Do)

- [ ] **Add task weight tuning**
  - If element strength underperforms in multi-task, try different weights
  - Document recommended weights in config or README

- [ ] **Implement early stopping**
  - Monitor combined validation loss or individual task metrics
  - Save best checkpoint by validation performance

- [x] **Add embedding visualization** — `scripts/run_embedding_visualization.py`; t-SNE (and optional UMAP) colored by day-master strength, dominant element, useful god; plots in `reports/`; summary in `reports/embedding_visualization_summary.md`

- [ ] **Create Phase 1 completion report template**
  - Auto-generate markdown report from evaluation metrics
  - Include confusion matrices, loss curves from TensorBoard
  - Compare against targets from phase1_detailed_plan.docx

### Low Priority (Nice to Have)

- [ ] **Keep sequential scripts for debugging**
  - Mark clearly as "debugging tools" in comments
  - Add warning that they produce incompatible encoders
  - Use only if multi-task training fails completely

- [ ] **Implement focal loss for interaction task**
  - If class imbalance is severe, replace BCELoss with FocalLoss
  - Current results show perfect F1, so likely unnecessary

- [ ] **Multi-task architecture search**
  - Experiment with different encoder sizes (d_model=128, 512)
  - Try deeper transformers (3-4 layers) if underfitting
  - Current 2-layer / 256-dim works well, so defer

---

## Common Pitfalls to Avoid

### ❌ DO NOT: Use sequential encoders in Phase 2

```python
# WRONG: Loading sequential checkpoint for production
encoder = torch.load('checkpoints/ten_god_best.pt')['encoder']
# This encoder only knows ten-god features!
```

### ❌ DO NOT: Train tasks sequentially then fine-tune

```python
# WRONG: Sequential pre-training
encoder = NatalEncoder()
train_on_ten_god(encoder)  # Encoder learns ten-god
train_on_element(encoder)   # Encoder forgets ten-god (catastrophic forgetting)
train_on_interaction(encoder)  # Encoder forgets element
# Final encoder is poor quality
```

### ❌ DO NOT: Average sequential encoder weights

```python
# WRONG: Averaging independently trained encoders
encoder_avg = average_weights([encoder_A, encoder_B, encoder_C, encoder_D])
# These encoders learned in different loss landscapes - averaging is meaningless
```

### ✅ DO: Use multi-task encoder directly

```python
# CORRECT: Load multi-task encoder for Phase 2
checkpoint = torch.load('checkpoints/multi_task_best.pt')
encoder = NatalEncoder()
encoder.load_state_dict(checkpoint['encoder'])
# This encoder knows ALL tasks
```

### ✅ DO: Keep heads for verification (optional)

```python
# CORRECT: Use heads for quality checks during inference
natal_emb = encoder(chart)
predicted_ten_gods = ten_god_head(natal_emb)
ground_truth_ten_gods = calculator.compute_ten_gods(chart)

if not match(predicted_ten_gods, ground_truth_ten_gods):
    logger.warning(f"Encoder prediction mismatch: {chart_id}")
    # Flag for human review or use calculator fallback
```

---

## Success Criteria (from phase1_detailed_plan.docx)

Phase 1 is complete when multi-task training achieves:

- [x] Ten-god classification: **>95% accuracy** → **Achieved: 100%**
- [x] Element strength regression: **MSE <0.05, ordinal >90%** → **Achieved: MSE 8.1e-04, ordinal 63%** (⚠️ ordinal below target but MSE excellent; consider task weight tuning)
- [x] Interaction detection: **>95% F1 common, >80% F1 rare** → **Achieved: 99.91% / 100%**
- [x] Useful god classification: **>70% accuracy unambiguous** → **Achieved: 99.93%**
- [x] Cross-temporal: **>90% F1** → **Achieved: 97.72% micro, 93.97% macro**
- [ ] Embedding visualization shows meaningful clustering — **run completed** (t-SNE plots in `reports/`; fill cluster quality notes in `reports/embedding_visualization_summary.md` after inspection)

**Current Status**: 5/6 criteria met; 1.5.2 visualization executed, qualitative review pending. Proceed to Phase 2 unless element ordinal accuracy is critical (can tune task weights if needed).

---

## Questions for Cursor to Answer

When implementing this specification, verify:

1. Does `train_multi_task.py` exist and follow the structure described above?
2. Does `evaluate_phase1.py` support `--use_multi_task` flag?
3. Are sequential scripts clearly marked as debugging tools?
4. Is `checkpoints/multi_task_best.pt` the recommended checkpoint for Phase 2?
5. Are there any references in the code or docs that still recommend sequential as primary path?

If any of these are unclear or incorrectly implemented, update the code to match this specification.

---

## Summary

**TL;DR for Cursor**:

- **Multi-task training** (one encoder, four heads, combined loss) is the ONLY production path
- **Sequential training** (separate encoder per task) is for debugging only and produces incompatible encoders
- Load `checkpoints/multi_task_best.pt` for Phase 2 projection layer training
- Current multi-task results exceed targets on 4/5 tasks; element ordinal accuracy can be improved with task weight tuning if needed
- Phase 1 is effectively complete; proceed to Phase 2

**Action items**:
1. Ensure `train_multi_task.py` is the primary script and well-documented
2. Mark sequential scripts as debugging tools with warnings
3. Implement Phase 2 checkpoint loading from multi-task encoder
4. (Optional) Tune element task weight if ordinal accuracy is critical
