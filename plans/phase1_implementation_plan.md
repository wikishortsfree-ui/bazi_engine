# Phase 1: Encoder Pre-training — Implementation Plan

**Source:** `phase1_detailed_plan.docx`  
**Aligned with:** Phase 0 ([bazi_engine](bazi_engine/), [task_0.8_implementation_plan.md](task_0.8_implementation_plan.md))  
**Estimated duration:** 22 days (~3 weeks)  
**Compute:** CPU or single GPU (e.g. RTX 3060/4060)

---

## Overview

Phase 1 trains the **natal** and **dynamic** encoders in isolation on supervised bāzì tasks before LLM integration (Phase 2). Pre-training gives encoders grounded, verifiable representations and avoids the LLM compensating for poor encoder outputs.

**Goals:**
- Train `natal_emb` to capture the full structural fingerprint of a birth chart.
- Train `dynamic_emb` for temporal overlays and cross-natal–temporal interactions.
- **Targets:** >95% ten-god accuracy; <0.05 MSE elemental strength; >95% F1 common interactions, >80% rare; >70% useful god on unambiguous; >90% F1 cross-temporal; meaningful embedding clusters.

**Out of scope:** LLM integration (Phase 2), RAG (Phase 5).

---

## Implementation todos

### 1.1 Encoder architecture (7 days)

- [ ] **1.1.1 Embedding layer** — `bazi_model/encoders/embeddings.py`
  - [ ] Add `PillarEmbedding`: `nn.Embedding` for stem(10), branch(12), ten_god(10), element(5), polarity(2), position(4); `d_emb` 32 or 64.
  - [ ] Per-pillar concat → vector length `5*d_emb`. Reuse [bazi_engine/encodings.py](bazi_engine/encodings.py) for vocab.
  - [ ] Unit test: output shape `batch × 4 × (5*d_emb)`.
- [ ] **1.1.2 Interaction feature encoding** — `bazi_model/encoders/interaction_features.py`
  - [ ] Add `InteractionFeatureEncoder`: binary (e.g. 64-dim), categorical (embedding), continuous (elemental_strengths, day_master_strength).
  - [ ] Single vector `d_interaction` ~80–120; document feature index mapping.
  - [ ] Unit test: output shape `batch × d_interaction`.
- [ ] **1.1.3 Natal encoder** — `bazi_model/encoders/natal_encoder.py`
  - [ ] Add `NatalEncoder`: [CLS] + 4 pillar vectors + interaction vector → 2-layer transformer (4 heads, d_model=256, d_ff=1024), dropout 0.1 → CLS output (batch, 256).
  - [ ] Forward test and ~2M parameter count check.
- [ ] **1.1.4 Dynamic encoder** — `bazi_model/encoders/dynamic_encoder.py`
  - [ ] Add `DynamicEncoder`: 4 temporal pillar slots + cross_interaction_vector; attention masking for absent pillars; learned `null_dynamic_emb` when no temporal context.
  - [ ] Tests: all present, only 大运+流年, natal-only, masking correctness.

### 1.2 Training data (3 days)

- [ ] **1.2.1 Random chart strategy** — `scripts/generate_training_charts.py`
  - [ ] Sample dates 1900–2025, times 00:00–23:59; optional flowing_year, flowing_month. Output `dates.csv` (birth_date, birth_time, timezone_offset, lat, lng, optional flowing_year, flowing_month).
  - [ ] Optional: diversity notebook (day master strength, dominant element, interaction types).
- [ ] **1.2.2 Batch chart computation** — `scripts/compute_charts.py`
  - [ ] Read `dates.csv` → multiprocessing + `BaziCalculator.from_datetime(dt, lat, lng)` → `chart.to_json(indent=None)` per line → `data/charts.jsonl` (100k+ lines).
  - [ ] Spot-check vs Phase 0; handle errors.
- [ ] **1.2.3 PyTorch Dataset** — `bazi_model/data/chart_dataset.py`
  - [ ] Dataset from `charts.jsonl`: parse → BaziChart (or from JSON) → `to_feature_arrays()` → tensors.
  - [ ] DataLoader with batching; filter for temporal subset when training cross-temporal head.

### 1.3 Supervised task heads (8 days)

- [ ] **1.3.1 Ten-god head** — `bazi_model/tasks/ten_god_head.py`
  - [ ] TenGodHead: natal_emb → ten-god logits (7×10 or 4×10 per doc). CrossEntropy; target >95% accuracy.
- [ ] **1.3.2 Element strength head** — `bazi_model/tasks/element_regressor.py`
  - [ ] ElementStrengthHead: natal_emb → (batch, 5) softmax. MSE or KLDiv; target MSE <0.05, ordinal acc >90%.
- [ ] **1.3.3 Interaction detector** — `bazi_model/tasks/interaction_detector.py`
  - [ ] InteractionHead: natal_emb → (batch, 64) sigmoid. BCEWithLogitsLoss + pos_weight for rare types; target >95% / >80% F1.
- [ ] **1.3.4 Useful god classifier** — `bazi_model/tasks/useful_god_classifier.py`
  - [ ] UsefulGodHead: natal_emb → 10 or 5 classes (see Clarifications). Target >70% on unambiguous.
- [ ] **1.3.5 Cross-temporal detector** — `bazi_model/tasks/cross_temporal_detector.py`
  - [ ] CrossTemporalHead: concat(natal_emb, dynamic_emb) → cross-interaction logits (16 or ~40 dims). Target >90% F1; train on temporal subset only.

### 1.4 Training infrastructure (2 days)

- [ ] **1.4.1 Trainer** — `bazi_model/training/trainer.py`
  - [ ] Train/val split (e.g. 90/10), AdamW weight_decay=0.01, lr 1e-3, cosine or ReduceLROnPlateau, early stopping (e.g. 5 epochs), best checkpoint, TensorBoard.
- [ ] **1.4.2 Training strategy**
  - [ ] Sequential training scripts (one per task).
  - [ ] Optional: multi-task script + ablation (sequential vs multi-task).

### 1.5 Evaluation and analysis (2 days)

**Detailed plan:** [phase1_5_evaluation_plan.md](phase1_5_evaluation_plan.md)

- [ ] **1.5.1 Quantitative** — `reports/phase1_evaluation.md`
  - [ ] Metrics table (accuracy, MSE/MAE/ordinal, per-class F1, confusion matrices); short error analysis.
- [ ] **1.5.2 Embedding visualization** — `notebooks/embedding_visualization.ipynb`
  - [ ] Extract natal_emb (1k–5k charts), t-SNE or UMAP, color by strength / dominant element / useful god; cluster quality notes.

---

## Package layout

```
bazi_model/
├── __init__.py
├── encoders/
│   ├── __init__.py
│   ├── embeddings.py           # PillarEmbedding
│   ├── interaction_features.py # InteractionFeatureEncoder
│   ├── natal_encoder.py        # NatalEncoder
│   └── dynamic_encoder.py      # DynamicEncoder
├── tasks/
│   ├── __init__.py
│   ├── ten_god_head.py
│   ├── element_regressor.py
│   ├── interaction_detector.py
│   ├── useful_god_classifier.py
│   └── cross_temporal_detector.py
├── data/
│   ├── __init__.py
│   └── chart_dataset.py        # PyTorch Dataset from charts.jsonl
└── training/
    ├── __init__.py
    └── trainer.py
```

**Supporting:** `scripts/generate_training_charts.py`, `scripts/compute_charts.py`, `data/charts.jsonl`, `notebooks/embedding_visualization.ipynb`, `reports/phase1_evaluation.md`, `checkpoints/`, `runs/`.

---

## Task summary (heads)

| Task        | Head                 | Input              | Output        | Loss              | Target              |
|------------|----------------------|--------------------|---------------|-------------------|---------------------|
| 1.3.1      | TenGodHead           | natal_emb (256)    | 7×10 or 4×10  | CrossEntropy       | >95% acc            |
| 1.3.2      | ElementStrengthHead  | natal_emb          | (5) softmax   | MSE / KLDiv        | MSE <0.05, ord >90% |
| 1.3.3      | InteractionHead      | natal_emb          | (64) sigmoid  | BCE + pos_weight   | F1 >95% / >80%      |
| 1.3.4      | UsefulGodHead        | natal_emb          | 10 or 5       | CrossEntropy       | >70% unamb.         |
| 1.3.5      | CrossTemporalHead    | natal+dynamic 512 | 66            | BCE                | >90% F1             |

---

## Dependencies (Phase 0)

- **BaziChart** [models.py](bazi_engine/models.py): `to_feature_arrays()` (stem_ids, branch_ids, interaction_vector 64, dynamic_interaction_vector 66, elemental_strengths, day_master_strength, useful_god_id, …), `to_json()`. Layout: [0:16] natal×大运, [16:32] natal×流年, [32:48] natal×流月, [48:52] 大运×流年, [52:56] 大运×流月, [56:60] 流年×流月, [60:65] elemental_delta, [65] useful_god_active.
- **Encodings** [encodings.py](bazi_engine/encodings.py): stem/branch/element/ten_god/useful_god IDs (frozen after Phase 1 starts).
- **Batch script** [batch_generate_charts.py](scripts/batch_generate_charts.py): CSV → BaziChart → JSONL (extend or reuse for dates.csv → charts.jsonl).

---

## Risks and mitigations

| Risk                         | Mitigation                                                                 |
|-----------------------------|----------------------------------------------------------------------------|
| Encoder not learning        | Check labels vs Phase 0; projection into transformer; lr range 1e-4–1e-2.  |
| Low F1 on rare interactions | pos_weight / focal loss; oversample or bias chart gen; class-balanced.   |
| Useful god ~40–50%          | Verify Phase 0 logic; try strong vs weak first; defer to Phase 3 if stuck. |
| Dynamic masking wrong       | Unit test mask; compare dynamic_emb with/without temporal; full-temporal ablation. |

---

## Clarifications to confirm

1. **Useful god head (resolved):** Phase 0 [BaziChart.to_feature_arrays()](bazi_engine/models.py) outputs `useful_god_id` from [encodings.USEFUL_GOD_TO_ID](bazi_engine/encodings.py): **5 element-based categories** (比劫=0, 食伤=1, 财=2, 官杀=3, 印=4) or -1 if ambiguous. So **vocab = 6** (0–4 + padding for -1) is correct; [interaction_features.py](bazi_model/encoders/interaction_features.py) matches this. For a 10 ten-god classifier, Phase 0 would need to expose ten-god useful god and the engine contract would change.
2. **Cross-temporal head size (resolved):** `dynamic_interaction_vector` is **66 dims** (60 binary + 5 elemental_delta + 1 useful_god_active). See architecture spec Section 2.3.1.
3. **Package location:** Use top-level `bazi_model/` next to `bazi_engine/`, or e.g. `bazi_engine/model/`?
4. **Ten-god head format:** Doc uses “ten_god_sequence length 7” (non–day-master). Output 7×10 logits or 4×10 (one per pillar stem)?

---

## Completion criteria (from docx)

Phase 1 is complete when:

- Ten-god classification >95% accuracy (validation).
- Elemental strength MSE <0.05, ordinal accuracy >90%.
- Interaction detection >95% F1 common, >80% F1 rare.
- Useful god >70% accuracy on unambiguous charts (defer acceptable if not reached).
- Cross-temporal interaction >90% F1.
- Embedding visualization shows meaningful clustering by chart type.
- Code documented; checkpoints saved; evaluation report written.

Then proceed to Phase 2 (projection layer + LLM integration).
