# Phase 1 Integration Checklist (pre–Task 1.2)

Resolved items before proceeding to Task 1.2 (training data).

## 1. Phase 0 calculator contract

**Source:** `BaziChart.to_feature_arrays()` in `bazi_engine/models.py`.  
**Tests:** `tests/integration/test_phase0_feature_contract.py`, `tests/test_to_feature_arrays_natal.py`, `tests/test_to_feature_arrays_dynamic.py`.

### Natal (all charts)

| Key | Shape | Notes |
|-----|--------|------|
| stem_ids | (4,) | int64 |
| branch_ids | (4,) | int64 |
| ten_god_stem_ids | (4,) | 0–10 (日主=10 for day stem); no -1 for day master |
| stem_elements | (4,) | int64 |
| stem_polarities | (4,) | int64 |
| pillar_positions | (4,) | int64 [0,1,2,3] |
| branch_elements | (4,) | int64 |
| branch_polarities | (4,) | int64 |
| ten_god_branch_ids | (12,) | int64; -1 for missing hidden-stem slot |
| interaction_vector | **(64)** | float64; spec in `docs/binary_vector_and_positions_spec.md` |
| elemental_strengths | (5,) | float64 |
| day_master_strength | (1,) | float64 |
| useful_god_id | (1,) | int64; 0–4 or -1 |

Note: Checklist previously said "interaction_vector (62)"; the actual contract is **64** dimensions.

### Dynamic (when `temporal_context` and `cross_interactions` are set)

| Key | Shape | Notes |
|-----|--------|------|
| dayun_stem_id | (1,) | -1 if absent |
| dayun_branch_id | (1,) | -1 if absent |
| liunian_stem_id | (1,) | -1 if absent |
| liunian_branch_id | (1,) | -1 if absent |
| liuyue_stem_id | (1,) | -1 if absent |
| liuyue_branch_id | (1,) | -1 if absent |
| pillar_presence_mask | (3,) | [dayun, liunian, liuyue]; 1=present, 0=absent |
| dynamic_interaction_vector | **(66)** | 60 binary + 5 elemental_delta + 1 useful_god_active |

---

## 2. Useful god vocabulary

**Resolved:** **6-class** (element-based + unknown).

- `encodings.USEFUL_GOD_TO_ID`: 比劫=0, 食伤=1, 财=2, 官杀=3, 印=4.
- `useful_god_id` in `to_feature_arrays()`: 0–4 or **-1** if ambiguous.
- Encoder: `InteractionFeatureEncoder` uses `USEFUL_GOD_VOCAB = 6` with `padding_idx=5` for -1.

An 11-class (ten gods + unknown) useful god would require a Phase 0 contract change (engine would need to expose ten-god useful god).

---

## 3. Dynamic interaction dimension

**Resolved:** **66** (`DYNAMIC_INTERACTION_DIM = 66`).

- Phase 0 `_dynamic_interaction_vector()` returns 66 dims: 60 binary (natal×大运/流年/流月, 大运×流年/流月, 流年×流月) + 5 elemental_delta + 1 useful_god_active.
- `ChartDataset` and `DynamicEncoder` use 66.

---

## 4. Attention masking (DynamicEncoder)

**Decision:** **Semantic masking** (current): absent temporal slots are replaced by a learned `null_dynamic_emb`; the model always sees 4 tokens (3 slots + cross). No `key_padding_mask` is passed to the transformer.

- **Verification:** `test_dynamic_encoder_masking_correctness` ensures that changing stem/branch IDs for an absent slot does not change the output (no leakage).
- **Optional later:** Add `key_padding_mask` so masked positions are excluded from attention (hard masking). Not required for Task 1.1.

---

## 5. Unit tests (Task 1.1)

| Check | Test | Status |
|-------|------|--------|
| PillarEmbedding output shape (batch, 4, 6*d_emb) | `test_pillar_embedding_output_shape` | ✅ |
| InteractionFeatureEncoder output (batch, d_interaction) | `test_interaction_feature_encoder_*` | ✅ |
| NatalEncoder random input → (batch, 256) | `test_natal_encoder_output_shape` | ✅ |
| NatalEncoder parameter count ~2M | `test_natal_encoder_parameter_count` (1.5M–2.5M) | ✅ |
| DynamicEncoder variable pillar presence → (batch, 256) | `test_dynamic_encoder_*_present`, `*_only` | ✅ |
| DynamicEncoder parameter count ~2M | `test_dynamic_encoder_parameter_count` (1.5M–2.5M) | ✅ |
| DynamicEncoder masking: absent slots don’t affect output | `test_dynamic_encoder_masking_correctness` | ✅ |

---

## 6. Running the contract and encoder tests

```bash
pytest tests/integration/test_phase0_feature_contract.py -v
pytest tests/test_pillar_embedding.py tests/test_interaction_feature_encoder.py tests/test_natal_encoder.py tests/test_dynamic_encoder.py -v
```
