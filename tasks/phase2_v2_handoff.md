# Phase 2 v2: Cross-Interaction Features + Multi-Epoch Training

## Status: COMPLETE — Results mixed (perplexity improved, factual consistency regressed)

## v2 Results vs v1 Baseline

| Metric | v1 Baseline | v2 Result | Change |
|--------|-------------|-----------|--------|
| Perplexity | 7.56 | **6.22** | −17.7% (better) |
| BLEU | 0.09 | **0.11** | +25.8% (better) |
| ROUGE-L | 0.21 | **0.23** | +11.7% (better) |
| Factual consistency | 44% | **28%** | **−16pp (worse)** |
| Val loss (best) | 2.0210 | **1.8254** | −9.7% (better) |

**Conclusion**: Cross-interaction features + 3 epochs improved all language metrics but factual grounding degraded. The frozen LLM generates more fluent hallucinations. Projection-only training has hit its ceiling for factual accuracy. **LoRA fine-tuning is the necessary next step.**

Full report: `reports/phase2_baseline_eval.md`

## What was done

### Session 1 (previous): v1 baseline + cross-interaction encoder
- Evaluated v1 baseline: perplexity 7.56, factual 44%
- Identified root cause: phantom ten-gods from dropped cross-interaction data
- Built `CrossInteractionEncoderMLX` (27→128→256, trainable)
- Wired into dataset, BaziLLM, train, and eval scripts (d_in: 512→768)

### Session 2 (this session): v2 training + evaluation
- **Added `--resume` flag** to `train_phase2.py`: full state restoration (model weights + optimizer moments + step counter + LR schedule position)
- **Trained v2** for 3 full epochs (3,300 steps) with pause/resume across multiple sessions
- Val loss curve: monotonically decreasing from 2.67 (step 100) → 1.83 (step 3300), no overfitting
- **Ran evaluation**: perplexity improved but factual consistency dropped from 44% → 28%

## What needs to happen next

### Priority 1: LoRA fine-tuning
The frozen LLM cannot learn to extract specific chart facts from virtual tokens via projection-only training. LoRA adapters on the Qwen attention layers would allow the LLM to learn chart-aware attention patterns. This is the most promising path to improving factual consistency.

### Priority 2: Hybrid prompt approach
Inject structured chart facts (ten-gods, interactions, elemental strengths) as explicit text in the prompt alongside virtual tokens. This gives the LLM direct access to chart data without needing to decode it from embeddings.

### Other options
- Constrained decoding / factual-consistency reward (RLHF/DPO with `factual_check.py`)
- More virtual tokens (16–32) for richer chart encoding
- Do NOT continue projection-only training — diminishing returns

## Architecture summary

```
Chart JSON
  ├─ NatalEncoderMLX (frozen, 256d) ────────────────┐
  ├─ DynamicEncoderMLX (frozen, 256d) ───────────────┤ concat → 768d
  └─ CrossInteractionEncoderMLX (trainable, 256d) ───┘     │
                                                    ProjectionLayer (trainable)
                                                    768 → 8 × 3584
                                                            │
                                                    8 virtual tokens
                                                            │
                                                    ┌───────┴───────┐
                                                    │ [virt] [query] │
                                                    └───────────────┘
                                                            │
                                                    Qwen 2.5 7B 4-bit (frozen)
                                                            │
                                                        logits
```

## Key files reference
- `bazi_model/encoders/cross_interaction_encoder_mlx.py` — cross-interaction encoder
- `bazi_model/data/phase2_dataset.py` — `_cross_interaction_features()`, `CROSS_FEATURE_DIM = 27`
- `bazi_model/model/bazi_llm.py` — `cross_interaction_encoder` param, `cross_features` in `__call__`
- `scripts/train_phase2.py` — training with cross features + `--resume` flag
- `scripts/evaluate_phase2.py` — evaluation with cross features
- `scripts/factual_check.py` — factual consistency checker
- `reports/phase2_baseline_eval.md` — v1 vs v2 eval report
- `checkpoints/phase2_baseline/checkpoint_best.npz` — v2 best (step 3300, val_loss 1.8254)
- `checkpoints/phase2_baseline/checkpoint_best_v1_512d.npz` — v1 backup (step 1100, val_loss 2.0210)

## Training log summary (v2)

| Step | Val Loss | Session |
|------|----------|---------|
| 100 | 2.6689 | Session 1: fresh start |
| 500 | 2.1081 | |
| 800 | 1.9950 | First beat of v1 (2.0210) |
| 900 | 1.9735 | Paused |
| 1000 | 1.9512 | Session 2: resumed from 900 |
| 1500 | 1.8883 | Paused |
| 2200 | 1.8462 | Session 3: resumed from 1500, paused |
| 2900 | 1.8297 | Session 4: resumed from 2200, paused |
| 3300 | 1.8254 | Session 5: resumed from 2900, completed epoch 3 |

## Lessons / gotchas
- `mlx_lm` v0.31.0: `generate_step` is not exported at top level; access via `mlx_lm.stream_generate.__globals__["generate_step"]`
- `generate_step` yields tokens that may be `int` (not `mx.array`); use `int(tok.item()) if hasattr(tok, "item") else int(tok)`
- Tokenizer `eos_token_ids` can be `int` not list; check `isinstance` before iterating
- The dynamic encoder gets called even for natal-only tasks (with null inputs returning zeros) — this is by design
- `_batch_to_mx` in both train and eval scripts must forward the `cross_features` key
- **Projection-only training cannot solve factual grounding** — richer embeddings improve fluency but the frozen LLM still hallucinates ten-gods because it never learns to attend to specific virtual token content
- **Resume feature is essential** for multi-day training on shared hardware — `--resume` restores model + optimizer + step counter + LR schedule
