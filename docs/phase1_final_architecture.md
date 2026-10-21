# Phase 1 Final Architecture

This document records the **final** Phase 1 encoder architecture and training strategy as implemented and evaluated. It is the single reference for what was actually built and why, including deviations from the original plan.

**Date:** 2026-03-08  
**Status:** Authoritative for Phase 1 production setup.

---

## 1. Encoder: 3-layer (deviation from original 2-layer plan)

### Original plan

The Phase 1 implementation plan (`plans/phase1_implementation_plan.md`) specified:

- **NatalEncoder:** 2-layer Transformer (4 heads, d_model=256, d_ff=1024) → CLS output (batch, 256).

### Final architecture

- **NatalEncoder:** **3-layer** Transformer, same d_model=256, d_ff=1024, 4 attention heads.
- **Rationale:** With multi-task training, element **ordinal accuracy** was the bottleneck (target >90%). Adding a third layer improved ordinal (81.45% vs 77.70% for 2-layer) without hurting ten-god (100% in both). The 3-layer encoder was chosen for best element/ordinal performance; ten-god already reached 100% with 2-layer multi-task, so the gain was not driven by depth for ten-god.
- **Deviation:** Plan said 2 layers; we use **3 layers** for the production checkpoint (`checkpoints/multi_task_3layer_1_6_1_1.pt`).

**Reference:** `docs/phase1_best_checkpoint.md` §2.6, §4.1; `plans/phase1_implementation_plan.md` §1.1.3.

---

## 2. Task weights (1, 6, 1, 1) and rationale

### Final weights

| Task        | Weight | Order in `--task_weights` |
|------------|--------|---------------------------|
| Ten-god     | 1      | first                     |
| Element     | **6**  | second                    |
| Interaction | 1      | third                     |
| Useful god  | 1      | fourth                    |

Example: `--task_weights 1,6,1,1`.

### Rationale: balancing loss magnitudes

- **Problem:** With equal weights (1,1,1,1), the **element** task (MSE + ordinal auxiliary) has smaller raw loss magnitude than the classification tasks (cross-entropy). The combined gradient was dominated by ten-god, interaction, and useful-god; element ordinal lagged (~63–77% vs target >90%).
- **Decision:** Upweight the element task so its loss contributes proportionally more to the total loss and the encoder gets stronger gradients for element/ordinal. Weights were tuned (e.g. via `tune_phase1.py`) with element weight in {1, 2, 3, 4, 5, 6}; **6** gave the best ordinal in our runs (81.45%) without degrading the other tasks.
- **Note:** This is **loss-magnitude balancing**: we do not change the loss functions themselves; we scale the element loss by 6 so it is not drowned out by the other tasks. Ten-god remains at weight 1 and still reaches 100% accuracy.

**Reference:** `docs/phase1_best_checkpoint.md` §2.3, §2.5, §3; `docs/design_decisions.md` §15 (save-by-composite when using unequal weights).

---

## 3. Multi-task training as the primary strategy

### Decision

- **Primary (production) path:** **Multi-task training only.** One shared `NatalEncoder`, four task heads (ten-god, element strength, interaction, useful god). One forward pass per batch; combined loss = weighted sum of per-task losses; single checkpoint with encoder + all heads.
- **Script:** `scripts/train_multi_task.py`. Checkpoint: `checkpoints/multi_task_3layer_1_6_1_1.pt` (or `--save_name` as chosen).
- **Sequential training** (one script per task, one encoder per task) is **not** the production path. It is for **debugging only**: if multi-task fails, sequential runs isolate which task is broken. Sequential produces four incompatible encoders and cannot yield a single representation for Phase 2.

### Why multi-task

- Evaluation showed multi-task: ten-god **100%**, useful-god **99.93%**, interaction and element much better than sequential.
- Sequential ten-god alone reached ~89%; sequential useful-god collapsed (~23%). A single shared encoder trained with all four tasks learns a better representation and meets or exceeds targets on all natal tasks.

**Reference:** `docs/phase1_training_strategy_clarification.md` (Executive Summary, Multi-Task Training, Sequential as debugging only); `docs/phase1_best_checkpoint.md` §2.1.

---

## 4. Cross-temporal trained separately

### Decision

- **Cross-temporal** (natal × 大运/流年/流月 interactions) is **trained separately** from the natal multi-task model.
- **Script:** `scripts/train_cross_temporal.py`.
- **Model:** Uses `DynamicEncoder` (and natal encoder for context), not the same training loop as the 4-task natal model. Dataset is filtered to charts with temporal data (e.g. 流年).
- **Checkpoint:** Separate checkpoint (e.g. `checkpoints/sequential/cross_temporal` or project convention); **not** bundled in `multi_task_3layer_1_6_1_1.pt`.

### Rationale

- Temporal data is a different distribution (charts with 流年/流月 overlays) and a different input (concat(natal_emb, dynamic_emb)).
- Mixing temporal and natal-only examples in one multi-task batch is awkward; keeping cross-temporal as a separate run keeps data and objectives clear.
- Phase 1 “best” checkpoint refers to the **natal** multi-task checkpoint; cross-temporal is evaluated and used independently.

**Reference:** `docs/phase1_training_strategy_clarification.md` § Script Responsibilities (`train_cross_temporal.py`), § Checkpoint structure; `docs/phase1_best_checkpoint.md` §1 (cross-temporal separate).

---

## 5. Summary table

| Aspect              | Original plan / default | Final (production)                          |
|---------------------|-------------------------|---------------------------------------------|
| NatalEncoder depth  | 2 layers                | **3 layers**                                |
| Task weights        | 1,1,1,1                 | **1, 6, 1, 1** (element = 6)                |
| Training strategy   | Sequential + optional multi-task | **Multi-task primary**; sequential debug only |
| Cross-temporal      | —                       | **Trained separately** (`train_cross_temporal.py`) |
| Best checkpoint     | —                       | `checkpoints/multi_task_3layer_1_6_1_1.pt`   |

---

## 6. Related docs

- **Best checkpoint process and metrics:** `docs/phase1_best_checkpoint.md`
- **Training strategy and script usage:** `docs/phase1_training_strategy_clarification.md`
- **Design choices (losses, save-by-composite, etc.):** `docs/design_decisions.md`
- **Original implementation plan:** `plans/phase1_implementation_plan.md`
