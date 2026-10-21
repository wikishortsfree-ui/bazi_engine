# Tasks module audit (bazi_model/tasks)

Review of `bazi_model/tasks/` for potential issues and optimizations.

---

## 1. Critical: CrossTemporalHead missing training API

**Issue:** `CrossTemporalHead` has no `loss()` or `forward_loss()`, while tests expect them and a unified training loop would need them.

- **Tests:** `tests/test_cross_temporal_head.py` imports `DYNAMIC_INTERACTION_DIM`, `NATAL_DIM`, `DYNAMIC_DIM` and calls `head.forward_loss(batch_dict, natal_emb, dynamic_emb)`. The module only exports `D_NATAL`, `D_DYNAMIC`, `N_DYNAMIC_DIMS` and has no `forward_loss`/`loss`.
- **Effect:** When PyTorch is installed, tests fail on import (missing names) and on execution (missing methods). One test also expects `hidden_dim=None` and `head.proj`, which do not exist in the current single-linear implementation.

**Recommendation:** Add `loss(logits, target)` (BCE with logits) and `forward_loss(batch, natal_emb, dynamic_emb)` to `CrossTemporalHead`. Export alias constants (`NATAL_DIM`, `DYNAMIC_DIM`, `DYNAMIC_INTERACTION_DIM`) for test compatibility. Update tests to match current design (single linear, no `hidden_dim`/`proj`).

---

## 2. Inconsistent `forward_loss` return shape/signature — FIXED

**Issue:** Return type and order of `forward_loss` differed across heads; a generic trainer had to branch on head type.

**Resolution:** All heads now use the same convention: `(loss, pred)` — loss first, then prediction (logits or probs). Trainer can always do `loss, pred = head.forward_loss(...)` and call `loss.backward()`; `pred` is available for metrics when needed.

| Head                 | Return value        |
|----------------------|--------------------|
| TenGodHead           | `(loss, logits)`   |
| ElementStrengthHead  | `(loss, probs)`    |
| UsefulGodHead        | `(loss, logits)`   |
| InteractionHead      | `(loss, logits)`   |
| CrossTemporalHead    | `(loss, logits)`   |

---

## 3. Inconsistent `forward()` signature (**kwargs)

**Issue:** Some heads accept `**kwargs`, others do not. A trainer that calls `head.forward(natal_emb, dynamic_emb=dynamic_emb)` would break on heads that don’t accept `**kwargs`.

- With `**kwargs`: `ElementStrengthHead`, `UsefulGodHead`, `CrossTemporalHead`.
- Without: `TenGodHead`, `InteractionHead`.

**Recommendation:** Add `**kwargs: object` to `forward()` in `TenGodHead` and `InteractionHead` so a generic caller can pass extra arguments (e.g. `dynamic_emb`) without errors.

---

## 4. Duplicated dimension constants

**Issue:** `D_NATAL = 256` (and similar) are defined in multiple task files and in encoders. Changing encoder output size would require updates in several places.

**Recommendation (low priority):** Centralize in one place (e.g. `bazi_model/encoders/natal_encoder.py` or a small `bazi_model/_constants.py`) and have tasks import from there. Reduces drift and magic numbers.

---

## 5. UsefulGodHead.loss in-place on clone

**Issue:** `target = useful_god_id.squeeze(-1).clone(); target[target == -1] = AMBIGUOUS_CLASS_ID` mutates the clone. Correct and safe, but slightly less readable than a pure expression.

**Recommendation (optional):** Use `torch.where(target == -1, torch.full_like(target, AMBIGUOUS_CLASS_ID), target)` (or similar) to avoid in-place mutation. Purely stylistic.

---

## 6. InteractionHead.forward_loss batch key

**Issue:** `forward_loss(batch, natal_emb, ...)` expects `batch["interaction_vector"]`. Dataset/collate may use the same key; ensure naming is consistent everywhere (e.g. `interaction_vector` vs `interaction_vector` in chart_dataset).

**Status:** No change needed if dataset and collate already use `interaction_vector`; just something to keep consistent when adding new datasets.

---

## 7. Pos_weight device

**Issue:** In `InteractionHead`, `pos_weight` is registered as a buffer. When the model is moved to GPU, buffers move with it. No bug if the module is always used on one device; ensure any code that constructs `pos_weight` from dataset stats does not leave it on CPU when the model is on GPU.

**Status:** `register_buffer` is correct; no change unless you construct `pos_weight` outside the module and pass a tensor that might be on the wrong device.

---

## Summary

- **Must fix:** CrossTemporalHead — add `loss`, `forward_loss`, and export aliases; align tests with current API.
- **Should consider:** Unify `forward_loss` return convention and add `**kwargs` to all `forward()` methods for a generic trainer.
- **Nice to have:** Centralize dimension constants; optional cleanup in UsefulGodHead.loss and doc updates.
