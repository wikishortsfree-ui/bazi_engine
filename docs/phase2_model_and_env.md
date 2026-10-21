# Phase 2: Model and Environment (Task 2.1)

This document records Phase 2 **environment and model preparation**: MLX setup, Qwen2.5-7B-Instruct conversion, and constants used by the projection layer and BaziLLM.

**Reference:** Phase 2 Task 2.1 implementation plan (environment and model preparation).

---

## 1. LLM: Qwen2.5-7B-Instruct

### Hugging Face path

- **Model:** Qwen2.5-7B-Instruct  
- **HF path:** `Qwen/Qwen2.5-7B-Instruct`

Use this path when downloading or converting the model (e.g. with `huggingface-hub` or `mlx_lm.convert`).

### LLM hidden size (constant)

- **`d_llm = 3584`** — hidden size of Qwen2.5-7B.

Use this value when defining the **projection layer** (encoder output → LLM hidden size) and in **BaziLLM** (frozen LLM + projection). Do not hardcode a different size; other Qwen variants may have different dimensions.

---

## 2. Converting Qwen to MLX

Convert the Hugging Face model to MLX format so it can be loaded with `mlx_lm` in Phase 2.

### Command

From the repo root (or any directory where you want the MLX model):

```bash
python -m mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-7B-Instruct \
  --mlx-path ./models/qwen2.5-7b-instruct-mlx
```

### Options

- **`--quantize`** — Optional. Quantize the model for lower memory (e.g. 4-bit or 8-bit). Omit for full-precision (fp16); expect ~14GB VRAM for 7B fp16.
- **Output path:** `models/qwen2.5-7b-instruct-mlx/` is the conventional location in this repo; scripts (e.g. `scripts/test_qwen_generation.py`) load from here.

### Prerequisites

- Phase 2 environment with `mlx`, `mlx-lm`, and `huggingface-hub` (see `requirements_phase2.txt`).
- Sufficient disk space for the converted weights (several GB).

---

## 3. Phase 1 encoders: checkpoint keys and freezing

### 3.1 Checkpoint keys

Phase 2 loads encoder weights from Phase 1 checkpoints. Use these keys and paths:

| Encoder   | Checkpoint file                                      | State-dict key          | Config (e.g. layers) |
| --------- | ---------------------------------------------------- | ----------------------- | ---------------------|
| **Natal** | `checkpoints/multi_task_3layer_1_6_1_1.pt`          | `"encoder"`             | `num_layers=3`, `d_model=256` |
| **Dynamic** | `checkpoints/cross_temporal_best.pt`               | `"dynamic_encoder"`     | `num_layers=2`, `d_model=256` |

- **Natal:** From multi-task training (`scripts/train_multi_task.py`). The saved checkpoint is a dict with keys such as `encoder`, `ten_god_head`, `element_head`, etc.; use only `ckpt["encoder"]` for the natal encoder.
- **Dynamic:** From cross-temporal training (`scripts/train_cross_temporal.py`). Use `ckpt["dynamic_encoder"]` for the dynamic encoder.

Converted MLX weights (from `scripts/convert_encoders_to_mlx.py`) are written to:

- `checkpoints/natal_encoder_mlx.npz`
- `checkpoints/dynamic_encoder_mlx.npz`

Parity tests in `tests/test_encoder_mlx_parity.py` compare PT vs MLX encoder outputs for the same inputs (within fp16 tolerance).

### 3.2 Encoder freezing (Phase 2 training)

For Phase 2 training (projection layer + frozen LLM), **all encoder parameters are frozen**: they are not updated by the Phase 2 loss.

- **PyTorch:** Set `requires_grad=False` for all encoder parameters (or exclude them from the optimizer).
- **MLX:** Do not include encoder parameters in the list of parameters passed to the optimizer; only the projection layer (and any other trainable Phase 2 parameters) receive gradients.

The encoders are used only in inference mode to produce natal/dynamic embeddings; their weights are fixed from Phase 1.

---

## 4. Related docs

- **Phase 1 checkpoint:** `docs/phase1_best_checkpoint.md` (encoder checkpoints for conversion to MLX).
- **Phase 2 deps and verify script:** `requirements_phase2.txt`, `scripts/verify_mlx.py`.
- **Qwen generation test:** `scripts/test_qwen_generation.py` (loads from `models/qwen2.5-7b-instruct-mlx`, runs short generation, reports memory).
- **Encoder parity (PT vs MLX):** `tests/test_encoder_mlx_parity.py` (same inputs, outputs within fp16 tolerance; see §3.1).
- **mlx_lm API for custom prefix (Task 2.2):** `docs/design_decisions.md` §17; verification script `scripts/inspect_mlx_lm_api.py`.
