#!/usr/bin/env bash
# End-to-end: Phase 1 encoders → MLX Qwen → encoder .npz → Phase 2 MLX training.
# Run from repo root: bash scripts/run_phase2_full_stack.sh
# Logs append to runs/phase2_full_stack.log
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/runs" "$ROOT/models" "$ROOT/checkpoints" "$ROOT/checkpoints/phase2_run"
LOG="$ROOT/runs/phase2_full_stack.log"
exec >>"$LOG" 2>&1

echo "======== phase2 full stack start $(date -Iseconds) pid=$$ ========"

MT="$ROOT/checkpoints/multi_task_3layer_1_6_1_1.pt"
if [[ ! -f "$MT" ]]; then
  echo "[1/5] Phase 1 multi-task (natal encoder) ..."
  python scripts/train_multi_task.py \
    --data "$ROOT/data/charts.jsonl" \
    --epochs 30 --batch_size 64 \
    --lr 0.0005 --task_weights 1,6,1,1 \
    --num_layers 3 \
    --val_ratio 0.1 --test_ratio 0.1 --seed 42 \
    --checkpoint_dir "$ROOT/checkpoints" --log_dir "$ROOT/runs/multi_task" \
    --save_by composite --ordinal_loss_weight 0.15 \
    --save_name multi_task_3layer_1_6_1_1.pt
else
  echo "[1/5] Skip multi-task (exists): $MT"
fi

CT="$ROOT/checkpoints/cross_temporal_best.pt"
if [[ ! -f "$CT" ]]; then
  echo "[2/5] Phase 1 cross-temporal ..."
  python scripts/train_cross_temporal.py \
    --data "$ROOT/data/charts.jsonl" \
    --epochs 60 \
    --checkpoint_dir "$ROOT/checkpoints" \
    --log_dir "$ROOT/runs/cross_temporal"
else
  echo "[2/5] Skip cross-temporal (exists): $CT"
fi

QWEN="$ROOT/models/qwen2.5-7b-instruct-mlx"
if [[ ! -f "$QWEN/config.json" ]]; then
  echo "[3/5] Converting Qwen2.5-7B-Instruct to MLX (large download) ..."
  python -m mlx_lm convert \
    --hf-path Qwen/Qwen2.5-7B-Instruct \
    --mlx-path "$QWEN"
else
  echo "[3/5] Skip MLX Qwen (exists): $QWEN"
fi

NE="$ROOT/checkpoints/natal_encoder_mlx.npz"
DE="$ROOT/checkpoints/dynamic_encoder_mlx.npz"
if [[ ! -f "$NE" ]] || [[ ! -f "$DE" ]]; then
  echo "[4/5] PyTorch encoders → MLX .npz ..."
  python scripts/convert_encoders_to_mlx.py \
    --natal_checkpoint "$MT" \
    --dynamic_checkpoint "$CT" \
    --output_dir "$ROOT/checkpoints"
else
  echo "[4/5] Skip encoder convert (exists): $NE $DE"
fi

echo "[5/5] Phase 2 training ..."
# 7B MLX + long sequences: micro-batches are slow (~1–3 min/step on Apple Silicon).
# batch 1 × accum 8 ≈ default effective batch 4×2; shorter max_length reduces memory and compile time.
python scripts/train_phase2.py \
  --train_data "$ROOT/data/phase2_train.jsonl" \
  --val_data "$ROOT/data/phase2_val.jsonl" \
  --llm_path "$QWEN" \
  --natal_weights "$NE" \
  --dynamic_weights "$DE" \
  --checkpoint_dir "$ROOT/checkpoints/phase2_run" \
  --batch_size 1 --accumulation_steps 8 \
  --max_length 768

echo "======== phase2 full stack done $(date -Iseconds) ========"
