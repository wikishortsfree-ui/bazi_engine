#!/usr/bin/env bash
# Run after multi-task training finishes. Re-evaluates the latest checkpoint
# and optionally runs hyperparameter tuning (short trials + full run).
# Usage:
#   ./scripts/run_after_multi_task_training.sh
#   ./scripts/run_after_multi_task_training.sh --tune

set -e
cd "$(dirname "$0")/.."
DATA="${DATA:-data/charts.jsonl}"
VAL=0.1
TEST=0.1
SEED=42

echo "=== 1. Evaluate multi-task checkpoint ==="
python scripts/evaluate_phase1.py \
  --data "$DATA" \
  --use_multi_task \
  --val_ratio "$VAL" \
  --test_ratio "$TEST" \
  --seed "$SEED" \
  --output reports/phase1_metrics_multi_task.json

echo ""
echo "=== 2. Optional: tune hyperparameters (short trials + full run) ==="
if [[ "${1:-}" == "--tune" ]]; then
  SAVE_BY="${TUNE_SAVE_BY:-composite}"
  ORDINAL="${ORDINAL_LOSS_WEIGHT:-0.15}"
  python scripts/tune_phase1.py \
    --data "$DATA" \
    --epochs_per_trial 5 \
    --n_trials 15 \
    --val_ratio "$VAL" \
    --test_ratio "$TEST" \
    --seed "$SEED" \
    --save_by "$SAVE_BY" \
    --ordinal_loss_weight "$ORDINAL" \
    --run_full_at_end \
    --output reports/tune_phase1_results.json
  echo "Tuning done. Re-run this script (without --tune) to evaluate the new checkpoint."
else
  echo "Skipped. Run with --tune to run tune_phase1.py (short trials + full 30-epoch run)."
  echo "Optional: TUNE_SAVE_BY=composite ORDINAL_LOSS_WEIGHT=0.15 ./scripts/run_after_multi_task_training.sh --tune"
fi
