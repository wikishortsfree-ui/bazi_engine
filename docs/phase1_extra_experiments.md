# Phase 1 extra experiments

Optional one-off runs to push element ordinal or explore capacity. Use after the main tune (1,4,1,1) is done.

## 1. Stronger element weight (1,5,1,1)

Single full run with element weight 5.0; save to a separate file so the current best is not overwritten.

```bash
python scripts/train_multi_task.py \
  --data data/charts.jsonl \
  --epochs 30 --batch_size 64 \
  --lr 0.0005 --task_weights 1,5,1,1 \
  --val_ratio 0.1 --test_ratio 0.1 --seed 42 \
  --checkpoint_dir checkpoints --log_dir runs/multi_task \
  --save_by composite --ordinal_loss_weight 0.15 \
  --save_name multi_task_1_5_1_1.pt
```

Then evaluate:

```bash
python scripts/evaluate_phase1.py --data data/charts.jsonl --use_multi_task \
  --multi_task_checkpoint checkpoints/multi_task_1_5_1_1.pt \
  --output reports/phase1_metrics_1_5_1_1.json
```

Compare `reports/phase1_metrics_1_5_1_1.json` (element ordinal, MSE) to your current best. Or run the full tune (15 trials) with `./scripts/run_after_multi_task_training.sh --tune` to search over weights 1–5.

## 2. Encoder capacity (3 layers or d_model=512)

**3 layers (same width):**

```bash
python scripts/train_multi_task.py \
  --data data/charts.jsonl \
  --epochs 30 --batch_size 64 \
  --lr 0.0005 --task_weights 1,4,1,1 \
  --num_layers 3 \
  --val_ratio 0.1 --test_ratio 0.1 --seed 42 \
  --checkpoint_dir checkpoints --log_dir runs/multi_task \
  --save_by composite --ordinal_loss_weight 0.15 \
  --save_name multi_task_3layer.pt
```

**Larger width (d_model=512):**

```bash
python scripts/train_multi_task.py \
  --data data/charts.jsonl \
  --epochs 30 --batch_size 64 \
  --lr 0.0005 --task_weights 1,4,1,1 \
  --d_model 512 \
  --val_ratio 0.1 --test_ratio 0.1 --seed 42 \
  --checkpoint_dir checkpoints --log_dir runs/multi_task \
  --save_by composite --ordinal_loss_weight 0.15 \
  --save_name multi_task_d512.pt
```

Note: `--encoder_checkpoint` is only used when `--d_model 256` (default); otherwise the load is skipped.

## 3. Ordinal error analysis

See how many ordinal errors are "borderline" (true strengths very close) vs clear:

```bash
python scripts/analyze_ordinal_errors.py \
  --data data/charts.jsonl \
  --checkpoint checkpoints/multi_task_best.pt \
  --output reports/ordinal_error_analysis.json
```

Interpretation: if most errors are borderline (e.g. >60%), the 90% ordinal target may be capped by label noise; if most are clear, more capacity or data may help.
