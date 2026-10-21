#!/usr/bin/env python3
"""Time one micro-batch at different max_length and batch_size to estimate total training time."""
from __future__ import annotations
import sys, time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from bazi_model.encoders.natal_encoder_mlx import NatalEncoderMLX
from bazi_model.encoders.dynamic_encoder_mlx import DynamicEncoderMLX
from bazi_model.encoders.cross_interaction_encoder_mlx import CrossInteractionEncoderMLX
from bazi_model.model.bazi_llm import BaziLLM
from bazi_model.projection import ProjectionLayer
from bazi_model.training.phase2_loss import compute_lm_loss
from bazi_model.data.phase2_dataset import Phase2ChartDataset, phase2_collate
from mlx_lm import load as mlx_lm_load

print("Loading model ...")
natal = NatalEncoderMLX(d_emb=32, d_model=256, num_layers=3)
natal.load_weights(str(_root / "checkpoints" / "natal_encoder_mlx.npz"))
dynamic = DynamicEncoderMLX(d_emb=32, d_model=256, num_layers=2)
dynamic.load_weights(str(_root / "checkpoints" / "dynamic_encoder_mlx.npz"))
cross = CrossInteractionEncoderMLX(d_input=27, d_hidden=128, d_model=256)
proj = ProjectionLayer(d_in=768, d_llm=3584, n_virtual_tokens=64)

llm, tok_wrapper = mlx_lm_load(str(_root / "models" / "qwen2.5-7b-instruct-mlx"))
tokenizer = getattr(tok_wrapper, "_tokenizer", tok_wrapper)

ds = Phase2ChartDataset(_root / "data" / "phase2_train.jsonl")
N_VIRTUAL = 64
TRAIN_SIZE = len(ds)
print(f"Train set: {TRAIN_SIZE} samples")

def make_batch_mx(indices, max_length):
    batch = phase2_collate([ds[i] for i in indices], tokenizer, max_length=max_length)
    out = {}
    for k, v in batch.items():
        if k in ("full_texts", "task_types"):
            out[k] = v
        elif k == "chart_natal_inputs":
            out[k] = {kk: mx.array(vv) for kk, vv in v.items()}
        elif k == "chart_dynamic_inputs":
            out[k] = {kk: mx.array(vv) for kk, vv in v.items()} if v else None
        elif isinstance(v, (list, tuple)):
            continue
        else:
            try: out[k] = mx.array(v)
            except: out[k] = v
    return out

configs = [
    # (max_length, batch_size, checkpointing, label)
    (960, 1, False, "ml=960 bs=1 no-ckpt"),
    (960, 1, True,  "ml=960 bs=1 ckpt"),
    (512, 1, False, "ml=512 bs=1 no-ckpt"),
    (384, 1, False, "ml=384 bs=1 no-ckpt"),
]

for ml, bs, ckpt, label in configs:
    model = BaziLLM(natal, dynamic, proj, llm, tokenizer,
                    cross_interaction_encoder=cross, max_length=ml, n_virtual_tokens=N_VIRTUAL)
    model.freeze_except_projection()
    model.gradient_checkpointing = ckpt

    def loss_fn(m, bmx):
        logits = m(bmx["chart_natal_inputs"], bmx["chart_dynamic_inputs"],
                   bmx["full_texts"], cross_features=bmx.get("cross_features"))
        L = bmx["labels"].shape[1]
        return compute_lm_loss(logits[:, N_VIRTUAL:N_VIRTUAL+L, :], bmx["labels"], bmx["loss_mask"])

    lg = nn.value_and_grad(model, loss_fn)
    bmx = make_batch_mx(list(range(bs)), ml)

    # warmup
    loss, grads = lg(model, bmx)
    mx.eval(loss)

    t0 = time.time()
    loss, grads = lg(model, bmx)
    mx.eval(loss)
    elapsed = time.time() - t0

    accum = 8
    steps_per_epoch = TRAIN_SIZE // (bs * accum)
    total_steps_3ep = steps_per_epoch * 3
    total_secs = total_steps_3ep * accum * elapsed
    total_hours = total_secs / 3600

    print(f"[{label}]  {elapsed:.1f}s/micro-batch  loss={float(loss):.3f}"
          f"  | 3 epochs ≈ {total_hours:.1f}h ({total_steps_3ep} steps × {accum} micro)")
