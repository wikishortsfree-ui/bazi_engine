#!/usr/bin/env python3
"""Minimal diagnostic: does mx.checkpoint work for BaziLLM backward?"""
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

print("Loading encoders ...")
natal = NatalEncoderMLX(d_emb=32, d_model=256, num_layers=3)
natal.load_weights(str(_root / "checkpoints" / "natal_encoder_mlx.npz"))
dynamic = DynamicEncoderMLX(d_emb=32, d_model=256, num_layers=2)
dynamic.load_weights(str(_root / "checkpoints" / "dynamic_encoder_mlx.npz"))
cross = CrossInteractionEncoderMLX(d_input=27, d_hidden=128, d_model=256)
proj = ProjectionLayer(d_in=768, d_llm=3584, n_virtual_tokens=64)

print("Loading Qwen ...")
llm, tok_wrapper = mlx_lm_load(str(_root / "models" / "qwen2.5-7b-instruct-mlx"))
tokenizer = getattr(tok_wrapper, "_tokenizer", tok_wrapper)

N_VIRTUAL = 64
MAX_LEN = 256  # short for quick test

model = BaziLLM(natal, dynamic, proj, llm, tokenizer,
                cross_interaction_encoder=cross, max_length=MAX_LEN, n_virtual_tokens=N_VIRTUAL)
model.freeze_except_projection()

print("Loading 1 sample ...")
ds = Phase2ChartDataset(_root / "data" / "phase2_train.jsonl")
batch = phase2_collate([ds[0]], tokenizer, max_length=MAX_LEN)
batch_mx = {}
for k, v in batch.items():
    if k in ("full_texts", "task_types"):
        batch_mx[k] = v
    elif k == "chart_natal_inputs":
        batch_mx[k] = {kk: mx.array(vv) for kk, vv in v.items()}
    elif k == "chart_dynamic_inputs":
        batch_mx[k] = {kk: mx.array(vv) for kk, vv in v.items()} if v else None
    elif isinstance(v, (list, tuple)):
        continue
    else:
        try:
            batch_mx[k] = mx.array(v)
        except Exception:
            batch_mx[k] = v

def loss_fn(m, bmx):
    logits = m(bmx["chart_natal_inputs"], bmx["chart_dynamic_inputs"],
               bmx["full_texts"], cross_features=bmx.get("cross_features"))
    L = bmx["labels"].shape[1]
    logits_text = logits[:, N_VIRTUAL:N_VIRTUAL + L, :]
    return compute_lm_loss(logits_text, bmx["labels"], bmx["loss_mask"])

loss_and_grad = nn.value_and_grad(model, loss_fn)

# --- Test 1: NO checkpointing ---
print("\n=== Test 1: NO gradient checkpointing ===")
model.gradient_checkpointing = False
t0 = time.time()
loss1, grads1 = loss_and_grad(model, batch_mx)
mx.eval(loss1)
t1 = time.time()
print(f"  Loss: {float(loss1):.4f}  Time: {t1-t0:.1f}s")

# --- Test 2: WITH checkpointing ---
print("\n=== Test 2: WITH gradient checkpointing ===")
model.gradient_checkpointing = True
t0 = time.time()
loss2, grads2 = loss_and_grad(model, batch_mx)
mx.eval(loss2)
t2 = time.time()
print(f"  Loss: {float(loss2):.4f}  Time: {t2-t0:.1f}s")

print("\nDone. If both tests complete, checkpointing works.")
