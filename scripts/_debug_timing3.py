#!/usr/bin/env python3
"""Time forward+backward with 4-bit quantized model (eval both loss AND grads)."""
from __future__ import annotations
import sys, time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

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

print("Loading 4-bit model ...")
natal = NatalEncoderMLX(d_emb=32, d_model=256, num_layers=3)
natal.load_weights(str(_root / "checkpoints" / "natal_encoder_mlx.npz"))
dynamic = DynamicEncoderMLX(d_emb=32, d_model=256, num_layers=2)
dynamic.load_weights(str(_root / "checkpoints" / "dynamic_encoder_mlx.npz"))
cross = CrossInteractionEncoderMLX(d_input=27, d_hidden=128, d_model=256)

llm, tok_wrapper = mlx_lm_load(str(_root / "models" / "qwen2.5-7b-instruct-mlx-4bit"))
tokenizer = getattr(tok_wrapper, "_tokenizer", tok_wrapper)

ds = Phase2ChartDataset(_root / "data" / "phase2_train.jsonl")
N_VIRTUAL = 64
TRAIN_SIZE = len(ds)

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
    (960, False, "4bit ml=960 no-ckpt"),
    (960, True,  "4bit ml=960 ckpt"),
    (768, False, "4bit ml=768 no-ckpt"),
]

for ml, ckpt, label in configs:
    proj = ProjectionLayer(d_in=768, d_llm=3584, n_virtual_tokens=N_VIRTUAL)
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
    bmx = make_batch_mx([0], ml)

    # warmup with FULL eval (loss + grads)
    loss, grads = lg(model, bmx)
    mx.eval(loss, grads)

    # timed runs
    times = []
    for _ in range(3):
        t0 = time.time()
        loss, grads = lg(model, bmx)
        mx.eval(loss, grads)
        times.append(time.time() - t0)

    avg = sum(times) / len(times)
    accum = 8
    steps_3ep = (TRAIN_SIZE // accum) * 3
    hours = steps_3ep * accum * avg / 3600

    print(f"[{label}]  {avg:.1f}s/micro (fwd+bwd)  loss={float(loss):.3f}  | 3 epochs ≈ {hours:.1f}h ({hours/24:.1f} days)")
