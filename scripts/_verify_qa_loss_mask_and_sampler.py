"""Empirically verify (a) loss-mask geometry and (b) sampler honoring for Stage 1 QA.

Verification (a) — loss mask:
- Build a small Stage1QADataset, run phase2_collate, then for each sample decode
  the input_ids slice that loss_mask flags as "loss-bearing" and check it matches
  the answer (modulo a boundary "\\n" or BOS/EOS artifact).
- Reports per-sample: prefix_token_count, masked_token_count, masked_text vs answer.

Verification (b) — sampler:
- Build the train subset the way scripts/train_projection_qa.py does
  (stratified_holdout_indices + get_category_weights_for_sampler), then draw
  N samples from np.random.choice(p=weights) and report the resulting
  per-bucket fraction. Compare against TARGET_CATEGORY_DISTRIBUTION.

Run: python scripts/_verify_qa_loss_mask_and_sampler.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from bazi_model.data.phase2_dataset import phase2_collate  # noqa: E402
from bazi_model.data.stage1_qa_dataset import (  # noqa: E402
    TARGET_CATEGORY_DISTRIBUTION,
    Stage1QADataset,
    _category_bucket,
    get_category_weights_for_sampler,
    stratified_holdout_indices,
)


def _load_qwen_tokenizer(llm_path: Path):
    """Use the same tokenizer wrapper the training script uses."""
    from mlx_lm import load as mlx_lm_load

    _, wrapper = mlx_lm_load(str(llm_path))
    tok = getattr(wrapper, "_tokenizer", wrapper)
    if not callable(tok):
        raise RuntimeError(f"Tokenizer ({type(tok)}) is not callable.")
    return tok


def _decode(tokenizer, ids):
    return tokenizer.decode(list(int(x) for x in ids))


def verify_loss_mask(args, tokenizer):
    print("\n=== (a) Loss-mask verification ===")
    ds = Stage1QADataset(args.qa_data, args.qa_charts, limit=args.mask_samples * 50)
    if len(ds) == 0:
        raise RuntimeError("Empty Stage1QADataset.")

    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds), size=min(args.mask_samples, len(ds)), replace=False)
    items = [ds[int(i)] for i in indices]
    batch = phase2_collate(items, tokenizer, max_length=args.max_length)

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    seq_len = input_ids.shape[1]

    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)

    n_ok = 0
    n_bad = 0
    leak_lines = 0
    miss_lines = 0
    for b, item in enumerate(items):
        question = item["input_sequence"]
        answer = item["target_sequence"]
        full_text = question + "\n" + answer

        full_ids = list(int(x) for x in input_ids[b])
        full_decoded = _decode(tokenizer, [t for t in full_ids if t != pad_id])

        prefix_ids = tokenizer(question, padding=False, truncation=False, return_tensors="np", add_special_tokens=True)["input_ids"]
        plen = int(prefix_ids.shape[-1] if hasattr(prefix_ids, "shape") else len(prefix_ids))

        mask_b = loss_mask[b]
        masked_positions = np.where(mask_b > 0.5)[0]

        # The set of input_ids tokens that are *predicted* (i.e. labels) by the
        # mask is input_ids[i+1] for masked i. We want to check that this
        # predicted-token set matches the answer span.
        predicted_token_ids = [int(input_ids[b, i + 1]) for i in masked_positions if i + 1 < seq_len]
        predicted_text = _decode(tokenizer, predicted_token_ids)

        ok = answer.strip() in predicted_text
        leaks_question = any(qchar in predicted_text for qchar in question if qchar.strip()) and not ok
        if ok and not leaks_question:
            n_ok += 1
        else:
            n_bad += 1
            if not ok:
                miss_lines += 1
            if leaks_question:
                leak_lines += 1

        if b < args.print_first:
            print(f"\n--- sample {b} (cat={item['category']}) ---")
            print(f"  question:        {question[:100]!r}")
            print(f"  answer:          {answer[:100]!r}")
            print(f"  prefix_token_n:  {plen}")
            print(f"  masked_count:    {len(masked_positions)}")
            print(f"  predicted_text:  {predicted_text[:160]!r}")
            print(f"  full_decoded:    {full_decoded[:160]!r}")
            if not ok:
                print("  WARNING: predicted_text does not contain answer.")
            if leaks_question:
                print("  WARNING: predicted_text appears to leak question characters.")

    print(f"\nLoss-mask result: {n_ok}/{len(items)} samples OK; {n_bad} bad ({miss_lines} missing answer, {leak_lines} leaked question).")
    return n_ok, n_bad


def verify_sampler(args):
    print("\n=== (b) Sampler verification ===")
    ds = Stage1QADataset(args.qa_data, args.qa_charts, limit=args.sampler_subset_size)
    print(f"  Loaded {len(ds)} QA pairs (subset_size={args.sampler_subset_size}).")

    raw_buckets = Counter(_category_bucket(rec["question_category"]) for rec in ds._records)
    print("  Empirical category distribution (raw):")
    for cat, n in raw_buckets.most_common():
        print(f"    {cat:25s}  {n / len(ds):.3f}  ({n})")

    train_idx, val_idx = stratified_holdout_indices(ds, per_category=args.qa_val_per_category, seed=0)
    print(f"  Stratified split: train={len(train_idx)}  val={len(val_idx)}")

    class _Subset:
        def __init__(self, parent, indices):
            self._parent = parent
            self._indices = indices
            self._records = [parent._records[int(i)] for i in indices]

        def __len__(self):
            return len(self._indices)

    train_ds = _Subset(ds, train_idx)
    weights = get_category_weights_for_sampler(train_ds)
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        print("  ERROR: weights sum to 0.")
        return 0, 1
    w = w / w.sum()

    train_buckets = Counter(_category_bucket(rec["question_category"]) for rec in train_ds._records)
    print("  Train subset category distribution (raw, before sampling):")
    for cat, n in train_buckets.most_common():
        print(f"    {cat:25s}  {n / len(train_ds):.3f}  ({n})")

    rng = np.random.default_rng(0)
    drawn = rng.choice(len(train_ds), size=args.sampler_draws, replace=True, p=w)
    drawn_buckets = Counter(
        _category_bucket(train_ds._records[int(i)]["question_category"]) for i in drawn
    )

    print(f"  Empirical sampled distribution ({args.sampler_draws} draws):")
    print(f"    {'bucket':25s}  {'sampled':>8s}  {'target':>8s}  {'delta':>8s}")
    n_ok = 0
    n_bad = 0
    for cat in TARGET_CATEGORY_DISTRIBUTION.keys():
        observed = drawn_buckets.get(cat, 0) / args.sampler_draws
        target = TARGET_CATEGORY_DISTRIBUTION[cat]
        delta = observed - target
        ok = abs(delta) < 0.02  # 2-percentage-point tolerance
        flag = "" if ok else "  *"
        print(f"    {cat:25s}  {observed:8.3f}  {target:8.3f}  {delta:+8.3f}{flag}")
        if ok:
            n_ok += 1
        else:
            n_bad += 1

    return n_ok, n_bad


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--qa_data", type=Path, default=_root / "data" / "stage1_qa.jsonl")
    p.add_argument("--qa_charts", type=Path, default=_root / "data" / "qa_charts.jsonl")
    p.add_argument("--llm_path", type=Path, default=_root / "models" / "qwen2.5-7b-instruct-mlx-4bit")
    p.add_argument("--max_length", type=int, default=320)
    p.add_argument("--mask_samples", type=int, default=20)
    p.add_argument("--print_first", type=int, default=4)
    p.add_argument("--sampler_subset_size", type=int, default=20000)
    p.add_argument("--sampler_draws", type=int, default=20000)
    p.add_argument("--qa_val_per_category", type=int, default=500)
    p.add_argument("--skip_mask", action="store_true")
    p.add_argument("--skip_sampler", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    rc = 0
    if not args.skip_mask:
        tokenizer = _load_qwen_tokenizer(args.llm_path)
        n_ok, n_bad = verify_loss_mask(args, tokenizer)
        if n_bad > 0:
            rc = 1
    if not args.skip_sampler:
        n_ok, n_bad = verify_sampler(args)
        if n_bad > 0:
            rc = 1

    print("\n" + ("PASS" if rc == 0 else "FAIL"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
