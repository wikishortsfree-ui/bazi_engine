#!/usr/bin/env python3
"""
Phase 2 evaluation: perplexity, text generation, and automated scoring.

Three sequential modes:
  A. Test-set perplexity — teacher-forcing cross-entropy over all test examples.
  B. Text generation — autoregressive generation from virtual tokens + query (stratified sample).
  C. Automated scoring — character-level BLEU, ROUGE-L, factual consistency on generated samples.

Model loading reuses the train_phase2 pattern: MLX encoders + projection + frozen Qwen 4-bit.

Usage (from repo root, Phase 2 env):
  python scripts/evaluate_phase2.py [options]
  python scripts/evaluate_phase2.py --skip_generation   # perplexity only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _parse_args():
    p = argparse.ArgumentParser(description="Phase 2 evaluation: perplexity + generation + scoring.")
    p.add_argument("--test_data", type=Path, default=_root / "data" / "phase2_test.jsonl")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size for perplexity pass.")
    p.add_argument("--n_virtual_tokens", type=int, default=64)
    p.add_argument("--max_length", type=int, default=960)
    p.add_argument("--gen_samples", type=int, default=50, help="Number of samples for generation.")
    p.add_argument("--gen_max_tokens", type=int, default=512)
    p.add_argument("--skip_generation", action="store_true", help="Run perplexity only, skip generation + scoring.")
    p.add_argument("--skip_perplexity", action="store_true", help="Skip perplexity, run generation + scoring only.")
    p.add_argument(
        "--output_dir", type=Path, default=_root / "reports",
        help="Directory for eval outputs (generated samples, report).",
    )
    p.add_argument("--llm_path", type=Path, default=_root / "models" / "qwen2.5-7b-instruct-mlx-4bit")
    p.add_argument("--natal_weights", type=Path, default=_root / "checkpoints" / "natal_encoder_mlx.npz")
    p.add_argument("--dynamic_weights", type=Path, default=_root / "checkpoints" / "dynamic_encoder_mlx.npz")
    p.add_argument(
        "--checkpoint", type=Path,
        default=_root / "checkpoints" / "phase2_baseline" / "checkpoint_best.npz",
        help="Phase 2 projection checkpoint (.npz).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# A. Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(model, dataset, tokenizer, args, n_virtual: int) -> dict[str, Any]:
    """Teacher-forcing perplexity over the full test set."""
    import mlx.core as mx
    from bazi_model.data.phase2_dataset import phase2_collate
    from bazi_model.training.phase2_loss import compute_lm_loss

    batch_size = args.batch_size
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    t0 = time.time()
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        batch = phase2_collate([dataset[i] for i in indices], tokenizer, max_length=args.max_length)
        if not batch:
            continue

        batch_mx = _batch_to_mx(batch, n_virtual)
        logits = model(
            batch_mx["chart_natal_inputs"], batch_mx["chart_dynamic_inputs"],
            batch_mx["full_texts"], cross_features=batch_mx.get("cross_features"),
        )
        L = batch_mx["labels"].shape[1]
        logits_text = logits[:, n_virtual : n_virtual + L, :]
        loss = compute_lm_loss(logits_text, batch_mx["labels"], batch_mx["loss_mask"])
        mask_sum = float(batch_mx["loss_mask"].sum())

        mx.eval(loss)
        total_loss += float(loss) * mask_sum
        total_tokens += mask_sum
        n_batches += 1

        if n_batches % 25 == 0:
            elapsed = time.time() - t0
            print(f"  ppl pass: {n_batches} batches, {total_tokens:.0f} tokens, {elapsed:.1f}s")

    elapsed = time.time() - t0
    mean_loss = total_loss / max(total_tokens, 1.0)
    ppl = math.exp(min(mean_loss, 20.0))
    print(f"  Perplexity: {ppl:.2f}  (mean_loss={mean_loss:.4f}, tokens={total_tokens:.0f}, {elapsed:.1f}s)")
    return {"perplexity": ppl, "mean_loss": mean_loss, "total_tokens": int(total_tokens), "elapsed_s": elapsed}


# ---------------------------------------------------------------------------
# B. Text generation
# ---------------------------------------------------------------------------

def _stratified_sample_indices(dataset, n: int, seed: int = 42) -> list[int]:
    """Stratified sample: proportional to task_type distribution, at least 1 per type."""
    import numpy as np
    rng = np.random.RandomState(seed)
    by_type: dict[str, list[int]] = defaultdict(list)
    for i in range(len(dataset)):
        tt = dataset._records[i]["task_type"]
        by_type[tt].append(i)

    total = len(dataset)
    selected: list[int] = []
    for tt, indices in by_type.items():
        frac = len(indices) / total
        k = max(1, round(frac * n))
        k = min(k, len(indices))
        chosen = rng.choice(indices, size=k, replace=False).tolist()
        selected.extend(chosen)

    if len(selected) > n:
        selected = rng.choice(selected, size=n, replace=False).tolist()
    return selected


def generate_from_chart(
    model,  # BaziLLM
    natal_inputs: dict[str, Any],
    dynamic_inputs: dict[str, Any] | None,
    query_text: str,
    *,
    cross_features=None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """
    Autoregressive generation: encode chart → project → virtual tokens → concat query emb → generate.

    Uses mlx_lm's generate_step with input_embeddings for the prompt phase,
    then decodes greedily (temperature=0) or with sampling.
    """
    import mlx.core as mx

    gs_fn = _get_generate_step_fn()

    natal_kw = {k: _to_mx(v) for k, v in natal_inputs.items()}
    natal_emb = model.natal_encoder(**natal_kw)

    if dynamic_inputs is not None:
        dynamic_kw = {k: _to_mx(v) for k, v in dynamic_inputs.items()}
    else:
        from bazi_model.model.bazi_llm import _null_dynamic_inputs
        dynamic_kw = _null_dynamic_inputs(1)
    dynamic_emb = model.dynamic_encoder(**dynamic_kw)

    parts = [natal_emb, dynamic_emb]
    if model.cross_interaction_encoder is not None:
        if cross_features is None:
            cross_features = mx.zeros((1, 27))
        cf = _to_mx(cross_features) if not isinstance(cross_features, mx.array) else cross_features
        cross_emb = model.cross_interaction_encoder(cf)
        parts.append(cross_emb)

    concat_emb = mx.concatenate(parts, axis=-1)
    virtual_tokens = model.projection_layer(concat_emb)  # (1, n_virt, d_llm)

    enc = model.tokenizer(
        [query_text], padding=False, truncation=True,
        max_length=model.max_length, return_tensors="np",
    )
    query_ids = mx.array(enc["input_ids"])  # (1, query_len)
    inner = getattr(model.llm_model, "model", model.llm_model)
    query_emb = inner.embed_tokens(query_ids)  # (1, query_len, d_llm)

    input_embeddings = mx.concatenate([virtual_tokens, query_emb], axis=1)  # (1, total, d_llm)
    input_embeddings = input_embeddings.squeeze(0)  # (total, d_llm)

    total_len = input_embeddings.shape[0]
    prompt = mx.zeros((total_len,), dtype=mx.int32)

    sampler = None
    if temperature > 0:
        def _temp_sampler(logprobs):
            scaled = logprobs / temperature
            return mx.random.categorical(scaled)
        sampler = _temp_sampler

    tokens = []
    eos_ids = set()
    if hasattr(model.tokenizer, "eos_token_id") and model.tokenizer.eos_token_id is not None:
        eos_ids.add(int(model.tokenizer.eos_token_id))
    eos_ids_raw = getattr(model.tokenizer, "eos_token_ids", None)
    if eos_ids_raw is not None:
        if isinstance(eos_ids_raw, (list, tuple, set)):
            eos_ids.update(int(t) for t in eos_ids_raw)
        else:
            eos_ids.add(int(eos_ids_raw))

    for tok, _logprobs in gs_fn(
        prompt, model.llm_model,
        input_embeddings=input_embeddings,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        tok_id = int(tok.item()) if hasattr(tok, "item") else int(tok)
        if tok_id in eos_ids:
            break
        tokens.append(tok_id)

    return model.tokenizer.decode(tokens)


def run_generation(model, dataset, args) -> list[dict[str, Any]]:
    """Generate text for a stratified sample of test examples."""
    indices = _stratified_sample_indices(dataset, args.gen_samples)
    results = []
    t0 = time.time()

    for i, idx in enumerate(indices):
        rec = dataset._records[idx]
        item = dataset[idx]
        natal_np = item["natal_inputs"]
        dynamic_np = item["dynamic_inputs"]
        cross_np = item.get("cross_features")

        natal_batch = {k: v[None, ...] for k, v in natal_np.items()}
        if dynamic_np is not None:
            dynamic_batch = {k: v[None, ...] for k, v in dynamic_np.items()}
        else:
            dynamic_batch = None
        cross_batch = cross_np[None, ...] if cross_np is not None else None

        query = rec.get("input_sequence") or (rec.get("task_token", "") + "\n" + rec.get("query", ""))

        gen_text = generate_from_chart(
            model, natal_batch, dynamic_batch, query,
            cross_features=cross_batch, max_tokens=args.gen_max_tokens,
        )

        reference = rec.get("target_sequence") or (rec.get("reasoning", "") + "\n" + rec.get("output", ""))

        result = {
            "index": int(idx),
            "task_type": rec["task_type"],
            "query": rec.get("query", ""),
            "reference": reference,
            "generated": gen_text,
            "chart": rec.get("chart"),
        }
        results.append(result)

        elapsed = time.time() - t0
        print(f"  gen [{i+1}/{len(indices)}] task={rec['task_type']}  {len(gen_text)} chars  ({elapsed:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# C. Automated scoring
# ---------------------------------------------------------------------------

def _char_tokenize(text: str) -> list[str]:
    """Character-level tokenization for Chinese BLEU/ROUGE."""
    return list(text.replace(" ", "").replace("\n", ""))


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu_char(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Character-level BLEU (modified precision) with brevity penalty."""
    ref_tok = _char_tokenize(reference)
    hyp_tok = _char_tokenize(hypothesis)

    if not hyp_tok:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ng = _ngrams(ref_tok, n)
        hyp_ng = _ngrams(hyp_tok, n)
        clipped = sum(min(c, ref_ng.get(ng, 0)) for ng, c in hyp_ng.items())
        total = max(sum(hyp_ng.values()), 1)
        precisions.append(clipped / total)

    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n
    bp = 1.0 if len(hyp_tok) >= len(ref_tok) else math.exp(1 - len(ref_tok) / max(len(hyp_tok), 1))
    return bp * math.exp(log_avg)


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 (longest common subsequence) on character-level tokens."""
    ref_tok = _char_tokenize(reference)
    hyp_tok = _char_tokenize(hypothesis)

    if not ref_tok or not hyp_tok:
        return 0.0

    m, n = len(ref_tok), len(hyp_tok)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref_tok[i - 1] == hyp_tok[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr

    lcs_len = prev[n]
    precision = lcs_len / n
    recall = lcs_len / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_scoring(gen_results: list[dict[str, Any]]) -> dict[str, Any]:
    """BLEU, ROUGE-L, and factual consistency on generated samples."""
    from scripts.factual_check import check_factual_consistency

    bleu_scores = []
    rouge_scores = []
    factual_ok_count = 0
    by_type: dict[str, dict[str, list]] = defaultdict(lambda: {"bleu": [], "rouge": [], "factual_ok": []})

    for r in gen_results:
        ref = r["reference"]
        gen = r["generated"]
        tt = r["task_type"]

        bleu = sentence_bleu_char(ref, gen)
        rouge = rouge_l_f1(ref, gen)
        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
        by_type[tt]["bleu"].append(bleu)
        by_type[tt]["rouge"].append(rouge)

        fc_record = {
            "chart": r["chart"],
            "reasoning": gen,
            "output": "",
            "query": r["query"],
        }
        issues = check_factual_consistency(fc_record)
        is_ok = len(issues) == 0
        if is_ok:
            factual_ok_count += 1
        by_type[tt]["factual_ok"].append(is_ok)
        r["bleu"] = bleu
        r["rouge_l"] = rouge
        r["factual_issues"] = issues
        r["factual_ok"] = is_ok

    n = len(gen_results)
    summary = {
        "n_samples": n,
        "bleu_avg": sum(bleu_scores) / max(n, 1),
        "rouge_l_avg": sum(rouge_scores) / max(n, 1),
        "factual_consistency_rate": factual_ok_count / max(n, 1),
        "factual_ok_count": factual_ok_count,
    }
    per_type = {}
    for tt, scores in by_type.items():
        k = len(scores["bleu"])
        per_type[tt] = {
            "n": k,
            "bleu_avg": sum(scores["bleu"]) / max(k, 1),
            "rouge_l_avg": sum(scores["rouge"]) / max(k, 1),
            "factual_consistency_rate": sum(scores["factual_ok"]) / max(k, 1),
        }
    summary["per_task_type"] = per_type

    print(f"  BLEU avg: {summary['bleu_avg']:.4f}")
    print(f"  ROUGE-L avg: {summary['rouge_l_avg']:.4f}")
    print(f"  Factual consistency: {factual_ok_count}/{n} ({summary['factual_consistency_rate']:.1%})")
    for tt, s in per_type.items():
        print(f"    {tt}: BLEU={s['bleu_avg']:.4f}  ROUGE-L={s['rouge_l_avg']:.4f}  "
              f"factual={s['factual_consistency_rate']:.1%}  (n={s['n']})")

    return summary


# ---------------------------------------------------------------------------
# Helpers (shared with train_phase2)
# ---------------------------------------------------------------------------

def _to_mx(arr):
    import mlx.core as mx
    if isinstance(arr, mx.array):
        return arr
    return mx.array(arr)


def _batch_to_mx(batch: dict, n_virtual: int):
    """Convert batch arrays (numpy) to MLX."""
    import mlx.core as mx
    out = {}
    for k, v in batch.items():
        if k in ("full_texts", "task_types"):
            out[k] = v
            continue
        if k == "chart_natal_inputs":
            out[k] = {kk: mx.array(vv) for kk, vv in v.items()}
            continue
        if k == "chart_dynamic_inputs":
            if v is None:
                out[k] = None
            else:
                out[k] = {kk: mx.array(vv) for kk, vv in v.items()}
            continue
        if isinstance(v, (list, tuple)):
            continue
        try:
            out[k] = mx.array(v)
        except Exception:
            out[k] = v
    return out


def _get_generate_step_fn():
    """Resolve mlx_lm's generate_step (not exported at top level in newer versions)."""
    import mlx_lm
    fn = mlx_lm.stream_generate
    gs = fn.__globals__.get("generate_step")
    if gs is not None:
        return gs
    raise ImportError("Could not find generate_step in mlx_lm internals.")


# ---------------------------------------------------------------------------
# Model loading (mirrors train_phase2.py)
# ---------------------------------------------------------------------------

def load_model(args):
    """Load BaziLLM with encoders, projection, cross encoder, frozen LLM. Returns (model, tokenizer)."""
    import numpy as np
    from mlx_lm import load as mlx_lm_load
    from mlx_lm.utils import does_model_support_input_embeddings

    from bazi_model.encoders.dynamic_encoder_mlx import DynamicEncoderMLX
    from bazi_model.encoders.natal_encoder_mlx import NatalEncoderMLX
    from bazi_model.encoders.cross_interaction_encoder_mlx import CrossInteractionEncoderMLX
    from bazi_model.model.bazi_llm import BaziLLM
    from bazi_model.projection import ProjectionLayer

    print("Loading encoders ...")
    natal_encoder = NatalEncoderMLX(d_emb=32, d_model=256, num_layers=3)
    natal_encoder.load_weights(str(args.natal_weights))
    dynamic_encoder = DynamicEncoderMLX(d_emb=32, d_model=256, num_layers=2)
    dynamic_encoder.load_weights(str(args.dynamic_weights))
    cross_encoder = CrossInteractionEncoderMLX(d_input=27, d_hidden=128, d_model=256)

    d_in = 768  # natal(256) + dynamic(256) + cross(256)
    projection = ProjectionLayer(d_in=d_in, d_llm=3584, n_virtual_tokens=args.n_virtual_tokens)
    if args.checkpoint.exists():
        print(f"Loading projection + cross encoder from {args.checkpoint} ...")
        _load_projection_from_checkpoint(projection, args.checkpoint, cross_encoder)
    else:
        print(f"Warning: checkpoint not found at {args.checkpoint}, using random weights.", file=sys.stderr)

    print(f"Loading LLM from {args.llm_path} ...")
    llm_model, tokenizer_wrapper = mlx_lm_load(str(args.llm_path))
    if not does_model_support_input_embeddings(llm_model):
        print("Error: model does not support input_embeddings.", file=sys.stderr)
        sys.exit(1)

    tokenizer = getattr(tokenizer_wrapper, "_tokenizer", tokenizer_wrapper)
    if not callable(tokenizer):
        print(f"Error: tokenizer ({type(tokenizer)}) is not callable.", file=sys.stderr)
        sys.exit(1)

    model = BaziLLM(
        natal_encoder, dynamic_encoder, projection, llm_model, tokenizer,
        cross_interaction_encoder=cross_encoder,
        max_length=args.max_length, n_virtual_tokens=args.n_virtual_tokens,
    )
    model.freeze_except_projection()
    model.projection_layer.freeze()
    model.cross_interaction_encoder.freeze()
    return model, tokenizer_wrapper


def _load_projection_from_checkpoint(projection, path: Path, cross_encoder=None) -> None:
    import numpy as np
    from bazi_model.encoders.natal_encoder_mlx import _flat_to_nested

    data = dict(np.load(path, allow_pickle=True))

    proj_flat = {
        k.replace("projection_layer.", ""): np.asarray(v)
        for k, v in data.items()
        if k.startswith("projection_layer.")
    }
    if proj_flat:
        nested = _flat_to_nested(proj_flat)
        projection.update(nested, strict=False)

    if cross_encoder is not None:
        cross_flat = {
            k.replace("cross_interaction_encoder.", ""): np.asarray(v)
            for k, v in data.items()
            if k.startswith("cross_interaction_encoder.")
        }
        if cross_flat:
            nested = _flat_to_nested(cross_flat)
            cross_encoder.update(nested, strict=False)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(output_dir: Path, ppl_result: dict | None, scoring_result: dict | None, gen_results: list | None):
    """Write markdown evaluation report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# Phase 2 Baseline Evaluation Report\n"]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if ppl_result:
        lines.append("## A. Test-set Perplexity\n")
        lines.append(f"- **Perplexity**: {ppl_result['perplexity']:.2f}")
        lines.append(f"- **Mean CE loss**: {ppl_result['mean_loss']:.4f}")
        lines.append(f"- **Total tokens**: {ppl_result['total_tokens']:,}")
        lines.append(f"- **Elapsed**: {ppl_result['elapsed_s']:.1f}s")
        target_met = ppl_result["perplexity"] < 15
        lines.append(f"- **Target (< 15)**: {'MET' if target_met else 'NOT MET'}\n")

    if scoring_result:
        lines.append("## B. Automated Scoring (on generated samples)\n")
        lines.append(f"- **Samples**: {scoring_result['n_samples']}")
        lines.append(f"- **BLEU avg**: {scoring_result['bleu_avg']:.4f}")
        lines.append(f"- **ROUGE-L avg**: {scoring_result['rouge_l_avg']:.4f}")
        fc_rate = scoring_result["factual_consistency_rate"]
        lines.append(f"- **Factual consistency**: {scoring_result['factual_ok_count']}/{scoring_result['n_samples']} "
                      f"({fc_rate:.1%})")
        fc_target = fc_rate > 0.9
        lines.append(f"- **Factual target (> 90%)**: {'MET' if fc_target else 'NOT MET'}\n")

        lines.append("### Per-task-type breakdown\n")
        lines.append("| Task Type | N | BLEU | ROUGE-L | Factual % |")
        lines.append("|-----------|---|------|---------|-----------|")
        for tt, s in scoring_result.get("per_task_type", {}).items():
            lines.append(f"| {tt} | {s['n']} | {s['bleu_avg']:.4f} | {s['rouge_l_avg']:.4f} | "
                          f"{s['factual_consistency_rate']:.1%} |")
        lines.append("")

    if gen_results:
        lines.append("## C. Sample Generations\n")
        good = [r for r in gen_results if r.get("factual_ok") and r.get("bleu", 0) > 0.05]
        bad = [r for r in gen_results if not r.get("factual_ok") or r.get("bleu", 0) < 0.02]
        examples = good[:3] + bad[:2]
        if not examples:
            examples = gen_results[:5]

        for i, ex in enumerate(examples):
            label = "Good" if ex.get("factual_ok") and ex.get("bleu", 0) > 0.02 else "Needs improvement"
            lines.append(f"### Example {i+1} ({label}, task={ex['task_type']})\n")
            lines.append(f"**Query**: {ex['query'][:200]}\n")
            lines.append(f"**Generated** (BLEU={ex.get('bleu', 0):.4f}, ROUGE-L={ex.get('rouge_l', 0):.4f}):")
            lines.append(f"```\n{ex['generated'][:500]}\n```\n")
            lines.append(f"**Reference**:")
            lines.append(f"```\n{ex['reference'][:500]}\n```\n")

    lines.append("## D. Deferred / Future Work\n")
    lines.append("- Attention analysis (virtual token attention patterns) — requires hooks into MLX Qwen internals.")
    lines.append("- Human evaluation with rubric (30-50 samples).")
    lines.append("- LoRA fine-tuning comparison.")
    lines.append("- Data augmentation experiments.\n")

    report_path = output_dir / "phase2_baseline_eval.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")

    if gen_results:
        gen_path = output_dir / "phase2_generated_samples.jsonl"
        with open(gen_path, "w", encoding="utf-8") as f:
            for r in gen_results:
                out = {k: v for k, v in r.items() if k != "chart"}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"Generated samples written to {gen_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    try:
        import mlx.core as mx  # noqa: F401
        from bazi_model.data.phase2_dataset import Phase2ChartDataset
    except ImportError as e:
        print(f"Error: missing dependencies. {e}", file=sys.stderr)
        print("Install Phase 2 deps: pip install -r requirements_phase2.txt", file=sys.stderr)
        return 1

    if not args.test_data.exists():
        print(f"Error: test data not found: {args.test_data}", file=sys.stderr)
        return 1

    print(f"Loading test data from {args.test_data} ...")
    dataset = Phase2ChartDataset(args.test_data)
    print(f"  {len(dataset)} test examples")
    type_counts = Counter(dataset._records[i]["task_type"] for i in range(len(dataset)))
    for tt, c in type_counts.items():
        print(f"    {tt}: {c}")

    model, tokenizer_wrapper = load_model(args)
    n_virtual = args.n_virtual_tokens

    ppl_result = None
    gen_results = None
    scoring_result = None

    # A. Perplexity
    if not args.skip_perplexity:
        print("\n=== A. Test-set Perplexity ===")
        ppl_result = compute_perplexity(model, dataset, model.tokenizer, args, n_virtual)

    # B. Generation + C. Scoring
    if not args.skip_generation:
        print(f"\n=== B. Text Generation ({args.gen_samples} samples) ===")
        gen_results = run_generation(model, dataset, args)

        print("\n=== C. Automated Scoring ===")
        scoring_result = run_scoring(gen_results)

    # Write report
    print("\n=== Writing report ===")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_report(args.output_dir, ppl_result, scoring_result, gen_results)

    if ppl_result:
        print(f"\nPerplexity: {ppl_result['perplexity']:.2f}  (target < 15: "
              f"{'PASS' if ppl_result['perplexity'] < 15 else 'FAIL'})")
    if scoring_result:
        fc = scoring_result["factual_consistency_rate"]
        print(f"Factual consistency: {fc:.1%}  (target > 90%: {'PASS' if fc > 0.9 else 'FAIL'})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
