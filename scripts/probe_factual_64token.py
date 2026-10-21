#!/usr/bin/env python3
"""
Sanity probe for 64-token projection: single-fact QA over Phase 2 test charts.

For each of 20 hand-crafted factual question types, this script:
- picks a random chart from data/phase2_test.jsonl,
- runs greedy inference (virtual tokens + question → Qwen),
- compares the answer against ground truth derived from BaziChart,
- reports per-category and overall accuracy, plus per-question details.

Usage (from repo root, Phase 2 env):
  python scripts/probe_factual_64token.py
  python scripts/probe_factual_64token.py --help
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probe factual recall with 64 virtual tokens on Phase 2 test charts.",
    )
    p.add_argument(
        "--test_data",
        type=Path,
        default=_root / "data" / "phase2_test.jsonl",
        help="Phase 2 test JSONL (charts + narratives).",
    )
    p.add_argument(
        "--n_virtual_tokens",
        type=int,
        default=63,
        help=(
            "Number of virtual tokens (must match projection checkpoint). "
            "Production layout is 7 slots * 9 virtual tokens/slot = 63."
        ),
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=960,
        help="Max text length for tokenizer (virtual + text <= 1024).",
    )
    p.add_argument(
        "--llm_path",
        type=Path,
        default=_root / "models" / "qwen2.5-7b-instruct-mlx-4bit",
        help="Path to MLX-converted Qwen model directory.",
    )
    p.add_argument(
        "--natal_mode",
        choices=("encoder", "embedding"),
        default="encoder",
        help=(
            "Match training: 'encoder' loads frozen NatalEncoderMLX from "
            "--natal_weights; 'embedding' instantiates EmbeddingNatalMLX and "
            "loads its weights from --projection_checkpoint (under the "
            "'natal_encoder.' prefix)."
        ),
    )
    p.add_argument(
        "--natal_weights",
        type=Path,
        default=_root / "checkpoints" / "natal_encoder_mlx.npz",
        help="Path to natal encoder MLX weights (only used for --natal_mode=encoder).",
    )
    p.add_argument(
        "--dynamic_weights",
        type=Path,
        default=_root / "checkpoints" / "dynamic_encoder_mlx.npz",
        help="Path to dynamic encoder MLX weights.",
    )
    p.add_argument(
        "--projection_checkpoint",
        type=Path,
        default=_root / "checkpoints" / "phase3_prep" / "checkpoint_best.npz",
        help="64-token projection checkpoint (.npz).",
    )
    p.add_argument(
        "--num_questions",
        type=int,
        default=20,
        help="Total number of single-fact questions to ask (default: 20).",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=64,
        help="Max generation length (tokens) per question.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for chart sampling.",
    )
    p.add_argument(
        "--output_json",
        type=Path,
        default=_root / "reports" / "probe_factual_64token_results.json",
        help="Output JSON with per-question results and summary.",
    )
    p.add_argument(
        "--hybrid_chart_text",
        action="store_true",
        help=(
            "Match the hybrid training format: prepend chart.to_prompt_text() "
            "to the question and use the training wrapper "
            "'{chart_text}\\n\\n问题：…\\n答：'. Required to fairly evaluate "
            "checkpoints trained with --hybrid_chart_text. See "
            "plans/hybrid-prompt-v1-and-lora_*.plan.md §6."
        ),
    )
    p.add_argument(
        "--zero_virtual_tokens",
        action="store_true",
        help=(
            "Replace the projection's virtual tokens with zeros at inference. "
            "Combined with --hybrid_chart_text this is the §7 ablation: "
            "'what does the projection add on top of the text block?'."
        ),
    )
    return p.parse_args()


def _to_mx(arr):
    import mlx.core as mx

    if isinstance(arr, mx.array):
        return arr
    return mx.array(arr)


_ANSWER_PREFIX_RE = None


def _extract_short_answer(pred: str) -> str:
    """Pull the likely short answer out of a rambling 64-token generation.

    The probe asks for one-character / one-word answers, but the model often
    emits the answer followed by a long 断语. Substring-matching the whole
    prediction gives false positives (e.g. "水10.0%" matching expected "水"
    when the model actually answered "木"). We want to keep only the first
    short answer chunk before any explanation.
    """
    import re

    global _ANSWER_PREFIX_RE
    if _ANSWER_PREFIX_RE is None:
        _ANSWER_PREFIX_RE = re.compile(r"^\s*(?:答案|回答|解答|答)\s*[:：]\s*")

    if not pred:
        return ""
    text = pred.strip()

    # Drop chat-template leakage (Qwen sometimes leaks <|endoftext|>Human: ...).
    for marker in ("<|endoftext|>", "<|im_end|>", "<|im_start|>", "Human:", "Assistant:"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()

    # First non-empty line.
    first_line = ""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break
    if not first_line:
        return ""

    # Strip repeated answer prefixes ("答案：X" -> "X").
    while True:
        m = _ANSWER_PREFIX_RE.match(first_line)
        if not m:
            break
        first_line = first_line[m.end():].strip()

    first_line = first_line.strip("『』「」\"'“”‘’ \t")

    # The model often emits "答案。然后解释..." — truncate at the first sentence
    # or clause boundary so a trailing 断语 cannot contribute to matching.
    # We intentionally exclude ASCII "." because it appears inside decimal
    # numbers like "0.25" for the percentile question.
    boundary = re.search(r"[。，；;]|,\s", first_line)
    if boundary and boundary.start() > 0:
        first_line = first_line[: boundary.start()]

    first_line = first_line.rstrip("。，,.;；:：!！?？ \t")

    # Factual answers here are short (at most ~6 chars). Cap to keep any
    # residual explanation out of the match window.
    return first_line[:20]


def _get_generate_step_fn():
    """
    Resolve mlx_lm's internal generate_step used by stream_generate.
    """
    import mlx_lm

    fn = mlx_lm.stream_generate
    gs = fn.__globals__.get("generate_step")
    if gs is not None:
        return gs
    raise ImportError("Could not find generate_step in mlx_lm internals.")


def generate_from_chart(
    model,
    natal_inputs: Dict[str, Any],
    dynamic_inputs: Dict[str, Any] | None,
    query_text: str,
    *,
    cross_features=None,
    max_tokens: int = 64,
    zero_virtual_tokens: bool = False,
) -> str:
    """
    Greedy generation from chart + question using virtual tokens as prefix.
    """
    import mlx.core as mx
    from bazi_model.model.bazi_llm import _null_dynamic_inputs

    gs_fn = _get_generate_step_fn()

    keys = getattr(model, "natal_input_keys", tuple(natal_inputs.keys()))
    natal_kw = {k: _to_mx(natal_inputs[k]) for k in keys if k in natal_inputs}
    natal_emb = model.natal_encoder(**natal_kw)

    if dynamic_inputs is not None:
        dynamic_kw = {k: _to_mx(v) for k, v in dynamic_inputs.items()}
    else:
        dynamic_kw = _null_dynamic_inputs(1)
    dynamic_emb = model.dynamic_encoder(**dynamic_kw)

    import mlx.core as mx

    slot_parts = [natal_emb, dynamic_emb[:, None, :]]
    if model.cross_interaction_encoder is not None:
        if cross_features is None:
            cf = mx.zeros((1, 27))
        else:
            cf = cross_features
            if not isinstance(cf, mx.array):
                cf = mx.array(cf)
        cross_emb = model.cross_interaction_encoder(cf)
        slot_parts.append(cross_emb[:, None, :])
    else:
        slot_parts.append(mx.zeros((1, 1, natal_emb.shape[-1])))

    slot_stack = mx.concatenate(slot_parts, axis=1)  # (1, 7, d_model)
    virtual_tokens = model.projection_layer(slot_stack)  # (1, n_virt, d_llm)
    if zero_virtual_tokens:
        # §7 ablation: keep the prompt geometry (and any positional effect of
        # the prefix) but strip the projection's contribution to logits.
        virtual_tokens = mx.zeros_like(virtual_tokens)

    enc = model.tokenizer(
        [query_text],
        padding=False,
        truncation=True,
        max_length=model.max_length,
        return_tensors="np",
    )
    query_ids = mx.array(enc["input_ids"])  # (1, L)
    inner = getattr(model.llm_model, "model", model.llm_model)
    query_emb = inner.embed_tokens(query_ids)  # (1, L, d_llm)

    input_embeddings = mx.concatenate([virtual_tokens, query_emb], axis=1)
    input_embeddings = input_embeddings.squeeze(0)  # (total_len, d_llm)

    total_len = input_embeddings.shape[0]
    prompt = mx.zeros((total_len,), dtype=mx.int32)

    eos_ids = set()
    if getattr(model.tokenizer, "eos_token_id", None) is not None:
        eos_ids.add(int(model.tokenizer.eos_token_id))
    eos_ids_raw = getattr(model.tokenizer, "eos_token_ids", None)
    if eos_ids_raw is not None:
        if isinstance(eos_ids_raw, (list, tuple, set)):
            eos_ids.update(int(t) for t in eos_ids_raw)
        else:
            eos_ids.add(int(eos_ids_raw))

    tokens: List[int] = []
    for tok, _logprobs in gs_fn(
        prompt,
        model.llm_model,
        input_embeddings=input_embeddings,
        max_tokens=max_tokens,
        sampler=None,  # greedy
    ):
        tok_id = int(tok.item()) if hasattr(tok, "item") else int(tok)
        if tok_id in eos_ids:
            break
        tokens.append(tok_id)

    return model.tokenizer.decode(tokens)


def load_bazi_llm(args):
    """
    Load BaziLLM with encoders, projection (64 tokens), cross encoder and frozen Qwen.
    """
    from mlx_lm import load as mlx_lm_load
    from mlx_lm.utils import does_model_support_input_embeddings

    from bazi_model.encoders.dynamic_encoder_mlx import DynamicEncoderMLX
    from bazi_model.encoders.embedding_natal_mlx import EmbeddingNatalMLX
    from bazi_model.encoders.natal_encoder_mlx import NatalEncoderMLX, _flat_to_nested
    from bazi_model.encoders.cross_interaction_encoder_mlx import CrossInteractionEncoderMLX
    from bazi_model.model.bazi_llm import BaziLLM
    from bazi_model.projection import ProjectionLayer
    import numpy as np

    if not args.llm_path.is_dir():
        raise FileNotFoundError(f"LLM path not found: {args.llm_path}")
    if args.natal_mode == "encoder" and not args.natal_weights.exists():
        raise FileNotFoundError(f"Natal encoder weights not found: {args.natal_weights}")
    if not args.dynamic_weights.exists():
        raise FileNotFoundError(f"Dynamic encoder weights not found: {args.dynamic_weights}")
    if not args.projection_checkpoint.exists():
        raise FileNotFoundError(f"Projection checkpoint not found: {args.projection_checkpoint}")

    print("Loading encoders ...")
    if args.natal_mode == "encoder":
        natal_encoder = NatalEncoderMLX(d_emb=32, d_model=256, num_layers=3)
        natal_encoder.load_weights(str(args.natal_weights))
        print("  natal: NatalEncoderMLX (pretrained)")
    else:
        natal_encoder = EmbeddingNatalMLX(d_emb=64, d_model=256)
        print("  natal: EmbeddingNatalMLX (weights from --projection_checkpoint)")
    dynamic_encoder = DynamicEncoderMLX(d_emb=32, d_model=256, num_layers=2)
    dynamic_encoder.load_weights(str(args.dynamic_weights))
    cross_encoder = CrossInteractionEncoderMLX(d_input=27, d_hidden=128, d_model=256)

    # Per-slot projection: 5 natal slots + 1 dynamic + 1 cross = 7 slots.
    n_slots = 7
    if args.n_virtual_tokens % n_slots != 0:
        raise ValueError(
            f"--n_virtual_tokens ({args.n_virtual_tokens}) must be divisible by "
            f"{n_slots} (per-slot projection); use 63 (= 7 * 9)."
        )
    n_virtual_per_slot = args.n_virtual_tokens // n_slots
    projection = ProjectionLayer(
        d_in_per_slot=256,
        d_llm=3584,
        n_slots=n_slots,
        n_virtual_per_slot=n_virtual_per_slot,
    )

    print(f"Loading projection + cross encoder from {args.projection_checkpoint} ...")
    data = dict(np.load(args.projection_checkpoint, allow_pickle=True))

    proj_flat = {
        k.replace("projection_layer.", ""): np.asarray(v)
        for k, v in data.items()
        if k.startswith("projection_layer.")
    }
    if proj_flat:
        nested = _flat_to_nested(proj_flat)
        projection.update(nested, strict=False)

    cross_flat = {
        k.replace("cross_interaction_encoder.", ""): np.asarray(v)
        for k, v in data.items()
        if k.startswith("cross_interaction_encoder.")
    }
    if cross_flat:
        nested = _flat_to_nested(cross_flat)
        cross_encoder.update(nested, strict=False)

    if args.natal_mode == "embedding":
        natal_flat = {
            k.replace("natal_encoder.", ""): np.asarray(v)
            for k, v in data.items()
            if k.startswith("natal_encoder.")
        }
        if not natal_flat:
            raise RuntimeError(
                "Projection checkpoint contains no 'natal_encoder.*' keys; "
                "either pass a checkpoint trained with --natal_mode=embedding or "
                "switch the probe to --natal_mode=encoder."
            )
        nested = _flat_to_nested(natal_flat)
        natal_encoder.update(nested, strict=False)

    print(f"Loading LLM from {args.llm_path} ...")
    llm_model, tokenizer_wrapper = mlx_lm_load(str(args.llm_path))
    if not does_model_support_input_embeddings(llm_model):
        raise RuntimeError("Loaded model does not support input_embeddings.")

    tokenizer = getattr(tokenizer_wrapper, "_tokenizer", tokenizer_wrapper)
    if not callable(tokenizer):
        raise TypeError(f"tokenizer ({type(tokenizer)}) is not callable; expected HF tokenizer.")

    model = BaziLLM(
        natal_encoder,
        dynamic_encoder,
        projection,
        llm_model,
        tokenizer,
        cross_interaction_encoder=cross_encoder,
        max_length=args.max_length,
        n_virtual_tokens=args.n_virtual_tokens,
    )
    model.freeze_except_projection()
    # For pure probing we also freeze projection and cross encoder (no gradients).
    model.projection_layer.freeze()
    if model.cross_interaction_encoder is not None:
        model.cross_interaction_encoder.freeze()
    return model


@dataclass
class ProbeQuestion:
    id: str
    category: str
    text_fn: Callable[[Any], str]
    answer_fn: Callable[[Any], Any]
    judge_fn: Callable[[str, Any], bool]
    # "categorical" = short-answer factual recall (pillars, ten gods, element
    # names, interaction pairs, class labels). "numeric" = numeric/regression
    # output (e.g. day-master percentile, element percentage). The summary
    # reports aggregates per bucket so we can gate on categorical accuracy
    # without being dragged down by hard numeric questions that the current
    # 64-token projection is not expected to nail.
    bucket: str = "categorical"


def _build_questions() -> List[ProbeQuestion]:
    """
    Define 20 single-fact questions across 5 categories.

    Ground truth is always derived from BaziChart (bazi_engine.models.BaziChart).
    """
    from bazi_engine.models import BaziChart, compute_element_strengths, classify_day_master

    def pillar_str(chart: BaziChart, attr: str) -> str:
        p = getattr(chart, f"{attr}_pillar")
        return f"{p.stem.value}{p.branch.value}"

    def stem_str(chart: BaziChart, attr: str) -> str:
        p = getattr(chart, f"{attr}_pillar")
        return p.stem.value

    def ten_god_stem(chart: BaziChart, attr: str) -> str:
        p = getattr(chart, f"{attr}_pillar")
        return p.ten_god_stem.value

    def strongest_element(chart: BaziChart) -> str:
        stems, branches = chart._stems_and_branches()
        strengths = compute_element_strengths(stems, branches)
        best = max(strengths.items(), key=lambda kv: kv[1])[0]
        return best

    def weakest_element(chart: BaziChart) -> str:
        stems, branches = chart._stems_and_branches()
        strengths = compute_element_strengths(stems, branches)
        best = min(strengths.items(), key=lambda kv: kv[1])[0]
        return best

    def dm_strength_percentile(chart: BaziChart) -> float:
        stems, branches = chart._stems_and_branches()
        from bazi_engine.models import compute_day_master_strength

        return float(compute_day_master_strength(stems, branches))

    def dm_strength_label(chart: BaziChart) -> str:
        stems, branches = chart._stems_and_branches()
        cls = classify_day_master(stems, branches)
        return cls

    def interactions_summary(chart: Any, which: str) -> List[Tuple[int, int]]:
        from bazi_engine.interactions import detect_all_interactions

        stems, branches = chart._stems_and_branches()
        ir = detect_all_interactions(stems, branches)
        if which == "branch_clashes":
            pairs = [(r["pos_a"], r["pos_b"]) for r in ir.branch_clashes]
        elif which == "stem_combinations":
            pairs = [(r["pos_a"], r["pos_b"]) for r in ir.stem_combinations]
        elif which == "harms":
            pairs = [(r["pos_a"], r["pos_b"]) for r in ir.harms]
        else:
            pairs = []
        # Normalize order
        return [tuple(sorted(p)) for p in pairs]

    def judge_exact_contains(pred: str, expected: str) -> bool:
        return expected in _extract_short_answer(pred)

    def judge_element(pred: str, expected: str) -> bool:
        # expected is 五行汉字 like "木", "火", etc.
        return expected in _extract_short_answer(pred)

    def judge_strength_label(pred: str, expected_label: str) -> bool:
        # Coarse bin 身强/身弱 on the short answer only, not the whole explanation.
        short = _extract_short_answer(pred)
        is_strong_gt = "强" in expected_label
        contains_strong = "身强" in short or "偏强" in short
        contains_weak = "身弱" in short or "偏弱" in short
        if is_strong_gt:
            return contains_strong and not contains_weak
        return contains_weak and not contains_strong

    def judge_percentile(pred: str, expected_value: float) -> bool:
        # Look for a percentage number within ±5 percentage points, in the short answer.
        import re

        short = _extract_short_answer(pred)
        nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", short)
        if not nums:
            return False
        expected_pct = expected_value * 100.0
        for s in nums:
            try:
                v = float(s)
            except ValueError:
                continue
            # Allow both percentile-as-integer (e.g. "51") and fraction (e.g. "0.51")
            if abs(v - expected_pct) <= 5.0:
                return True
            if abs(v - expected_value) <= 0.05:
                return True
        return False

    def judge_pairs(pred: str, expected_pairs: List[Tuple[int, int]]) -> bool:
        short = _extract_short_answer(pred)
        if not expected_pairs:
            # Ground truth has no such interactions; accept explicit "none" answers only.
            return (
                "无" in short
                or "没有" in short
                or "不存在" in short
                or "没明显" in short
            )
        labels = ("年", "月", "日", "时")
        for a, b in expected_pairs:
            if 0 <= a < 4 and 0 <= b < 4:
                pair_str1 = labels[a] + labels[b]
                pair_str2 = labels[b] + labels[a]
                if pair_str1 in short or pair_str2 in short:
                    return True
        return False

    def judge_day_stem_ten_god(pred: str, expected: str) -> bool:
        # Day stem's ten-god relative to itself is 比肩 by convention; the engine
        # returns "日主" as a display label. Accept either wording.
        short = _extract_short_answer(pred)
        return "比肩" in short or "日主" in short or (expected and expected in short)

    questions: List[ProbeQuestion] = []

    # 1. Pillar identity (4)
    questions.append(
        ProbeQuestion(
            id="pillar-day-stem",
            category="pillar_identity",
            text_fn=lambda c: "日主天干是什么？请只回答一个汉字。",
            answer_fn=lambda c: stem_str(c, "day"),
            judge_fn=judge_exact_contains,
        )
    )
    questions.append(
        ProbeQuestion(
            id="pillar-year",
            category="pillar_identity",
            text_fn=lambda c: "年柱是什么？请回答天干地支，例如“甲子”。",
            answer_fn=lambda c: pillar_str(c, "year"),
            judge_fn=judge_exact_contains,
        )
    )
    questions.append(
        ProbeQuestion(
            id="pillar-month",
            category="pillar_identity",
            text_fn=lambda c: "月柱是什么？请回答天干地支，例如“甲子”。",
            answer_fn=lambda c: pillar_str(c, "month"),
            judge_fn=judge_exact_contains,
        )
    )
    questions.append(
        ProbeQuestion(
            id="pillar-hour",
            category="pillar_identity",
            text_fn=lambda c: "时柱是什么？请回答天干地支，例如“甲子”。",
            answer_fn=lambda c: pillar_str(c, "hour"),
            judge_fn=judge_exact_contains,
        )
    )

    # 2. Ten gods (4)
    questions.append(
        ProbeQuestion(
            id="tengod-year-stem",
            category="ten_gods",
            text_fn=lambda c: "年干的十神是什么？请直接回答“比肩”“正财”等。",
            answer_fn=lambda c: ten_god_stem(c, "year"),
            judge_fn=judge_exact_contains,
        )
    )
    questions.append(
        ProbeQuestion(
            id="tengod-month-stem",
            category="ten_gods",
            text_fn=lambda c: "月干的十神是什么？请直接回答“比肩”“正财”等。",
            answer_fn=lambda c: ten_god_stem(c, "month"),
            judge_fn=judge_exact_contains,
        )
    )
    questions.append(
        ProbeQuestion(
            id="tengod-day-stem",
            category="ten_gods",
            text_fn=lambda c: "日干的十神是什么？请直接回答“比肩”“正财”等。",
            answer_fn=lambda c: ten_god_stem(c, "day"),
            judge_fn=judge_day_stem_ten_god,
        )
    )
    questions.append(
        ProbeQuestion(
            id="tengod-hour-stem",
            category="ten_gods",
            text_fn=lambda c: "时干的十神是什么？请直接回答“比肩”“正财”等。",
            answer_fn=lambda c: ten_god_stem(c, "hour"),
            judge_fn=judge_exact_contains,
        )
    )

    # 3. Five elements (4)
    questions.append(
        ProbeQuestion(
            id="element-strongest",
            category="five_elements",
            text_fn=lambda c: "五行中哪个最旺？请只回答“木火土金水”之一。",
            answer_fn=strongest_element,
            judge_fn=judge_element,
        )
    )
    questions.append(
        ProbeQuestion(
            id="element-weakest",
            category="five_elements",
            text_fn=lambda c: "五行中哪个最弱？请只回答“木火土金水”之一。",
            answer_fn=weakest_element,
            judge_fn=judge_element,
        )
    )
    questions.append(
        ProbeQuestion(
            id="element-water-strong",
            category="five_elements",
            text_fn=lambda c: "这个八字水五行是偏强还是偏弱？请只回答“偏强”或“偏弱”。",
            answer_fn=lambda c: "偏强" if strongest_element(c) == "水" else "偏弱",
            judge_fn=judge_exact_contains,
        )
    )
    questions.append(
        ProbeQuestion(
            id="element-metal-strong",
            category="five_elements",
            text_fn=lambda c: "这个八字金五行是偏强还是偏弱？请只回答“偏强”或“偏弱”。",
            answer_fn=lambda c: "偏强" if strongest_element(c) == "金" else "偏弱",
            judge_fn=judge_exact_contains,
        )
    )

    # 4. Day master strength (4)
    questions.append(
        ProbeQuestion(
            id="dm-strong-weak",
            category="day_master_strength",
            text_fn=lambda c: "日主是身强还是身弱？请只回答“身强”或“身弱”。",
            answer_fn=dm_strength_label,
            judge_fn=judge_strength_label,
        )
    )
    questions.append(
        ProbeQuestion(
            id="dm-strength-percentile",
            category="day_master_strength",
            text_fn=lambda c: "日主强度大约是多少百分位？请给出一个0到100之间的数字即可。",
            answer_fn=dm_strength_percentile,
            judge_fn=judge_percentile,
            bucket="numeric",
        )
    )
    questions.append(
        ProbeQuestion(
            id="dm-classification",
            category="day_master_strength",
            text_fn=lambda c: "请用一个词回答此命局的身强身弱分类，例如“身弱”“偏强”“过旺”等。",
            answer_fn=dm_strength_label,
            judge_fn=judge_exact_contains,
        )
    )
    questions.append(
        ProbeQuestion(
            id="dm-useful-god-primary",
            category="day_master_strength",
            text_fn=lambda c: "此命局用神十神是什么？请只回答“印”“比劫”“财”“食伤”“官杀”之一。",
            answer_fn=lambda c: (c._compute_useful_god(0.0, "")) if False else "",  # placeholder; actual below
            judge_fn=judge_exact_contains,
        )
    )
    # Override the last answer_fn with explicit logic using private helper from models.
    from bazi_engine.models import _compute_useful_god, ELEMENTS

    def useful_god_primary(chart: Any) -> str:
        stems, branches = chart._stems_and_branches()
        from bazi_engine.models import compute_day_master_strength

        dm_pct = compute_day_master_strength(stems, branches)
        ug = _compute_useful_god(dm_pct, chart.day_pillar.stem.element)
        if not ug or ug.get("is_ambiguous", True):
            return "不明"
        return ug.get("primary", "不明")

    questions[-1] = ProbeQuestion(
        id="dm-useful-god-primary",
        category="day_master_strength",
        text_fn=lambda c: "此命局用神十神大类是什么？请只回答“印”“比劫”“财”“食伤”“官杀”之一，若不明确可答“不明”。",
        answer_fn=useful_god_primary,
        judge_fn=lambda pred, exp: exp == "不明" and ("不明" in pred or "难判" in pred)
        or (exp != "不明" and exp in pred),
    )

    # 5. Interactions (4)
    questions.append(
        ProbeQuestion(
            id="interactions-branch-clash",
            category="interactions",
            text_fn=lambda c: "地支有哪些六冲？若没有请回答“没有明显六冲”。",
            answer_fn=lambda c: interactions_summary(c, "branch_clashes"),
            judge_fn=judge_pairs,
        )
    )
    questions.append(
        ProbeQuestion(
            id="interactions-stem-combo",
            category="interactions",
            text_fn=lambda c: "天干有哪些合化或合局？若没有请回答“没有明显天干合化”。",
            answer_fn=lambda c: interactions_summary(c, "stem_combinations"),
            judge_fn=judge_pairs,
        )
    )
    questions.append(
        ProbeQuestion(
            id="interactions-harms",
            category="interactions",
            text_fn=lambda c: "地支有没有刑害之类的不良组合？若没有请回答“没有明显刑害”。",
            answer_fn=lambda c: interactions_summary(c, "harms"),
            judge_fn=judge_pairs,
        )
    )
    questions.append(
        ProbeQuestion(
            id="interactions-any",
            category="interactions",
            text_fn=lambda c: "这个八字总体来看，地支冲合多还是少？请只回答“多”或“少”。",
            answer_fn=lambda c: "多" if interactions_summary(c, "branch_clashes") else "少",
            judge_fn=lambda pred, exp: exp in _extract_short_answer(pred),
        )
    )

    return questions


def _load_phase2_dataset(path: Path):
    from bazi_model.data.phase2_dataset import Phase2ChartDataset

    if not path.exists():
        raise FileNotFoundError(f"Phase 2 test data not found: {path}")
    return Phase2ChartDataset(path)


def _chart_from_record(rec: Dict[str, Any]):
    from bazi_engine.models import BaziChart

    chart_data = rec.get("chart")
    if chart_data is None:
        raise ValueError("Record missing 'chart' field.")
    # chart_data is already a dict produced by BaziChart.to_json(); serialize back to string.
    return BaziChart.from_json(json.dumps(chart_data, ensure_ascii=False))


def run_probe(args) -> Dict[str, Any]:
    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading test data from {args.test_data} ...")
    dataset = _load_phase2_dataset(args.test_data)
    n = len(dataset)
    print(f"  {n} charts available")
    if n == 0:
        raise RuntimeError("Phase 2 test dataset is empty.")

    print("Loading BaziLLM (64-token projection) ...")
    model = load_bazi_llm(args)

    questions = _build_questions()
    if args.num_questions < len(questions):
        # Truncate deterministically if caller requests fewer than 20.
        questions = questions[: args.num_questions]
    elif args.num_questions > len(questions):
        print(
            f"Warning: requested num_questions={args.num_questions} > defined={len(questions)}; "
            "using each question once.",
            file=sys.stderr,
        )

    results: List[Dict[str, Any]] = []
    by_category: Dict[str, Dict[str, int]] = {}
    by_bucket: Dict[str, Dict[str, int]] = {}

    for q in questions:
        idx = random.randrange(0, n)
        rec = dataset._records[idx]
        item = dataset[idx]
        chart = _chart_from_record(rec)

        natal_np = item["natal_inputs"]
        dynamic_np = item["dynamic_inputs"]
        cross_np = item.get("cross_features")

        natal_batch = {k: v[None, ...] for k, v in natal_np.items()}
        if dynamic_np is not None:
            dynamic_batch = {k: v[None, ...] for k, v in dynamic_np.items()}
        else:
            dynamic_batch = None
        cross_batch = cross_np[None, ...] if cross_np is not None else None

        question_text = q.text_fn(chart)
        if args.hybrid_chart_text:
            # Match training format from bazi_model/data/stage1_qa_dataset.py:
            #   {chart_text}\n\n问题：{question}\n答：
            chart_text = chart.to_prompt_text()
            full_query = f"{chart_text}\n\n问题：{question_text}\n答："
        else:
            full_query = (
                "根据下面这个八字，只回答一个简短事实性问题，不要解释。\n"
                f"问题：{question_text}\n"
                "请直接给出答案。"
            )

        answer_gt = q.answer_fn(chart)
        pred_text = generate_from_chart(
            model,
            natal_batch,
            dynamic_batch,
            full_query,
            cross_features=cross_batch,
            max_tokens=args.max_tokens,
            zero_virtual_tokens=args.zero_virtual_tokens,
        )
        is_correct = bool(q.judge_fn(pred_text, answer_gt))

        cat = q.category
        stats = by_category.setdefault(cat, {"n": 0, "correct": 0})
        stats["n"] += 1
        if is_correct:
            stats["correct"] += 1

        bucket = q.bucket
        bstats = by_bucket.setdefault(bucket, {"n": 0, "correct": 0})
        bstats["n"] += 1
        if is_correct:
            bstats["correct"] += 1

        results.append(
            {
                "question_id": q.id,
                "category": q.category,
                "bucket": bucket,
                "chart_index": int(idx),
                "chart_name": rec.get("name", ""),
                "question": question_text,
                "query_text": full_query,
                "expected": answer_gt,
                "prediction": pred_text,
                "correct": is_correct,
            }
        )
        print(
            f"[{q.category}/{bucket}] {q.id}: "
            f"chart={idx}  name={rec.get('name', '')!r}  "
            f"{'OK' if is_correct else 'MISS'}"
        )

    total = len(results)
    correct_total = sum(1 for r in results if r["correct"])
    overall_acc = correct_total / total if total else 0.0

    print("\n=== Probe summary ===")
    print(f"Total questions: {total}")
    print(f"Overall accuracy: {overall_acc:.3f}  ({correct_total}/{total})")
    print("Per bucket:")
    for bucket in ("categorical", "numeric"):
        bstats = by_bucket.get(bucket)
        if not bstats or not bstats["n"]:
            continue
        acc_b = bstats["correct"] / bstats["n"]
        print(f"  {bucket}: {acc_b:.3f}  ({bstats['correct']}/{bstats['n']})")
    print("Per category:")
    for cat, stats in by_category.items():
        n_cat = stats["n"]
        c_cat = stats["correct"]
        acc_cat = c_cat / n_cat if n_cat else 0.0
        print(f"  {cat}: {acc_cat:.3f}  ({c_cat}/{n_cat})")

    summary = {
        "total_questions": total,
        "correct": correct_total,
        "overall_accuracy": overall_acc,
        "per_category": {
            cat: {
                "n": stats["n"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["n"] if stats["n"] else 0.0,
            }
            for cat, stats in by_category.items()
        },
        "per_bucket": {
            bucket: {
                "n": stats["n"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["n"] if stats["n"] else 0.0,
            }
            for bucket, stats in by_bucket.items()
        },
    }
    return {"summary": summary, "results": results}


def main() -> int:
    args = _parse_args()

    try:
        import mlx.core as mx  # noqa: F401
        import mlx.nn as nn  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError as e:
        print("Error: missing MLX / numpy dependencies.", file=sys.stderr)
        print("Install Phase 2 deps: pip install -r requirements_phase2.txt", file=sys.stderr)
        print(f"ImportError: {e}", file=sys.stderr)
        return 1

    try:
        probe_out = run_probe(args)
    except Exception as e:
        print(f"Error while running probe: {e}", file=sys.stderr)
        return 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(probe_out, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

