#!/usr/bin/env python3
"""
Generate Stage 2 multi-fact QA pairs from a chart JSONL file.

Stage 2 answers compose 2--4 verified atomic facts per question (e.g. "which
pillar has 正官 and what are its hidden stems"). Every composed answer is
verified against the BaziChart object before writing; any inconsistency aborts
the run.

Usage (from repo root):
  python scripts/generate_stage2_qa.py \\
    --input data/charts.jsonl \\
    --output data/stage2_qa.jsonl \\
    --num_per_chart 10 \\
    --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.qa_templates import (
    QATemplate,
    applicable_templates,
    build_stage2_templates,
)
from scripts.qa_utils import stable_chart_id, write_jsonl_line


_WORKER_TEMPLATES: list[QATemplate] | None = None


def _worker_init() -> None:
    global _WORKER_TEMPLATES
    _WORKER_TEMPLATES = build_stage2_templates()


def _process_chart(
    args: tuple[int, str, int, int],
) -> tuple[str, dict[str, Any], list[dict[str, Any]], list[str]]:
    idx, raw_line, num_per_chart, base_seed = args
    assert _WORKER_TEMPLATES is not None

    from bazi_engine.models import BaziChart

    rec = json.loads(raw_line)
    chart_json = rec.get("chart") if isinstance(rec, dict) and "chart" in rec else rec
    chart_id = stable_chart_id(chart_json)
    chart = BaziChart.from_json(json.dumps(chart_json, ensure_ascii=False))

    apps = applicable_templates(_WORKER_TEMPLATES, chart_json)
    rng = random.Random(base_seed + idx)
    k = min(num_per_chart, len(apps))
    sampled = rng.sample(apps, k) if k < len(apps) else apps

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for tmpl in sampled:
        try:
            question = tmpl.text_fn(chart_json)
            answer = tmpl.answer_fn(chart_json)
        except Exception as e:
            errors.append(f"{chart_id}|{tmpl.id}|answer_fn raised: {e!r}")
            continue
        if not isinstance(answer, str):
            errors.append(f"{chart_id}|{tmpl.id}|answer not a string: {type(answer).__name__}")
            continue
        try:
            ok = tmpl.verify_fn(chart_json, chart, answer)
        except Exception as e:
            errors.append(f"{chart_id}|{tmpl.id}|verify_fn raised: {e!r}")
            continue
        if not ok:
            errors.append(f"{chart_id}|{tmpl.id}|verify_fn returned False; answer={answer!r}")
            continue
        records.append(
            {
                "chart_id": chart_id,
                "question": question,
                "answer": answer,
                "question_category": tmpl.category,
                "template_id": tmpl.id,
            }
        )
    return chart_id, chart_json, records, errors


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Stage 2 multi-fact QA pairs from a chart JSONL."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=_root / "data" / "charts.jsonl",
        help="Chart JSONL.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_root / "data" / "stage2_qa.jsonl",
        help="Output QA JSONL (one QA record per line; references chart_id).",
    )
    p.add_argument(
        "--charts_output",
        type=Path,
        default=_root / "data" / "qa_charts.jsonl",
        help="Shared companion charts JSONL (same default as Stage 1 generator).",
    )
    p.add_argument(
        "--num_per_chart",
        type=int,
        default=10,
        help="Number of sampled templates per chart.",
    )
    p.add_argument("--max_total", type=int, default=0, help="Stop after writing this many records (0 = no cap).")
    p.add_argument("--max_charts", type=int, default=0, help="Stop after processing this many charts.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (mp.cpu_count() or 2) - 1),
    )
    p.add_argument("--chunk_size", type=int, default=64)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    charts_output: Path = args.charts_output
    charts_output.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    written = 0
    charts_written = 0
    seen_chart_ids: set[str] = set()
    per_category: Counter[str] = Counter()
    per_template: Counter[str] = Counter()
    total_errors: list[str] = []

    def _iter_work():
        with args.input.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if args.max_charts and i >= args.max_charts:
                    return
                yield (i, line, args.num_per_chart, args.seed)

    with (
        args.output.open("w", encoding="utf-8") as fout,
        charts_output.open("w", encoding="utf-8") as fcharts,
        mp.Pool(processes=args.workers, initializer=_worker_init) as pool,
    ):
        for chart_id, chart_json, records, errors in pool.imap_unordered(
            _process_chart, _iter_work(), chunksize=args.chunk_size
        ):
            if errors:
                total_errors.extend(errors)
                if len(total_errors) > 20:
                    break
            if records and chart_id not in seen_chart_ids:
                write_jsonl_line(fcharts, {"chart_id": chart_id, "chart": chart_json})
                seen_chart_ids.add(chart_id)
                charts_written += 1
            for rec in records:
                if args.max_total and written >= args.max_total:
                    break
                write_jsonl_line(fout, rec)
                written += 1
                per_category[rec["question_category"]] += 1
                per_template[rec["template_id"]] += 1
            if args.max_total and written >= args.max_total:
                break
            if written and written % 100_000 == 0:
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else 0.0
                print(
                    f"[stage2] wrote={written}  charts={charts_written}  elapsed={elapsed:.1f}s  rate={rate:.0f}/s",
                    flush=True,
                )

    if total_errors:
        print("Verification failures (first 20):", file=sys.stderr)
        for e in total_errors[:20]:
            print(f"  {e}", file=sys.stderr)
        return 2

    elapsed = time.time() - t0
    print(f"\n[stage2] done: {written} QA records -> {args.output}")
    print(f"[stage2] charts: {charts_written} -> {charts_output}")
    print(f"[stage2] elapsed: {elapsed:.1f}s  rate: {written / max(elapsed, 1e-6):.0f}/s")
    print("\n[stage2] per-category:")
    for cat, n in per_category.most_common():
        print(f"  {cat}: {n}")
    print("\n[stage2] per-template:")
    for tid, n in per_template.most_common():
        print(f"  {tid}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
