"""
Qwen-Plus bulk interpretation generation: selected_charts.jsonl → qwen_raw.jsonl

Reads selected charts, assigns task types per the Phase 2 target distribution
(本命分析 40%, 流年 30%, 流月 10%, 八字问答 20%), builds prompts via phase2_prompts,
calls Qwen-Plus (Dashscope), parses JSON responses, and writes one JSONL record
per (chart, task_type) with fields: chart, task_type, query, reasoning, output.

For 流年/流月 tasks, computes dynamic context (大运/流年/流月 + cross interactions)
and attaches to the chart JSON before prompt building and output.

Resumable: skips records already present in the output file (matched by a
composite key). Supports concurrent API workers.

Requires: pip install dashscope python-dotenv
Set DASHSCOPE_API_KEY in .env or the environment.

Usage:
  python scripts/qwen_bulk_generate.py
  python scripts/qwen_bulk_generate.py --workers 10 --limit 100
  python scripts/qwen_bulk_generate.py --input data/selected_charts.jsonl --output data/qwen_raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

try:
    import dashscope
except ImportError:
    dashscope = None

from scripts.phase2_prompts import (
    _QA_TEMPLATES,
    build_prompt,
)
from scripts.prepare_phase2_dataset import TASK_TYPES, TARGET_DISTRIBUTION

DASHSCOPE_MODEL = "qwen-plus"


# ── events index (for grounded temporal year selection) ──────────────────────

def _chart_identity(record_or_chart: dict | Any, *, record: dict | None = None) -> str:
    """
    Stable identity for a chart record: birth_datetime + name.

    Accepts either a full record (with "chart" and "name" fields) or a chart dict
    directly. The optional ``record`` kwarg provides the outer record when only the
    chart dict is passed (for name lookup).
    """
    if isinstance(record_or_chart, dict) and "chart" in record_or_chart:
        chart = record_or_chart.get("chart") or {}
        name = record_or_chart.get("name") or ""
    elif record is not None:
        chart = record_or_chart if isinstance(record_or_chart, dict) else {}
        name = record.get("name") or ""
    else:
        chart = record_or_chart if isinstance(record_or_chart, dict) else {}
        name = ""
    bdt = (chart.get("metadata") or {}).get("birth_datetime", "")
    return f"{bdt}|{name}"


def load_events_index(events_path: Path) -> dict[str, list[dict]]:
    """
    Load astrodatabank_events.jsonl and index by chart identity (birth_datetime + name).

    Returns {identity: [event_dicts_with_year]} — only events
    that have a non-null year are included.
    """
    index: dict[str, list[dict]] = {}
    if not events_path.exists():
        return index
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = _chart_identity(rec)
            if not cid:
                continue
            events = rec.get("events") or []
            dated = [e for e in events if isinstance(e.get("year"), int)]
            if dated:
                index[cid] = dated
    return index


# ── task assignment ──────────────────────────────────────────────────────────

def assign_task_types(
    records: list[dict],
    seed: int,
) -> list[tuple[dict, str, str | None]]:
    """
    Assign task types to records following TARGET_DISTRIBUTION.

    Returns list of (record, task_type, qa_question_or_None).
    Each record gets exactly one task type. The assignment is deterministic
    for a given seed.
    """
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)

    n = len(records)
    counts = {}
    for task, frac in TARGET_DISTRIBUTION.items():
        counts[task] = max(1, round(n * frac))
    remainder = n - sum(counts.values())
    if remainder > 0:
        counts["本命分析"] += remainder
    elif remainder < 0:
        for task in ("八字问答", "流月", "流年", "本命分析"):
            can_remove = min(-remainder, counts[task] - 1)
            counts[task] -= can_remove
            remainder += can_remove
            if remainder >= 0:
                break

    assignments: list[tuple[dict, str, str | None]] = []
    idx = 0
    for task in TASK_TYPES:
        for _ in range(counts.get(task, 0)):
            if idx >= n:
                break
            rec = records[idx]
            question = rng.choice(_QA_TEMPLATES) if task == "八字问答" else None
            assignments.append((rec, task, question))
            idx += 1

    return assignments


# ── dynamic context (for 流年/流月) ──────────────────────────────────────────

def _pick_temporal_year_month(
    chart_dict: dict,
    task_type: str,
    rng: random.Random,
    events: list[dict] | None = None,
) -> tuple[int, int | None, dict | None]:
    """
    Pick a (year, month_or_None, event_or_None) for temporal tasks.

    Strategy: if events are available for this chart, pick one at random
    (preferring events with month for 流月 tasks). Otherwise fall back to
    a random age 20–55 after birth.

    The returned event dict (if any) is for metadata only — its content
    must NOT be passed to the LLM prompt (to avoid validation leakage).
    """
    birth_dt_str = (chart_dict.get("metadata") or {}).get("birth_datetime", "")
    try:
        birth_year = int(birth_dt_str[:4])
    except (ValueError, IndexError):
        birth_year = 1980

    if events:
        if task_type == "流月":
            with_month = [e for e in events if isinstance(e.get("month"), int) and 1 <= e["month"] <= 12]
            pool = with_month if with_month else events
        else:
            pool = events

        # Filter to adult years (age >= 10) to avoid birth/infancy events
        adult = [e for e in pool if e["year"] >= birth_year + 10]
        pool = adult if adult else pool

        event = rng.choice(pool)
        year = event["year"]
        em = event.get("month")
        month = em if task_type == "流月" and isinstance(em, int) and 1 <= em <= 12 else None
        if task_type == "流月" and month is None:
            month = rng.randint(1, 12)
        return year, month, event

    age = rng.randint(20, 55)
    year = birth_year + age
    month = rng.randint(1, 12) if task_type == "流月" else None
    return year, month, None


def build_chart_with_dynamic(
    chart_dict: dict,
    year: int,
    month: int | None,
    gender: int,
) -> dict | None:
    """Attach temporal_context and cross_interactions to chart JSON for the given year/month."""
    from bazi_engine.calculator import BaziCalculator
    from bazi_engine.models import BaziChart
    from bazi_engine.temporal import (
        TemporalContext,
        compute_liunian,
        compute_liuyue,
        detect_cross_interactions,
    )

    try:
        chart = BaziChart.from_json(json.dumps(chart_dict))
    except Exception:
        return None

    dayun_seq = BaziCalculator().compute_dayun(chart, gender, max_periods=10)
    dayun = dayun_seq.get_period_for_year(year)
    liunian = compute_liunian(year, day_master_stem=chart.day_pillar.stem.value)
    liuyue = (
        compute_liuyue(year, month, day_master_stem=chart.day_pillar.stem.value)
        if month is not None and 1 <= month <= 12
        else None
    )
    ctx = TemporalContext(dayun=dayun, liunian=liunian, liuyue=liuyue, liuri=None)
    stems = [p.stem.value for p in [chart.year_pillar, chart.month_pillar, chart.day_pillar, chart.hour_pillar]]
    branches = [p.branch.value for p in [chart.year_pillar, chart.month_pillar, chart.day_pillar, chart.hour_pillar]]
    cross = detect_cross_interactions(stems, branches, ctx, useful_god=None)

    chart_with_dyn = BaziChart(
        birth_dt=chart.birth_dt,
        solar_dt=chart.solar_dt,
        lat=chart.lat,
        lng=chart.lng,
        year_pillar=chart.year_pillar,
        month_pillar=chart.month_pillar,
        day_pillar=chart.day_pillar,
        hour_pillar=chart.hour_pillar,
        sect=chart.sect,
        temporal_context=ctx,
        cross_interactions=cross,
    )
    return chart_with_dyn._to_json_dict()


# ── Dashscope API call ───────────────────────────────────────────────────────

_CALL_TIMEOUT_SECONDS = 120


def _call_qwen(
    system_prompt: str,
    user_prompt: str,
    model: str = DASHSCOPE_MODEL,
    max_retries: int = 3,
) -> dict[str, str] | None:
    """
    Call Dashscope Qwen to get a JSON response with 'reasoning' and 'output'.

    Returns {"reasoning": ..., "output": ...} on success, None on failure.
    Retries on transient errors with exponential backoff.
    Each attempt has a hard timeout to prevent indefinite hangs.
    """
    if dashscope is None:
        raise RuntimeError("dashscope not installed; run: pip install dashscope")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(max_retries):
        try:
            with ThreadPoolExecutor(max_workers=1) as timeout_pool:
                future = timeout_pool.submit(
                    dashscope.Generation.call,
                    model=model,
                    messages=messages,
                    result_format="message",
                )
                response = future.result(timeout=_CALL_TIMEOUT_SECONDS)
        except TimeoutError:
            print(f"  [timeout] API call timed out after {_CALL_TIMEOUT_SECONDS}s "
                  f"(attempt {attempt + 1}/{max_retries})", file=sys.stderr)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None

        if response.status_code == 429:
            time.sleep(2 ** (attempt + 1))
            continue
        if response.status_code != 200:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None

        try:
            content = response.output.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            return None

        return _parse_json_response(content)

    return None


def _parse_json_response(content: str) -> dict[str, str] | None:
    """Extract reasoning and output from LLM response text."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    json_match = re.search(r"\{[\s\S]*\}", content)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None

    reasoning = str(data.get("reasoning") or "").strip()
    output = str(data.get("output") or "").strip()

    if not reasoning and not output:
        return None

    return {"reasoning": reasoning, "output": output}


# ── per-record worker ────────────────────────────────────────────────────────

def _process_one(
    record: dict,
    task_type: str,
    question: str | None,
    model: str,
    gender: int,
    rng_seed: int,
    events: list[dict] | None = None,
) -> tuple[dict | None, bool]:
    """
    Process one (record, task_type) pair: build prompt, call API, parse response.

    Events (if provided) are used ONLY for picking the temporal year — their
    content is never passed to the LLM prompt.

    Returns (output_record_or_None, is_api_error).
    """
    chart_raw = record.get("chart")
    if not isinstance(chart_raw, dict):
        return None, False

    chart_for_prompt = chart_raw
    rng = random.Random(rng_seed)
    source_event = None

    if task_type in ("流年", "流月"):
        year, month, source_event = _pick_temporal_year_month(
            chart_raw, task_type, rng, events=events,
        )
        chart_for_prompt = build_chart_with_dynamic(chart_raw, year, month, gender)
        if chart_for_prompt is None:
            return None, False

    try:
        prompt = build_prompt(chart_for_prompt, task_type, question=question)
    except (ValueError, KeyError, TypeError):
        return None, False

    result = _call_qwen(prompt["system_prompt"], prompt["user_prompt"], model=model)
    if result is None:
        return None, True

    out: dict[str, Any] = {
        "chart": chart_for_prompt,
        "task_type": task_type,
        "query": prompt["query"],
        "reasoning": result["reasoning"],
        "output": result["output"],
    }
    if "name" in record:
        out["name"] = record["name"]
    if "selection_source" in record:
        out["selection_source"] = record["selection_source"]
    if source_event is not None:
        out["_ground_truth_event"] = {
            "year": source_event.get("year"),
            "month": source_event.get("month"),
            "type": source_event.get("type"),
        }

    return out, False


# ── resume support ───────────────────────────────────────────────────────────

def _record_key(record: dict) -> str:
    """Unique key for a processed record: chart_identity + task_type."""
    chart_id = _chart_identity(record)
    task_type = record.get("task_type") or ""
    return f"{chart_id}||{task_type}"


def _load_done_keys(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(_record_key(r))
    return done


def _assignment_key(record: dict, task_type: str, _question: str | None) -> str:
    """Key for a pending assignment; mirrors _record_key on the output side."""
    chart_id = _chart_identity(record)
    return f"{chart_id}||{task_type}"


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qwen-Plus bulk interpretation generation → qwen_raw.jsonl"
    )
    parser.add_argument(
        "--input", "-i", type=str, default="data/selected_charts.jsonl",
        help="Input JSONL with chart records",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="data/qwen_raw.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--model", type=str, default=DASHSCOPE_MODEL,
        help="Dashscope model name (default: qwen-plus)",
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Number of concurrent API workers (default: 10)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max records to process (after resume, before task assignment)",
    )
    parser.add_argument(
        "--default-gender", type=int, default=1, choices=(0, 1),
        help="Gender for 大运 computation in temporal tasks (0=female, 1=male)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for task assignment and temporal year/month selection",
    )
    parser.add_argument(
        "--events", type=str, default="data/astrodatabank_events.jsonl",
        help="Events JSONL for grounded temporal year selection (default: data/astrodatabank_events.jsonl)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Overwrite output instead of resuming from existing progress",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show assignment stats without calling the API",
    )
    args = parser.parse_args()

    if dashscope is None and not args.dry_run:
        print("Error: dashscope not installed. Run: pip install dashscope", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key and not args.dry_run:
        print("Error: DASHSCOPE_API_KEY not set", file=sys.stderr)
        return 1
    base_url = os.environ.get("DASHSCOPE_BASE_URL", "").strip() or None
    if base_url and dashscope is not None:
        dashscope.base_http_api_url = base_url

    # Load input records
    print(f"Loading {input_path}...")
    records: list[dict] = []
    skipped = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not rec.get("chart"):
                skipped += 1
                continue
            records.append(rec)

    if args.limit is not None:
        records = records[: args.limit]

    print(f"Loaded {len(records)} records ({skipped} skipped)")

    # Load events index for grounded temporal year selection
    events_path = Path(args.events)
    events_index: dict[str, list[dict]] = {}
    if events_path.exists():
        events_index = load_events_index(events_path)
        print(f"Events index: {len(events_index)} charts with dated events")
    else:
        print(f"Events file not found ({events_path}); using random year selection for temporal tasks")

    # Assign task types
    assignments = assign_task_types(records, args.seed)
    task_counts: dict[str, int] = {}
    for _, task, _ in assignments:
        task_counts[task] = task_counts.get(task, 0) + 1
    print(f"Task assignment ({len(assignments)} total):")
    for task in TASK_TYPES:
        cnt = task_counts.get(task, 0)
        pct = cnt / len(assignments) * 100 if assignments else 0
        print(f"  {task}: {cnt} ({pct:.1f}%)")

    if events_index:
        temporal_with_events = sum(
            1 for rec, task, _ in assignments
            if task in ("流年", "流月")
            and _chart_identity(rec) in events_index
        )
        temporal_total = sum(1 for _, task, _ in assignments if task in ("流年", "流月"))
        print(f"Temporal tasks with event grounding: {temporal_with_events}/{temporal_total}")

    if args.dry_run:
        print("Dry run — exiting without API calls.")
        return 0

    # Resume support
    done_keys: set[str] = set()
    if not args.no_resume:
        done_keys = _load_done_keys(output_path)
    if done_keys:
        print(f"Resuming: {len(done_keys)} records already in {output_path}")

    pending: list[tuple[dict, str, str | None, int, list[dict] | None]] = []
    for i, (rec, task, question) in enumerate(assignments):
        ak = _assignment_key(rec, task, question)
        if ak in done_keys:
            continue
        cid = _chart_identity(rec)
        rec_events = events_index.get(cid) if task in ("流年", "流月") else None
        pending.append((rec, task, question, args.seed + i, rec_events))

    total = len(pending)
    print(f"Pending: {total} to process")
    if total == 0:
        print("Nothing to do.")
        return 0

    file_mode = "a" if done_keys else "w"
    processed = 0
    api_errors = 0
    unexpected_errors = 0
    skipped_build = 0
    t_start = time.monotonic()

    with open(output_path, file_mode, encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_info = {
                executor.submit(
                    _process_one, rec, task, question, args.model,
                    args.default_gender, rng_seed, rec_events,
                ): (rec, task)
                for rec, task, question, rng_seed, rec_events in pending
            }

            for future in as_completed(future_to_info):
                rec, task = future_to_info[future]
                try:
                    result, is_error = future.result()
                except Exception as exc:
                    unexpected_errors += 1
                    if unexpected_errors <= 5:
                        print(f"\n  unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
                    processed += 1
                    continue

                if is_error:
                    api_errors += 1
                elif result is None:
                    skipped_build += 1
                else:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()

                processed += 1
                if processed % 20 == 0 or processed == total:
                    elapsed = time.monotonic() - t_start
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (total - processed) / rate if rate > 0 else 0
                    err_total = api_errors + unexpected_errors
                    print(
                        f"  {processed}/{total}"
                        f" ({rate:.1f}/s, ETA {eta / 60:.0f}m)"
                        f" errors={err_total}"
                        f" build_skip={skipped_build}",
                        end="\r",
                    )

    elapsed = time.monotonic() - t_start
    written = processed - api_errors - unexpected_errors - skipped_build
    print(
        f"\nDone: {processed}/{total} processed in {elapsed / 60:.1f}m,"
        f" {written} written, {api_errors} API errors,"
        f" {unexpected_errors} unexpected errors,"
        f" {skipped_build} build failures."
    )
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
