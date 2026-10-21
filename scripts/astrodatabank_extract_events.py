"""
Extract life events from Astrodatabank biographies via Qwen (Dashscope).

Reads selected_charts.jsonl (chart + name, biography, ...), filters to bio >= 400
chars by default, calls Qwen-Plus to extract structured events per biography, and
writes astrodatabank_events.jsonl with an `events` array per record.

Supports concurrent API calls (--workers N) for faster throughput.

Resumable: if the output file already exists, previously processed records are
skipped (matched by (name, date, time) key) and new records are appended.

Optional: produce 流年/流月 example records (chart with temporal_context and
cross_interactions for each event year/month) for Phase 2 temporal training.

Requires: pip install dashscope python-dotenv. Set DASHSCOPE_API_KEY in .env or
the environment. For US Virginia (not CN/SG) keys, set
DASHSCOPE_BASE_URL=https://dashscope-us.aliyuncs.com/api/v1 in .env.

Usage:
  python scripts/astrodatabank_extract_events.py --workers 10
  python scripts/astrodatabank_extract_events.py --input data/selected_charts.jsonl --min-bio-length 400
  python scripts/astrodatabank_extract_events.py --liunian-output data/astrodatabank_liunian_examples.jsonl --workers 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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

DASHSCOPE_MODEL = "qwen-plus"
# US Virginia endpoint; set DASHSCOPE_BASE_URL to this in .env when using a US (not CN/SG) API key
DASHSCOPE_US_VA_BASE = "https://dashscope-us.aliyuncs.com/api/v1"
EVENTS_SCHEMA = """Array of objects, each with:
- "year": integer (Gregorian year when the event occurred; use null if unknown)
- "month": integer or null (1-12 if known, else null)
- "type": string (e.g. "career", "marriage", "award", "death", "birth", "education", "relocation", "health", "other")
- "description": string (short English description)"""

EXTRACT_SYSTEM = f"""You extract life events from biography text. Output valid JSON only: a single object with one key "events" whose value is an array of event objects. Each event must have: year (int or null), month (int 1-12 or null), type (string), description (string). Use null for unknown year or month. If no clear events, use "events": [].
Schema: {EVENTS_SCHEMA}"""


def _call_qwen_extract_events(biography: str, model: str = DASHSCOPE_MODEL) -> list[dict] | None:
    """Call Dashscope to extract events from biography text. Returns list of event dicts or None on failure."""
    if not biography or not biography.strip():
        return []
    if dashscope is None:
        raise RuntimeError("dashscope not installed; run: pip install dashscope")
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY environment variable is not set")
    base_url = os.environ.get("DASHSCOPE_BASE_URL", "").strip() or None
    if base_url:
        dashscope.base_http_api_url = base_url
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": f"Extract life events from this biography. Output JSON only.\n\n{biography[:12000]}"},
    ]
    try:
        response = dashscope.Generation.call(
            model=model,
            messages=messages,
            result_format="message",
        )
    except Exception:
        return None
    if response.status_code != 200:
        return None
    try:
        content = response.output.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return None
    content = content.strip()
    json_match = re.search(r"\{[\s\S]*\}", content)
    if not json_match:
        return None
    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None
    events = data.get("events")
    if not isinstance(events, list):
        return []
    out = []
    for e in events:
        if not isinstance(e, dict):
            continue
        year = e.get("year")
        if year is not None and not isinstance(year, int):
            try:
                year = int(year)
            except (TypeError, ValueError):
                year = None
        month = e.get("month")
        if month is not None and not isinstance(month, int):
            try:
                month = int(month)
            except (TypeError, ValueError):
                month = None
        if month is not None and not (1 <= month <= 12):
            month = None
        out.append({
            "year": year,
            "month": month,
            "type": str(e.get("type") or "other").strip() or "other",
            "description": str(e.get("description") or "").strip(),
        })
    return out


def _record_key(record: dict) -> tuple[str, str, str]:
    """Unique key for an input record: (name, date, time)."""
    return (record.get("name", ""), record.get("date", ""), record.get("time", ""))


def _load_done_keys(path: Path) -> set[tuple[str, str, str]]:
    """Read already-processed output records and return their keys for resume."""
    done: set[tuple[str, str, str]] = set()
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


def _build_chart_with_dynamic(
    chart_dict: dict,
    event_year: int,
    event_month: int | None,
    gender: int,
) -> dict | None:
    """Build chart JSON with dynamic (temporal_context + cross_interactions) for the given event year/month."""
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
    dayun = dayun_seq.get_period_for_year(event_year)
    liunian = compute_liunian(event_year, day_master_stem=chart.day_pillar.stem.value)
    liuyue = (
        compute_liuyue(event_year, event_month, day_master_stem=chart.day_pillar.stem.value)
        if event_month is not None and 1 <= event_month <= 12
        else None
    )
    ctx = TemporalContext(dayun=dayun, liunian=liunian, liuyue=liuyue, liuri=None)
    stems = [
        chart.year_pillar.stem.value,
        chart.month_pillar.stem.value,
        chart.day_pillar.stem.value,
        chart.hour_pillar.stem.value,
    ]
    branches = [
        chart.year_pillar.branch.value,
        chart.month_pillar.branch.value,
        chart.day_pillar.branch.value,
        chart.hour_pillar.branch.value,
    ]
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


def _process_record(
    record: dict,
    model: str,
    default_gender: int,
    produce_liunian: bool,
) -> tuple[dict, list[dict], bool]:
    """Worker: extract events from one record, optionally build liunian rows.

    Returns (record_with_events, liunian_rows, is_api_error).
    """
    biography = (record.get("biography") or "").strip()
    chart_raw = record.get("chart")

    events = _call_qwen_extract_events(biography, model=model)
    is_error = events is None
    if events is None:
        events = []

    liunian_rows: list[dict] = []
    if produce_liunian and events and isinstance(chart_raw, dict):
        for ev in events:
            year = ev.get("year")
            if year is None or not isinstance(year, int):
                continue
            month = ev.get("month") if isinstance(ev.get("month"), int) and 1 <= ev.get("month", 0) <= 12 else None
            task_type = "流月" if month is not None else "流年"
            chart_with_dyn = _build_chart_with_dynamic(chart_raw, year, month, default_gender)
            if chart_with_dyn is None:
                continue
            query = f"请分析此造在{year}年" + (f"{month}月" if month else "") + "的运势。"
            liunian_rows.append({
                "chart": chart_with_dyn,
                "task_type": task_type,
                "query": query,
                "reasoning": "",
                "output": "",
                "event_year": year,
                "event_month": month,
                "event_type": ev.get("type"),
                "event_description": ev.get("description"),
                "name": record.get("name"),
            })

    record_out = {**record, "events": events}
    return record_out, liunian_rows, is_error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract life events from Astrodatabank bios via Qwen (Dashscope); optional 流年/流月 examples"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/selected_charts.jsonl",
        help="Input JSONL path (default: data/selected_charts.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/astrodatabank_events.jsonl",
        help="Output event-annotated JSONL path",
    )
    parser.add_argument(
        "--liunian-output",
        type=str,
        default=None,
        help="Optional: write 流年/流月 example records (chart with dynamic) to this JSONL",
    )
    parser.add_argument(
        "--default-gender",
        type=int,
        default=1,
        choices=(0, 1),
        help="Gender for 大运 when building 流年/流月 (0=female, 1=male); used only with --liunian-output",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DASHSCOPE_MODEL,
        help="Dashscope model name (default: qwen-plus)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between API calls (rate limit)",
    )
    parser.add_argument(
        "--min-bio-length",
        type=int,
        default=400,
        help="Skip records with biography shorter than this (default: 400)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max records to process, excluding skipped/resumed (default: all)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite output instead of resuming from existing progress",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of concurrent API workers (default: 10)",
    )
    args = parser.parse_args()

    if dashscope is None:
        print("Error: dashscope not installed. Run: pip install dashscope", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    liunian_path = Path(args.liunian_output) if args.liunian_output else None
    if liunian_path:
        liunian_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys: set[tuple[str, str, str]] = set()
    if not args.no_resume:
        done_keys = _load_done_keys(output_path)
    if done_keys:
        print(f"Resuming: {len(done_keys)} records already in {output_path}")

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("Error: DASHSCOPE_API_KEY not set", file=sys.stderr)
        return 1

    base_url = os.environ.get("DASHSCOPE_BASE_URL", "").strip() or None
    if base_url:
        dashscope.base_http_api_url = base_url

    eligible: list[dict] = []
    skipped_bio = 0
    skipped_other = 0
    resumed = 0

    print(f"Loading and filtering {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_other += 1
                continue
            if not record.get("chart"):
                skipped_other += 1
                continue
            if len((record.get("biography") or "").strip()) < args.min_bio_length:
                skipped_bio += 1
                continue
            if _record_key(record) in done_keys:
                resumed += 1
                continue
            eligible.append(record)

    if args.limit is not None:
        eligible = eligible[: args.limit]

    total = len(eligible)
    print(
        f"Eligible: {total} to process"
        f" ({resumed} resumed, {skipped_bio} short bio, {skipped_other} other skips)"
    )
    if total == 0:
        print("Nothing to do.")
        return 0

    file_mode = "a" if done_keys else "w"
    produce_liunian = liunian_path is not None
    processed = 0
    events_total = 0
    liunian_count = 0
    api_errors = 0
    t_start = time.monotonic()

    with open(output_path, file_mode, encoding="utf-8") as f_out:
        f_liu = open(liunian_path, file_mode, encoding="utf-8") if liunian_path else None
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_rec = {
                    executor.submit(
                        _process_record, rec, args.model, args.default_gender, produce_liunian
                    ): rec
                    for rec in eligible
                }
                for future in as_completed(future_to_rec):
                    try:
                        record_out, liunian_rows, is_error = future.result()
                    except Exception:
                        api_errors += 1
                        continue

                    if is_error:
                        api_errors += 1
                    events = record_out.get("events", [])
                    events_total += len(events)

                    if f_liu and liunian_rows:
                        for row in liunian_rows:
                            f_liu.write(json.dumps(row, ensure_ascii=False) + "\n")
                        liunian_count += len(liunian_rows)
                        f_liu.flush()

                    f_out.write(json.dumps(record_out, ensure_ascii=False) + "\n")
                    f_out.flush()
                    processed += 1

                    if processed % 10 == 0 or processed == total:
                        elapsed = time.monotonic() - t_start
                        rate = processed / elapsed if elapsed > 0 else 0
                        eta = (total - processed) / rate if rate > 0 else 0
                        print(
                            f"  {processed}/{total}"
                            f" ({rate:.1f}/s, ETA {eta/60:.0f}m)"
                            f" events={events_total}"
                            f" errors={api_errors}"
                            f" liunian={liunian_count}",
                            end="\r",
                        )
        finally:
            if f_liu:
                f_liu.close()

    elapsed = time.monotonic() - t_start
    print(
        f"\nDone: {processed}/{total} processed in {elapsed/60:.1f}m,"
        f" {resumed} resumed, {api_errors} API errors,"
        f" {events_total} events, {liunian_count} 流年/流月 examples."
    )
    print(f"Output: {output_path}")
    if liunian_path:
        print(f"流年/流月 examples: {liunian_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
