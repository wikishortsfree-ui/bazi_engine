"""
Claude Tier 2 regeneration and merge → synthetic_final.jsonl

Takes Tier 2 examples (score 60–79) from the QC scoring pipeline, regenerates
interpretations with Claude (Anthropic), replaces reasoning/output, and merges
with Tier 1 to produce the final synthetic dataset.

Requires: pip install anthropic python-dotenv
Set ANTHROPIC_API_KEY in .env or the environment.

Usage:
  # Regenerate Tier 2 records
  python scripts/claude_tier2_boost.py -i data/synthetic_scored.jsonl -o data/claude_improved.jsonl

  # Regenerate and merge with Tier 1
  python scripts/claude_tier2_boost.py -i data/synthetic_scored.jsonl -o data/claude_improved.jsonl \\
      --merge data/synthetic_final.jsonl

  # Dry run (show Tier 2 stats without calling API)
  python scripts/claude_tier2_boost.py -i data/synthetic_scored.jsonl --dry-run

  # Use a pre-filtered tier2 list (from phase2_quality_filter --tier2-list)
  python scripts/claude_tier2_boost.py -i data/tier2_list.jsonl -o data/claude_improved.jsonl --all-records
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
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

from scripts.phase2_prompts import build_prompt

CLAUDE_MODEL = "claude-3-5-sonnet-latest"
MAX_TOKENS = 2048

_QC_FIELDS = frozenset(
    {"factual_issues", "language_issues", "score", "tier", "factual_ok", "language_ok"}
)


# ── record identity (mirrors qwen_bulk_generate pattern) ────────────────────


def _record_key(record: dict) -> str:
    chart = record.get("chart") or {}
    bdt = (chart.get("metadata") or {}).get("birth_datetime", "")
    name = record.get("name", "")
    task_type = record.get("task_type", "")
    return f"{bdt}|{name}||{task_type}"


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
                done.add(_record_key(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return done


# ── JSON response parsing (same logic as qwen_bulk_generate) ────────────────


def _parse_json_response(content: str) -> dict[str, str] | None:
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


# ── Claude API call ─────────────────────────────────────────────────────────


def _call_claude(
    client: Any,
    system_prompt: str,
    user_prompt: str,
    model: str = CLAUDE_MODEL,
    max_tokens: int = MAX_TOKENS,
    max_retries: int = 3,
) -> dict[str, str] | None:
    import anthropic as _anthropic

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except _anthropic.RateLimitError:
            time.sleep(2 ** (attempt + 2))
            continue
        except _anthropic.APIStatusError as e:
            if e.status_code >= 500 and attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            return None
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None

        try:
            content = response.content[0].text
        except (AttributeError, IndexError, TypeError):
            return None

        return _parse_json_response(content)

    return None


# ── per-record worker ────────────────────────────────────────────────────────


def _strip_qc_fields(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in _QC_FIELDS}


def _process_one(
    client: Any,
    record: dict,
    model: str,
    max_tokens: int,
    max_retries: int,
) -> tuple[dict | None, bool]:
    """
    Regenerate one record with Claude.

    Returns (improved_record_or_None, is_api_error).
    """
    chart = record.get("chart")
    if not isinstance(chart, dict):
        return None, False

    task_type = record.get("task_type", "")
    question = record.get("query") if task_type == "八字问答" else None

    try:
        prompt = build_prompt(chart, task_type, question=question)
    except (ValueError, KeyError, TypeError):
        return None, False

    result = _call_claude(
        client,
        prompt["system_prompt"],
        prompt["user_prompt"],
        model=model,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
    if result is None:
        return None, True

    out = _strip_qc_fields(record)
    out["reasoning"] = result["reasoning"]
    out["output"] = result["output"]
    out["_claude_regenerated"] = True
    return out, False


# ── I/O helpers ──────────────────────────────────────────────────────────────


def _load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── regeneration loop ────────────────────────────────────────────────────────


def _run_regeneration(
    tier2_records: list[dict],
    output_path: Path,
    *,
    model: str,
    max_tokens: int,
    max_retries: int,
    workers: int,
    no_resume: bool,
    api_key: str,
) -> int:
    """Run the Claude regeneration loop. Returns count of records written."""
    done_keys: set[str] = set()
    if not no_resume:
        done_keys = _load_done_keys(output_path)
    if done_keys:
        print(f"Resuming: {len(done_keys)} records already in {output_path}")

    pending = [r for r in tier2_records if _record_key(r) not in done_keys]
    total = len(pending)
    print(f"Pending: {total} to process")
    if total == 0:
        print("Nothing to regenerate.")
        return 0

    client = anthropic.Anthropic(api_key=api_key)
    file_mode = "a" if done_keys else "w"
    processed = 0
    api_errors = 0
    unexpected_errors = 0
    build_failures = 0
    written = 0
    t_start = time.monotonic()

    with open(output_path, file_mode, encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_rec = {
                executor.submit(
                    _process_one, client, rec, model, max_tokens, max_retries,
                ): rec
                for rec in pending
            }

            for future in as_completed(future_to_rec):
                try:
                    result, is_error = future.result()
                except Exception as exc:
                    unexpected_errors += 1
                    if unexpected_errors <= 5:
                        print(
                            f"\n  unexpected error: {type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )
                    processed += 1
                    continue

                if is_error:
                    api_errors += 1
                elif result is None:
                    build_failures += 1
                else:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
                    written += 1

                processed += 1
                if processed % 10 == 0 or processed == total:
                    elapsed = time.monotonic() - t_start
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (total - processed) / rate if rate > 0 else 0
                    print(
                        f"  {processed}/{total}"
                        f" ({rate:.1f}/s, ETA {eta / 60:.0f}m)"
                        f" written={written} errors={api_errors + unexpected_errors}"
                        f" build_fail={build_failures}",
                        end="\r",
                    )

    elapsed = time.monotonic() - t_start
    print(
        f"\nRegeneration done: {processed}/{total} in {elapsed / 60:.1f}m,"
        f" {written} written, {api_errors} API errors,"
        f" {unexpected_errors} unexpected, {build_failures} build failures."
    )
    return written


# ── merge ────────────────────────────────────────────────────────────────────


def _run_merge(
    scored_records: list[dict],
    improved_path: Path,
    merge_path: Path,
    *,
    include_unimproved: bool = False,
) -> None:
    improved = _load_records(improved_path)
    improved_keys = {_record_key(r) for r in improved}

    tier1 = [r for r in scored_records if r.get("tier") == 1]
    tier2 = [r for r in scored_records if r.get("tier") == 2]
    unimproved = [r for r in tier2 if _record_key(r) not in improved_keys]

    merged: list[dict] = []
    for r in tier1:
        merged.append(_strip_qc_fields(r))
    for r in improved:
        merged.append(_strip_qc_fields(r))
    if include_unimproved:
        for r in unimproved:
            out = _strip_qc_fields(r)
            out["_tier2_no_claude"] = True
            merged.append(out)

    _write_jsonl(merge_path, merged)

    tier3_count = sum(1 for r in scored_records if r.get("tier") == 3)
    print(
        f"\nMerge → {merge_path}:"
        f"\n  Tier 1 (kept):           {len(tier1)}"
        f"\n  Tier 2 (Claude):         {len(improved)}"
    )
    if include_unimproved and unimproved:
        print(f"  Tier 2 (unimproved):     {len(unimproved)}")
    elif unimproved:
        print(f"  Tier 2 (dropped):        {len(unimproved)}")
    print(
        f"  Tier 3 (dropped):        {tier3_count}"
        f"\n  Total in final:          {len(merged)}"
    )


# ── rescore comparison ───────────────────────────────────────────────────────


def _run_rescore(
    tier2_records: list[dict],
    improved_path: Path,
) -> None:
    from scripts.phase2_quality_filter import score_record

    improved = _load_records(improved_path)
    if not improved:
        print("No improved records to rescore.")
        return

    improved_keys = {_record_key(r) for r in improved}
    originals = {
        _record_key(r): r.get("score", 0)
        for r in tier2_records
        if _record_key(r) in improved_keys
    }

    new_scores: list[int] = []
    old_scores: list[int] = []
    for rec in improved:
        scored = score_record(rec)
        new_scores.append(scored["score"])
        old_scores.append(originals.get(_record_key(rec), 0))

    old_avg = sum(old_scores) / len(old_scores) if old_scores else 0
    new_avg = sum(new_scores) / len(new_scores) if new_scores else 0
    new_tier1 = sum(1 for s in new_scores if s >= 80)
    new_tier2 = sum(1 for s in new_scores if 60 <= s < 80)
    new_tier3 = sum(1 for s in new_scores if s < 60)
    print(
        f"\nRescore ({len(improved)} records):"
        f"\n  Original avg: {old_avg:.1f}"
        f"\n  Claude avg:   {new_avg:.1f} ({new_avg - old_avg:+.1f})"
        f"\n  New tiers:    T1={new_tier1} T2={new_tier2} T3={new_tier3}"
    )


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Claude Tier 2 regeneration and merge → synthetic_final.jsonl",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/synthetic_scored.jsonl",
        help="Scored JSONL from phase2_quality_filter (default: data/synthetic_scored.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/claude_improved.jsonl",
        help="Claude-improved output JSONL (default: data/claude_improved.jsonl)",
    )
    parser.add_argument(
        "--merge",
        type=str,
        default=None,
        help="Merge Tier 1 + improved Tier 2 → path (e.g. data/synthetic_final.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=CLAUDE_MODEL,
        help=f"Claude model name (default: {CLAUDE_MODEL})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"Max output tokens per API call (default: {MAX_TOKENS})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Concurrent API workers (default: 5)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per record on transient errors (default: 3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max Tier 2 records to regenerate (for testing)",
    )
    parser.add_argument(
        "--all-records",
        action="store_true",
        help="Treat all input records as targets (for pre-filtered tier2 lists)",
    )
    parser.add_argument(
        "--include-unimproved",
        action="store_true",
        help="In merge: keep Tier 2 records that Claude failed to regenerate",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Re-score improved records and print comparison vs originals",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite output instead of resuming from existing progress",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show stats without calling the API",
    )
    args = parser.parse_args()

    if anthropic is None and not args.dry_run:
        print(
            "Error: anthropic not installed. Run: pip install anthropic",
            file=sys.stderr,
        )
        return 1

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.dry_run:
        print("Error: ANTHROPIC_API_KEY not set in env or .env", file=sys.stderr)
        return 1

    all_records = _load_records(input_path)
    print(f"Loaded {len(all_records)} records from {input_path}")

    tier2_records: list[dict]
    if args.all_records:
        tier2_records = all_records
    else:
        tier2_records = [r for r in all_records if r.get("tier") == 2]

    tier_counts: dict[Any, int] = {}
    for r in all_records:
        t = r.get("tier", "?")
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print(f"Tier distribution: {dict(sorted(tier_counts.items(), key=lambda x: str(x[0])))}")
    print(f"Tier 2 targets: {len(tier2_records)}")

    task_counts: dict[str, int] = {}
    for r in tier2_records:
        task_counts[r.get("task_type", "?")] = task_counts.get(r.get("task_type", "?"), 0) + 1
    if task_counts:
        print(f"Tier 2 by task_type: {dict(sorted(task_counts.items()))}")

    if args.limit is not None:
        tier2_records = tier2_records[: args.limit]
        print(f"Limited to {len(tier2_records)} records")

    if args.dry_run:
        print("Dry run — exiting without API calls.")
        if args.merge:
            tier1_count = sum(1 for r in all_records if r.get("tier") == 1)
            print(
                f"Merge would produce: ~{tier1_count} Tier 1"
                f" + {len(tier2_records)} improved"
                f" = ~{tier1_count + len(tier2_records)}"
            )
        return 0

    _run_regeneration(
        tier2_records,
        output_path,
        model=args.model,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        workers=args.workers,
        no_resume=args.no_resume,
        api_key=api_key,
    )

    if args.rescore:
        _run_rescore(tier2_records, output_path)

    if args.merge:
        _run_merge(
            all_records,
            output_path,
            Path(args.merge),
            include_unimproved=args.include_unimproved,
        )

    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
