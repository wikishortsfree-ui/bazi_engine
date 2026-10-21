"""
Language quality heuristics for Phase 2 curated records (reasoning + output).

No LLM or remote API calls: length counts, phrase lists, and simple repetition stats only.

Flags short text, excessive generic filler phrases, thin classical terminology,
and high repeated-sentence overlap.

Usage:
  python scripts/language_quality_check.py --input data/qwen_raw.jsonl -o data/qwen_lang_checked.jsonl
  python scripts/language_quality_check.py -i data/qwen_raw.jsonl --print-summary
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Filler / template phrases (Chinese); counts contribute to language_issues.
_GENERIC_PHRASES: tuple[str, ...] = (
    "综上所述",
    "总的来说",
    "总而言之",
    "需要注意的是",
    "值得注意的是",
    "不难看出",
    "毫无疑问",
    "毫无疑问的是",
    "当然",
    "此外",
    "另外",
    "因此",
    "所以",
    "总之",
    "换言之",
    "也就是说",
    "从某种意义上说",
    "在一定程度上",
)

# Domain terms — reward presence (training-quality signal).
_CLASSICAL_TERMS: tuple[str, ...] = (
    "用神",
    "十神",
    "刑冲合害",
    "天干",
    "地支",
    "大运",
    "流年",
    "流月",
    "月令",
    "格局",
    "身强",
    "身弱",
    "日主",
    "喜忌",
    "纳音",
)


def _split_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    parts = re.split(r"[。！？\n]+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 8]


def _repetition_score(sentences: list[str]) -> tuple[float, int]:
    """Return (duplicate_ratio, max_count) for exact duplicate sentences."""
    if len(sentences) < 2:
        return 0.0, 0
    c = Counter(sentences)
    dup = sum(n for s, n in c.items() if n > 1)
    max_ct = max(c.values())
    return dup / len(sentences), max_ct


def check_language_quality(
    record: dict[str, Any],
    *,
    min_reasoning_chars: int = 100,
    min_output_chars: int = 200,
    max_generic_hits: int = 8,
    min_classical_terms: int = 2,
    max_duplicate_ratio: float = 0.35,
    max_duplicate_sentence_count: int = 4,
) -> list[dict[str, str]]:
    """
    Return issue dicts with type + message. Heuristic only.
    """
    issues: list[dict[str, str]] = []
    reasoning = str(record.get("reasoning") or "")
    output = str(record.get("output") or "")
    combined = f"{reasoning}\n{output}"

    if len(reasoning) < min_reasoning_chars:
        issues.append(
            {
                "type": "short_reasoning",
                "message": f"reasoning length {len(reasoning)} < {min_reasoning_chars}",
            }
        )
    if len(output) < min_output_chars:
        issues.append(
            {
                "type": "short_output",
                "message": f"output length {len(output)} < {min_output_chars}",
            }
        )

    generic_hits = 0
    hit_terms: list[str] = []
    for phrase in _GENERIC_PHRASES:
        n = combined.count(phrase)
        if n:
            generic_hits += n
            hit_terms.append(f"{phrase}×{n}")
    if generic_hits > max_generic_hits:
        issues.append(
            {
                "type": "generic_phrases",
                "message": f"generic phrase hits {generic_hits} (>{max_generic_hits}): {', '.join(hit_terms[:12])}",
            }
        )

    classical_found = [t for t in _CLASSICAL_TERMS if t in combined]
    if len(classical_found) < min_classical_terms:
        issues.append(
            {
                "type": "thin_terminology",
                "message": f"classical/domain term hits {len(classical_found)} < {min_classical_terms} (found: {classical_found})",
            }
        )

    sents = _split_sentences(combined)
    dup_ratio, max_dup = _repetition_score(sents)
    if dup_ratio > max_duplicate_ratio or max_dup > max_duplicate_sentence_count:
        issues.append(
            {
                "type": "repetition",
                "message": f"duplicate sentence ratio {dup_ratio:.2f} (max single sentence repeats {max_dup})",
            }
        )

    return issues


def _annotate_record(record: dict[str, Any], issues: list[dict[str, str]]) -> dict[str, Any]:
    out = dict(record)
    out["language_issues"] = issues
    out["language_ok"] = len(issues) == 0
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Language quality check for Phase 2 JSONL records.")
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument("--output", "-o", type=Path, default=None)
    p.add_argument("--print-summary", action="store_true")
    p.add_argument("--min-reasoning", type=int, default=100)
    p.add_argument("--min-output", type=int, default=200)
    p.add_argument("--max-generic", type=int, default=8)
    p.add_argument("--min-classical", type=int, default=2)
    args = p.parse_args(argv)

    total = 0
    with_issues = 0
    by_type: dict[str, int] = {}

    out_f = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                issues = check_language_quality(
                    rec,
                    min_reasoning_chars=args.min_reasoning,
                    min_output_chars=args.min_output,
                    max_generic_hits=args.max_generic,
                    min_classical_terms=args.min_classical,
                )
                total += 1
                if issues:
                    with_issues += 1
                    for it in issues:
                        by_type[it["type"]] = by_type.get(it["type"], 0) + 1
                out_f.write(json.dumps(_annotate_record(rec, issues), ensure_ascii=False) + "\n")
    finally:
        if args.output:
            out_f.close()

    if args.print_summary:
        print(
            f"language_quality_check: lines={total} with_issues={with_issues} by_type={by_type}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
