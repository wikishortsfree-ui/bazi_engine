"""
Shared utilities for Stage 1 / Stage 2 QA dataset generation.

Loaders, stable chart IDs, and helpers for writing JSONL records. Templates and
category logic live in `scripts/qa_templates.py`.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def stable_chart_id(chart_json: dict[str, Any]) -> str:
    """
    Stable 16-char hex ID from the chart's invariant identity fields.

    Uses birth_datetime + timezone_offset + lat + lng + sect so two records
    derived from the same chart always share the same ID.
    """
    md = chart_json.get("metadata") or {}
    key = "|".join(
        str(md.get(k, ""))
        for k in ("birth_datetime", "timezone_offset", "lat", "lng", "sect")
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def iter_chart_records(path: Path) -> Iterator[dict[str, Any]]:
    """
    Stream chart records from a JSONL file.

    Accepts two formats transparently:
    - Bare chart dicts (one full `BaziChart.to_json()` per line, e.g. data/charts.jsonl).
    - Wrapped records `{"chart": {...}}` (e.g. data/synthetic_charts.jsonl).

    Yields the raw chart dict (with keys like `metadata`, `natal_pillars`, ...).
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "chart" in rec and isinstance(rec["chart"], dict):
                yield rec["chart"]
            else:
                yield rec


def write_jsonl_line(f, record: dict[str, Any]) -> None:
    """Write a single compact JSONL line."""
    f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    f.write("\n")
