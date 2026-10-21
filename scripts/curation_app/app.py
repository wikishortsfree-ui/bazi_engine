"""
Human curation web UI for Phase 2 data.

Loads scored/final JSONL, displays one record at a time (chart, reasoning,
interpretation, QC score, issues), and lets the reviewer accept / reject / flag
each record with optional notes.  Review state is persisted to a sidecar JSON
so progress survives restarts.  Accepted records can be exported to a curated
JSONL matching the prepare_phase2_dataset schema.

Usage:
    python scripts/curation_app/app.py                         # defaults
    python scripts/curation_app/app.py -i data/synthetic_final.jsonl
    python scripts/curation_app/app.py --port 5050 --debug
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for

ROOT = Path(__file__).resolve().parents[2]

app = Flask(__name__)

# ── Global state (populated in main) ──────────────────────────────────────────

RECORDS: list[dict] = []
STATE_PATH: Path = Path()
REVIEW_STATE: dict[str, dict] = {}  # str(index) -> {action, notes, timestamp}
INPUT_PATH: Path = Path()
BIO_LOOKUP: dict[str, str] = {}  # name -> biography text


def _load_bio_lookup(path: Path) -> dict[str, str]:
    """Build name → biography index from astrodatabank_bazi.jsonl."""
    lookup: dict[str, str] = {}
    if not path.exists():
        return lookup
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = rec.get("name", "")
            bio = rec.get("biography", "")
            if name and bio:
                lookup[name] = bio
    return lookup


def _load_records(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_state(path: Path) -> dict[str, dict]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_state() -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(REVIEW_STATE, f, ensure_ascii=False, indent=2)


def _progress() -> dict:
    total = len(RECORDS)
    reviewed = sum(1 for v in REVIEW_STATE.values() if v.get("action"))
    accepted = sum(1 for v in REVIEW_STATE.values() if v.get("action") == "accept")
    rejected = sum(1 for v in REVIEW_STATE.values() if v.get("action") == "reject")
    flagged = sum(1 for v in REVIEW_STATE.values() if v.get("action") == "flag")
    return {
        "total": total,
        "reviewed": reviewed,
        "accepted": accepted,
        "rejected": rejected,
        "flagged": flagged,
        "remaining": total - reviewed,
        "pct": round(100 * reviewed / total, 1) if total else 0,
    }


def _extract_chart_display(chart: dict) -> dict:
    """Build a display-friendly summary of the chart."""
    pillars = chart.get("natal_pillars", {})
    pillar_labels = ["year", "month", "day", "hour"]
    pillar_data = []
    for label in pillar_labels:
        p = pillars.get(label, {})
        pillar_data.append({
            "label": label,
            "stem": p.get("stem", "?"),
            "branch": p.get("branch", "?"),
            "element_stem": p.get("element_stem", ""),
            "element_branch": p.get("element_branch", ""),
        })

    ten_gods = chart.get("ten_gods", {})
    dm = chart.get("day_master", {})
    useful = chart.get("useful_god", {})
    strengths = chart.get("elemental_strengths", {})
    interactions = chart.get("interactions", {})
    metadata = chart.get("metadata", {})

    interaction_summary = []
    for itype in [
        "stem_combinations", "stem_clashes",
        "branch_six_combinations", "branch_clashes",
        "three_harmony", "harms", "punishments", "destructions",
    ]:
        items = interactions.get(itype, [])
        if items:
            interaction_summary.append({"type": itype, "count": len(items), "items": items})

    return {
        "pillars": pillar_data,
        "ten_gods": ten_gods,
        "day_master": dm,
        "useful_god": useful,
        "strengths": strengths,
        "interactions_summary": interaction_summary,
        "metadata": metadata,
    }


def _build_index(filter_tier: str | None, filter_status: str | None, filter_task: str | None) -> list[int]:
    """Return record indices matching the active filters."""
    indices = []
    for i, rec in enumerate(RECORDS):
        quality = rec.get("quality", {})
        tier = quality.get("tier")
        task_type = rec.get("task_type", "")
        review = REVIEW_STATE.get(str(i), {})
        action = review.get("action", "")

        if filter_tier and filter_tier != "all":
            if str(tier) != filter_tier:
                continue
        if filter_status and filter_status != "all":
            if filter_status == "unreviewed" and action:
                continue
            elif filter_status in ("accept", "reject", "flag") and action != filter_status:
                continue
        if filter_task and filter_task != "all":
            if task_type != filter_task:
                continue

        indices.append(i)
    return indices


_TERM_EXTRACTORS = {
    "phantom_ten_god": lambda msg: _extract_quoted(msg) or _extract_mentions(msg),
    "composite_ten_god": lambda msg: _extract_quoted(msg),
    "strength_mismatch": lambda msg: ["身强"] if "身强" in msg else (["身弱"] if "身弱" in msg else []),
    "phantom_interaction": lambda msg: _extract_interaction_terms(msg),
    "useful_god_mismatch": lambda msg: ["用神"],
}


def _extract_quoted(msg: str) -> list[str]:
    """Extract single-quoted terms from issue messages."""
    import re
    return re.findall(r"'([^']+)'", msg)


def _extract_mentions(msg: str) -> list[str]:
    """Extract 'mentions X not present' pattern."""
    import re
    m = re.search(r"mentions\s+(\S+)\s+not", msg)
    return [m.group(1)] if m else []


def _extract_interaction_terms(msg: str) -> list[str]:
    """Extract branch/stem pair from interaction issue messages."""
    import re
    m = re.search(r"(?:clash|六合|六害|相破|combination)\s+(\S)(\S)", msg)
    if m:
        return [m.group(1) + m.group(2), m.group(2) + m.group(1)]
    return []


def _collect_highlight_terms(record: dict) -> list[str]:
    """Extract terms to highlight from all factual/language issues."""
    terms: list[str] = []
    for issue in record.get("factual_issues", []) + record.get("language_issues", []):
        itype = issue.get("type", "")
        msg = issue.get("message", "")
        extractor = _TERM_EXTRACTORS.get(itype)
        if extractor:
            terms.extend(extractor(msg))
    return list(dict.fromkeys(t for t in terms if t))


ELEMENT_COLORS = {
    "木": "#22c55e", "木/wood": "#22c55e", "wood": "#22c55e",
    "火": "#ef4444", "火/fire": "#ef4444", "fire": "#ef4444",
    "土": "#eab308", "土/earth": "#eab308", "earth": "#eab308",
    "金": "#f59e0b", "金/metal": "#f59e0b", "metal": "#f59e0b",
    "水": "#3b82f6", "水/water": "#3b82f6", "water": "#3b82f6",
}


@app.route("/")
def index():
    return redirect(url_for("review", idx=0))


@app.route("/review/<int:idx>")
def review(idx: int):
    filter_tier = request.args.get("tier", "all")
    filter_status = request.args.get("status", "all")
    filter_task = request.args.get("task", "all")

    filtered = _build_index(filter_tier, filter_status, filter_task)
    if not filtered:
        return render_template(
            "review.html",
            record=None,
            chart_display=None,
            progress=_progress(),
            current_idx=0,
            filtered_pos=0,
            filtered_total=0,
            review_data={},
            filter_tier=filter_tier,
            filter_status=filter_status,
            filter_task=filter_task,
            element_colors=ELEMENT_COLORS,
        )

    pos = max(0, min(idx, len(filtered) - 1))
    real_idx = filtered[pos]
    rec = RECORDS[real_idx]
    chart = rec.get("chart", {})
    chart_display = _extract_chart_display(chart)
    review_data = REVIEW_STATE.get(str(real_idx), {})
    biography = BIO_LOOKUP.get(rec.get("name", ""), "")
    highlight_terms = _collect_highlight_terms(rec)

    return render_template(
        "review.html",
        record=rec,
        chart_display=chart_display,
        progress=_progress(),
        current_idx=real_idx,
        filtered_pos=pos,
        filtered_total=len(filtered),
        review_data=review_data,
        filter_tier=filter_tier,
        filter_status=filter_status,
        filter_task=filter_task,
        element_colors=ELEMENT_COLORS,
        biography=biography,
        highlight_terms=highlight_terms,
    )


@app.route("/submit_review", methods=["POST"])
def submit_review():
    data = request.form
    real_idx = data.get("record_idx", "0")
    action = data.get("action", "")
    notes = data.get("notes", "").strip()
    filter_tier = data.get("filter_tier", "all")
    filter_status = data.get("filter_status", "all")
    filter_task = data.get("filter_task", "all")
    nav = data.get("nav", "next")

    if action in ("accept", "reject", "flag"):
        REVIEW_STATE[real_idx] = {
            "action": action,
            "notes": notes,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _save_state()

    filtered = _build_index(filter_tier, filter_status, filter_task)
    current_real = int(real_idx)

    if current_real in filtered:
        current_pos = filtered.index(current_real)
    else:
        current_pos = 0

    if nav == "next":
        next_pos = min(current_pos + 1, len(filtered) - 1) if filtered else 0
    elif nav == "prev":
        next_pos = max(current_pos - 1, 0)
    else:
        next_pos = current_pos

    return redirect(url_for(
        "review",
        idx=next_pos,
        tier=filter_tier,
        status=filter_status,
        task=filter_task,
    ))


@app.route("/api/progress")
def api_progress():
    return jsonify(_progress())


@app.route("/export")
def export_page():
    progress = _progress()
    return render_template("export.html", progress=progress, input_path=str(INPUT_PATH))


@app.route("/do_export", methods=["POST"])
def do_export():
    output_path = Path(request.form.get("output_path", "data/curated_synthetic.jsonl"))
    include_flagged = request.form.get("include_flagged") == "on"

    exported = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(RECORDS):
            review = REVIEW_STATE.get(str(i), {})
            action = review.get("action", "")
            if action == "accept" or (include_flagged and action == "flag"):
                out = {
                    "chart": rec["chart"],
                    "task_type": rec["task_type"],
                    "query": rec["query"],
                    "reasoning": rec["reasoning"],
                    "output": rec["output"],
                    "human_review": {
                        "action": action,
                        "notes": review.get("notes", ""),
                        "timestamp": review.get("timestamp", ""),
                    },
                }
                if "name" in rec:
                    out["name"] = rec["name"]
                if "selection_source" in rec:
                    out["selection_source"] = rec["selection_source"]
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                exported += 1

    return render_template(
        "export.html",
        progress=_progress(),
        input_path=str(INPUT_PATH),
        exported=exported,
        output_path=str(output_path),
    )


def main() -> int:
    global RECORDS, STATE_PATH, REVIEW_STATE, INPUT_PATH, BIO_LOOKUP

    parser = argparse.ArgumentParser(description="Phase 2 human curation web UI")
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=ROOT / "data" / "synthetic_final.jsonl",
        help="Scored/final JSONL to review (default: data/synthetic_final.jsonl)",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help="Review state JSON (default: <input>.curation_state.json)",
    )
    parser.add_argument(
        "--bios",
        type=Path,
        default=ROOT / "data" / "astrodatabank_bazi.jsonl",
        help="Astrodatabank JSONL with biographies for reference (default: data/astrodatabank_bazi.jsonl)",
    )
    parser.add_argument("--port", type=int, default=5050, help="Server port (default: 5050; 5000 is often taken by AirPlay on macOS)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    INPUT_PATH = args.input
    if not INPUT_PATH.exists():
        print(f"Error: input file not found: {INPUT_PATH}", file=sys.stderr)
        return 1

    STATE_PATH = args.state or INPUT_PATH.with_suffix(".curation_state.json")
    RECORDS = _load_records(INPUT_PATH)
    REVIEW_STATE = _load_state(STATE_PATH)
    BIO_LOOKUP = _load_bio_lookup(args.bios)

    print(f"Loaded {len(RECORDS)} records from {INPUT_PATH}")
    print(f"Biographies: {len(BIO_LOOKUP)} loaded from {args.bios}")
    print(f"Review state: {STATE_PATH} ({len(REVIEW_STATE)} reviewed)")
    print(f"Starting server at http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
