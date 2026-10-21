"""Tests for scripts/qa_templates.py and scripts/qa_utils.py.

Covers:
- Template counts and category coverage
- Template answer correctness against a deterministic fixture chart
- Absence handling ("无" for missing interactions / unapplicable templates)
- Verify functions reject intentionally wrong answers
- Shared utilities (stable_chart_id, iter_chart_records, write_jsonl_line)
- Applicability filter (requires_dynamic, applies)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bazi_engine.calculator import BaziCalculator
from bazi_engine.models import BaziChart
from bazi_engine.temporal import TemporalContext, compute_liunian, detect_cross_interactions

from scripts.qa_templates import (
    QATemplate,
    applicable_templates,
    build_stage1_templates,
    build_stage2_templates,
)
from scripts.qa_utils import iter_chart_records, stable_chart_id, write_jsonl_line


# -----------------------------------------------------------------------------
# Fixture charts
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chart_natal() -> tuple[dict, BaziChart]:
    """Deterministic natal chart (no dynamic context)."""
    dt = datetime(1988, 7, 15, 10, 30, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, lat=31.2304, lng=121.4737)
    chart_json = json.loads(chart.to_json(indent=None))
    return chart_json, chart


@pytest.fixture(scope="module")
def chart_with_dynamic() -> tuple[dict, BaziChart]:
    """Chart with temporal context + cross-interactions attached."""
    dt = datetime(1990, 3, 20, 14, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, lat=39.9042, lng=116.4074)
    stems, branches = chart._stems_and_branches()
    liunian = compute_liunian(2024, day_master_stem=stems[2])
    ctx = TemporalContext(dayun=None, liunian=liunian, liuyue=None)
    chart.temporal_context = ctx
    chart.cross_interactions = detect_cross_interactions(stems, branches, ctx)
    chart_json = json.loads(chart.to_json(indent=None))
    return chart_json, chart


# -----------------------------------------------------------------------------
# Template counts and coverage
# -----------------------------------------------------------------------------


def test_stage1_template_count_and_categories():
    templates = build_stage1_templates()
    assert len(templates) >= 50, "Stage 1 must have at least 50 templates"
    categories = {t.category for t in templates}
    expected = {
        "pillar_identity",
        "ten_gods",
        "five_elements",
        "day_master_strength",
        "interactions",
        "hidden_stems",
        "dynamic",
    }
    assert expected <= categories, f"missing categories: {expected - categories}"


def test_stage2_template_count_and_categories():
    templates = build_stage2_templates()
    assert len(templates) >= 10, "Stage 2 must have at least 10 templates"
    categories = {t.category for t in templates}
    expected = {
        "relational",
        "conditional",
        "cross_pillar",
        "analytical",
        "interaction_chain",
        "temporal_relational",
        "summary",
    }
    assert expected <= categories, f"missing categories: {expected - categories}"


def test_template_ids_are_unique():
    all_ids = [t.id for t in build_stage1_templates() + build_stage2_templates()]
    assert len(set(all_ids)) == len(all_ids), "template ids must be unique"


# -----------------------------------------------------------------------------
# Correctness against a real chart
# -----------------------------------------------------------------------------


def _run_template(tmpl: QATemplate, chart_json: dict, chart: BaziChart) -> str:
    q = tmpl.text_fn(chart_json)
    a = tmpl.answer_fn(chart_json)
    assert isinstance(q, str) and q, f"{tmpl.id}: text_fn returned empty"
    assert isinstance(a, str), f"{tmpl.id}: answer_fn must return str, got {type(a)}"
    assert tmpl.verify_fn(chart_json, chart, a), (
        f"{tmpl.id}: verify_fn rejected its own answer_fn output; answer={a!r}"
    )
    return a


def test_all_stage1_templates_pass_on_natal(chart_natal):
    chart_json, chart = chart_natal
    for tmpl in applicable_templates(build_stage1_templates(), chart_json):
        _run_template(tmpl, chart_json, chart)


def test_all_stage2_templates_pass_on_natal(chart_natal):
    chart_json, chart = chart_natal
    for tmpl in applicable_templates(build_stage2_templates(), chart_json):
        _run_template(tmpl, chart_json, chart)


def test_dynamic_templates_run_on_dynamic_chart(chart_with_dynamic):
    chart_json, chart = chart_with_dynamic
    dynamic_templates = [t for t in build_stage1_templates() if t.requires_dynamic]
    applicable = applicable_templates(dynamic_templates, chart_json)
    assert len(applicable) > 0, "expected some dynamic templates to apply"
    for tmpl in applicable:
        _run_template(tmpl, chart_json, chart)


def test_dynamic_templates_excluded_on_natal_only(chart_natal):
    chart_json, _ = chart_natal
    dynamic_templates = [t for t in build_stage1_templates() if t.requires_dynamic]
    assert applicable_templates(dynamic_templates, chart_json) == []


# -----------------------------------------------------------------------------
# Absence handling
# -----------------------------------------------------------------------------


def test_absence_produces_wu_for_empty_interactions():
    """When an interaction family is empty, the answer must be '无'."""
    # Build a fake chart_json with all interactions empty.
    chart_json = {
        "metadata": {"birth_datetime": "1990-01-01T00:00:00+00:00", "lat": 0, "lng": 0, "sect": 2},
        "interactions": {
            k: []
            for k in (
                "stem_combinations", "stem_clashes", "branch_six_combinations",
                "branch_clashes", "three_harmony", "harms", "punishments", "destructions",
            )
        },
    }
    templates = [
        t for t in build_stage1_templates()
        if t.id.startswith("interaction-") and not t.id.startswith("interaction-count")
        and t.id != "interaction-any"
    ]
    for t in templates:
        a = t.answer_fn(chart_json)
        assert a == "无", f"{t.id}: expected '无' for empty interactions, got {a!r}"


# -----------------------------------------------------------------------------
# Verify rejects wrong answers
# -----------------------------------------------------------------------------


def test_verify_rejects_wrong_pillar_stem(chart_natal):
    """Swap the answer with a different stem; verify_fn must return False."""
    chart_json, chart = chart_natal
    templates = {t.id: t for t in build_stage1_templates()}
    tmpl = templates["pillar-day-stem"]
    correct = tmpl.answer_fn(chart_json)
    wrong = "甲" if correct != "甲" else "乙"
    assert tmpl.verify_fn(chart_json, chart, correct)
    assert not tmpl.verify_fn(chart_json, chart, wrong), "verify accepted wrong stem"


def test_verify_rejects_wrong_pillar_full(chart_natal):
    chart_json, chart = chart_natal
    templates = {t.id: t for t in build_stage1_templates()}
    tmpl = templates["pillar-year"]
    correct = tmpl.answer_fn(chart_json)
    assert tmpl.verify_fn(chart_json, chart, correct)
    assert not tmpl.verify_fn(chart_json, chart, "甲子非本柱")


def test_verify_rejects_wrong_strongest_element(chart_natal):
    chart_json, chart = chart_natal
    templates = {t.id: t for t in build_stage1_templates()}
    tmpl = templates["element-strongest"]
    correct = tmpl.answer_fn(chart_json)
    wrong = next(e for e in ("木", "火", "土", "金", "水") if e != correct)
    assert tmpl.verify_fn(chart_json, chart, correct)
    assert not tmpl.verify_fn(chart_json, chart, wrong)


def test_verify_rejects_wrong_classification(chart_natal):
    chart_json, chart = chart_natal
    templates = {t.id: t for t in build_stage1_templates()}
    tmpl = templates["dm-strong-weak"]
    correct = tmpl.answer_fn(chart_json)
    wrong = next(w for w in ("身强", "中和", "身弱") if w != correct)
    assert tmpl.verify_fn(chart_json, chart, correct)
    assert not tmpl.verify_fn(chart_json, chart, wrong)


# -----------------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------------


def test_stable_chart_id_is_deterministic(chart_natal):
    chart_json, _ = chart_natal
    assert stable_chart_id(chart_json) == stable_chart_id(chart_json)


def test_stable_chart_id_differs_for_different_charts(chart_natal, chart_with_dynamic):
    a, _ = chart_natal
    b, _ = chart_with_dynamic
    assert stable_chart_id(a) != stable_chart_id(b)


def test_write_jsonl_line_roundtrip(tmp_path: Path):
    record = {"question": "日主是什么？", "answer": "壬", "n": 1}
    out = tmp_path / "out.jsonl"
    with out.open("w", encoding="utf-8") as f:
        write_jsonl_line(f, record)
        write_jsonl_line(f, record)
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        assert json.loads(line) == record


def test_iter_chart_records_handles_bare_and_wrapped(tmp_path: Path, chart_natal):
    chart_json, _ = chart_natal
    p_bare = tmp_path / "bare.jsonl"
    p_wrapped = tmp_path / "wrapped.jsonl"
    with p_bare.open("w", encoding="utf-8") as f:
        f.write(json.dumps(chart_json, ensure_ascii=False) + "\n")
    with p_wrapped.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"chart": chart_json, "extra": "ignored"}, ensure_ascii=False) + "\n")
    for path in (p_bare, p_wrapped):
        loaded = list(iter_chart_records(path))
        assert len(loaded) == 1
        assert loaded[0]["metadata"]["birth_datetime"] == chart_json["metadata"]["birth_datetime"]
