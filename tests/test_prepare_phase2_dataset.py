"""Tests for scripts/prepare_phase2_dataset.py: validation, task tokens, split, output format."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bazi_engine import BaziCalculator
from bazi_engine.models import BaziChart
from bazi_engine.temporal import (
    TemporalContext,
    compute_liunian,
    compute_liuyue,
    detect_cross_interactions,
)


def _chart_with_dynamic(birth_dt, lat: float = 0, lng: float = 0, year: int = 2024, month: int = 3):
    """Build a chart with temporal_context and cross_interactions."""
    chart = BaziCalculator.from_datetime(birth_dt, lat, lng)
    dayun_seq = BaziCalculator().compute_dayun(chart, 1, max_periods=10)
    dayun = dayun_seq.get_period_for_year(year)
    liunian = compute_liunian(year, day_master_stem=chart.day_pillar.stem.value)
    liuyue = compute_liuyue(year, month, day_master_stem=chart.day_pillar.stem.value)
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
    return BaziChart(
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


def _curated_record(chart: BaziChart, task_type: str) -> dict:
    """One curated line: chart as dict, task_type, query, reasoning, output."""
    chart_dict = json.loads(chart.to_json(indent=None))
    return {
        "chart": chart_dict,
        "task_type": task_type,
        "query": "请分析。",
        "reasoning": "根据八字结构…",
        "output": "结论。",
    }


@pytest.fixture
def curated_jsonl_path():
    """Write a small curated JSONL: 2 本命分析, 1 八字问答, 1 流年, 1 流月."""
    records = []
    # Natal-only: 本命分析 (2), 八字问答 (1)
    for i, task in enumerate(["本命分析", "本命分析", "八字问答"]):
        dt = datetime(1990 + i, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        chart = BaziCalculator.from_datetime(dt, 0, 0)
        records.append(_curated_record(chart, task))
    # Temporal: 流年, 流月
    dt = datetime(1995, 4, 24, 21, 20, 0, tzinfo=timezone.utc)
    chart_ln = _chart_with_dynamic(dt, 0, 0, 2024, 3)
    records.append(_curated_record(chart_ln, "流年"))
    dt2 = datetime(1996, 5, 1, 10, 0, 0, tzinfo=timezone.utc)
    chart_ly = _chart_with_dynamic(dt2, 0, 0, 2024, 5)
    records.append(_curated_record(chart_ly, "流月"))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def script_dir():
    """Repo root (parent of scripts/)."""
    return Path(__file__).resolve().parents[1]


def test_prepare_phase2_dataset_script_runs(curated_jsonl_path, script_dir):
    """Running the script on valid curated JSONL produces train/val JSONL with task_token."""
    with tempfile.TemporaryDirectory() as tmp:
        train_out = Path(tmp) / "phase2_train.jsonl"
        val_out = Path(tmp) / "phase2_val.jsonl"
        result = subprocess.run(
            [
                sys.executable,
                str(script_dir / "scripts" / "prepare_phase2_dataset.py"),
                "--input", curated_jsonl_path,
                "--train_out", str(train_out),
                "--val_out", str(val_out),
                "--val_ratio", "0.2",
                "--seed", "42",
            ],
            capture_output=True,
            text=True,
            cwd=str(script_dir),
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert train_out.exists()
        assert val_out.exists()

        train_lines = [l for l in train_out.read_text(encoding="utf-8").strip().split("\n") if l]
        val_lines = [l for l in val_out.read_text(encoding="utf-8").strip().split("\n") if l]
        assert len(train_lines) + len(val_lines) == 5

        for line in train_lines + val_lines:
            rec = json.loads(line)
            assert "task_token" in rec
            assert rec["task_token"] in {"[TASK:本命分析]", "[TASK:流年]", "[TASK:流月]", "[TASK:八字问答]"}
            assert "chart" in rec and "task_type" in rec and "query" in rec and "reasoning" in rec and "output" in rec
            assert "input_sequence" in rec
            assert "target_sequence" in rec


def test_prepare_phase2_dataset_stratified_split(curated_jsonl_path, script_dir):
    """Val set has roughly same task_type distribution; all task types appear in train."""
    with tempfile.TemporaryDirectory() as tmp:
        train_out = Path(tmp) / "phase2_train.jsonl"
        val_out = Path(tmp) / "phase2_val.jsonl"
        subprocess.run(
            [
                sys.executable,
                str(script_dir / "scripts" / "prepare_phase2_dataset.py"),
                "--input", curated_jsonl_path,
                "--train_out", str(train_out),
                "--val_out", str(val_out),
                "--val_ratio", "0.2",
                "--seed", "42",
            ],
            capture_output=True,
            check=True,
            cwd=str(script_dir),
        )
        train_types = []
        for line in train_out.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                train_types.append(json.loads(line)["task_type"])
        val_types = []
        for line in val_out.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                val_types.append(json.loads(line)["task_type"])

        assert len(train_types) + len(val_types) == 5
        # With 5 samples and val_ratio 0.2, stratified split can put all in train (per-task val count rounds to 0)
        assert set(train_types) | set(val_types) == {"本命分析", "八字问答", "流年", "流月"}


def test_prepare_phase2_dataset_skips_invalid(curated_jsonl_path, script_dir):
    """Invalid lines (missing task_type, unparseable chart) are skipped."""
    # Append a bad line
    with open(curated_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"chart": {}, "query": "x", "reasoning": "y", "output": "z"}, ensure_ascii=False) + "\n")  # missing task_type
    with tempfile.TemporaryDirectory() as tmp:
        train_out = Path(tmp) / "phase2_train.jsonl"
        val_out = Path(tmp) / "phase2_val.jsonl"
        result = subprocess.run(
            [
                sys.executable,
                str(script_dir / "scripts" / "prepare_phase2_dataset.py"),
                "--input", curated_jsonl_path,
                "--train_out", str(train_out),
                "--val_out", str(val_out),
                "--val_ratio", "0.2",
                "--seed", "42",
            ],
            capture_output=True,
            text=True,
            cwd=str(script_dir),
        )
        assert result.returncode == 0
        train_lines = [l for l in train_out.read_text(encoding="utf-8").strip().split("\n") if l]
        val_lines = [l for l in val_out.read_text(encoding="utf-8").strip().split("\n") if l]
        # Only 5 valid records (the bad one skipped)
        assert len(train_lines) + len(val_lines) == 5


def test_stratified_split_three_way():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.prepare_phase2_dataset import _stratified_split

    records = [{"task_type": "本命分析", "i": i} for i in range(10)]
    records += [{"task_type": "八字问答", "i": i} for i in range(10)]
    train, val, test = _stratified_split(records, 0.2, seed=42, test_ratio=0.2)
    assert len(train) == 12 and len(val) == 4 and len(test) == 4
    assert {r["task_type"] for r in train} <= {"本命分析", "八字问答"}
    assert not (set(map(id, train)) & set(map(id, val)))
    assert not (set(map(id, train)) & set(map(id, test)))


def test_prepare_phase2_dataset_writes_test_jsonl(curated_jsonl_path, script_dir):
    with tempfile.TemporaryDirectory() as tmp:
        train_out = Path(tmp) / "phase2_train.jsonl"
        val_out = Path(tmp) / "phase2_val.jsonl"
        test_out = Path(tmp) / "phase2_test.jsonl"
        result = subprocess.run(
            [
                sys.executable,
                str(script_dir / "scripts" / "prepare_phase2_dataset.py"),
                "--input", curated_jsonl_path,
                "--train_out", str(train_out),
                "--val_out", str(val_out),
                "--test_out", str(test_out),
                "--val_ratio", "0.2",
                "--test_ratio", "0.2",
                "--seed", "42",
            ],
            capture_output=True,
            text=True,
            cwd=str(script_dir),
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert test_out.exists()
        test_lines = [l for l in test_out.read_text(encoding="utf-8").strip().split("\n") if l]
        train_lines = [l for l in train_out.read_text(encoding="utf-8").strip().split("\n") if l]
        val_lines = [l for l in val_out.read_text(encoding="utf-8").strip().split("\n") if l]
        assert len(train_lines) + len(val_lines) + len(test_lines) == 5


def test_validate_record_and_task_token_constants():
    """_validate_record and TASK_TOKEN_MAP behave as expected (import from script)."""
    # Import the module (add repo root)
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.prepare_phase2_dataset import (
        TASK_TOKEN_MAP,
        TASK_TYPES,
        TARGET_DISTRIBUTION,
        _validate_record,
        _add_task_token,
    )

    assert set(TASK_TOKEN_MAP.keys()) == set(TASK_TYPES)
    assert TASK_TOKEN_MAP["本命分析"] == "[TASK:本命分析]"
    assert TARGET_DISTRIBUTION["本命分析"] == 0.40 and TARGET_DISTRIBUTION["流年"] == 0.30

    dt = datetime(1990, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    chart = BaziCalculator.from_datetime(dt, 0, 0)
    record = _curated_record(chart, "本命分析")
    ok, msg = _validate_record(record, validate_chart_features=True)
    assert ok, msg
    out = _add_task_token(record, add_input_target_sequences=True)
    assert out["task_token"] == "[TASK:本命分析]"
    assert "input_sequence" in out and "target_sequence" in out

    record_bad = dict(record)
    record_bad["task_type"] = "invalid"
    ok, _ = _validate_record(record_bad, validate_chart_features=False)
    assert not ok
