"""
Spot-check: same CSV row produces consistent chart JSON from compute_charts.py
and batch_generate_charts.py (natal and, when applicable, dynamic).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

ROOT = Path(__file__).resolve().parent.parent
BATCH_SCRIPT = ROOT / "scripts" / "batch_generate_charts.py"
COMPUTE_SCRIPT = ROOT / "scripts" / "compute_charts.py"


def _run(cmd: list[str]) -> tuple[int, str, str]:
    r = subprocess.run(
        [sys.executable] + [str(x) for x in cmd],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    return r.returncode, r.stdout, r.stderr


def _normalize_metadata_for_compare(data: dict) -> dict:
    """Keep only fields that should match across both scripts (exclude calculation_timestamp)."""
    meta = (data.get("metadata") or {}).copy()
    meta.pop("calculation_timestamp", None)
    meta.pop("calculator_version", None)
    return meta


def test_same_row_natal_batch_vs_compute():
    """Same CSV row: batch_generate_charts (no dynamic) and compute_charts (no flowing_year) produce consistent natal chart."""
    csv_content = """birth_datetime,timezone_offset,lat,lng
2003-04-24 21:20:00,8,37.35,-121.96
"""
    with NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        csv_path = f.name
    with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as batch_out:
        batch_path = batch_out.name
    with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as compute_out:
        compute_path = compute_out.name

    try:
        code_batch, _, stderr_batch = _run(
            [BATCH_SCRIPT, "--input", csv_path, "--output", batch_path]
        )
        assert code_batch == 0, f"batch_generate_charts failed: {stderr_batch}"

        code_compute, _, stderr_compute = _run(
            [COMPUTE_SCRIPT, "--input", csv_path, "--output", compute_path, "--workers", "1"]
        )
        assert code_compute == 0, f"compute_charts failed: {stderr_compute}"

        batch_lines = Path(batch_path).read_text(encoding="utf-8").strip().split("\n")
        compute_lines = Path(compute_path).read_text(encoding="utf-8").strip().split("\n")
        assert len(batch_lines) == 1 and len(compute_lines) == 1

        batch_data = json.loads(batch_lines[0])
        compute_data = json.loads(compute_lines[0])

        assert _normalize_metadata_for_compare(batch_data) == _normalize_metadata_for_compare(
            compute_data
        ), "metadata (excluding timestamp) should match"
        assert batch_data.get("natal_pillars") == compute_data.get(
            "natal_pillars"
        ), "natal_pillars should match"
        assert batch_data.get("day_master") == compute_data.get(
            "day_master"
        ), "day_master should match"
    finally:
        Path(csv_path).unlink(missing_ok=True)
        Path(batch_path).unlink(missing_ok=True)
        Path(compute_path).unlink(missing_ok=True)


def test_same_row_with_dynamic_batch_vs_compute():
    """Same row with dynamic: batch (--include-dynamic --current-date) and compute (flowing_year + flowing_month + gender) produce consistent dynamic section."""
    # Use current-date 2024-06-15 so batch uses month=6 for 流月; compute uses flowing_month=6.
    csv_content = """birth_datetime,timezone_offset,lat,lng,gender,flowing_year,flowing_month
2003-04-24 21:20:00,8,37.35,-121.96,male,2024,6
"""
    with NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        csv_path = f.name
    with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as batch_out:
        batch_path = batch_out.name
    with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as compute_out:
        compute_path = compute_out.name

    try:
        code_batch, _, stderr_batch = _run(
            [
                BATCH_SCRIPT,
                "--input", csv_path,
                "--output", batch_path,
                "--include-dynamic",
                "--current-date", "2024-06-15",
            ]
        )
        assert code_batch == 0, f"batch_generate_charts failed: {stderr_batch}"

        code_compute, _, stderr_compute = _run(
            [COMPUTE_SCRIPT, "--input", csv_path, "--output", compute_path, "--workers", "1"]
        )
        assert code_compute == 0, f"compute_charts failed: {stderr_compute}"

        batch_lines = Path(batch_path).read_text(encoding="utf-8").strip().split("\n")
        compute_lines = Path(compute_path).read_text(encoding="utf-8").strip().split("\n")
        assert len(batch_lines) == 1 and len(compute_lines) == 1

        batch_data = json.loads(batch_lines[0])
        compute_data = json.loads(compute_lines[0])

        assert batch_data.get("natal_pillars") == compute_data.get("natal_pillars")
        assert "dynamic" in batch_data and batch_data["dynamic"] is not None
        assert "dynamic" in compute_data and compute_data["dynamic"] is not None

        bd = batch_data["dynamic"]
        cd = compute_data["dynamic"]
        assert bd.get("da_yun") == cd.get("da_yun"), "da_yun should match"
        assert bd.get("liu_nian") == cd.get("liu_nian"), "liu_nian should match"
        assert bd.get("liu_yue") == cd.get("liu_yue"), "liu_yue should match"
    finally:
        Path(csv_path).unlink(missing_ok=True)
        Path(batch_path).unlink(missing_ok=True)
        Path(compute_path).unlink(missing_ok=True)


def test_compute_charts_birth_date_and_time_columns():
    """compute_charts accepts birth_date + birth_time (no birth_datetime) and produces same chart as batch with combined datetime."""
    csv_compute = """birth_date,birth_time,timezone_offset,lat,lng
2003-04-24,21:20:00,8,37.35,-121.96
"""
    csv_batch = """birth_datetime,timezone_offset,lat,lng
2003-04-24 21:20:00,8,37.35,-121.96
"""
    with NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_compute)
        csv_compute_path = f.name
    with NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_batch)
        csv_batch_path = f.name
    with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as out1:
        compute_path = out1.name
    with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as out2:
        batch_path = out2.name

    try:
        code_c, _, stderr_c = _run(
            [COMPUTE_SCRIPT, "--input", csv_compute_path, "--output", compute_path, "--workers", "1"]
        )
        assert code_c == 0, stderr_c
        code_b, _, stderr_b = _run(
            [BATCH_SCRIPT, "--input", csv_batch_path, "--output", batch_path]
        )
        assert code_b == 0, stderr_b

        c_data = json.loads(Path(compute_path).read_text(encoding="utf-8").strip().split("\n")[0])
        b_data = json.loads(Path(batch_path).read_text(encoding="utf-8").strip().split("\n")[0])
        assert c_data.get("natal_pillars") == b_data.get("natal_pillars")
        assert c_data["metadata"]["birth_datetime"] == b_data["metadata"]["birth_datetime"]
    finally:
        for p in (csv_compute_path, csv_batch_path, compute_path, batch_path):
            Path(p).unlink(missing_ok=True)
