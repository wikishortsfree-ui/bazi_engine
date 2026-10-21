"""Integration tests for scripts/batch_generate_charts.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

# Project root
ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "batch_generate_charts.py"


def _run_script(args: list[str], cwd: str | Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_batch_script_creates_jsonl_with_five_lines():
    """Run script on temp CSV with 5 rows; output JSONL has 5 lines."""
    csv_content = """birth_datetime,timezone_offset,gender
1990-05-15 14:30:00,8,male
2003-04-24 21:20:00,8,male
1985-11-03 08:45:00,-5,female
2000-01-01 00:00:00,0,male
1995-07-20 12:00:00,9,female
"""
    with NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        input_path = f.name
    with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        output_path = f.name

    try:
        result = _run_script(["--input", input_path, "--output", output_path])
        assert result.returncode == 0, (result.stdout, result.stderr)
        lines = Path(output_path).read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5, f"Expected 5 lines, got {len(lines)}"
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def test_batch_script_each_line_valid_json():
    """Each line of output JSONL is valid JSON with expected keys."""
    csv_content = """birth_datetime,timezone_offset,gender
1990-05-15 14:30:00,8,male
2003-04-24 21:20:00,8,male
"""
    with NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        input_path = f.name
    with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        output_path = f.name

    try:
        result = _run_script(["--input", input_path, "--output", output_path])
        assert result.returncode == 0
        for i, line in enumerate(Path(output_path).read_text(encoding="utf-8").strip().split("\n")):
            data = json.loads(line)
            assert "metadata" in data
            assert "natal_pillars" in data
            assert "metadata" in data and "birth_datetime" in data["metadata"]
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def test_batch_script_birth_datetime_in_output():
    """Charts in output have correct birth_datetime from input CSV."""
    csv_content = """birth_datetime,timezone_offset,gender
1990-05-15 14:30:00,8,male
"""
    with NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        input_path = f.name
    with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        output_path = f.name

    try:
        result = _run_script(["--input", input_path, "--output", output_path])
        assert result.returncode == 0
        line = Path(output_path).read_text(encoding="utf-8").strip().split("\n")[0]
        data = json.loads(line)
        assert "1990-05-15" in data["metadata"]["birth_datetime"]
        assert "14:30" in data["metadata"]["birth_datetime"]
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def test_batch_script_skip_errors():
    """With --skip-errors, invalid row is skipped and script still exits 0."""
    csv_content = """birth_datetime,timezone_offset,gender
1990-05-15 14:30:00,8,male
not-a-date,8,male
2003-04-24 21:20:00,8,male
"""
    with NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        input_path = f.name
    with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        output_path = f.name

    try:
        result = _run_script(
            ["--input", input_path, "--output", output_path, "--skip-errors"]
        )
        assert result.returncode == 0
        lines = Path(output_path).read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2  # two valid rows
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
