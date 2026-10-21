"""
Tests for scripts/astrodatabank_export_to_jsonl.py.
Uses tests/fixtures/c_sample_mini.zip (minimal ADB export format).
"""

import json
from pathlib import Path

import pytest


def test_export_script_produces_valid_jsonl():
    from scripts.astrodatabank_export_to_jsonl import parse_c_sample_zip

    fixture_zip = Path(__file__).resolve().parents[0] / "fixtures" / "c_sample_mini.zip"
    assert fixture_zip.exists(), f"Fixture missing: {fixture_zip}"
    zip_bytes = fixture_zip.read_bytes()
    records = parse_c_sample_zip(zip_bytes, min_rodden="A")
    assert len(records) == 2
    required = (
        "name",
        "date",
        "time",
        "longitude",
        "latitude",
        "location",
        "rodden_rating",
        "category",
        "biography",
        "source",
        "source_url",
    )
    for rec in records:
        for key in required:
            assert key in rec, f"Missing key {key}"
        assert -180 <= rec["longitude"] <= 180
        assert -90 <= rec["latitude"] <= 90
        assert rec["rodden_rating"] in ("AA", "A")
    assert records[0]["name"] == "Test Person"
    assert records[0]["category"] == "Technology"
    assert records[1]["name"] == "Another Person"
    assert records[1]["latitude"] == pytest.approx(51.5074)


def test_export_script_cli_with_fixture(tmp_path):
    import subprocess

    fixture_zip = Path(__file__).resolve().parents[0] / "fixtures" / "c_sample_mini.zip"
    out = tmp_path / "out.jsonl"
    r = subprocess.run(
        [
            "python",
            "scripts/astrodatabank_export_to_jsonl.py",
            "--output",
            str(out),
            "--input-zip",
            str(fixture_zip),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, (r.stdout, r.stderr)
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert "name" in obj and "longitude" in obj
