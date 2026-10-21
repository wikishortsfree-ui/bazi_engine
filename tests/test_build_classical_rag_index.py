"""Unit tests for scripts/build_classical_rag_index.py (chunking; optional integration)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.build_classical_rag_index import token_id_windows


def test_token_id_windows_empty():
    assert token_id_windows([], chunk_size=10, overlap=2) == []


def test_token_id_windows_single_chunk():
    ids = list(range(5))
    assert token_id_windows(ids, chunk_size=10, overlap=2) == [ids]


def test_token_id_windows_overlap():
    ids = list(range(20))
    chunks = token_id_windows(ids, chunk_size=8, overlap=2)
    assert len(chunks) >= 2
    assert chunks[0] == list(range(8))
    step = 8 - 2
    assert chunks[1][0] == step


@pytest.mark.skipif(
    os.environ.get("BAZI_TEST_CLASSICAL_RAG") != "1",
    reason="set BAZI_TEST_CLASSICAL_RAG=1 to run (downloads embedding model)",
)
def test_build_classical_rag_index_smoke(tmp_path):
    pytest.importorskip("faiss")
    pytest.importorskip("sentence_transformers")
    import subprocess

    out_index = tmp_path / "idx.faiss"
    out_meta = tmp_path / "meta.json"
    r = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_classical_rag_index.py"),
            "--input_dir",
            str(repo_root / "data" / "classical_texts"),
            "--out_index",
            str(out_index),
            "--out_meta",
            str(out_meta),
            "--chunk_tokens",
            "64",
            "--overlap_tokens",
            "8",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert out_index.is_file()
    assert out_meta.is_file()
