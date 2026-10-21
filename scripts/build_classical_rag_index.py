"""
Chunk classical .txt files, embed with BGE (sentence-transformers), build FAISS index.

Expects UTF-8 text under data/classical_texts/ (see data/classical_texts/README.md).

Usage:
  python scripts/build_classical_rag_index.py
  python scripts/build_classical_rag_index.py --input_dir data/classical_texts --chunk_tokens 300 --overlap_tokens 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def token_id_windows(
    token_ids: list[int],
    chunk_size: int,
    overlap: int,
) -> list[list[int]]:
    """Non-overlapping-adjacent windows over token ids; step = chunk_size - overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)
    step = max(1, chunk_size - overlap)
    if not token_ids:
        return []
    chunks: list[list[int]] = []
    start = 0
    n = len(token_ids)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(token_ids[start:end])
        if end >= n:
            break
        start += step
    return chunks


def _load_text_files(input_dir: Path, glob_pat: str) -> list[tuple[str, str]]:
    paths = sorted(input_dir.glob(glob_pat))
    out: list[tuple[str, str]] = []
    for p in paths:
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8").strip()
        if text:
            out.append((str(p.resolve()), text))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FAISS + metadata for classical text RAG.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/classical_texts"),
        help="Directory of UTF-8 .txt files",
    )
    parser.add_argument("--glob", dest="glob_pat", default="*.txt", help="Glob under input_dir")
    parser.add_argument("--chunk_tokens", type=int, default=300, help="Target chunk size in tokens")
    parser.add_argument("--overlap_tokens", type=int, default=50, help="Token overlap between chunks")
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-small-zh-v1.5",
        help="sentence-transformers model name",
    )
    parser.add_argument(
        "--out_index",
        type=Path,
        default=Path("data/classical_faiss.index"),
        help="Output FAISS index path",
    )
    parser.add_argument(
        "--out_meta",
        type=Path,
        default=Path("data/classical_metadata.json"),
        help="Output JSON metadata (list of chunk records)",
    )
    args = parser.parse_args()

    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(
            "Missing dependency: pip install faiss-cpu sentence-transformers",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        return 1

    input_dir = args.input_dir
    if not input_dir.is_dir():
        print(f"Error: input_dir is not a directory: {input_dir}", file=sys.stderr)
        return 1

    files = _load_text_files(input_dir, args.glob_pat)
    if not files:
        print(
            f"Error: no non-empty .txt files matched {input_dir}/{args.glob_pat}",
            file=sys.stderr,
        )
        return 1

    print(f"Loading embedding model {args.model!r} …")
    model = SentenceTransformer(args.model)
    tokenizer = model.tokenizer

    chunk_texts: list[str] = []
    chunk_sources: list[str] = []
    for source_path, full_text in files:
        enc = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_tensors=None,
        )
        ids = enc["input_ids"]
        if not isinstance(ids, list):
            ids = list(ids)
        for win in token_id_windows(ids, args.chunk_tokens, args.overlap_tokens):
            piece = tokenizer.decode(win, skip_special_tokens=True).strip()
            if piece:
                chunk_texts.append(piece)
                chunk_sources.append(source_path)

    if not chunk_texts:
        print("Error: no chunks after tokenization (text too short?)", file=sys.stderr)
        return 1

    print(f"Encoding {len(chunk_texts)} chunks …")
    embeddings = model.encode(
        chunk_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    arr = np.asarray(embeddings, dtype=np.float32)
    dim = arr.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(arr)

    metadata = [
        {
            "chunk_id": i,
            "source_path": chunk_sources[i],
            "text": chunk_texts[i],
        }
        for i in range(len(chunk_texts))
    ]

    args.out_index.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.out_index))
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "chunk_tokens": args.chunk_tokens,
                "overlap_tokens": args.overlap_tokens,
                "num_chunks": len(metadata),
                "embedding_dim": dim,
                "chunks": metadata,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote {args.out_index} ({len(chunk_texts)} vectors, dim={dim})")
    print(f"Wrote {args.out_meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
