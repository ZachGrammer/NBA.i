"""
build_index.py
--------------
Loads processed semantic documents and builds a FAISS vector index
using Ollama's nomic-embed-text embedding model.

Each document is expected to be a dict:
{
  "text": "...",
  "metadata": {...}
}

The index and matching metadata are persisted to /vectorstore so they can be
loaded at query time without re-embedding.
"""

from __future__ import annotations

import json
import os
import pickle

import faiss
import numpy as np
import ollama

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_PATH = os.path.join(PROJECT_ROOT, "data", "processed_docs", "nba_docs.json")
VECTORSTORE_DIR = os.path.join(PROJECT_ROOT, "vectorstore")
INDEX_PATH = os.path.join(VECTORSTORE_DIR, "nba.index")
META_PATH = os.path.join(VECTORSTORE_DIR, "nba_meta.pkl")

os.makedirs(VECTORSTORE_DIR, exist_ok=True)

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768


def embed_text(text: str) -> np.ndarray:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return np.array(response["embedding"], dtype=np.float32)


def embed_batch(texts: list[str], batch_size: int = 32) -> np.ndarray:
    all_vectors = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        print(
            f"  [embed] Processing batch {i // batch_size + 1} "
            f"({i}–{min(i + batch_size, total)} of {total})..."
        )
        batch_vecs = [embed_text(t) for t in batch]
        all_vectors.extend(batch_vecs)

    return np.vstack(all_vectors)


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"[build_index] FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_index(index: faiss.Index, docs: list[dict]) -> None:
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(docs, f)
    print(f"[build_index] Index saved -> {INDEX_PATH}")
    print(f"[build_index] Metadata saved -> {META_PATH}")


def load_index() -> tuple[faiss.Index, list[dict]]:
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"No FAISS index found at {INDEX_PATH}. "
            "Run `python -m rag.build_index` first."
        )

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        docs = pickle.load(f)

    print(f"[build_index] Loaded index with {index.ntotal} vectors.")
    return index, docs


if __name__ == "__main__":
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(
            f"Documents not found at {DOCS_PATH}. "
            "Run `python -m rag.ingest` first."
        )

    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    if not isinstance(docs, list):
        raise ValueError("nba_docs.json must contain a list of documents.")

    normalized_docs = []
    for doc in docs:
        if isinstance(doc, str):
            normalized_docs.append({"text": doc, "metadata": {}})
        elif isinstance(doc, dict) and "text" in doc:
            normalized_docs.append({
                "text": str(doc["text"]),
                "metadata": doc.get("metadata", {}),
            })
        else:
            raise ValueError("Each document must be either a string or a dict with a 'text' field.")

    texts = [doc["text"] for doc in normalized_docs]

    print(f"[build_index] Loaded {len(normalized_docs)} documents.")

    print("[build_index] Doc type counts:")
    type_counts = {}
    season_counts = {}
    for doc in normalized_docs:
        meta = doc.get("metadata", {})
        doc_type = meta.get("doc_type", "unknown")
        season = meta.get("season", "unknown")
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        season_counts[season] = season_counts.get(season, 0) + 1

    for k, v in sorted(type_counts.items()):
        print(f"  - {k}: {v}")

    print("[build_index] Season counts:")
    for k, v in sorted(season_counts.items(), reverse=True):
        print(f"  - {k}: {v}")

    print("[build_index] Embedding with model='nomic-embed-text'...")
    vectors = embed_batch(texts, batch_size=32)

    index = build_faiss_index(vectors)
    save_index(index, normalized_docs)

    print("[build_index] Done.")