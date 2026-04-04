"""
build_index.py
--------------
Loads processed narrative documents and builds a FAISS vector index
using Ollama's nomic-embed-text embedding model.

The index and the raw texts are persisted to /vectorstore so they can be
loaded at query time without re-embedding.

Usage
-----
    # Step 1 – ingest first (only needed when data refreshes)
    python -m rag.ingest

    # Step 2 – build / rebuild the FAISS index
    python -m rag.build_index
"""

import os
import json
import pickle
import numpy as np
import faiss
import ollama

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_PATH = os.path.join(PROJECT_ROOT, "data", "processed_docs", "nba_docs.json")
VECTORSTORE_DIR = os.path.join(PROJECT_ROOT, "vectorstore")
INDEX_PATH = os.path.join(VECTORSTORE_DIR, "nba.index")
META_PATH = os.path.join(VECTORSTORE_DIR, "nba_meta.pkl")  # stores raw texts + ids

os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBED_MODEL = "nomic-embed-text"   # pulled via: ollama pull nomic-embed-text
EMBED_DIM = 768                    # nomic-embed-text output dimension


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_text(text: str) -> np.ndarray:
    """
    Call the local Ollama embedding endpoint for a single text string.
    Returns a 1-D float32 numpy array.
    """
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    vector = response["embedding"]
    return np.array(vector, dtype=np.float32)


def embed_batch(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of texts in batches to avoid memory spikes.
    Returns a 2-D float32 numpy array of shape (len(texts), EMBED_DIM).
    """
    all_vectors = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        print(f"  [embed] Processing batch {i // batch_size + 1} "
              f"({i}–{min(i + batch_size, total)} of {total})...")

        batch_vecs = [embed_text(t) for t in batch]
        all_vectors.extend(batch_vecs)

    return np.vstack(all_vectors)


# ---------------------------------------------------------------------------
# FAISS index building
# ---------------------------------------------------------------------------

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """
    Build an L2-normalized inner-product FAISS index (cosine similarity).

    Why IndexFlatIP + normalization?
      - Exact nearest-neighbor search (no approximation errors at MVP scale)
      - Cosine similarity via L2-normalised vectors + dot product
      - Easy to swap for IndexIVFFlat later if corpus grows > 100k docs
    """
    # L2-normalise so inner product == cosine similarity
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product (cosine after L2 norm)
    index.add(vectors)
    print(f"[build_index] FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_index(index: faiss.Index, texts: list[str]) -> None:
    """Persist the FAISS index and matching raw text list."""
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(texts, f)
    print(f"[build_index] Index saved → {INDEX_PATH}")
    print(f"[build_index] Metadata saved → {META_PATH}")


def load_index() -> tuple[faiss.Index, list[str]]:
    """Load a previously built FAISS index + raw texts from disk."""
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"No FAISS index found at {INDEX_PATH}. "
            "Run `python -m rag.build_index` first."
        )
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        texts = pickle.load(f)
    print(f"[build_index] Loaded index with {index.ntotal} vectors.")
    return index, texts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load processed documents
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(
            f"Documents not found at {DOCS_PATH}. "
            "Run `python -m rag.ingest` first."
        )

    with open(DOCS_PATH) as f:
        docs: list[str] = json.load(f)

    print(f"[build_index] Loaded {len(docs)} documents.")

    # 2. Embed
    print(f"[build_index] Embedding with model='{EMBED_MODEL}'...")
    vectors = embed_batch(docs, batch_size=32)

    # 3. Build FAISS index
    index = build_faiss_index(vectors)

    # 4. Save
    save_index(index, docs)
    print("[build_index] Done.")
