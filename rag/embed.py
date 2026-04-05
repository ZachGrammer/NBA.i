# rag/embed.py

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "chunks.json"
EMBEDDINGS_PATH = BASE_DIR / "data" / "embeddings.npy"
METADATA_PATH = BASE_DIR / "data" / "chunk_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(input_path: Path) -> list[dict]:
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_texts(chunks: list[dict]) -> list[str]:
    return [chunk["text"] for chunk in chunks]


def build_metadata(chunks: list[dict]) -> list[dict]:
    metadata = []
    for chunk in chunks:
        metadata.append(
            {
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
            }
        )
    return metadata


def create_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.astype("float32")


def save_embeddings(embeddings: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)


def save_metadata(metadata: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    chunks = load_chunks(INPUT_PATH)
    texts = extract_texts(chunks)
    metadata = build_metadata(chunks)

    embeddings = create_embeddings(texts, MODEL_NAME)

    save_embeddings(embeddings, EMBEDDINGS_PATH)
    save_metadata(metadata, METADATA_PATH)

    print(f"Saved {len(embeddings)} embeddings to {EMBEDDINGS_PATH}")
    print(f"Saved {len(metadata)} metadata records to {METADATA_PATH}")