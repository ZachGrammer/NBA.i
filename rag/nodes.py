"""
nodes.py
--------
LangGraph / RAG nodes for the NBA local RAG app.

Supports:
- structured query routing for leaderboard / recent / shot-zone / comparison questions
- semantic FAISS fallback for open-ended questions
- metadata-aware semantic retrieval
- player-aware and doc-type-aware reranking
- season-aware semantic query expansion
- retrieval_mode tracking for debugging / UI display
"""

from __future__ import annotations

import os
import pickle
import re
from typing import Any

import faiss
import numpy as np
import ollama

from rag.prompts import ANSWER_PROMPT
from rag.query_router import classify_query, extract_season_reference, LATEST_COMPLETED_SEASON
from rag.structured_answers import (
    answer_structured_query,
    extract_player_names,
    load_season_stats,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_DIR = os.path.join(PROJECT_ROOT, "vectorstore")
INDEX_PATH = os.path.join(VECTORSTORE_DIR, "nba.index")
META_PATH = os.path.join(VECTORSTORE_DIR, "nba_meta.pkl")

EMBED_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.3"
TOP_K = 6

CORPUS_DOC_TYPES = {
    "corpus_overview",
    "season_coverage",
    "coverage_by_doc_type",
}

_index = None
_metadata = None


def _load_vectorstore() -> tuple[Any, list[dict]]:
    global _index, _metadata

    if _index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. "
                "Run `python -m rag.build_index` first."
            )
        _index = faiss.read_index(INDEX_PATH)

    if _metadata is None:
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(
                f"Metadata file not found at {META_PATH}. "
                "Run `python -m rag.build_index` first."
            )
        with open(META_PATH, "rb") as f:
            _metadata = pickle.load(f)

    return _index, _metadata


def get_embedding(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def _generate_llm_answer(question: str, context: str) -> str:
    prompt = ANSWER_PROMPT.format(question=question, context=context)
    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an NBA statistics assistant. "
                    "Use only the provided context. "
                    "If the context is insufficient, say so clearly. "
                    "Do not invent stats, rankings, player comparisons, or season context."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"].strip()


def _normalize_doc(item: Any) -> dict:
    if isinstance(item, dict):
        return {
            "text": str(item.get("text", "")),
            "metadata": item.get("metadata", {}),
        }
    return {"text": str(item), "metadata": {}}


def _get_all_docs() -> list[dict]:
    _, metadata = _load_vectorstore()
    docs = []
    for item in metadata:
        doc = _normalize_doc(item)
        if doc["text"]:
            docs.append(doc)
    return docs


def _cosine_score(query_vec: np.ndarray, doc_vec: np.ndarray) -> float:
    q = query_vec / np.linalg.norm(query_vec)
    d = doc_vec / np.linalg.norm(doc_vec)
    return float(np.dot(q, d))


def rank_docs_by_query(query: str, docs: list[dict], top_k: int = TOP_K) -> list[dict]:
    if not docs:
        return []

    query_vec = np.array(get_embedding(query), dtype="float32")
    scored = []

    for doc in docs:
        doc_vec = np.array(get_embedding(doc["text"]), dtype="float32")
        score = _cosine_score(query_vec, doc_vec)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def retrieve_semantic_docs_global(query: str, k: int = TOP_K) -> list[dict]:
    index, metadata = _load_vectorstore()
    query_embedding = np.array([get_embedding(query)], dtype="float32")
    faiss.normalize_L2(query_embedding)
    _, indices = index.search(query_embedding, k)

    docs: list[dict] = []
    for idx in indices[0]:
        if idx == -1:
            continue
        docs.append(_normalize_doc(metadata[idx]))
    return docs


def format_semantic_context(docs: list[dict]) -> str:
    if not docs:
        return "No relevant context was retrieved."
    return "\n\n".join(doc["text"] for doc in docs)


def _resolve_semantic_season(query: str, route_info: dict | None = None) -> str | None:
    route_info = route_info or {}
    if route_info.get("intent") == "corpus_lookup":
        return extract_season_reference(query)
    return extract_season_reference(query) or LATEST_COMPLETED_SEASON


def expand_semantic_query(query: str, route_info: dict | None = None) -> str:
    q = query.lower().strip()
    route_info = route_info or {}
    season = _resolve_semantic_season(query, route_info)

    if route_info.get("intent") == "corpus_lookup":
        season_part = f" {season}" if season else ""
        return (
            f"{query}{season_part} corpus overview season coverage coverage by doc type "
            "available seasons covered players recent game coverage shot profile coverage"
        )

    if "what kind of player is" in q:
        player = re.sub(r"(?i)what kind of player is", "", query).replace("?", "").strip()
        return f"{player} {season} player summary season summary recent summary shot style"

    if "tell me about" in q:
        player = re.sub(r"(?i)tell me about", "", query).replace("?", "").strip()
        return f"{player} {season} player summary season summary recent summary shot style"

    if "how has" in q and ("been playing" in q or "performed" in q):
        return f"{query} {season} recent summary recent games season summary"

    if "how is" in q and ("playing" in q or "performing" in q):
        return f"{query} {season} recent summary recent games season summary"

    if "is " in q and " good" in q:
        player = query.replace("?", "").strip()
        return f"{player} {season} season summary recent summary shot style"

    if "differ as players" in q or "different as players" in q:
        return f"{query} {season} player comparison season summary shot style recent summary"

    return f"{query} {season}" if season else query


def extract_query_players(query: str) -> list[str]:
    season_df = load_season_stats()
    if season_df.empty or "player_name" not in season_df.columns:
        return []

    valid_names = season_df["player_name"].dropna().astype(str).unique().tolist()
    return extract_player_names(query, valid_names, max_players=2)


def filter_docs_by_player_name(docs: list[dict], player_names: list[str]) -> list[dict]:
    if not player_names:
        return docs

    player_names_lower = [p.lower() for p in player_names]
    filtered = []

    for doc in docs:
        metadata = doc.get("metadata", {})
        doc_type = str(metadata.get("doc_type", "")).strip()
        meta_player = str(metadata.get("player_name", "")).lower().strip()

        if doc_type in CORPUS_DOC_TYPES:
            filtered.append(doc)
            continue

        if meta_player and meta_player in player_names_lower:
            filtered.append(doc)

    return filtered


def doc_type_priority_for_query(query: str, route_info: dict | None = None) -> list[str]:
    q = query.lower()
    route_info = route_info or {}

    if route_info.get("intent") == "corpus_lookup":
        return [
            "corpus_overview",
            "season_coverage",
            "coverage_by_doc_type",
            "season_summary",
            "recent_summary",
            "shot_profile_overall",
        ]

    if "what kind of player is" in q or "tell me about" in q or ("is " in q and " good" in q):
        return ["season_style", "season_summary", "recent_summary", "shot_style", "shot_profile_overall"]

    if "how has" in q or "how is" in q:
        return ["recent_summary", "recent_game", "season_summary", "season_style"]

    if "shot profile" in q or "from the corners" in q or "above the break" in q:
        return ["shot_style", "shot_profile_overall", "shot_profile_split", "season_summary"]

    if "differ as players" in q or "different as players" in q:
        return ["season_style", "season_summary", "recent_summary", "shot_style", "shot_profile_overall"]

    return [
        "season_summary",
        "recent_summary",
        "season_style",
        "shot_style",
        "shot_profile_overall",
        "recent_game",
        "shot_profile_split",
        "corpus_overview",
        "season_coverage",
        "coverage_by_doc_type",
    ]


def rerank_docs_by_doc_type(docs: list[dict], preferred_order: list[str]) -> list[dict]:
    order_map = {doc_type: idx for idx, doc_type in enumerate(preferred_order)}

    def _rank(doc: dict) -> tuple[int, str]:
        doc_type = doc.get("metadata", {}).get("doc_type", "zzz_unknown")
        return (order_map.get(doc_type, 999), doc.get("text", ""))

    return sorted(docs, key=_rank)


def route_question(state: dict) -> dict:
    question = state["question"]
    route_info = classify_query(question)

    print("\n[ROUTER DEBUG]", route_info)

    state["route_info"] = route_info
    state["retrieval_mode"] = "unresolved"
    state["structured_answer_found"] = False
    return state


def try_structured_answer(state: dict) -> dict:
    question = state["question"]
    route_info = state.get("route_info", {})

    if route_info.get("route") != "structured":
        state["structured_answer_found"] = False
        return state

    print("[STRUCTURED ATTEMPT]", route_info)

    result = answer_structured_query(route_info, question)
    if result:
        state["answer"] = result["answer"]
        state["final_answer"] = result["answer"]
        state["retrieval_mode"] = result.get("mode", "structured")
        state["retrieved_docs"] = result.get("rows", [])
        state["context"] = result["answer"]
        state["structured_answer_found"] = True
    else:
        print("[STRUCTURED FAILED -> FALLBACK]")
        state["structured_answer_found"] = False

    return state


def retrieve_context(state: dict) -> dict:
    if state.get("structured_answer_found"):
        return state

    question = state["question"]
    route_info = state.get("route_info", {})

    expanded_query = expand_semantic_query(question, route_info=route_info)

    if route_info.get("intent") == "corpus_lookup":
        player_names = []
    else:
        player_names = extract_query_players(question)

    all_docs = _get_all_docs()

    if player_names:
        candidate_docs = filter_docs_by_player_name(all_docs, player_names)
    else:
        candidate_docs = all_docs

    preferred_doc_types = doc_type_priority_for_query(question, route_info=route_info)
    candidate_docs = rerank_docs_by_doc_type(candidate_docs, preferred_doc_types)

    if candidate_docs:
        docs = rank_docs_by_query(expanded_query, candidate_docs, top_k=TOP_K)
    else:
        docs = retrieve_semantic_docs_global(expanded_query, k=TOP_K)

    print("\n[SEMANTIC DEBUG] question:", question)
    print("[SEMANTIC DEBUG] expanded query:", expanded_query)
    print("[SEMANTIC DEBUG] detected players:", player_names)
    print("[SEMANTIC DEBUG] candidate docs after filtering:", len(candidate_docs))
    print("[SEMANTIC DEBUG] retrieved docs:")
    for i, doc in enumerate(docs, start=1):
        print(f"[{i}] {doc['text'][:300]}")

    context = format_semantic_context(docs)
    state["retrieved_docs"] = docs
    state["context"] = context
    state["retrieval_mode"] = "semantic_faiss"
    return state


def answer_question(state: dict) -> dict:
    if state.get("structured_answer_found"):
        return state

    question = state["question"]
    context = state.get("context", "")
    answer = _generate_llm_answer(question, context)

    state["answer"] = answer
    state["final_answer"] = answer
    return state


def run_rag_pipeline(question: str) -> dict:
    state = {"question": question}
    state = route_question(state)
    state = try_structured_answer(state)
    state = retrieve_context(state)
    state = answer_question(state)
    return state


if __name__ == "__main__":
    while True:
        try:
            q = input("\nAsk a question (or 'quit'): ").strip()
            if q.lower() in {"quit", "exit"}:
                break

            result = run_rag_pipeline(q)

            print("\n--- Retrieval mode ---")
            print(result.get("retrieval_mode"))

            print("\n--- Answer ---")
            print(result.get("final_answer", result.get("answer", "No answer generated.")))
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"\n[error] {exc}")