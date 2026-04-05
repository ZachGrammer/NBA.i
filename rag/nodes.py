"""
nodes.py
--------
Individual LangGraph node functions for the NBA RAG pipeline.

Each function receives the shared NBAGraphState dict and returns a
partial-state dict with only the keys it modifies — LangGraph merges
updates automatically.

Nodes
-----
1. retrieve_context   – FAISS semantic search → retrieved_docs
2. analyze_matchup    – Clean + rank docs → context string
3. generate_answer    – Ollama llama3 generation → final_answer
"""

import numpy as np
import ollama
import faiss

from rag.state import NBAGraphState
from rag.prompts import MATCHUP_ANALYSIS_PROMPT
from rag.build_index import load_index, embed_text, EMBED_MODEL

# ---------------------------------------------------------------------------
# Module-level index cache (loaded once, reused across calls)
# ---------------------------------------------------------------------------
_INDEX: faiss.Index | None = None
_TEXTS: list[str] | None = None

GENERATION_MODEL = "llama3.2"     # ollama pull llama3.2
TOP_K = 6                          # number of chunks to retrieve


def _get_index() -> tuple[faiss.Index, list[str]]:
    """Lazy-load the FAISS index once per process."""
    global _INDEX, _TEXTS
    if _INDEX is None:
        _INDEX, _TEXTS = load_index()
    return _INDEX, _TEXTS


# ---------------------------------------------------------------------------
# Node 1 – retrieve_context
# ---------------------------------------------------------------------------

def retrieve_context(state: NBAGraphState) -> dict:
    """
    Embed the user question and retrieve the top-K most semantically
    similar NBA documents from the FAISS index.

    Input state keys used : question
    Output state keys set : retrieved_docs
    """
    question = state["question"]
    print(f"[retrieve_context] Question: {question!r}")

    # 1. Embed the query
    q_vec = embed_text(question).reshape(1, -1)

    # 2. L2-normalise to match how the index was built
    faiss.normalize_L2(q_vec)

    # 3. Search
    index, texts = _get_index()
    scores, indices = index.search(q_vec, TOP_K)

    # 4. Collect matching documents (filter score > 0 to drop garbage)
    retrieved = [
        texts[idx]
        for score, idx in zip(scores[0], indices[0])
        if score > 0.0 and idx < len(texts)
    ]

    print(f"[retrieve_context] Retrieved {len(retrieved)} docs.")
    return {"retrieved_docs": retrieved}


# ---------------------------------------------------------------------------
# Node 2 – analyze_matchup
# ---------------------------------------------------------------------------

def analyze_matchup(state: NBAGraphState) -> dict:
    """
    Deduplicate and format retrieved documents into a single context
    string ready for the LLM prompt.

    This node is intentionally kept deterministic (no LLM call) so that
    retrieval quality can be evaluated independently from generation.

    Input state keys used : retrieved_docs
    Output state keys set : context
    """
    docs = state.get("retrieved_docs", [])

    if not docs:
        return {"context": "No relevant statistics found for this query."}

    # Deduplicate while preserving insertion order
    seen: set[str] = set()
    unique_docs: list[str] = []
    for doc in docs:
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)

    # Number each chunk so the LLM can cite sources
    context_lines = [f"[{i+1}] {doc}" for i, doc in enumerate(unique_docs)]
    context = "\n\n".join(context_lines)

    print(f"[analyze_matchup] Context built from {len(unique_docs)} unique docs.")
    return {"context": context}


# ---------------------------------------------------------------------------
# Node 3 – generate_answer
# ---------------------------------------------------------------------------

def generate_answer(state: NBAGraphState) -> dict:
    """
    Call Ollama's llama3.3 model with the retrieved context to produce
    a structured, data-grounded NBA analysis.

    Input state keys used : question, context
    Output state keys set : final_answer
    """
    question = state["question"]
    context = state.get("context", "")

    # Fill in the prompt template
    prompt = MATCHUP_ANALYSIS_PROMPT.format(
        context=context,
        question=question,
    )

    print(f"[generate_answer] Calling Ollama ({GENERATION_MODEL})...")

    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert NBA analyst. Be concise, precise, "
                    "and always ground your answers in the provided statistics."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": 0.2,   # low temp = more deterministic / factual
            "num_predict": 512,    # cap tokens to keep responses tight
        },
    )

    answer = response["message"]["content"].strip()
    print(f"[generate_answer] Answer generated ({len(answer)} chars).")
    return {"final_answer": answer}
