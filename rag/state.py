"""
state.py
--------
Defines the shared state object that flows through every node in the
LangGraph pipeline.

This updated version supports both:
- structured answers from CSV-backed logic
- semantic retrieval + LLM fallback
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class NBAGraphState(TypedDict, total=False):
    """
    Central state bag passed between every LangGraph node.

    Fields
    ------
    question
        The raw user question.

    route_info
        Routing decision from query_router.py, including route, intent,
        stat, timeframe, zone, and comparison flags.

    retrieval_mode
        Human-readable label describing how the question was answered,
        e.g. "structured_season_leaderboard" or "semantic_faiss".

    structured_answer_found
        True if structured logic answered the question directly.

    retrieved_docs
        Either:
        - a list of semantic text chunks from FAISS, or
        - structured rows used to answer the question.

    context
        Final context string passed to the LLM in semantic mode, or the
        structured answer text when a structured path succeeds.

    answer
        Final answer returned to the app.

    final_answer
        Backward-compatible alias for older app/graph code that still
        expects the field name "final_answer".
    """

    question: str
    route_info: dict[str, Any]
    retrieval_mode: str
    structured_answer_found: bool
    retrieved_docs: list[Any]
    context: Optional[str]
    answer: Optional[str]
    final_answer: Optional[str]