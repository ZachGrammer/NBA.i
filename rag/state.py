"""
state.py
--------
Defines the shared state object that flows through every node in the
LangGraph pipeline.  Using TypedDict keeps the schema explicit and
lets LangGraph validate data at each edge.
"""

from typing import TypedDict, List, Optional


class NBAGraphState(TypedDict):
    """
    Central state bag passed between every LangGraph node.

    Fields
    ------
    question       : The raw user question (e.g. "Who wins the matchup tonight?")
    retrieved_docs : Raw document chunks returned by FAISS retrieval.
    context        : Cleaned / concatenated text fed into the LLM prompt.
    final_answer   : The model-generated answer returned to the caller.
    """

    question: str
    retrieved_docs: List[str]
    context: Optional[str]
    final_answer: Optional[str]
