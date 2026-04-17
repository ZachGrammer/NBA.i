"""
graph.py
--------
Compiles the LangGraph workflow for the NBA RAG system.

Flow
----
    question
        │
        ▼
    route_question
        │
        ▼
    try_structured_answer
        ├── if structured answer found ──► END
        └── otherwise ───────────────────► retrieve_context
                                              │
                                              ▼
                                          answer_question
                                              │
                                              ▼
                                              END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from rag.nodes import (
    answer_question,
    retrieve_context,
    route_question,
    try_structured_answer,
)
from rag.state import NBAGraphState


def _after_structured(state: NBAGraphState) -> str:
    """
    Decide whether to stop after structured handling or continue
    into semantic retrieval.
    """
    if state.get("structured_answer_found"):
        return "done"
    return "semantic"


def build_graph():
    """
    Construct and compile the NBA RAG LangGraph.

    Usage:
        graph = build_graph()
        result = graph.invoke({"question": "Who scored the most in their last 5 games?"})
        print(result["final_answer"])
    """
    workflow = StateGraph(NBAGraphState)

    workflow.add_node("route_question", route_question)
    workflow.add_node("try_structured_answer", try_structured_answer)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("answer_question", answer_question)

    workflow.set_entry_point("route_question")
    workflow.add_edge("route_question", "try_structured_answer")

    workflow.add_conditional_edges(
        "try_structured_answer",
        _after_structured,
        {
            "done": END,
            "semantic": "retrieve_context",
        },
    )

    workflow.add_edge("retrieve_context", "answer_question")
    workflow.add_edge("answer_question", END)

    return workflow.compile()


if __name__ == "__main__":
    graph = build_graph()

    test_questions = [
        "Who are the best 3 point shooters this season?",
        "Who scored the most points in their last 5 games?",
        "Who are the best corner 3 shooters?",
        "Compare Stephen Curry and Damian Lillard",
        "How has LeBron James performed in the last 5 games?",
    ]

    for q in test_questions:
        print("\n" + "=" * 60)
        print(f"Q: {q}")
        print("=" * 60)
        result = graph.invoke({"question": q})
        print(result.get("final_answer", result.get("answer", "No answer generated.")))
        print(f"\nMode: {result.get('retrieval_mode', 'unknown')}")