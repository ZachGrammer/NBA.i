"""
graph.py
--------
Compiles the deterministic LangGraph workflow for the NBA RAG system.

Flow
----
    question
        │
        ▼
    retrieve_context     ← FAISS semantic search
        │
        ▼
    analyze_matchup      ← deduplicate + format context
        │
        ▼
    generate_answer      ← Ollama llama3.3 generation
        │
        ▼
    END

Usage
-----
    from rag.graph import build_graph

    graph = build_graph()
    result = graph.invoke({"question": "Who dominates rebounds tonight?"})
    print(result["final_answer"])
"""

from langgraph.graph import StateGraph, END

from rag.state import NBAGraphState
from rag.nodes import retrieve_context, analyze_matchup, generate_answer


def build_graph() -> StateGraph:
    """
    Construct and compile the NBA RAG LangGraph.

    Returns a compiled graph that can be invoked with:
        graph.invoke({"question": "..."})

    The graph is deterministic — no conditional branches at MVP.
    Add branching (e.g. query rewriting on low-confidence retrieval)
    in future iterations.
    """

    # 1. Initialise the state graph with our typed state schema
    workflow = StateGraph(NBAGraphState)

    # 2. Register nodes (name → function)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("analyze_matchup", analyze_matchup)
    workflow.add_node("generate_answer", generate_answer)

    # 3. Define directed edges (execution order)
    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "analyze_matchup")
    workflow.add_edge("analyze_matchup", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # 4. Compile into an executable runnable
    return workflow.compile()


# ---------------------------------------------------------------------------
# Quick smoke-test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    graph = build_graph()

    test_questions = [
        "Who has the matchup advantage tonight?",
        "Which player is likely to outperform their average?",
        "Who dominates rebounds in this matchup?",
        "How has LeBron James performed in the last 5 games?",
        "What trends suggest an upset tonight?",
    ]

    for q in test_questions:
        print("\n" + "=" * 60)
        print(f"Q: {q}")
        print("=" * 60)
        result = graph.invoke({"question": q})
        print(result["final_answer"])
