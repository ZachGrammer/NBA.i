"""
main.py
-------
Interactive CLI entrypoint for the NBA RAG system.

Run
---
    python main.py

The FAISS index must already be built before using this script:
    python -m rag.ingest
    python -m rag.build_index
"""

from rag.graph import build_graph
from rag.nodes import GENERATION_MODEL


def main():
    print("\n" + "█" * 50)
    print("  NBA.i — RAG Intelligence System")
    print(f"  Powered by Ollama {GENERATION_MODEL} + FAISS")
    print("█" * 50 + "\n")

    # Compile the graph once (loads FAISS index on first query)
    graph = build_graph()

    print("Ask me anything about NBA matchups, player performance, or trends.")
    print("Type 'exit' or press Ctrl-C to quit.\n")

    while True:
        try:
            question = input("🏀  Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit] Goodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("[exit] Goodbye!")
            break

        result = graph.invoke({"question": question})
        print(f"\n📊  Answer:\n{result['final_answer']}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()
