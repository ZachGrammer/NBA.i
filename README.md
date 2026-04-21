NBA.i

A local, modular Retrieval-Augmented Generation (RAG) system for NBA
analytics, player evaluation, and data-driven Q&A. Powered by Ollama,
LangGraph, and FAISS — runs 100% locally, no API keys required.

------------------------------------------------------------------------

OVERVIEW

NBA.i is a hybrid system that combines:

-   Structured analytics (deterministic answers) for stats-based
    questions
-   Semantic retrieval (FAISS + embeddings) for open-ended questions
-   Corpus-aware reasoning for questions about available data

------------------------------------------------------------------------

ARCHITECTURE

NBA.i/ ├── main.py ├── requirements.txt ├── rag/ │ ├── state.py │ ├──
prompts.py │ ├── ingest.py │ ├── build_index.py │ ├── nodes.py │ ├──
structured_answers.py │ ├── query_router.py │ └── graph.py ├──
vectorstore/ └── data/processed_docs/

------------------------------------------------------------------------

DATA COVERAGE

Seasons: 2024–25, 2023–24, 2022–23, 2021–22, 2020–21 (2025–26 excluded
due to incomplete data)

Data Types: - Season stats (all players) - Recent games (top 50
scorers) - Shot profiles (top 60 players)

------------------------------------------------------------------------

RUNNING THE SYSTEM

1.  Ingest Data: python -m rag.ingest

2.  Build Index: python -m rag.build_index

3.  Run: streamlit run app.py

SUMMARY

NBA.i is a hybrid analytics engine that: - Uses structured logic where
possible - Uses LLM reasoning where needed - Understands its dataset -
Runs entirely locally
