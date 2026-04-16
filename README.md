# NBA.i

> A local, modular Retrieval-Augmented Generation (RAG) system for NBA analytics and matchup intelligence.
> Powered by **Ollama**, **LangGraph**, and **FAISS** — runs 100% locally, no API keys required.

---

## Architecture

```
NBA.i/
├── main.py                    ← interactive CLI
├── requirements.txt
├── rag/
│   ├── __init__.py
│   ├── state.py               ← shared LangGraph state schema
│   ├── prompts.py             ← LLM prompt templates
│   ├── ingest.py              ← nba_api → narrative documents
│   ├── build_index.py         ← embed docs → FAISS index
│   ├── nodes.py               ← retrieve / analyze / generate nodes
│   └── graph.py               ← LangGraph workflow compiler
├── vectorstore/               ← FAISS index + metadata (auto-created)
└── data/
    └── processed_docs/        ← JSON narrative documents (auto-created)
```

### LangGraph Flow

```
question → retrieve_context → analyze_matchup → generate_answer → END
```

| Node | Role |
|---|---|
| `retrieve_context` | Embeds the question with `nomic-embed-text`, searches FAISS |
| `analyze_matchup` | Deduplicates + formats retrieved chunks into a context string |
| `generate_answer` | Calls `llama3.3` via Ollama with the context-grounded prompt |

---

## Prerequisites

### 1 — Install Ollama

```bash
# macOS
brew install ollama

# Then pull the two required models
ollama pull nomic-embed-text
ollama pull llama3.3
```

### 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Running the System

### Step 1 — Ingest NBA data

```bash
python -m rag.ingest
```

Pulls season averages + last-5-game logs for the top 50 scorers.
Writes narrative documents to `data/processed_docs/nba_docs.json`.

### Step 2 — Build the FAISS vector index

```bash
python -m rag.build_index
```

Embeds all documents with `nomic-embed-text` and saves a cosine-similarity
FAISS index to `vectorstore/`.

### Step 3 — Ask questions

```bash
python main.py
```

Example questions:
- *"Who has the matchup advantage tonight?"*
- *"Which player is likely to outperform their average?"*
- *"Who dominates rebounds in this matchup?"*
- *"How has LeBron James performed in the last 5 games?"*
- *"What trends suggest an upset tonight?"*

---

## Refreshing Data

Re-run steps 1 and 2 any time you want fresh stats:

```bash
python -m rag.ingest && python -m rag.build_index
```

---

## Next Steps

- [ ] Add **FastAPI** server (`api/server.py`) for HTTP access
- [ ] Add a **query rewriting** node to improve retrieval precision
- [ ] Ingest **play-by-play** and **injury reports** as additional doc types
- [ ] Add **team-level** narrative documents alongside player docs
- [ ] Implement **live data refresh** via a scheduled cron job
- [ ] Add evaluation harness (RAGAS or custom metrics)
