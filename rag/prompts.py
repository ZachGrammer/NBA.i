"""
prompts.py
----------
All LLM prompt templates live here so they can be versioned, A/B tested,
and swapped without touching graph logic.
"""

# ---------------------------------------------------------------------------
# Matchup analysis prompt
# Receives: context (retrieved NBA narrative docs), question (user query)
# ---------------------------------------------------------------------------
MATCHUP_ANALYSIS_PROMPT = """\
You are an expert NBA analyst with deep knowledge of player statistics,
team dynamics, and game strategy.

Use ONLY the context below to answer the question. If the context does not
contain enough information, say so — do not hallucinate stats.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {question}

Provide a structured, data-driven analysis. Lead with the key insight,
then support it with statistics from the context.
"""

# ---------------------------------------------------------------------------
# Retrieval query rewriting prompt  (optional future node)
# ---------------------------------------------------------------------------
REWRITE_QUERY_PROMPT = """\
Rewrite the following NBA question into a concise search query that will
surface the most relevant player and team statistics:

Original question: {question}
Rewritten query:
"""
