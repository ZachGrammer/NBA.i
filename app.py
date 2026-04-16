"""
app.py
------
Streamlit web frontend for the NBA.I RAG Intelligence System.

Run
---
    streamlit run app.py

Requirements
------------
- FAISS index must be built first:
    python -m rag.ingest
    python -m rag.build_index
- Ollama must be running locally with llama3.2 + nomic-embed-text pulled.
"""

import random
import streamlit as st
from rag.graph import build_graph
from rag.nodes import GENERATION_MODEL

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NBA.I — Intelligence System",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Orbitron:wght@700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Background ── */
    .stApp {
        background: #0a0e1a;
        min-height: 100vh;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stSidebar"] { display: none; }
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* ── Top header bar ── */
    .top-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 18px 40px 16px 40px;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        background: rgba(10, 14, 26, 0.95);
        backdrop-filter: blur(12px);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    .logo-group {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .logo-icon {
        width: 42px;
        height: 42px;
        background: #c0392b;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        box-shadow: 0 0 18px rgba(192, 57, 43, 0.5);
        flex-shrink: 0;
    }
    .logo-text {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.5rem;
        font-weight: 900;
        color: #c0392b;
        letter-spacing: 2px;
    }
    .logo-sub {
        color: #6b7280;
        font-size: 0.78rem;
        font-weight: 400;
        margin-top: 1px;
    }
    .badge-live {
        background: #166534;
        color: #86efac;
        border: 1px solid #166534;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        padding: 5px 14px;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-dot {
        width: 7px;
        height: 7px;
        background: #4ade80;
        border-radius: 50%;
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.4; }
    }
    .header-right {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* ── Main content area ── */
    .main-area {
        max-width: 860px;
        margin: 0 auto;
        padding: 32px 24px 120px 24px;
    }

    /* ── Chip row ── */
    .chip-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 18px;
    }

    /* ── Input section ── */
    .input-label {
        font-size: 0.72rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #4b5563;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* ── Streamlit input override ── */
    .stTextInput > div > div > input {
        background: #1c2333 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
        color: #f1f5f9 !important;
        font-size: 1.05rem !important;
        padding: 16px 20px !important;
        caret-color: #63b3ed;
        transition: border 0.2s, box-shadow 0.2s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: rgba(99, 179, 237, 0.4) !important;
        box-shadow: 0 0 0 3px rgba(99, 179, 237, 0.1) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder { color: #4b5563 !important; }

    /* ── Ask button ── */
    button[kind="secondary"][data-testid="baseButton-secondary"]#send_btn,
    div[data-testid="column"]:has(button[kind="secondary"]) .stButton > button[key="send_btn"] {
        background: #1c2333 !important;
        color: #9ca3af !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        height: 56px !important;
        width: 100% !important;
        transition: all 0.2s !important;
        letter-spacing: 0.3px !important;
    }

    /* ── Chip buttons — target by key prefix ── */
    button[data-testid="baseButton-secondary"][kind="secondary"] {
        background: #1c2333 !important;
        color: #d1d5db !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 24px !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
        padding: 7px 18px !important;
        transition: all 0.2s !important;
    }
    button[data-testid="baseButton-secondary"][kind="secondary"]:hover {
        background: #2d3748 !important;
        border-color: rgba(99, 179, 237, 0.45) !important;
        color: #93c5fd !important;
        transform: translateY(-1px) !important;
    }

    /* ── Ask button override ── */
    button[key="send_btn"] {
        border-radius: 14px !important;
        height: 56px !important;
    }

    /* ── Chat bubbles ── */
    .bubble-user {
        display: flex;
        justify-content: flex-end;
        margin: 12px 0;
        animation: fadein 0.3s ease;
    }
    .bubble-user-inner {
        background: #1d4ed8;
        color: #fff;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        max-width: 68%;
        font-size: 0.93rem;
        line-height: 1.65;
        box-shadow: 0 4px 16px rgba(29, 78, 216, 0.35);
    }
    .bubble-ai {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin: 12px 0;
        animation: fadein 0.3s ease;
    }
    .bubble-ai-icon {
        font-size: 1.3rem;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .bubble-ai-inner {
        background: #141c2e;
        border: 1px solid rgba(255,255,255,0.07);
        color: #d1d5db;
        padding: 14px 18px;
        border-radius: 4px 18px 18px 18px;
        max-width: 85%;
        font-size: 0.92rem;
        line-height: 1.8;
    }
    .bubble-ai-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #4b9de8;
        margin-bottom: 5px;
    }
    @keyframes fadein {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Collapsible Summaries ── */
    details {
        margin-top: 5px;
    }
    summary {
        list-style: none;
        cursor: pointer;
        color: #4b9de8;
        font-size: 0.8rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 5px;
        transition: opacity 0.2s;
    }
    summary:hover { opacity: 0.8; }
    summary::-webkit-details-marker { display: none; }
    summary::before {
        content: "▶";
        font-size: 0.6rem;
        transition: transform 0.2s;
    }
    details[open] summary::before {
        transform: rotate(90deg);
    }

    /* ── Empty state ── */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 80px 0 40px 0;
        gap: 16px;
        color: #4b5563;
        font-size: 0.95rem;
    }
    .empty-ball { font-size: 3.2rem; }

    /* ── Settings popover ── */
    [data-testid="stPopover"] button {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #9ca3af !important;
        border-radius: 10px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 5px 14px !important;
        transition: all 0.2s !important;
    }
    [data-testid="stPopover"] button:hover {
        background: rgba(255,255,255,0.1) !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stPopoverBody"] {
        background: #141c2e !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 14px !important;
        padding: 16px !important;
        color: #d1d5db !important;
        min-width: 260px !important;
    }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.06) !important; margin: 20px 0 !important; }

    /* ── Scrollable chat ── */
    .chat-scroll {
        max-height: 520px;
        overflow-y: auto;
        padding-right: 4px;
        margin-bottom: 24px;
    }
    .chat-scroll::-webkit-scrollbar { width: 4px; }
    .chat-scroll::-webkit-scrollbar-track { background: transparent; }
    .chat-scroll::-webkit-scrollbar-thumb {
        background: rgba(99,179,237,0.25);
        border-radius: 4px;
    }

    /* ── Clear button fix ── */
    #clear-btn > button {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        color: #f87171 !important;
        border-radius: 10px !important;
        font-size: 0.82rem !important;
        width: 100% !important;
    }
    #clear-btn > button:hover {
        background: rgba(239, 68, 68, 0.2) !important;
    }

    /* spinner colour */
    .stSpinner > div { color: #63b3ed !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    with st.spinner("Loading RAG graph…"):
        st.session_state.graph = build_graph()

if "chip_prefill" not in st.session_state:
    st.session_state.chip_prefill = ""

# ---------------------------------------------------------------------------
# Shared query runner (used by chips AND the input form)
# ---------------------------------------------------------------------------
_LOADING_PHRASES = [
    "🏀  Shot clock winding down…",
    "🎯  Pulling from deep…",
    "🏋️  At the free throw line…",
    "📊  Breaking down the film…",
    "⚡  Fast break in progress…",
    "🔍  Reading the defence…",
    "🧠  Running the play…",
    "🎙️  Checking with the analyst…",
]

def run_query(question: str) -> None:
    """Invoke the RAG graph and append messages to session state."""
    question = question.strip()
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    phrase = random.choice(_LOADING_PHRASES)
    with st.spinner(phrase):
        try:
            result = st.session_state.graph.invoke({"question": question})
            answer = result.get(
                "final_answer",
                "Sorry, I couldn't generate an answer. Please try again.",
            )
        except Exception as exc:
            answer = f"⚠️ An error occurred: {exc}"
    st.session_state.messages.append({"role": "ai", "content": answer})

# ---------------------------------------------------------------------------
# Suggestion chips
# ---------------------------------------------------------------------------
SUGGESTED = [
    "Who dominates rebounds?",
    "Best 3-point shooters",
    "Assists leaders this season",
    "LeBron career stats",
    "Double-double under 25",
    "Best scoring average",
]

# ---------------------------------------------------------------------------
# ── Header / Navbar ──
# ---------------------------------------------------------------------------
nav_left, nav_right = st.columns([5, 2])

with nav_left:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px; padding: 10px 0 8px 0;">
            <div style="display:flex; align-items:center; gap:0; font-family:'Orbitron',sans-serif; font-weight:900; letter-spacing:1px;">
                <span style="color:#c0392b; font-size:1.6rem;">NBA</span>
                <span style="color:#ffffff; font-size:1.6rem; margin-left:2px;">.I</span>
            </div>
            <div style="width:1px; height:22px; background:rgba(255,255,255,0.2); flex-shrink:0;"></div>
            <div style="color:#6b7280; font-size:0.82rem; font-weight:400; white-space:nowrap;">
                Real-time basketball intelligence · nba_api + Llama 3.3
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_right:
    rc1, rc2, rc3 = st.columns([2, 2, 1])

    with rc1:
        with st.popover("⚙ Settings"):
            st.markdown(
                "<div style='color:#9ca3af; font-size:0.7rem; letter-spacing:2px;"
                "text-transform:uppercase; font-weight:600; margin-bottom:12px;'>System Info</div>",
                unsafe_allow_html=True,
            )
            questions_asked = sum(1 for m in st.session_state.messages if m["role"] == "user")
            st.markdown(f"**Model:** `{GENERATION_MODEL}`")
            st.markdown("**Vector Store:** FAISS · Local")
            st.markdown("**Embeddings:** nomic-embed-text")
            st.markdown(f"**Queries this session:** {questions_asked}")
            st.markdown(f"**Messages total:** {len(st.session_state.messages)}")
            st.markdown("---")
            st.markdown(
                "<div style='color:#9ca3af; font-size:0.7rem; letter-spacing:2px;"
                "text-transform:uppercase; font-weight:600; margin-bottom:12px;'>Actions</div>",
                unsafe_allow_html=True,
            )
            if st.button("🗑 Clear conversation", key="clear_btn", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chip_prefill = ""
                st.rerun()

    with rc2:
        st.markdown(
            """
            <div style="display:flex; align-items:center; justify-content:center; height:38px;">
                <div class="badge-live">
                    <div class="badge-dot"></div> Live data
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with rc3:
        st.markdown(
            """
            <div style="display:flex; align-items:center; justify-content:center; height:38px;">
                <div style="background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12);
                            border-radius:8px; padding:5px 10px; color:#9ca3af;
                            font-size:0.9rem; cursor:pointer; letter-spacing:2px;">···</div>
            </div>
            """,
            unsafe_allow_html=True,
        )



# ---------------------------------------------------------------------------
# ── Main area (centred) ──
# ---------------------------------------------------------------------------
_, center, _ = st.columns([1, 5, 1])

with center:

    # ── Suggestion chips ──
    chip_cols = st.columns(len(SUGGESTED))
    for i, suggestion in enumerate(SUGGESTED):
        with chip_cols[i]:
            if st.button(suggestion, key=f"chip_{i}"):
                st.session_state.chip_prefill = suggestion
                st.rerun()

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # ── Input bar (form so Enter key submits) ──
    with st.form(key="chat_form", clear_on_submit=True):
        input_col, btn_col = st.columns([6, 1])
        with input_col:
            user_input = st.text_input(
                label="Your question",
                placeholder="Ask anything about NBA players, games, or stats...",
                value=st.session_state.chip_prefill,
                label_visibility="collapsed",
            )
        with btn_col:
            send_clicked = st.form_submit_button("Ask →")
        # Capture value here while it's still available (before clear_on_submit wipes it)
        if send_clicked and user_input and user_input.strip():
            st.session_state["_submit_question"] = user_input.strip()
            st.session_state.chip_prefill = ""  # clear the pre-fill after submit

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<hr style='margin:0; border-color:rgba(255,255,255,0.06);'>",
        unsafe_allow_html=True,
    )

    # ── Chat history or empty state ──
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-ball">🏀</div>
                <div>Ask a question above or tap a suggestion to get started.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f"""<div class="bubble-user">
                            <div class="bubble-user-inner">{msg["content"]}</div>
                        </div>""",
                    unsafe_allow_html=True,
                )
            else:
                full_text = msg["content"].replace("\n", "<br>")
                display_html = f"""
                    <details open>
                        <summary style="margin-bottom: 8px; color: #4b9de8; font-size: 0.72rem; letter-spacing: 1px;">
                            COLLAPSE ANALYSIS
                        </summary>
                        <div style="margin-top: 5px;">
                            {full_text}
                        </div>
                    </details>
                """
                
                st.markdown(
                    f"""<div class="bubble-ai">
                            <div class="bubble-ai-icon">🏀</div>
                            <div>
                                <div class="bubble-ai-label">NBA.I Analysis</div>
                                <div class="bubble-ai-inner">{display_html}</div>
                            </div>
                        </div>""",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ── Process submitted question ──
# ---------------------------------------------------------------------------
if st.session_state.get("_submit_question"):
    question = st.session_state.pop("_submit_question")
    run_query(question)
    st.rerun()
