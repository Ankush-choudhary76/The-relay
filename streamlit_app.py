"""The Relay — Streamlit Web UI.

Interactive frontend for question-answering strictly grounded in 3 research papers.
Demonstrates multi-stage pipeline, verified citations, and latency telemetry.
"""

import asyncio
import os
import re
import sys
import time
from pathlib import Path
import streamlit as st

# Setup project root and sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from relay.chunking import chunk_papers
from relay.config import settings
from relay.groq_provider import GroqProvider
from relay.math_format import format_math_unicode
from relay.papers import extract_all_papers
from relay.pipeline import RelayPipeline
from relay.retrieval import DocumentRetriever
from relay.schemas import RelayRequest
from relay.standin_provider import StandinProvider
from standin.app import quota_tracker as standin_quota_tracker

# Page configuration
st.set_page_config(
    page_title="The Relay — Research Paper Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #616161;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .citation-box {
        background-color: #e8f5e9;
        border-left: 4px solid #2e7d32;
        padding: 12px;
        margin-top: 10px;
        border-radius: 4px;
    }
    .refusal-box {
        background-color: #fff3e0;
        border-left: 4px solid #ef6c00;
        padding: 12px;
        margin-top: 10px;
        border-radius: 4px;
    }
    .answer-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e8f5e9 100%);
        border-left: 5px solid #1E88E5;
        border-radius: 8px;
        padding: 20px 24px;
        margin: 12px 0;
        font-size: 1.02rem;
        line-height: 1.7;
        color: #212121;
    }
    .answer-card p { margin-bottom: 0.6em; }
    .refusal-card {
        background: linear-gradient(135deg, #fff8e1 0%, #fff3e0 100%);
        border-left: 5px solid #ef6c00;
        border-radius: 8px;
        padding: 20px 24px;
        margin: 12px 0;
        font-size: 1.02rem;
        line-height: 1.7;
        color: #4e342e;
    }
    .cite-badge {
        display: inline-block;
        background: #e3f2fd;
        color: #1565c0;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }
    .cite-detail-row {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin-top: 6px;
    }
    .cite-detail-item {
        background: #f5f5f5;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 0.88rem;
    }
    .cite-detail-item b { color: #424242; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing Relay Pipeline and indexing research papers...")
def initialize_engine():
    """Extracts papers, caches text, and builds unified multi-provider pipeline once."""
    extracted = extract_all_papers()
    paper_pages = {
        p.paper_id: {page.page_number: page.text for page in p.pages}
        for p in extracted
    }
    chunks = chunk_papers(extracted, chunk_size_tokens=600, overlap_tokens=90)
    standin = StandinProvider()
    groq = GroqProvider()
    pipeline = RelayPipeline(
        chunks=chunks,
        paper_pages_text=paper_pages,
        standin_provider=standin,
        groq_provider=groq,
    )
    return extracted, paper_pages, pipeline


# Load cached engine resources
try:
    extracted_papers, paper_pages_text, shared_pipeline = initialize_engine()
except Exception as e:
    st.error(f"Error loading papers: {e}. Please ensure papers are in data/papers/.")
    st.stop()


def clean_answer_text(raw: str) -> str:
    """Remove inline [Citation: ...] markers and format math for clean display."""
    # Strip citation markers (displayed separately below)
    cleaned = re.sub(
        r'\s*\[(?:Citation:\s*)?[^\]]+arXiv:[^\]]+\]\s*\.?',
        '',
        raw,
    )
    # Collapse multiple blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    # Convert any remaining raw math notation to Unicode
    cleaned = format_math_unicode(cleaned.strip())
    return cleaned

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.title("⚡ The Relay")

    st.divider()

    st.subheader("🛠️ Engine Configuration")

    provider_choice = st.radio(
        "Model Provider:",
        options=["Stand-in (Safe Simulator)", "Real Groq API"],
        index=0,
        help="Stand-in runs in-process without consuming real quota. Real Groq API uses your configured GROQ_API_KEY.",
    )
    provider_code = "groq" if "Real Groq" in provider_choice else "standin"

    if provider_code == "groq":
        st.info(f"Target: `{settings.groq_model}`\n\nKey: {'✅ Detected' if bool(settings.groq_api_key) else '❌ Missing GROQ_API_KEY'}")
    else:
        st.caption("Target: `standin-calibrated-lpu` (Safe in-process simulator)")

    retrieval_choice = st.selectbox(
        "Retrieval Setting:",
        options=[
            "Setting B (600 tok, top-5, rerank) [Recommended]",
            "Setting A (350 tok, top-3, no rerank)",
            "Setting C (900 tok, top-8, rerank)",
        ],
        index=0,
    )

    setting_code = "setting_b"
    chunk_size = 600
    top_k = 5
    use_rerank = True

    if "Setting A" in retrieval_choice:
        setting_code = "setting_a"
        chunk_size = 350
        top_k = 3
        use_rerank = False
    elif "Setting C" in retrieval_choice:
        setting_code = "setting_c"
        chunk_size = 900
        top_k = 8
        use_rerank = True



# -------------------------------------------------------------
# MAIN CONTENT AREA
# -------------------------------------------------------------
st.markdown('<div class="main-header">The Relay: Research Paper Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Strictly grounded question answering over three target arXiv papers with automated citation verification and latency telemetry.</div>',
    unsafe_allow_html=True,
)

# Sample query buttons — populate session state so the form picks them up
st.markdown("**Quick Example Prompts:**")
col1, col2, col3, col4 = st.columns(4)

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""

if col1.button("Attention Mechanism", use_container_width=True):
    st.session_state.pending_prompt = "What is Scaled Dot-Product Attention in the Transformer?"
if col2.button("Mimi Frame Rate", use_container_width=True):
    st.session_state.pending_prompt = "What is the frame rate of the Mimi neural audio codec?"
if col3.button("Dynamic Chunking", use_container_width=True):
    st.session_state.pending_prompt = "How does dynamic chunking reduce attention complexity?"
if col4.button("Refusal Test (Out of scope)", use_container_width=True):
    st.session_state.pending_prompt = "What is the capital of France?"

# Query Input Form
with st.form("query_form"):
    query_text = st.text_area(
        "Enter your engineering question:",
        value=st.session_state.pending_prompt,
        placeholder="e.g. State the formula for Scaled Dot-Product Attention, or ask about Mimi codec...",
        height=100,
    )
    submit_button = st.form_submit_button("⚡ Run Relay Pipeline", use_container_width=True)

if submit_button and query_text.strip():
    # Clear the pending prompt after submission
    st.session_state.pending_prompt = ""

    # Build chunks for the selected setting and configure retriever
    with st.spinner("Executing 6-Stage Pipeline (Validation → Intent → BM25 Retrieval → Evidence → Generation → Verification)..."):
        chunks = chunk_papers(
            extracted_papers,
            chunk_size_tokens=chunk_size,
            overlap_tokens=int(chunk_size * 0.15),
        )
        shared_pipeline.chunks = chunks
        shared_pipeline.retriever = DocumentRetriever(chunks)

        if provider_code == "standin":
            # Reset stand-in quota tracker so interactive testing starts fresh
            standin_quota_tracker.reset()

        req = RelayRequest(
            question=query_text.strip(),
            retrieval_setting=setting_code,
            provider=provider_code,
        )

        t_ui_start = time.perf_counter()
        resp = asyncio.run(shared_pipeline.execute(req))
        t_ui_end = time.perf_counter()

    # -------------------------------------------------------------
    # RESULTS DISPLAY
    # -------------------------------------------------------------
    st.divider()

    # ── Metrics Row ──
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        if resp.status == "ok":
            delta_label = "🚫 Refusal" if resp.is_refusal else "✅ Answered"
        else:
            delta_label = f"⚠️ {resp.status.upper()}"
        st.metric("Status", resp.status.upper(), delta=delta_label)
    with m2:
        provider_badge = f"{resp.provider.upper()}"
        mode_label = "Simulated" if resp.is_simulated else "Real API"
        st.metric("Provider", provider_badge, delta=mode_label)
    with m3:
        ttft_val = f"{resp.first_token_latency_ms:.0f} ms" if resp.first_token_latency_ms else "—"
        st.metric("TTFT", ttft_val)
    with m4:
        st.metric("Latency", f"{resp.latency_ms:.0f} ms")
    with m5:
        st.metric("Tokens", f"{resp.input_tokens} in / {resp.output_tokens} out")

    st.markdown("")

    # ── Answer Card ──
    if resp.status in ("error", "rate_limited", "timeout"):
        st.markdown(f"#### ⚠️ Pipeline {resp.status.replace('_', ' ').title()}")
        err_msg = resp.refusal_reason or "An error occurred while contacting the model provider."
        st.error(f"**Error Details:** {err_msg}")
    elif resp.is_refusal:
        st.markdown("#### 🚫 Pipeline Refusal")
        reason_text = resp.refusal_reason or "The question is outside the scope of the loaded research papers."
        st.markdown(
            f'<div class="refusal-card">'
            f'<p><strong>This question cannot be answered.</strong></p>'
            f'<p>{reason_text}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("#### 💡 Answer")
        display_text = clean_answer_text(resp.answer)
        # Render via st.markdown() so bold, bullets, blockquotes,
        # tables, and Unicode math all display correctly
        st.markdown(display_text)

    # ── Citations ──
    if resp.citations:
        st.markdown("#### 📑 Verified Citations")
        for i, c in enumerate(resp.citations, 1):
            with st.expander(
                f"📄 {c.paper_title}  —  Page {c.page_number}",
                expanded=(len(resp.citations) <= 3),
            ):
                st.markdown(
                    f'<div class="cite-detail-row">'
                    f'<div class="cite-detail-item"><b>arXiv:</b> {c.arxiv_id}{c.version}</div>'
                    f'<div class="cite-detail-item"><b>Page:</b> {c.page_number}</div>'
                    f'<div class="cite-detail-item"><b>Section:</b> {c.section or "General"}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if c.supporting_passage:
                    st.markdown(f"> {c.supporting_passage}")
    elif not resp.is_refusal:
        st.info("No explicit citations were extracted or verified for this answer.")

    # ── Stage Timings ──
    with st.expander("⏱️ Pipeline Stage Breakdown", expanded=False):
        t = resp.stage_timings
        st.markdown(
            f"""
| Stage | Duration |
|:------|-------:|
| 1. Request Validation | `{t.validation_ms:.2f} ms` |
| 2. Intent Analysis | `{t.intent_ms:.2f} ms` |
| 3. BM25 Retrieval | `{t.retrieval_ms:.2f} ms` |
| 4. Evidence Evaluation | `{t.evidence_eval_ms:.2f} ms` |
| 5. Queue Wait | `{t.queue_wait_ms:.2f} ms` |
| 6. Answer Generation | `{t.generation_ms:.2f} ms` |
| 7. Citation Verification | `{t.verification_ms:.2f} ms` |
| **Total Pipeline** | **`{t.total_ms:.2f} ms`** |
"""
        )

st.markdown("---")
st.caption("The Relay — Research Paper QA Pipeline")

