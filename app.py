"""
Streamlit chat UI for the PDF-RAG system.

Run with:  ``streamlit run app.py``

Upload a PDF in the sidebar → the ingestion pipeline extracts pages,
generates metadata via LLM, and embeds everything into ChromaDB.
Then ask questions in the chat — answers are grounded in the PDF with
page-level citations.
"""

import logging
import os
import tempfile
import time

import streamlit as st

import config  # NOTE: importing config triggers setup_logging()
from ingest import ingest_pdf
from prompts import RAG_QA_PROMPT
from retriever import HybridRetriever

log = logging.getLogger("app")

# ── Page config ───────────────────────────────────────────────────────

st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="wide")

# ── Session state defaults ────────────────────────────────────────────

for key, default in {
    "pdf_hash": None,
    "documents": None,
    "retriever": None,
    "chat_history": [],
    "pdf_name": None,
    "pdf_page_count": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helper functions ──────────────────────────────────────────────────

def format_chat_history(history: list[dict], max_turns: int = 5) -> str:
    """Format recent chat history as a string for the QA prompt."""
    recent = history[-(max_turns * 2) :]  # each turn = user + assistant
    if not recent:
        return "(No prior conversation)"
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def format_context(documents) -> str:
    """Format retrieved documents into a context string for the QA prompt."""
    parts = []
    for doc in documents:
        page = doc.metadata.get("page_number", "?")
        title = doc.metadata.get("title", "")
        header = f"[Page {page}]"
        if title:
            header += f" {title}"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str) -> str:
    """Retrieve context, build prompt, invoke LLM, return answer."""
    log.info("=" * 60)
    log.info("QA PIPELINE START — question: %s", config.log_preview(question, 300))
    log.info("=" * 60)
    qa_t0 = time.perf_counter()

    # ── Retrieval ──
    retriever = st.session_state.retriever
    docs = retriever.retrieve(question)

    # ── Context building ──
    context = format_context(docs)
    history = format_chat_history(st.session_state.chat_history)
    log.debug("Chat history sent to LLM:\n%s", config.log_preview(history, 500))
    log.debug("Context sent to LLM (%d chars):\n%s", len(context), config.log_preview(context, 1000))

    prompt = RAG_QA_PROMPT.format(
        chat_history=history,
        context=context,
        question=question,
    )
    log.debug("Full QA prompt (%d chars):\n%s", len(prompt), config.log_preview(prompt, 1500))

    # ── LLM call ──
    llm = config.get_llm(temperature=0.3)
    log.info("QA LLM call — model=%s, temperature=0.3", config.LLM_MODEL_NAME)
    t0 = time.perf_counter()
    response = llm.invoke(prompt)
    llm_time = time.perf_counter() - t0
    log.info("QA LLM response — %.2fs, %d chars", llm_time, len(response.content))
    log.debug("QA LLM response text:\n%s", config.log_preview(response.content, 1000))

    total_time = time.perf_counter() - qa_t0
    log.info("QA PIPELINE COMPLETE — %.2fs total", total_time)
    return response.content


def clear_pdf_state():
    """Reset all PDF-related session state."""
    st.session_state.pdf_hash = None
    st.session_state.documents = None
    st.session_state.retriever = None
    st.session_state.chat_history = []
    st.session_state.pdf_name = None
    st.session_state.pdf_page_count = None


# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📄 PDF RAG Chat")

    uploaded_file = st.file_uploader(
        "Upload a PDF", type=["pdf"], key="pdf_uploader"
    )

    if uploaded_file is not None and uploaded_file.name != st.session_state.pdf_name:
        log.info("New PDF uploaded via UI: %s", uploaded_file.name)
        # Save to temp file (PyMuPDF needs a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            progress_bar = st.progress(0, text="Starting ingestion...")
            status_text = st.empty()

            def progress_callback(current, total):
                progress_bar.progress(
                    current / total,
                    text=f"Processing page {current}/{total}...",
                )

            pdf_hash, documents = ingest_pdf(
                tmp_path,
                progress_callback=progress_callback,
            )

            progress_bar.progress(1.0, text="Done!")

            st.session_state.pdf_hash = pdf_hash
            st.session_state.documents = documents
            st.session_state.retriever = HybridRetriever(pdf_hash, documents)
            st.session_state.chat_history = []
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.pdf_page_count = len(documents)

            log.info("PDF ready — hash=%s, pages=%d", pdf_hash, len(documents))
            status_text.success(f"Loaded {len(documents)} pages.")

        except Exception as e:
            log.exception("Ingestion failed for %s", uploaded_file.name)
            st.error(f"Ingestion failed: {e}")

        finally:
            os.unlink(tmp_path)

    # File info
    if st.session_state.pdf_name:
        st.divider()
        st.markdown(f"**File:** {st.session_state.pdf_name}")
        st.markdown(f"**Pages:** {st.session_state.pdf_page_count}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("Remove PDF", use_container_width=True):
                clear_pdf_state()
                st.rerun()

# ── Main area ─────────────────────────────────────────────────────────

if st.session_state.pdf_name is None:
    st.info("Upload a PDF in the sidebar to get started.")
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if question := st.chat_input("Ask a question about the PDF..."):
        # Show user message
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = answer_question(question)
                except Exception as e:
                    answer = f"Error generating answer: {e}"
            st.markdown(answer)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
