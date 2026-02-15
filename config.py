"""
Central configuration for the PDF-RAG pipeline.

All tunable parameters — model endpoints, retrieval hyperparameters, logging
settings, and ingestion knobs — live here.  No magic numbers elsewhere.

Importing this module automatically initialises logging (file + console).
"""

import logging
import os
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Logging ───────────────────────────────────────────────────────────
LOG_FILE = os.path.join(PROJECT_DIR, "rag_pipeline.log")
LOG_LEVEL = logging.DEBUG          # file handler level
LOG_CONSOLE_LEVEL = logging.INFO   # console handler level
LOG_PREVIEW_CHARS = 500            # max chars shown for long strings in logs


def setup_logging() -> None:
    """Configure root logger with file + console handlers.

    Idempotent — safe to call multiple times (e.g. on Streamlit re-runs).
    The file handler writes at DEBUG level to ``rag_pipeline.log``; the
    console handler writes at INFO level.
    """
    root = logging.getLogger()
    # Avoid adding duplicate handlers on Streamlit re-runs
    if any(getattr(h, "_rag_marker", False) for h in root.handlers):
        return

    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — verbose
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(fmt)
    fh._rag_marker = True  # type: ignore[attr-defined]
    root.addHandler(fh)

    # Console handler — concise
    ch = logging.StreamHandler()
    ch.setLevel(LOG_CONSOLE_LEVEL)
    ch.setFormatter(fmt)
    ch._rag_marker = True  # type: ignore[attr-defined]
    root.addHandler(ch)


def log_preview(text: str, max_chars: int = LOG_PREVIEW_CHARS) -> str:
    """Return a truncated preview of *text* suitable for log messages.

    If *text* exceeds *max_chars*, it is trimmed and a total-length note
    is appended (e.g. ``"... (2048 chars total)"``).
    """
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... ({len(text)} chars total)"


# Initialise logging on import so every module that does ``import config``
# automatically gets the handlers registered.
setup_logging()

# ── LLM (llama.cpp server, OpenAI-compatible) ────────────────────────
LLM_BASE_URL: str = "http://127.0.0.1:9736/v1"
LLM_MODEL_NAME: str = "llama-3.2-3b"


def get_llm(**kwargs: Any) -> ChatOpenAI:
    """Create a ChatOpenAI instance pointed at the local llama.cpp server.

    Any keyword arguments are forwarded to ``ChatOpenAI``.  The
    ``temperature`` keyword defaults to 0.1 if not provided.
    """
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key="not-needed",
        model=LLM_MODEL_NAME,
        temperature=kwargs.pop("temperature", 0.1),
        **kwargs,
    )


# ── Embeddings (Ollama) ──────────────────────────────────────────────
EMBEDDING_MODEL: str = "nomic-embed-text"
EMBEDDING_BASE_URL: str = "http://127.0.0.1:11434"


def get_embeddings() -> OllamaEmbeddings:
    """Create an OllamaEmbeddings instance for ``nomic-embed-text``."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=EMBEDDING_BASE_URL,
    )


# ── ChromaDB ─────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.path.join(PROJECT_DIR, "chroma_store")

# ── Retrieval parameters (tuned via eval_retrieval.py) ───────────────
SEMANTIC_TOP_K: int = 5       # candidates returned by ChromaDB vector search
BM25_TOP_K: int = 7           # candidates returned by BM25 keyword search
FINAL_TOP_K: int = 5          # documents sent to the LLM after RRF fusion
RRF_K: int = 5                # RRF smoothing constant: score = 1/(k + rank)

# Stopwords removed from BM25 corpus and queries to reduce noise.
# ~90 common English function words.  Removing these lets domain-specific
# terms drive BM25 scoring (see PLAN.md §12f for motivation).
BM25_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "and", "or", "but", "if", "then", "so", "because", "as", "of", "in",
    "on", "at", "to", "for", "with", "by", "from", "about", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "up", "down",
    "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "who", "whom", "which",
    "what", "when", "where", "how", "why",
    "not", "no", "nor", "very", "too", "also", "just", "more", "most",
    "much", "many", "each", "every", "all", "any", "some",
})

# ── Ingestion parameters ─────────────────────────────────────────────
MAX_CURRENT_PAGE_CHARS: int | None = None      # None = no truncation (full page)
MAX_ADJACENT_PAGE_CHARS: int | None = None     # None = no truncation (full page)
LLM_RETRY_ATTEMPTS: int = 2                    # retries per page if LLM output is empty
