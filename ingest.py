"""
PDF ingestion pipeline: extract → LLM metadata → embed → store in ChromaDB.

Flow:
    PDF file  →  PyMuPDF page extraction  →  LLM title/keyword extraction
    →  embed raw page text  →  persist in ChromaDB collection (keyed by PDF hash)

Re-uploading the same PDF is a cache hit (SHA-256 hash match) — skips LLM
processing and loads directly from ChromaDB.
"""

import hashlib
import logging
import re
import time
from typing import Callable

import chromadb
import fitz  # PyMuPDF
from langchain_core.documents import Document

import config
from prompts import PAGE_PROCESSING_PROMPT

log = logging.getLogger("ingest")


# ── PDF extraction ────────────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list[str]:
    """Extract text from each page of a PDF using PyMuPDF."""
    log.info("Extracting pages from %s", pdf_path)
    t0 = time.perf_counter()
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    elapsed = time.perf_counter() - t0
    log.info(
        "Extracted %d pages in %.2fs  (char counts: %s)",
        len(pages),
        elapsed,
        [len(p) for p in pages],
    )
    return pages


def get_pdf_hash(pdf_path: str) -> str:
    """Return first 16 chars of SHA-256 hash of the PDF file."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    digest = h.hexdigest()[:16]
    log.info("PDF hash: %s  (%s)", digest, pdf_path)
    return digest


# ── ChromaDB helpers ──────────────────────────────────────────────────

def _get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client (path from ``config.CHROMA_PERSIST_DIR``)."""
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


def collection_exists(pdf_hash: str) -> bool:
    """Check if a ChromaDB collection already exists for this PDF."""
    client = _get_chroma_client()
    existing = [c.name for c in client.list_collections()]
    found = pdf_hash in existing
    log.info(
        "Cache check for '%s': %s  (existing collections: %s)",
        pdf_hash,
        "HIT" if found else "MISS",
        existing,
    )
    return found


# ── LLM page processing ──────────────────────────────────────────────

def _truncate(text: str, max_chars: int | None) -> str:
    """Truncate *text* to *max_chars*.  Returns the full string if *max_chars* is ``None``."""
    if max_chars is None or not text:
        return text or ""
    return text[:max_chars]


def process_page_with_llm(
    llm: "ChatOpenAI", page_texts: list[str], page_index: int
) -> dict[str, str | int]:
    """Send a page (with adjacent-page context) to the LLM for metadata extraction.

    Returns a dict with keys ``title``, ``keywords``, and ``page_number``.
    Retries up to ``config.LLM_RETRY_ATTEMPTS`` times if the LLM returns
    empty fields.
    """
    page_number = page_index + 1
    total = len(page_texts)
    current = _truncate(page_texts[page_index], config.MAX_CURRENT_PAGE_CHARS)

    prev = (
        _truncate(page_texts[page_index - 1], config.MAX_ADJACENT_PAGE_CHARS)
        if page_index > 0
        else "N/A (first page)"
    )
    nxt = (
        _truncate(page_texts[page_index + 1], config.MAX_ADJACENT_PAGE_CHARS)
        if page_index < total - 1
        else "N/A (last page)"
    )

    prompt = PAGE_PROCESSING_PROMPT.format(
        prev_page=prev,
        current_page=current,
        next_page=nxt,
        page_number=page_number,
    )

    log.info("LLM ingestion request — page %d/%d", page_number, total)
    log.debug(
        "LLM prompt (page %d):\n%s", page_number, config.log_preview(prompt, 1000)
    )

    for attempt in range(config.LLM_RETRY_ATTEMPTS):
        t0 = time.perf_counter()
        response = llm.invoke(prompt)
        elapsed = time.perf_counter() - t0

        log.debug(
            "LLM raw response (page %d, attempt %d, %.2fs):\n%s",
            page_number,
            attempt + 1,
            elapsed,
            config.log_preview(response.content, 1000),
        )

        parsed = _parse_structured_output(response.content, page_number)

        if any(parsed[k] for k in ("title", "keywords")):
            log.info(
                "LLM page %d parsed OK (attempt %d, %.2fs) — title: %s",
                page_number,
                attempt + 1,
                elapsed,
                config.log_preview(parsed["title"], 100),
            )
            return parsed

        log.warning(
            "LLM page %d attempt %d — all fields empty, retrying",
            page_number,
            attempt + 1,
        )

    log.error("LLM page %d — all %d attempts produced empty output", page_number, config.LLM_RETRY_ATTEMPTS)
    # Return whatever we got on last attempt
    return parsed


def _parse_structured_output(text: str, page_number: int) -> dict[str, str | int]:
    """Parse the LLM's ``TITLE: … / KEYWORDS: …`` output into a dict."""
    def _extract(label: str) -> str:
        pattern = rf"{label}:\s*(.+?)(?=\n(?:TITLE|KEYWORDS):|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    return {
        "title": _extract("TITLE"),
        "keywords": _extract("KEYWORDS"),
        "page_number": page_number,
    }


# ── Document creation & storage ──────────────────────────────────────

def create_document_from_processed_page(
    processed: dict, raw_text: str
) -> Document:
    """Build a LangChain Document from the structured LLM output.

    page_content (EMBEDDED) = actual page text from the PDF.
    metadata["title"]       = short title (kept for display / logging).
    metadata["keywords"]    = comma-separated keywords (kept for BM25).
    """
    return Document(
        page_content=raw_text,
        metadata={
            "page_number": processed["page_number"],
            "title": processed["title"],
            "keywords": processed["keywords"],
        },
    )


def store_documents(pdf_hash: str, documents: list[Document]) -> None:
    """Embed and store documents in a ChromaDB collection."""
    log.info("Storing %d documents in ChromaDB collection '%s'", len(documents), pdf_hash)
    client = _get_chroma_client()
    embeddings = config.get_embeddings()
    collection = client.get_or_create_collection(name=pdf_hash)

    for i, doc in enumerate(documents):
        page = doc.metadata.get("page_number", i)
        t0 = time.perf_counter()
        embedding = embeddings.embed_query(doc.page_content)
        embed_time = time.perf_counter() - t0
        log.debug(
            "Embedding page %d — %d dims, %.3fs  (text: %d chars)",
            page,
            len(embedding),
            embed_time,
            len(doc.page_content),
        )

        doc_id = f"{pdf_hash}_page_{i}"
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc.page_content],
            metadatas=[doc.metadata],
        )
        log.debug("ChromaDB ADD id=%s  (page %d)", doc_id, page)

    log.info("All %d documents stored in ChromaDB", len(documents))


def load_cached_documents(pdf_hash: str) -> list[Document]:
    """Load documents from an existing ChromaDB collection."""
    log.info("Loading cached documents from collection '%s'", pdf_hash)
    t0 = time.perf_counter()
    client = _get_chroma_client()
    collection = client.get_collection(name=pdf_hash)
    results = collection.get(include=["documents", "metadatas"])

    documents = []
    for content, meta in zip(results["documents"], results["metadatas"]):
        documents.append(Document(page_content=content, metadata=meta))

    # Sort by page number
    documents.sort(key=lambda d: d.metadata.get("page_number", 0))
    elapsed = time.perf_counter() - t0
    log.info("Loaded %d cached documents in %.3fs", len(documents), elapsed)
    return documents


# ── Master ingestion function ────────────────────────────────────────

def ingest_pdf(
    pdf_path: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[str, list[Document]]:
    """Run the full ingestion pipeline: extract → LLM metadata → embed → store.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file on disk.
    progress_callback:
        Optional ``(current_page, total_pages) -> None`` callable invoked
        after each page is processed (used by the Streamlit progress bar).

    Returns
    -------
    tuple[str, list[Document]]
        ``(pdf_hash, documents)`` where *pdf_hash* is the SHA-256 fingerprint
        used as the ChromaDB collection name, and *documents* is the list of
        LangChain ``Document`` objects (one per page).
    """
    log.info("=" * 60)
    log.info("INGESTION START — %s", pdf_path)
    log.info("=" * 60)
    pipeline_t0 = time.perf_counter()

    pdf_hash = get_pdf_hash(pdf_path)

    # Cache hit — skip LLM processing
    if collection_exists(pdf_hash):
        documents = load_cached_documents(pdf_hash)
        log.info("INGESTION COMPLETE (cache hit) — %d docs, %.2fs",
                 len(documents), time.perf_counter() - pipeline_t0)
        return pdf_hash, documents

    page_texts = extract_pages(pdf_path)
    if not page_texts:
        log.error("PDF contains no extractable text: %s", pdf_path)
        raise ValueError("PDF contains no extractable text.")

    llm = config.get_llm()
    log.info("LLM initialised — model=%s, base_url=%s",
             config.LLM_MODEL_NAME, config.LLM_BASE_URL)
    documents = []

    for i, raw_text in enumerate(page_texts):
        # Skip pages with negligible text
        if len(raw_text.strip()) < 20:
            log.info("Page %d skipped (only %d chars of text)", i + 1, len(raw_text.strip()))
            processed = {
                "title": f"Page {i + 1} (minimal content)",
                "keywords": "",
                "page_number": i + 1,
            }
        else:
            processed = process_page_with_llm(llm, page_texts, i)

        doc = create_document_from_processed_page(processed, raw_text)
        documents.append(doc)

        if progress_callback:
            progress_callback(i + 1, len(page_texts))

    store_documents(pdf_hash, documents)
    elapsed = time.perf_counter() - pipeline_t0
    log.info("=" * 60)
    log.info("INGESTION COMPLETE — %d docs, %.2fs total", len(documents), elapsed)
    log.info("=" * 60)
    return pdf_hash, documents
