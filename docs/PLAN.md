# PDF RAG System -- Implementation Plan

## 1. Project File Structure

```
PDF-RAG/
├── requirements.txt
├── app.py                  # Streamlit UI (entry point: streamlit run app.py)
├── ingest.py               # Ingestion pipeline: PDF -> structured chunks -> ChromaDB + BM25
├── retriever.py            # Hybrid retrieval: semantic + BM25 + RRF re-ranking
├── prompts.py              # All prompt templates (ingestion + QA)
├── config.py               # Constants and configuration (model names, paths, chunk params)
└── chroma_store/           # Auto-created: persistent ChromaDB data directory
```

Five Python files plus a config file. That is the entire codebase. No `src/` directory, no packages, no `__init__.py`. Flat and simple.

---

## 2. `config.py` -- Central Configuration

This file holds all magic strings and tunable parameters in one place.

```python
# config.py
import os

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PERSIST_DIR = os.path.join(PROJECT_DIR, "chroma_store")

# Ollama models
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Retrieval
SEMANTIC_TOP_K = 5
BM25_TOP_K = 5
FINAL_TOP_K = 5          # after RRF fusion
RRF_K = 60               # constant for reciprocal rank fusion

# BM25 stopwords (see errata 12f)
BM25_STOPWORDS = frozenset({...})  # ~90 common English stopwords, defined in config.py

# Collection naming
CHROMA_COLLECTION_PREFIX = "pdf_rag_"
```

---

## 3. `prompts.py` -- Prompt Templates

Two prompt templates are needed.

### 3a. Ingestion Prompt (Page Processing) — v2

This prompt is sent to the LLM once per page during ingestion. It receives the current page text plus the previous and next page texts for context awareness. The LLM returns only a **title** and **keywords** — the raw page text is embedded directly (not the LLM output).

```python
# prompts.py

PAGE_PROCESSING_PROMPT = """\
You are a document analyst. Given the text of a PDF page along with its surrounding context, produce a title and keywords.

PREVIOUS PAGE TEXT:
{prev_page}

CURRENT PAGE TEXT (Page {page_number}):
{current_page}

NEXT PAGE TEXT:
{next_page}

Output EXACTLY in this format (no extra text before or after):

TITLE: <short descriptive title for this page's content>
KEYWORDS: <5-15 comma-separated keywords or key phrases>"""
```

Key design decisions:
- **v2 change:** The prompt was trimmed from 4 fields (title, context_meaning, summary, keywords) to 2 fields (title, keywords). Since raw page text is now embedded directly, the LLM-generated summary and context are no longer needed — they were wasted tokens.
- The previous/next page text gives the LLM context about document flow. For page 1, the previous page section says "N/A (first page)." For the last page, next page says "N/A (last page)."
- The output is plain text with labeled sections, not JSON. Parsing is done with simple regex in Python.
- Keywords are comma-separated for easy extraction and BM25 tokenization.

### 3b. RAG QA Prompt (Query Answering)

```python
RAG_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions about a PDF document. Use ONLY the provided context to answer. If the context does not contain enough information to answer, say so clearly.

IMPORTANT: Always cite page numbers when referencing information. Use the format [Page X] inline in your response.

If multiple pages are relevant, cite each one where the information appears."""),
    ("human", """Chat history:
{chat_history}

Context from the document:
{context}

Question: {question}

Answer the question using the context above. Cite page numbers using [Page X] format.""")
])
```

Key design decisions:
- Chat history is injected as formatted text rather than using LangChain's message-based memory. This is simpler and avoids the complexity of `RunnableWithMessageHistory` for a project this size. The chat history is a formatted string of the last N turns.
- Context is a concatenated block of the structured chunk texts, each prefixed with its page number.
- The system prompt explicitly instructs citation format.

---

## 4. `ingest.py` -- Ingestion Pipeline

This is the core of the system. Here is the detailed design.

### 4a. PDF Text Extraction

```python
# ingest.py
import pymupdf  # PyMuPDF (newer import name; fitz also works)
import hashlib
import os
import re
from typing import Optional
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
import chromadb
from config import *
from prompts import PAGE_PROCESSING_PROMPT


def extract_pages(pdf_path: str) -> list[str]:
    """Extract text from each page of a PDF. Returns list of page texts (0-indexed)."""
    doc = pymupdf.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        pages.append(text)
    doc.close()
    return pages
```

### 4b. PDF Fingerprinting (Cache Check)

To avoid reprocessing the same PDF, we compute a hash of the file content and use it as the ChromaDB collection name.

```python
def get_pdf_hash(pdf_path: str) -> str:
    """Compute SHA-256 hash of PDF file for cache identification."""
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]  # first 16 chars is sufficient


def get_collection_name(pdf_hash: str) -> str:
    """ChromaDB collection name from PDF hash."""
    return f"{CHROMA_COLLECTION_PREFIX}{pdf_hash}"


def collection_exists(pdf_hash: str) -> bool:
    """Check if a ChromaDB collection already exists for this PDF."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    existing = [c.name for c in client.list_collections()]
    return get_collection_name(pdf_hash) in existing
```

### 4c. LLM Page Processing

This is the slow step. Each page is sent to Llama 3.2 3B with adjacent context. A callback function reports progress to the Streamlit UI.

```python
def process_page_with_llm(
    llm: ChatOllama,
    page_texts: list[str],
    page_index: int,       # 0-based
    doc_title: str,
    total_pages: int,
) -> dict:
    """Process a single page through the LLM to get title + keywords.
    
    Returns dict with keys: title, keywords, page_number
    """
    page_number = page_index + 1  # 1-based for display
    
    # Build adjacent context
    prev_text = page_texts[page_index - 1] if page_index > 0 else "N/A -- this is the first page."
    prev_num = page_number - 1 if page_index > 0 else "N/A"
    next_text = page_texts[page_index + 1] if page_index < len(page_texts) - 1 else "N/A -- this is the last page."
    next_num = page_number + 1 if page_index < len(page_texts) - 1 else "N/A"
    
    # Invoke the LLM
    chain = PAGE_PROCESSING_PROMPT | llm
    response = chain.invoke({
        "page_number": page_number,
        "doc_title": doc_title,
        "total_pages": total_pages,
        "prev_page_num": prev_num,
        "prev_page_text": prev_text[:2000],      # truncate to avoid context overflow
        "current_page_text": page_texts[page_index][:3000],
        "next_page_num": next_num,
        "next_page_text": next_text[:2000],
    })
    
    # Parse the structured response
    return parse_llm_page_output(response.content, page_number)


def _parse_structured_output(text: str, page_number: int) -> dict:
    """Parse the LLM's structured output (title + keywords only) into a dict."""
    def _extract(label: str) -> str:
        pattern = rf"{label}:\s*(.+?)(?=\n(?:TITLE|KEYWORDS):|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    return {
        "title": _extract("TITLE"),
        "keywords": _extract("KEYWORDS"),
        "page_number": page_number,
    }
```

### 4d. Document Creation (v2 — embed raw text)

Each parsed page becomes a LangChain `Document`. The **raw page text** is `page_content` (and gets embedded). LLM-generated title and keywords are kept as metadata for BM25 and display.

```python
def create_document_from_processed_page(processed: dict, raw_text: str) -> Document:
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
```

ChromaDB stores three things per document:
- **embedding** — derived from `page_content` (raw text), used ONLY for semantic search
- **page_content** — the text that got embedded (raw PDF page text)
- **metadata** — `page_number`, `title`, `keywords` — carried along but NOT embedded or searched by ChromaDB

### 4e. ChromaDB Storage

```python
def store_documents(documents: list[Document], pdf_hash: str):
    """Store documents in ChromaDB with embeddings."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection_name = get_collection_name(pdf_hash)
    
    # Delete existing collection if present (for re-upload scenario)
    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add documents with embeddings
    for i, doc in enumerate(documents):
        embedding = embeddings.embed_query(doc.page_content)
        collection.add(
            ids=[f"page_{doc.metadata['page_number']}"],
            embeddings=[embedding],
            documents=[doc.page_content],
            metadatas=[{
                "page_number": doc.metadata["page_number"],
                "title": doc.metadata["title"],
                "keywords": doc.metadata["keywords"],
            }]
        )
```

Note: We use the raw ChromaDB client API rather than LangChain's `Chroma` wrapper. This gives us full control over collection naming, persistence, and the ability to check for existing collections. The LangChain `Chroma` wrapper adds complexity we do not need since we also need direct access for the retriever.

### 4f. Master Ingestion Function

```python
def ingest_pdf(
    pdf_path: str,
    doc_title: str,
    progress_callback=None,   # callable(current_page, total_pages, status_message)
) -> tuple[str, list[Document]]:
    """
    Full ingestion pipeline. Returns (pdf_hash, documents).
    
    progress_callback is called after each page to update the Streamlit UI.
    """
    pdf_hash = get_pdf_hash(pdf_path)
    
    # Check cache
    if collection_exists(pdf_hash):
        if progress_callback:
            progress_callback(0, 0, "PDF already processed. Loading from cache...")
        documents = load_cached_documents(pdf_hash)
        return pdf_hash, documents
    
    # Extract pages
    if progress_callback:
        progress_callback(0, 0, "Extracting text from PDF...")
    page_texts = extract_pages(pdf_path)
    total_pages = len(page_texts)
    
    # Initialize LLM
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,   # low temperature for consistent structured output
    )
    
    # Process each page
    documents = []
    for i in range(total_pages):
        if progress_callback:
            progress_callback(i + 1, total_pages, f"Processing page {i + 1}/{total_pages} with LLM...")
        
        processed = process_page_with_llm(llm, page_texts, i, doc_title, total_pages)
        doc = create_document_from_processed_page(processed, page_texts[i])
        documents.append(doc)
    
    # Store in ChromaDB
    if progress_callback:
        progress_callback(total_pages, total_pages, "Storing embeddings in ChromaDB...")
    store_documents(documents, pdf_hash)
    
    return pdf_hash, documents
```

### 4g. Loading Cached Documents

```python
def load_cached_documents(pdf_hash: str) -> list[Document]:
    """Load previously processed documents from ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(get_collection_name(pdf_hash))
    
    results = collection.get(include=["documents", "metadatas"])
    
    documents = []
    for doc_text, metadata in zip(results["documents"], results["metadatas"]):
        documents.append(Document(
            page_content=doc_text,
            metadata=metadata
        ))
    
    # Sort by page number
    documents.sort(key=lambda d: d.metadata["page_number"])
    return documents
```

---

## 5. `retriever.py` -- Hybrid Retrieval Pipeline

### 5a. Architecture

The retrieval pipeline has three stages:
1. **Semantic search**: Query the ChromaDB collection using embedding similarity.
2. **BM25 keyword search**: Query an in-memory BM25 index built from the keywords field of each document.
3. **Reciprocal Rank Fusion (RRF)**: Combine the two ranked lists into one final ranking.

The BM25 index is built in-memory from the `Document` objects each time a PDF is loaded (it is fast -- milliseconds for hundreds of pages). It is NOT persisted; it is reconstructed from the ChromaDB-stored documents.

### 5b. Implementation

```python
# retriever.py
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import chromadb
from config import *


class HybridRetriever:
    """Hybrid retriever combining ChromaDB semantic search with BM25 keyword search."""
    
    def __init__(self, pdf_hash: str, documents: list[Document]):
        self.pdf_hash = pdf_hash
        self.documents = documents  # all page documents, sorted by page number
        
        # Initialize ChromaDB client and collection
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.chroma_client.get_collection(get_collection_name(pdf_hash))
        
        # Initialize embeddings model
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        
        # Build BM25 index from keywords
        self._build_bm25_index()
    
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, split, and remove stopwords."""
        return [t for t in text.lower().split() if t not in config.BM25_STOPWORDS]

    def _build_bm25_index(self):
        """Build in-memory BM25 index from document keywords + page text."""
        # v2: page_content is raw text, no more raw_text metadata
        self.bm25_corpus = []
        for doc in self.documents:
            keywords = doc.metadata.get("keywords", "")
            tokens = []
            for kw in keywords.split(","):
                tokens.extend(self._tokenize(kw.strip()))
            # page_content is the actual PDF page text (v2)
            tokens.extend(self._tokenize(doc.page_content))
            self.bm25_corpus.append(tokens)

        self.bm25 = BM25Okapi(self.bm25_corpus)
    
    def semantic_search(self, query: str, top_k: int = SEMANTIC_TOP_K) -> list[tuple[Document, float]]:
        """Perform semantic search via ChromaDB. Returns list of (document, score) tuples."""
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(self.documents)),
            include=["documents", "metadatas", "distances"]
        )
        
        scored_docs = []
        for doc_text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # ChromaDB cosine distance: lower = more similar
            # Convert to similarity score (1 - distance for cosine)
            similarity = 1 - distance
            doc = Document(page_content=doc_text, metadata=metadata)
            scored_docs.append((doc, similarity))
        
        return scored_docs
    
    def bm25_search(self, query: str, top_k: int = BM25_TOP_K) -> list[tuple[Document, float]]:
        """Perform BM25 keyword search. Returns list of (document, score) tuples."""
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Pair documents with scores and sort
        scored_docs = list(zip(self.documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]
    
    def reciprocal_rank_fusion(
        self,
        semantic_results: list[tuple[Document, float]],
        bm25_results: list[tuple[Document, float]],
        k: int = RRF_K,
    ) -> list[Document]:
        """
        Combine two ranked lists using Reciprocal Rank Fusion.
        
        RRF score for document d = sum over all rankers R of: 1 / (k + rank_R(d))
        
        This is a placeholder re-ranker that can be swapped for a cross-encoder later.
        """
        # Build a mapping from page_number -> cumulative RRF score
        rrf_scores = {}       # page_number -> score
        doc_lookup = {}       # page_number -> Document
        
        # Score from semantic search ranking
        for rank, (doc, _) in enumerate(semantic_results):
            page_num = doc.metadata["page_number"]
            doc_lookup[page_num] = doc
            rrf_scores[page_num] = rrf_scores.get(page_num, 0) + 1.0 / (k + rank + 1)
        
        # Score from BM25 ranking
        for rank, (doc, _) in enumerate(bm25_results):
            page_num = doc.metadata["page_number"]
            doc_lookup[page_num] = doc
            rrf_scores[page_num] = rrf_scores.get(page_num, 0) + 1.0 / (k + rank + 1)
        
        # Sort by RRF score descending
        sorted_pages = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [doc_lookup[page_num] for page_num, _ in sorted_pages[:FINAL_TOP_K]]
    
    def retrieve(self, query: str) -> list[Document]:
        """Full hybrid retrieval pipeline. Returns top documents."""
        semantic_results = self.semantic_search(query)
        bm25_results = self.bm25_search(query)
        fused_results = self.reciprocal_rank_fusion(semantic_results, bm25_results)
        return fused_results
```

### 5c. Design Notes on the Re-ranker

The `reciprocal_rank_fusion` method is the placeholder re-ranker. To swap it for a proper cross-encoder later:
1. Add a cross-encoder model (e.g., `sentence-transformers/ms-marco-MiniLM-L-6-v2` via the `sentence-transformers` library, or a locally-hosted model through Ollama).
2. After RRF produces its candidate list, re-score each candidate against the query using the cross-encoder.
3. Re-sort by cross-encoder scores.
4. The method signature stays the same; only the internals change.

---

## 6. `app.py` -- Streamlit UI

### 6a. Layout Design

```
+---------------------------------------------------------------+
|                    PDF Document Chat                          |
+------------------+--------------------------------------------+
| SIDEBAR          |  MAIN AREA                                 |
|                  |                                            |
| [Upload PDF]     |  +--------------------------------------+ |
| File: report.pdf |  | Chat Messages                        | |
| Status: Ready    |  |                                      | |
| Pages: 42        |  | User: What is the main argument?     | |
|                  |  |                                      | |
| [Clear Chat]     |  | Assistant: The main argument         | |
| [Remove PDF]     |  | presented in the document is...      | |
|                  |  | [Page 3] [Page 7]                   | |
|                  |  |                                      | |
| Processing:      |  | User: Tell me more about...          | |
| ████████░░ 80%   |  |                                      | |
| Page 8/10...     |  | Assistant: ...                       | |
|                  |  +--------------------------------------+ |
|                  |  [Ask a question about the document...] |
+------------------+--------------------------------------------+
```

### 6b. Implementation

```python
# app.py
import streamlit as st
import tempfile
import os
from langchain_ollama import ChatOllama
from config import *
from ingest import ingest_pdf, load_cached_documents, collection_exists, get_pdf_hash
from retriever import HybridRetriever
from prompts import RAG_QA_PROMPT


def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "pdf_hash": None,
        "documents": None,
        "retriever": None,
        "chat_history": [],        # list of {"role": "user"|"assistant", "content": str}
        "pdf_name": None,
        "processing": False,
        "total_pages": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def format_chat_history(history: list[dict], max_turns: int = 5) -> str:
    """Format recent chat history as a string for the prompt."""
    recent = history[-(max_turns * 2):]  # last N exchanges
    if not recent:
        return "No previous conversation."
    
    formatted = []
    for msg in recent:
        role = "Human" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)


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
    """Run the RAG pipeline to answer a question."""
    retriever = st.session_state.retriever
    
    # Retrieve relevant documents
    relevant_docs = retriever.retrieve(question)
    
    # Format context and chat history
    context = format_context(relevant_docs)
    chat_history_str = format_chat_history(st.session_state.chat_history)
    
    # Query the LLM
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,
    )
    
    chain = RAG_QA_PROMPT | llm
    response = chain.invoke({
        "chat_history": chat_history_str,
        "context": context,
        "question": question,
    })
    
    return response.content


def main():
    st.set_page_config(page_title="PDF Document Chat", layout="wide")
    st.title("PDF Document Chat")
    init_session_state()
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Document")
        
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        
        if uploaded_file is not None and uploaded_file.name != st.session_state.pdf_name:
            # New file uploaded -- process it
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.chat_history = []
            st.session_state.processing = True
            
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Progress display
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(current, total, message):
                if total > 0:
                    progress_bar.progress(current / total)
                    st.session_state.total_pages = total
                status_text.text(message)
            
            # Run ingestion
            try:
                pdf_hash, documents = ingest_pdf(
                    tmp_path,
                    doc_title=uploaded_file.name,
                    progress_callback=progress_callback,
                )
                st.session_state.pdf_hash = pdf_hash
                st.session_state.documents = documents
                st.session_state.retriever = HybridRetriever(pdf_hash, documents)
                st.session_state.processing = False
                status_text.text(f"Ready! {len(documents)} pages processed.")
                progress_bar.progress(1.0)
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                st.session_state.processing = False
            finally:
                os.unlink(tmp_path)
        
        # Display current document info
        if st.session_state.pdf_name:
            st.info(f"Current: {st.session_state.pdf_name}")
            st.text(f"Pages: {st.session_state.total_pages}")
            
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
            
            if st.button("Remove PDF"):
                st.session_state.pdf_hash = None
                st.session_state.documents = None
                st.session_state.retriever = None
                st.session_state.chat_history = []
                st.session_state.pdf_name = None
                st.session_state.total_pages = 0
                st.rerun()
    
    # --- MAIN CHAT AREA ---
    if st.session_state.retriever is None:
        st.info("Upload a PDF document in the sidebar to begin.")
        return
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = answer_question(prompt)
            st.markdown(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
```

### 6c. Key Streamlit Design Decisions

**Why plain `session_state` for chat history instead of LangChain memory**: `ConversationBufferMemory` is deprecated. Its replacements (`RunnableWithMessageHistory`, LangGraph) add significant complexity for what is essentially a list of strings. Storing `chat_history` as a list of dicts in `st.session_state` is the simplest possible approach and gives us full control over formatting for the prompt.

**Why `progress_callback` instead of streaming**: LLM processing during ingestion is a page-by-page batch operation, not a streaming chat. A progress bar with page count is the right UX pattern. The `st.progress()` bar plus a text status shows exactly where we are.

**Why `tempfile` for the upload**: Streamlit's `file_uploader` gives us an in-memory `BytesIO` object. PyMuPDF requires a file path. We write to a temp file, process it, then delete it.

---

## 7. `requirements.txt`

```
streamlit>=1.30.0
langchain-core>=0.3.0
langchain-ollama>=0.3.0
langchain-community>=0.3.0
chromadb>=1.0.0
pymupdf>=1.24.0
rank-bm25>=0.2.2
```

Note: `langchain-community` is listed because `BM25Retriever` lives there if we later decide to use LangChain's built-in version. For this plan, we use `rank_bm25` directly for more control, but the dependency is useful to have available.

---

## 8. Data Flow Diagrams

### Ingestion Flow (v2)

```
PDF File
    |
    v
PyMuPDF (extract_pages)
    |
    v
List[str] -- raw text per page
    |
    v
For each page i:
    +--> Build context: page[i-1], page[i], page[i+1]
    +--> PAGE_PROCESSING_PROMPT.format(...)
    +--> LLM.invoke() -> parse -> dict{title, keywords}
    +--> create_document_from_processed_page():
    |        page_content = raw page text       (EMBEDDED)
    |        metadata     = {page_number, title, keywords}  (NOT embedded)
    |
    v
List[Document]
    |
    +---> embed_query(doc.page_content)  -- embeds the raw page text
    +---> ChromaDB.collection.add(embedding, page_content, metadata)
    |
    v
Persistent ChromaDB (chroma_store/ directory)
    ├── embedding    → vector of raw page text (semantic search)
    ├── page_content → raw page text (what got embedded)
    └── metadata     → {page_number, title, keywords} (along for the ride)
```

### Retrieval Flow (v2)

```
User Query
    |
    +-------+-------+
    |               |
    v               v
Semantic Search   BM25 Search
(ChromaDB vectors (in-memory rank_bm25
 over raw text)    over keywords + raw text)
    |               |
    v               v
Top K docs        Top K docs
(with scores)     (with scores)
    |               |
    +-------+-------+
            |
            v
    Reciprocal Rank Fusion
            |
            v
    Top K fused documents
            |
            v
    format_context():
      [Page X] title
      raw page text (from page_content)
            |
            v
    RAG_QA_PROMPT.format(context, chat_history, question)
            |
            v
    LLM.invoke()
            |
            v
    Response with [Page X] citations
```

---

## 9. Implementation Order

This is the sequence for building the system, designed so that each step is independently testable.

### Phase 1: Foundation (config + prompts)
1. Create `config.py` with all constants.
2. Create `prompts.py` with both prompt templates.
3. Verify Ollama is running with both models pulled (`ollama pull llama3.2:3b` and `ollama pull nomic-embed-text`).

### Phase 2: Ingestion Pipeline
4. Implement `extract_pages()` in `ingest.py`. Test with a sample PDF to verify page text extraction.
5. Implement `get_pdf_hash()` and `collection_exists()`.
6. Implement `process_page_with_llm()` and `parse_llm_page_output()`. Test on a single page first -- this is the most critical function to get right. Tune the prompt if the LLM output does not parse correctly.
7. Implement `create_document_from_processed_page()`.
8. Implement `store_documents()`. Test the full ingestion on a small (3-5 page) PDF.
9. Implement `load_cached_documents()`. Verify that re-running ingestion on the same PDF skips LLM processing and loads from cache.

### Phase 3: Retrieval Pipeline
10. Implement `HybridRetriever.__init__()` and `_build_bm25_index()`.
11. Implement `semantic_search()`. Test with known queries against the ingested PDF.
12. Implement `bm25_search()`. Test the same queries.
13. Implement `reciprocal_rank_fusion()`. Verify that combining both result sets produces a reasonable merged ranking.
14. Wire up `retrieve()` as the single entry point.

### Phase 4: Streamlit UI
15. Build the basic Streamlit skeleton: sidebar with file upload, main area with chat.
16. Wire up file upload to the ingestion pipeline with progress callback.
17. Wire up chat input to the retrieval pipeline and LLM.
18. Add chat history display and session state management.
19. Add "Clear Chat" and "Remove PDF" buttons.

### Phase 5: Polish
20. Handle edge cases: empty pages, very short PDFs (1-2 pages), PDFs with no extractable text.
21. Add error handling for Ollama connection failures.
22. Test the full end-to-end flow with a real-world PDF (20+ pages).

---

## 10. Potential Challenges and Mitigations

**Challenge 1: LLM output parsing failures.**
Llama 3.2 3B may not always produce perfectly formatted output. The `parse_llm_page_output()` function uses regex with `re.DOTALL` and provides fallback default values. If a section is missing, the page still gets a basic `"Page N"` title and empty fields rather than crashing. Consider adding a retry (up to 2 attempts) if the parse returns all empty fields.

**Challenge 2: Ingestion speed.**
Processing 50 pages with Llama 3.2 3B locally could take 10-30 minutes depending on hardware. The progress bar is essential. The caching mechanism (PDF hash check) ensures this is a one-time cost per unique PDF. Consider adding a "Cancel" button using Streamlit's `st.stop()` mechanism.

**Challenge 3: Context window limits on Llama 3.2 3B.**
The model has a default 8192 token context window. Three pages of text (prev + current + next) plus the system prompt could approach or exceed this. The truncation in `process_page_with_llm()` (2000 chars for adjacent pages, 3000 for current) is a safety measure. If pages are very dense, consider reducing these limits or using `num_ctx` parameter in ChatOllama to increase the window.

**Challenge 4: Streamlit re-run behavior.**
Streamlit re-runs the entire script on every interaction. The `session_state` design ensures that the retriever, documents, and chat history persist across re-runs. The `progress_callback` only fires during the initial upload processing, not on subsequent re-runs.

**Challenge 5: BM25 index not persisted.**
The BM25 index is rebuilt from ChromaDB-stored documents each time the app starts or a cached PDF is loaded. For a single PDF (max a few hundred pages), this takes milliseconds and is not a problem. If this ever becomes slow, the `rank_bm25` object could be pickled to disk alongside the ChromaDB store.

---

## 11. Future Extension Points

The design explicitly supports these future enhancements without structural changes:

- **Cross-encoder re-ranker**: Replace the `reciprocal_rank_fusion` method body. No interface changes needed.
- **Streaming responses**: Change `chain.invoke()` in `answer_question()` to `chain.stream()` and use `st.write_stream()`.
- **Multiple PDFs**: Change `session_state` to hold a list of PDF hashes and retrievers. The `HybridRetriever` could accept multiple collections.
- **Better chunking**: The 1-page-1-chunk strategy could be refined to split very long pages or merge very short ones. The `Document` creation step is the only place that changes.

---

### Critical Files for Implementation

- `/Users/abhi/Desktop/ALL DEV RELATED FOLDERS/AI/claude-projects/PDF-RAG/ingest.py` - Core ingestion pipeline: PDF extraction, LLM page processing, ChromaDB storage, and cache management. This is the most complex file and where most debugging will happen.
- `/Users/abhi/Desktop/ALL DEV RELATED FOLDERS/AI/claude-projects/PDF-RAG/retriever.py` - Hybrid retrieval: semantic search, BM25 search, and reciprocal rank fusion. This is the retrieval brain and the future swap point for a cross-encoder.
- `/Users/abhi/Desktop/ALL DEV RELATED FOLDERS/AI/claude-projects/PDF-RAG/app.py` - Streamlit UI: file upload, progress tracking, chat interface, session state. The user-facing entry point.
- `/Users/abhi/Desktop/ALL DEV RELATED FOLDERS/AI/claude-projects/PDF-RAG/prompts.py` - Both prompt templates (ingestion and QA). These will require iterative tuning based on Llama 3.2 3B's actual output quality.
- `/Users/abhi/Desktop/ALL DEV RELATED FOLDERS/AI/claude-projects/PDF-RAG/config.py` - Central configuration. All tunable parameters in one place, making experimentation easy.

---

## 12. Post-Implementation Changes (Errata)

The following changes were made after the original plan was implemented. They diverge from the descriptions in sections above.

### 12a. ~~Raw text included in QA context~~ (superseded by 12h — v2 embed raw)

**Superseded.** In v2 the raw page text IS `page_content`, so there is no separate `raw_text` metadata field. See 12h.

### 12b. Pipeline-wide logging

**Changed:** `config.py`, `ingest.py`, `retriever.py`, `app.py`.

Comprehensive logging was added using Python's standard `logging` module:
- `config.py` — `setup_logging()` configures a file handler (`rag_pipeline.log`, DEBUG level) and a console handler (INFO level). Runs automatically on import. Includes `log_preview()` helper for truncating long strings.
- `ingest.py` — Logs PDF extraction (page counts, char lengths), hashing, cache hits/misses, LLM prompts and raw responses (DEBUG), parsed results, embedding timings, ChromaDB storage operations, and full pipeline timing.
- `retriever.py` — Logs BM25 index construction, semantic search results with similarity scores, BM25 results with scores, RRF per-document contributions and fused scores with source labels, and final ranked output.
- `app.py` — Logs file uploads, QA pipeline start/end with timing, context and chat history sent to LLM (DEBUG), LLM response timing and content.

Monitor with: `tail -f rag_pipeline.log`

### 12c. Ingestion prompt cleanup

**Changed:** `prompts.py` (section 3a above).

Removed a stray `RAW_TEXT:` output instruction from `PAGE_PROCESSING_PROMPT`. The raw text is already captured directly by PyMuPDF during extraction — asking the LLM to reproduce it wasted tokens and the parser (`_parse_structured_output`) never extracted it.

### 12d. ChromaDB data viewer

**Added:** `db_viewer.py` (not in original plan).

A standalone Streamlit app (`streamlit run db_viewer.py`) for inspecting ChromaDB contents. Displays all collections, documents table, metadata table, embedding statistics and heatmap, and expandable raw data view.

### 12e. Retrieval evaluation framework

**Added:** `eval_retrieval.py`, `input_data/eval_questions.json` (not in original plan).

A standalone evaluation script that measures retrieval quality against a ground-truth set of 20 questions with expected pages and categories (single-page, multi-page, keyword-heavy, semantic). Produces a detailed report with per-question diagnostics (semantic ranks, BM25 ranks, RRF fusion table, token overlaps, near-miss analysis) and aggregate metrics (hit rate, recall@K, precision@K, MRR, per-category breakdown, carrier analysis).

Run with: `python eval_retrieval.py`

### 12f. BM25 stopword filtering

**Changed:** `config.py`, `retriever.py`, `eval_retrieval.py` (section 5b above).

Added `BM25_STOPWORDS` frozenset to `config.py` containing ~90 common English stopwords. Added `HybridRetriever._tokenize()` static method that lowercases, splits, and removes stopwords. Applied to both BM25 corpus construction and query tokenization. The eval script also uses `_tokenize()` for consistent diagnostics.

**Motivation:** Evaluation showed BM25 rankings were dominated by stopword matches (e.g., "the", "is", "and"). For Q4 ("What is the Guardian microservice and what does it do?"), Page 5 ranked first in BM25 (score 2.2040) purely from stopword overlap ("and", "is", "it", "the"), while the correct Page 4 was pushed to rank 3. The keyword-heavy question category had only 50% hit rate and MRR of 0.17 — far worse than all other categories. Stopword removal lets domain-specific terms drive BM25 scoring.

### 12g. ~~Raw page text indexed in BM25 corpus~~ (superseded by 12h — v2 embed raw)

**Superseded.** In v2, `page_content` IS the raw page text, so BM25 naturally indexes it. No separate `raw_text` metadata needed. See 12h.

### 12h. v2 — Embed raw page text (branch `v2_embed_raw`)

**Changed:** `ingest.py`, `retriever.py`, `app.py`, `db_viewer.py`, `CLAUDE.md`, `PLAN.md`.

Major architectural change to what gets embedded vs stored as metadata.

**v1 flow (broken at scale):**
```
page_content (EMBEDDED) = Title + Context + Summary + Keywords  (LLM-generated)
metadata["raw_text"]    = actual page text                      (NOT embedded)
metadata["title"]       = title
metadata["keywords"]    = keywords
```

**v2 flow:**
```
page_content (EMBEDDED) = actual page text                      (raw PDF text)
metadata["title"]       = title                                 (kept for display/logging)
metadata["keywords"]    = keywords                              (kept for BM25)
metadata["raw_text"]    = REMOVED (redundant — it's now page_content)
metadata["summary"]     = REMOVED (not stored)
```

**What changed in each file:**
- `prompts.py` — `PAGE_PROCESSING_PROMPT` trimmed from 4 fields (TITLE, CONTEXT_MEANING, SUMMARY, KEYWORDS) to 2 fields (TITLE, KEYWORDS). Context meaning and summary were wasted LLM output tokens since they are no longer stored or embedded.
- `ingest.py` — `_parse_structured_output()` updated to only parse TITLE and KEYWORDS. `create_document_from_processed_page()` now sets `page_content=raw_text` and only stores `page_number`, `title`, `keywords` in metadata. Skip-page fallback dict trimmed to match.
- `retriever.py` — `_build_bm25_index()` tokenizes `metadata["keywords"]` + `page_content` (raw text). No more `raw_text` or `summary` metadata references.
- `app.py` — `format_context()` shows `[Page X] title` header + `page_content` (raw text). No more `raw_text` or `summary` metadata references.
- `db_viewer.py` — Comment update only (cosmetic).

**Motivation:** In v1, semantic search operated on LLM-generated summaries which could paraphrase, omit, or hallucinate terms from the original text. Embedding the raw page text directly means semantic search matches against the actual source material. This also simplifies the data model — no redundant `raw_text` metadata field, no need to juggle which text blob goes where. The ingestion prompt was also trimmed to only produce the 2 fields we actually keep (title, keywords), saving LLM output tokens and latency per page.

**Breaking change:** Existing ChromaDB collections from v1 must be deleted and PDFs re-ingested. The old collections have the structured summary as `page_content` and raw text in `metadata["raw_text"]`, which is the inverse of v2.
