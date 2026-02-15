# PDF-RAG: Local PDF Question-Answering with Hybrid Retrieval

A local-first RAG (Retrieval-Augmented Generation) system for PDF files. Upload a PDF, ask questions, get answers with page-level citations — all running on your machine with no API keys or cloud services.

**Key features:**
- Hybrid retrieval (semantic search + BM25 keyword search + Reciprocal Rank Fusion)
- Page-level citations in every answer (`[Page 3]`)
- PDF caching via SHA-256 fingerprint — re-uploads are instant
- Built-in evaluation framework with per-question diagnostics
- Chat UI with conversation history via Streamlit

## Architecture

```
                              ┌─────────────────────────┐
                              │      Streamlit UI        │
                              │       (app.py)           │
                              └────────┬────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                   │
              ┌─────▼─────┐    ┌──────▼──────┐    ┌──────▼──────┐
              │  Ingestion │    │  Retrieval  │    │     LLM     │
              │ (ingest.py)│    │(retriever.py│    │   Answer    │
              └─────┬──────┘    └──────┬──────┘    └──────┬──────┘
                    │                  │                   │
         ┌──────────┤          ┌──────┴──────┐            │
         │          │          │             │            │
    ┌────▼───┐ ┌────▼────┐ ┌──▼───┐   ┌─────▼──┐  ┌─────▼─────┐
    │ PyMuPDF│ │ Llama   │ │Chroma│   │  BM25  │  │  Llama    │
    │ (PDF   │ │ 3.2 3B  │ │  DB  │   │(rank-  │  │  3.2 3B   │
    │ parse) │ │(metadata)│ │(vec) │   │ bm25)  │  │ (answer)  │
    └────────┘ └─────────┘ └──────┘   └────────┘  └───────────┘
```

### Ingestion Flow

```
PDF  →  PyMuPDF (page extraction)  →  LLM (title + keywords per page)
     →  Embed raw page text (nomic-embed-text)  →  Store in ChromaDB
```

Each page becomes a ChromaDB document with:
- **embedding** — vector derived from the raw page text (used for semantic search)
- **page_content** — the actual PDF page text (what gets embedded)
- **metadata** — `page_number`, `title`, `keywords` (carried along, not embedded)

### Retrieval Flow

```
User Query
    ├── Semantic Search (ChromaDB vector similarity on raw page text)
    └── BM25 Search (keyword matching on page text + LLM-generated keywords)
         │
         └── Reciprocal Rank Fusion (merge both ranked lists)
              │
              └── Top-K documents → LLM → Answer with [Page X] citations
```

## Prerequisites

You need two local services running before starting the app:

### 1. Ollama (embeddings)

```bash
# Install Ollama: https://ollama.com
ollama pull nomic-embed-text
# Verify it's running on port 11434:
curl http://127.0.0.1:11434/api/embeddings -d '{"model":"nomic-embed-text","prompt":"test"}'
```

### 2. llama.cpp server (LLM)

Run a Llama 3.2 3B model (or any GGUF model) via llama.cpp's OpenAI-compatible server on port 9736:

```bash
# Example with llama-server:
llama-server -m llama-3.2-3b.gguf --port 9736

# Verify:
curl http://127.0.0.1:9736/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
```

> Both endpoints are configurable in `config.py` if your setup differs.

## Setup

```bash
git clone <repo-url> && cd PDF-RAG

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Chat UI

```bash
streamlit run app.py
```

1. Upload a PDF in the sidebar (first upload triggers LLM processing — subsequent uploads of the same file are cached)
2. Ask questions in the chat input
3. Answers include `[Page X]` citations grounded in the PDF

### ChromaDB Viewer

Inspect the raw data stored in ChromaDB (documents, metadata, embeddings):

```bash
streamlit run db_viewer.py
```

### Monitoring

Watch the pipeline log in real time:

```bash
tail -f rag_pipeline.log
```

## Project Structure

```
PDF-RAG/
├── app.py                 # Streamlit chat UI (entry point)
├── config.py              # All tunable parameters (models, retrieval, logging)
├── prompts.py             # Prompt templates (ingestion + QA)
├── ingest.py              # PDF → raw text + LLM metadata → ChromaDB
├── retriever.py           # Hybrid retrieval (semantic + BM25 + RRF)
├── db_viewer.py           # ChromaDB data viewer (streamlit run db_viewer.py)
├── eval_retrieval.py      # Retrieval evaluation with full diagnostics
├── requirements.txt       # Pinned Python dependencies
├── input_data/            # Evaluation test sets
│   ├── Project_Aurora/    # 5-page technical spec + 20 eval questions
│   └── chunking_strategies_data/  # 84-page research paper + eval questions
├── docs/
│   └── PLAN.md            # Detailed implementation plan and design decisions
├── chroma_store/          # Auto-created: persistent ChromaDB data (gitignored)
└── rag_pipeline.log       # Auto-created: pipeline debug log (gitignored)
```

## Configuration

All tunable parameters are in `config.py`. Nothing is scattered across files.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_BASE_URL` | `http://127.0.0.1:9736/v1` | llama.cpp server endpoint |
| `LLM_MODEL_NAME` | `llama-3.2-3b` | Model name for chat completions |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `EMBEDDING_BASE_URL` | `http://127.0.0.1:11434` | Ollama server endpoint |
| `CHROMA_PERSIST_DIR` | `./chroma_store` | ChromaDB storage directory |
| `SEMANTIC_TOP_K` | `5` | Candidates from vector search |
| `BM25_TOP_K` | `7` | Candidates from BM25 keyword search |
| `FINAL_TOP_K` | `5` | Documents sent to LLM after RRF fusion |
| `RRF_K` | `5` | RRF smoothing constant: `score = 1/(k + rank)` |
| `LLM_RETRY_ATTEMPTS` | `2` | Retries per page if LLM output is empty |
| `MAX_CURRENT_PAGE_CHARS` | `None` | Truncation limit for current page (None = full) |
| `MAX_ADJACENT_PAGE_CHARS` | `None` | Truncation limit for adjacent pages (None = full) |

## Evaluation

The evaluation framework measures whether the hybrid retriever returns the correct pages for questions with known ground truth. It is a first-class part of the project, not an afterthought.

### Running an evaluation

```bash
# Auto-detect the single ChromaDB collection:
python eval_retrieval.py

# Specify a PDF (finds its collection by hash):
python eval_retrieval.py --pdf-path input_data/Project_Aurora/Project_Aurora_Technical_Spec.pdf

# Use a specific question set:
python eval_retrieval.py --questions input_data/Project_Aurora/eval_questions.json

# Override FINAL_TOP_K:
python eval_retrieval.py --top-k 4

# Redirect output to a report file:
python eval_retrieval.py --questions input_data/Project_Aurora/eval_questions.json \
  > input_data/Project_Aurora/eval_report.txt
```

### Metrics reported

| Metric | What it measures |
|--------|-----------------|
| **Hit Rate** | % of questions where at least one expected page appears in top-K |
| **Recall@K** | Fraction of expected pages found in top-K |
| **Precision@K** | Fraction of top-K results that are expected pages |
| **MRR** | Mean Reciprocal Rank of the first correct result |

The report also includes:
- Per-question breakdown: semantic ranks, BM25 ranks, RRF fusion table
- Token overlap analysis (which query terms matched which corpus tokens)
- Carrier analysis (did semantic, BM25, or both contribute each hit?)
- Near-miss detection (pages ranked K+1 or K+2)
- Per-category breakdown (single-page, multi-page, keyword-heavy, semantic)

### Adding a new test set

1. Create a directory under `input_data/` with the PDF file
2. Add an `eval_questions.json` — a JSON array of objects:

```json
[
  {
    "question": "What is the budget for the project?",
    "expected_pages": [1, 5],
    "category": "multi-page"
  }
]
```

3. Ingest the PDF via the Streamlit UI (or script)
4. Run: `python eval_retrieval.py --pdf-path <pdf> --questions <json>`

### Existing test sets

| Test set | PDF | Pages | Questions | Description |
|----------|-----|-------|-----------|-------------|
| `Project_Aurora/` | Project Aurora Technical Spec | 5 | 20 | Fictional technical spec — small, fast to evaluate |
| `chunking_strategies_data/` | Evaluating Chunking Strategies | 84 | 50+ | Research paper — tests retrieval at scale |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| UI | Streamlit |
| LLM | Llama 3.2 3B via llama.cpp (OpenAI-compatible API) |
| Embeddings | nomic-embed-text via Ollama |
| Vector Store | ChromaDB (persistent) |
| PDF Parsing | PyMuPDF (fitz) |
| Keyword Search | rank-bm25 (BM25Okapi) |
| LLM Integration | LangChain (langchain-openai, langchain-ollama) |

## License

This project is for educational and personal use.
