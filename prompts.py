"""
Prompt templates for the PDF-RAG pipeline.

PAGE_PROCESSING_PROMPT — used during ingestion to extract metadata per page.
RAG_QA_PROMPT           — used at query time to answer questions with citations.
"""

PAGE_PROCESSING_PROMPT: str = """\
You are a document analyst. Extract metadata from this PDF page.

PAGE {page_number} TEXT:
{current_page}

Respond EXACTLY in this format with no extra text:

TITLE: <concise title capturing this page's specific topic — avoid generic titles>
KEYWORDS: <10-20 comma-separated terms: include specific names, acronyms, technical terms, metrics, method names, and any unique identifiers found on this page — prefer exact terms from the text over paraphrases>"""

RAG_QA_PROMPT: str = """\
You are a helpful assistant answering questions about a PDF document.

Rules:
- Answer ONLY from the retrieved context below. Never use outside knowledge.
- Cite every claim with [Page X] inline, placed at the end of the sentence before the period.
- If multiple pages support a point, cite all: [Page 3][Page 7].
- If the context is insufficient, state what information is missing and which pages you reviewed.
- Be concise. Do not repeat the question or preamble with "Based on the context..."

CHAT HISTORY:
{chat_history}

RETRIEVED CONTEXT:
{context}

QUESTION: {question}

Answer:"""
