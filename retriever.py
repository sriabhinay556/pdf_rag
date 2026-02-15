"""
Hybrid retrieval: semantic search (ChromaDB) + BM25 keyword search + RRF re-ranking.

The ``HybridRetriever`` class is the single entry point.  Given a natural-language
query it returns the top-K most relevant ``Document`` objects from the ingested PDF.
"""

import logging
import time

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

import config
from ingest import _get_chroma_client

log = logging.getLogger("retriever")


class HybridRetriever:
    """Combines semantic (ChromaDB) and keyword (BM25) search with RRF re-ranking.

    Parameters
    ----------
    pdf_hash:
        The SHA-256 fingerprint used as the ChromaDB collection name.
    documents:
        All page ``Document`` objects for the PDF (sorted by page number).
    """

    def __init__(self, pdf_hash: str, documents: list[Document]) -> None:
        self.pdf_hash = pdf_hash
        self.documents = documents
        self.embeddings = config.get_embeddings()
        log.info(
            "HybridRetriever init — collection='%s', %d documents",
            pdf_hash,
            len(documents),
        )
        self._build_bm25_index()

    # ── BM25 index ────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, split, and remove stopwords."""
        return [t for t in text.lower().split() if t not in config.BM25_STOPWORDS]

    def _build_bm25_index(self) -> None:
        """Build an in-memory BM25 index from document keywords + page text."""
        t0 = time.perf_counter()
        self.bm25_corpus = []
        for doc in self.documents:
            keywords = doc.metadata.get("keywords", "")
            # Tokenize: split keywords on commas, then split each on whitespace
            tokens = []
            for kw in keywords.split(","):
                tokens.extend(self._tokenize(kw.strip()))
            # Include words from page content (actual PDF page text)
            tokens.extend(self._tokenize(doc.page_content))
            self.bm25_corpus.append(tokens)

        self.bm25 = BM25Okapi(self.bm25_corpus)
        elapsed = time.perf_counter() - t0
        total_tokens = sum(len(c) for c in self.bm25_corpus)
        log.info(
            "BM25 index built — %d docs, %d total tokens, %.3fs",
            len(self.bm25_corpus),
            total_tokens,
            elapsed,
        )
        for i, doc in enumerate(self.documents):
            page = doc.metadata.get("page_number", i)
            log.debug(
                "  BM25 corpus page %d — %d tokens",
                page,
                len(self.bm25_corpus[i]),
            )

    # ── Semantic search ───────────────────────────────────────────────

    def semantic_search(self, query: str) -> list[tuple[Document, float]]:
        """Query ChromaDB for semantically similar documents."""
        log.info("Semantic search — query: %s", config.log_preview(query, 200))

        t0 = time.perf_counter()
        query_embedding = self.embeddings.embed_query(query)
        embed_time = time.perf_counter() - t0
        log.debug(
            "Query embedded — %d dims, %.3fs",
            len(query_embedding),
            embed_time,
        )

        client = _get_chroma_client()
        collection = client.get_collection(name=self.pdf_hash)

        n_results = min(config.SEMANTIC_TOP_K, len(self.documents))
        t0 = time.perf_counter()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        query_time = time.perf_counter() - t0
        log.debug("ChromaDB query — n_results=%d, %.3fs", n_results, query_time)

        scored_docs = []
        for content, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            doc = Document(page_content=content, metadata=meta)
            # ChromaDB returns L2 distance by default; convert to similarity
            similarity = 1.0 / (1.0 + distance)
            scored_docs.append((doc, similarity))

        log.info(
            "Semantic results — %d docs returned:",
            len(scored_docs),
        )
        for rank, (doc, score) in enumerate(scored_docs, start=1):
            page = doc.metadata.get("page_number", "?")
            title = doc.metadata.get("title", "")
            log.info(
                "  rank %d: Page %-3s | similarity=%.5f | %s",
                rank,
                page,
                score,
                config.log_preview(title, 80),
            )

        return scored_docs

    # ── BM25 search ───────────────────────────────────────────────────

    def bm25_search(self, query: str) -> list[tuple[Document, float]]:
        """Score documents against the query using BM25."""
        query_tokens = self._tokenize(query)
        log.info(
            "BM25 search — query tokens: %s",
            query_tokens,
        )

        t0 = time.perf_counter()
        scores = self.bm25.get_scores(query_tokens)
        elapsed = time.perf_counter() - t0

        # Log all non-zero scores for transparency
        all_scored = [
            (doc.metadata.get("page_number", i), score)
            for i, (doc, score) in enumerate(zip(self.documents, scores))
            if score > 0
        ]
        all_scored.sort(key=lambda x: x[1], reverse=True)
        log.debug(
            "BM25 all non-zero scores (%.3fs): %s",
            elapsed,
            [(f"Page {p}", f"{s:.4f}") for p, s in all_scored],
        )

        scored_docs = []
        for doc, score in zip(self.documents, scores):
            if score > 0:
                scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_k = scored_docs[: config.BM25_TOP_K]

        log.info("BM25 results — %d non-zero, returning top %d:", len(scored_docs), len(top_k))
        for rank, (doc, score) in enumerate(top_k, start=1):
            page = doc.metadata.get("page_number", "?")
            title = doc.metadata.get("title", "")
            log.info(
                "  rank %d: Page %-3s | bm25=%.5f | %s",
                rank,
                page,
                score,
                config.log_preview(title, 80),
            )

        return top_k

    # ── Reciprocal Rank Fusion ────────────────────────────────────────

    def reciprocal_rank_fusion(
        self,
        semantic_results: list[tuple[Document, float]],
        bm25_results: list[tuple[Document, float]],
    ) -> list[Document]:
        """Merge two ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
        k = config.RRF_K
        doc_scores: dict[int, float] = {}  # page_number → fused score
        doc_map: dict[int, Document] = {}

        log.info("RRF fusion — k=%d, semantic=%d results, bm25=%d results",
                 k, len(semantic_results), len(bm25_results))

        for rank, (doc, orig_score) in enumerate(semantic_results, start=1):
            page = doc.metadata.get("page_number", 0)
            rrf_contrib = 1.0 / (k + rank)
            doc_scores[page] = doc_scores.get(page, 0) + rrf_contrib
            doc_map[page] = doc
            log.debug(
                "  RRF semantic rank %d → Page %d: 1/(%d+%d) = %.6f  (orig_similarity=%.5f)",
                rank, page, k, rank, rrf_contrib, orig_score,
            )

        for rank, (doc, orig_score) in enumerate(bm25_results, start=1):
            page = doc.metadata.get("page_number", 0)
            rrf_contrib = 1.0 / (k + rank)
            doc_scores[page] = doc_scores.get(page, 0) + rrf_contrib
            doc_map[page] = doc
            log.debug(
                "  RRF bm25    rank %d → Page %d: 1/(%d+%d) = %.6f  (orig_bm25=%.5f)",
                rank, page, k, rank, rrf_contrib, orig_score,
            )

        # Sort by fused score descending
        ranked_pages = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        log.info("RRF fused scores (all %d candidates):", len(ranked_pages))
        for page, score in ranked_pages:
            in_semantic = any(d.metadata.get("page_number") == page for d, _ in semantic_results)
            in_bm25 = any(d.metadata.get("page_number") == page for d, _ in bm25_results)
            source = []
            if in_semantic:
                source.append("semantic")
            if in_bm25:
                source.append("bm25")
            log.info("  Page %-3d | rrf=%.6f | sources: %s", page, score, "+".join(source))

        final = [doc_map[page] for page, _ in ranked_pages[: config.FINAL_TOP_K]]

        log.info(
            "RRF final top-%d: %s",
            config.FINAL_TOP_K,
            [f"Page {d.metadata.get('page_number', '?')}" for d in final],
        )
        return final

    # ── Main retrieve ─────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[Document]:
        """Run hybrid retrieval: semantic + BM25 → RRF → top-K."""
        log.info("-" * 60)
        log.info("RETRIEVAL START — query: %s", config.log_preview(query, 200))
        log.info("-" * 60)
        t0 = time.perf_counter()

        semantic_results = self.semantic_search(query)
        bm25_results = self.bm25_search(query)
        final = self.reciprocal_rank_fusion(semantic_results, bm25_results)

        elapsed = time.perf_counter() - t0
        log.info("RETRIEVAL COMPLETE — %d docs returned, %.3fs total", len(final), elapsed)
        return final
