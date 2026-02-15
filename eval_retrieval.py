#!/usr/bin/env python3
"""
Retrieval evaluation script for the PDF-RAG pipeline.

Measures whether the hybrid retriever (semantic + BM25 + RRF) returns
the correct pages for a set of eval questions with known ground truth.

Outputs a detailed report with per-question diagnostics (semantic ranks,
BM25 ranks, RRF fusion table, token overlaps, near-miss analysis) and
aggregate metrics (hit rate, recall@K, precision@K, MRR, per-category
breakdown, carrier analysis).

Usage::

    python eval_retrieval.py                                           # auto-detect collection
    python eval_retrieval.py --pdf-path input_data/Project_Aurora/Project_Aurora_Technical_Spec.pdf
    python eval_retrieval.py --top-k 4                                 # override FINAL_TOP_K
    python eval_retrieval.py --questions input_data/Project_Aurora/eval_questions.json

Adding a new test set:
    1. Create a directory under ``input_data/`` with the PDF and an ``eval_questions.json``.
    2. The JSON should be a list of objects with ``question``, ``expected_pages``, and ``category``.
    3. Run: ``python eval_retrieval.py --pdf-path <pdf> --questions <json>``
"""

import argparse
import json
import sys
from pathlib import Path

from langchain_core.documents import Document

import config
from ingest import (
    _get_chroma_client,
    collection_exists,
    get_pdf_hash,
    load_cached_documents,
)
from retriever import HybridRetriever

EVAL_QUESTIONS_PATH = Path(__file__).parent / "input_data" / "eval_questions.json"


# ── Helpers ──────────────────────────────────────────────────────────────


def load_eval_questions(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def detect_collection() -> str:
    """Auto-detect the ChromaDB collection (must be exactly one)."""
    client = _get_chroma_client()
    collections = [c.name for c in client.list_collections()]
    if len(collections) == 0:
        print("ERROR: No ChromaDB collections found. Ingest a PDF first.")
        sys.exit(1)
    if len(collections) == 1:
        return collections[0]
    print(f"Multiple collections found: {collections}")
    print("Use --pdf-path to specify which PDF's collection to evaluate.")
    sys.exit(1)


def get_page_number(doc: Document) -> int:
    return doc.metadata.get("page_number", 0)


def get_title(doc: Document) -> str:
    return doc.metadata.get("title", "")


# ── Metric calculators ──────────────────────────────────────────────────


def recall_at_k(retrieved_pages: list[int], expected_pages: list[int]) -> float:
    if not expected_pages:
        return 0.0
    hits = len(set(retrieved_pages) & set(expected_pages))
    return hits / len(expected_pages)


def precision_at_k(retrieved_pages: list[int], expected_pages: list[int]) -> float:
    if not retrieved_pages:
        return 0.0
    hits = len(set(retrieved_pages) & set(expected_pages))
    return hits / len(retrieved_pages)


def reciprocal_rank(retrieved_pages: list[int], expected_pages: list[int]) -> float:
    expected_set = set(expected_pages)
    for i, page in enumerate(retrieved_pages, start=1):
        if page in expected_set:
            return 1.0 / i
    return 0.0


def hit_at_k(retrieved_pages: list[int], expected_pages: list[int]) -> bool:
    return bool(set(retrieved_pages) & set(expected_pages))


# ── Diagnostic retrieval (calls individual stages) ──────────────────────


def run_diagnostic_retrieval(
    retriever: HybridRetriever,
    query: str,
    final_top_k: int,
) -> dict:
    """Run each retrieval stage individually and collect full diagnostics."""

    # -- Semantic search --
    query_embedding = retriever.embeddings.embed_query(query)
    client = _get_chroma_client()
    collection = client.get_collection(name=retriever.pdf_hash)

    n_results = min(config.SEMANTIC_TOP_K, len(retriever.documents))
    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    semantic_results = []
    for content, meta, distance in zip(
        raw_results["documents"][0],
        raw_results["metadatas"][0],
        raw_results["distances"][0],
    ):
        doc = Document(page_content=content, metadata=meta)
        similarity = 1.0 / (1.0 + distance)
        semantic_results.append((doc, similarity, distance))

    # -- BM25 search (get ALL scores, not just top-K) --
    query_tokens = HybridRetriever._tokenize(query)
    bm25_scores = retriever.bm25.get_scores(query_tokens)

    bm25_all = []
    for i, (doc, score) in enumerate(zip(retriever.documents, bm25_scores)):
        bm25_all.append((doc, score, i))

    bm25_nonzero = [(doc, score, idx) for doc, score, idx in bm25_all if score > 0]
    bm25_nonzero.sort(key=lambda x: x[1], reverse=True)
    bm25_top_k = bm25_nonzero[: config.BM25_TOP_K]

    # -- Token overlap analysis --
    token_overlaps = {}
    for doc, score, idx in bm25_nonzero:
        page = get_page_number(doc)
        corpus_tokens = set(retriever.bm25_corpus[idx])
        query_token_set = set(query_tokens)
        overlap = corpus_tokens & query_token_set
        token_overlaps[page] = overlap

    # -- RRF fusion (full table) --
    k = config.RRF_K
    rrf_table = {}  # page -> {semantic_rank, bm25_rank, semantic_score, bm25_score, fused}
    doc_map = {}

    for rank, (doc, sim, dist) in enumerate(semantic_results, start=1):
        page = get_page_number(doc)
        rrf_table[page] = {
            "semantic_rank": rank,
            "semantic_sim": sim,
            "semantic_dist": dist,
            "bm25_rank": None,
            "bm25_score": 0.0,
            "fused": 1.0 / (k + rank),
        }
        doc_map[page] = doc

    for rank, (doc, score, idx) in enumerate(bm25_top_k, start=1):
        page = get_page_number(doc)
        if page in rrf_table:
            rrf_table[page]["bm25_rank"] = rank
            rrf_table[page]["bm25_score"] = score
            rrf_table[page]["fused"] += 1.0 / (k + rank)
        else:
            rrf_table[page] = {
                "semantic_rank": None,
                "semantic_sim": 0.0,
                "semantic_dist": None,
                "bm25_rank": rank,
                "bm25_score": score,
                "fused": 1.0 / (k + rank),
            }
            doc_map[page] = doc

    # Sort by fused score
    ranked = sorted(rrf_table.items(), key=lambda x: x[1]["fused"], reverse=True)
    final_pages = [page for page, _ in ranked[:final_top_k]]
    final_docs = [doc_map[page] for page in final_pages]

    return {
        "query_embedding_dim": len(query_embedding),
        "query_tokens": query_tokens,
        "semantic_results": semantic_results,
        "bm25_all": bm25_all,
        "bm25_nonzero": bm25_nonzero,
        "bm25_top_k": bm25_top_k,
        "token_overlaps": token_overlaps,
        "rrf_table": rrf_table,
        "rrf_ranked": ranked,
        "final_pages": final_pages,
        "final_docs": final_docs,
    }


# ── Report printing ─────────────────────────────────────────────────────


def print_header(text: str, char: str = "=", width: int = 80):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_subheader(text: str, char: str = "-", width: int = 60):
    print(f"\n  {char * width}")
    print(f"  {text}")
    print(f"  {char * width}")


def print_config_snapshot(retriever: HybridRetriever):
    """Print retrieval config and corpus stats at the top of the report."""
    print_header("RETRIEVAL EVALUATION REPORT")

    print("\n  Config:")
    print(f"    SEMANTIC_TOP_K  = {config.SEMANTIC_TOP_K}")
    print(f"    BM25_TOP_K      = {config.BM25_TOP_K}")
    print(f"    FINAL_TOP_K     = {config.FINAL_TOP_K}")
    print(f"    RRF_K           = {config.RRF_K}")
    print(f"    Embedding model = {config.EMBEDDING_MODEL}")
    print(f"    Collection      = {retriever.pdf_hash}")
    print(f"    Total docs      = {len(retriever.documents)}")

    total_tokens = sum(len(c) for c in retriever.bm25_corpus)
    avg_tokens = total_tokens / len(retriever.bm25_corpus) if retriever.bm25_corpus else 0
    print(f"\n  BM25 corpus stats:")
    print(f"    Total tokens    = {total_tokens}")
    print(f"    Avg tokens/doc  = {avg_tokens:.0f}")
    for i, doc in enumerate(retriever.documents):
        page = get_page_number(doc)
        print(f"    Page {page}: {len(retriever.bm25_corpus[i])} tokens")


def print_question_report(
    idx: int,
    question: dict,
    diag: dict,
    final_top_k: int,
):
    """Print full diagnostic report for a single question."""
    q_text = question["question"]
    expected = question["expected_pages"]
    category = question.get("category", "")
    retrieved = diag["final_pages"]

    print_header(f"Q{idx + 1}: {q_text}", char="━")
    print(f"  Expected pages: {expected}   Category: {category}")

    # -- Semantic search results --
    print_subheader("Semantic Search Results")
    sims = [sim for _, sim, _ in diag["semantic_results"]]
    if len(sims) >= 2:
        spread = max(sims) - min(sims)
        print(f"  Score spread: {spread:.5f} (max={max(sims):.5f}, min={min(sims):.5f})")
    print(f"  Query embedding dim: {diag['query_embedding_dim']}")
    print()
    print(f"  {'Rank':<6} {'Page':<6} {'L2 Dist':<12} {'Similarity':<12} Title")
    print(f"  {'─'*6} {'─'*6} {'─'*12} {'─'*12} {'─'*40}")
    for rank, (doc, sim, dist) in enumerate(diag["semantic_results"], start=1):
        page = get_page_number(doc)
        title = get_title(doc)[:50]
        marker = " ✓" if page in expected else ""
        print(f"  {rank:<6} {page:<6} {dist:<12.5f} {sim:<12.5f} {title}{marker}")

    # -- BM25 search results --
    print_subheader("BM25 Search Results (all non-zero)")
    print(f"  Query tokens: {diag['query_tokens']}")
    print()
    print(f"  {'Rank':<6} {'Page':<6} {'BM25 Score':<12} Title")
    print(f"  {'─'*6} {'─'*6} {'─'*12} {'─'*40}")
    for rank, (doc, score, idx_) in enumerate(diag["bm25_nonzero"], start=1):
        page = get_page_number(doc)
        title = get_title(doc)[:50]
        top_marker = " [top-K]" if rank <= config.BM25_TOP_K else ""
        exp_marker = " ✓" if page in expected else ""
        print(f"  {rank:<6} {page:<6} {score:<12.4f} {title}{top_marker}{exp_marker}")

    # Token overlap
    if diag["token_overlaps"]:
        print(f"\n  Token overlap (query ∩ corpus):")
        for page, overlap in sorted(diag["token_overlaps"].items()):
            if overlap:
                print(f"    Page {page}: {sorted(overlap)}")

    # Flag expected pages that scored 0 in BM25
    bm25_pages_nonzero = {get_page_number(doc) for doc, _, _ in diag["bm25_nonzero"]}
    bm25_misses = [p for p in expected if p not in bm25_pages_nonzero]
    if bm25_misses:
        print(f"\n  ⚠ BM25 scored 0 for expected pages: {bm25_misses}")

    # -- RRF fusion table --
    print_subheader("RRF Fusion Table")
    print(f"  {'Page':<6} {'Sem Rank':<10} {'BM25 Rank':<10} {'Fused':<12} Source")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*12} {'─'*20}")
    for page, info in diag["rrf_ranked"]:
        sem_rank = str(info["semantic_rank"]) if info["semantic_rank"] else "—"
        bm25_rank = str(info["bm25_rank"]) if info["bm25_rank"] else "—"
        sources = []
        if info["semantic_rank"]:
            sources.append("semantic")
        if info["bm25_rank"]:
            sources.append("bm25")
        source_str = "+".join(sources)
        exp_marker = " ✓" if page in expected else ""
        print(f"  {page:<6} {sem_rank:<10} {bm25_rank:<10} {info['fused']:<12.6f} {source_str}{exp_marker}")

    # Flag single-source pages
    single_source = [
        (page, "semantic-only" if info["semantic_rank"] and not info["bm25_rank"] else "bm25-only")
        for page, info in diag["rrf_ranked"]
        if bool(info["semantic_rank"]) != bool(info["bm25_rank"])
    ]
    if single_source:
        print(f"\n  Single-source pages: {[(p, s) for p, s in single_source]}")

    # Check semantic/BM25 top-1 agreement
    sem_top1 = get_page_number(diag["semantic_results"][0][0]) if diag["semantic_results"] else None
    bm25_top1 = get_page_number(diag["bm25_top_k"][0][0]) if diag["bm25_top_k"] else None
    if sem_top1 and bm25_top1 and sem_top1 != bm25_top1:
        print(f"\n  ⚠ Semantic/BM25 top-1 disagree: semantic=Page {sem_top1}, BM25=Page {bm25_top1}")

    # -- Final top-K --
    print_subheader(f"Final Top-{final_top_k}")
    for rank, page in enumerate(retrieved, start=1):
        info = diag["rrf_table"][page]
        exp_marker = " ✓" if page in expected else " ✗"
        print(f"  {rank}. Page {page} (fused={info['fused']:.6f}){exp_marker}")

    # -- Verdict --
    is_hit = hit_at_k(retrieved, expected)
    recall = recall_at_k(retrieved, expected)
    prec = precision_at_k(retrieved, expected)
    rr = reciprocal_rank(retrieved, expected)
    found = sorted(set(retrieved) & set(expected))
    missed = sorted(set(expected) - set(retrieved))

    print_subheader("Verdict")
    print(f"  {'HIT' if is_hit else 'MISS'}  |  Recall={recall:.2f}  Precision={prec:.2f}  MRR={rr:.2f}")
    print(f"  Found: {found}   Missed: {missed}")

    # -- Failure analysis --
    if missed:
        print(f"\n  Failure analysis for missed pages:")
        all_rrf_pages = [page for page, _ in diag["rrf_ranked"]]
        for mp in missed:
            if mp in all_rrf_pages:
                actual_rank = all_rrf_pages.index(mp) + 1
                info = diag["rrf_table"][mp]
                near = " ← NEAR MISS" if actual_rank <= final_top_k + 2 else ""
                print(f"    Page {mp}: actual RRF rank={actual_rank}, fused={info['fused']:.6f}{near}")
            else:
                print(f"    Page {mp}: not in RRF candidates at all (absent from both semantic and BM25 top-K)")

    false_positives = sorted(set(retrieved) - set(expected))
    if false_positives:
        print(f"\n  False positives (retrieved but not expected):")
        for fp in false_positives:
            info = diag["rrf_table"][fp]
            sem_s = f"sim={info['semantic_sim']:.5f}" if info["semantic_rank"] else "not in semantic"
            bm25_s = f"bm25={info['bm25_score']:.4f}" if info["bm25_rank"] else "not in bm25"
            print(f"    Page {fp}: {sem_s}, {bm25_s}")

    # Build per-expected-page rank lookups for decomposition table
    sem_rank_map = {
        get_page_number(doc): rank
        for rank, (doc, _, _) in enumerate(diag["semantic_results"], start=1)
    }
    bm25_rank_map = {
        get_page_number(doc): rank
        for rank, (doc, _, _) in enumerate(diag["bm25_nonzero"], start=1)
    }
    rrf_rank_map = {
        page: rank
        for rank, (page, _) in enumerate(diag["rrf_ranked"], start=1)
    }

    return {
        "hit": is_hit,
        "recall": recall,
        "precision": prec,
        "mrr": rr,
        "found": found,
        "missed": missed,
        "false_positives": false_positives,
        "sem_top1": sem_top1,
        "bm25_top1": bm25_top1,
        "sem_rank_map": sem_rank_map,
        "bm25_rank_map": bm25_rank_map,
        "rrf_rank_map": rrf_rank_map,
        "diag": diag,
    }


def print_aggregate_report(
    results: list[dict],
    questions: list[dict],
    final_top_k: int,
):
    """Print aggregate metrics and diagnostics."""
    n = len(results)
    total_hits = sum(1 for r in results if r["hit"])
    avg_recall = sum(r["recall"] for r in results) / n
    avg_precision = sum(r["precision"] for r in results) / n
    avg_mrr = sum(r["mrr"] for r in results) / n
    hit_rate = total_hits / n

    print_header(f"AGGREGATE RESULTS ({n} questions, top-{final_top_k})")

    # Summary table
    print(f"\n  {'Metric':<20} {'Value':<10}")
    print(f"  {'─'*20} {'─'*10}")
    print(f"  {'Hit Rate':<20} {hit_rate:.2%}")
    print(f"  {'Avg Recall@K':<20} {avg_recall:.2%}")
    print(f"  {'Avg Precision@K':<20} {avg_precision:.2%}")
    print(f"  {'MRR':<20} {avg_mrr:.4f}")
    print(f"  {'Hits':<20} {total_hits}/{n}")
    print(f"  {'Misses':<20} {n - total_hits}/{n}")

    # Per-category breakdown
    categories = sorted(set(q.get("category", "unknown") for q in questions))
    if len(categories) > 1:
        print_subheader("Per-Category Breakdown")
        print(f"  {'Category':<16} {'N':>4} {'Hit%':>8} {'Recall':>8} {'Prec':>8} {'MRR':>8}")
        print(f"  {'─'*16} {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        for cat in categories:
            cat_indices = [i for i, q in enumerate(questions) if q.get("category", "unknown") == cat]
            cat_results = [results[i] for i in cat_indices]
            cat_n = len(cat_results)
            cat_hits = sum(1 for r in cat_results if r["hit"])
            cat_recall = sum(r["recall"] for r in cat_results) / cat_n
            cat_prec = sum(r["precision"] for r in cat_results) / cat_n
            cat_mrr = sum(r["mrr"] for r in cat_results) / cat_n
            print(f"  {cat:<16} {cat_n:>4} {cat_hits/cat_n:>8.0%} {cat_recall:>8.2%} {cat_prec:>8.2%} {cat_mrr:>8.4f}")

    # Semantic vs BM25 agreement
    agree_count = sum(1 for r in results if r["sem_top1"] == r["bm25_top1"])
    print_subheader("Aggregate Diagnostics")
    print(f"  Semantic/BM25 top-1 agreement: {agree_count}/{n} ({agree_count/n:.0%})")

    # Source contribution for correct retrievals
    sem_only = 0
    bm25_only = 0
    both = 0
    for r in results:
        for page in r["found"]:
            info = r["diag"]["rrf_table"].get(page, {})
            has_sem = info.get("semantic_rank") is not None
            has_bm25 = info.get("bm25_rank") is not None
            if has_sem and has_bm25:
                both += 1
            elif has_sem:
                sem_only += 1
            elif has_bm25:
                bm25_only += 1

    total_correct = sem_only + bm25_only + both
    if total_correct > 0:
        print(f"\n  Source contribution for correctly retrieved pages ({total_correct} total):")
        print(f"    Both sources:    {both} ({both/total_correct:.0%})")
        print(f"    Semantic only:   {sem_only} ({sem_only/total_correct:.0%})")
        print(f"    BM25 only:       {bm25_only} ({bm25_only/total_correct:.0%})")

    # Failure breakdown
    missed_questions = [r for r in results if not r["hit"]]
    if missed_questions:
        sem_miss = 0
        bm25_miss = 0
        rrf_issue = 0
        for r in missed_questions:
            for mp in r["missed"]:
                diag = r["diag"]
                in_sem = any(
                    get_page_number(doc) == mp
                    for doc, _, _ in diag["semantic_results"]
                )
                bm25_pages = {get_page_number(doc) for doc, _, _ in diag["bm25_nonzero"]}
                in_bm25 = mp in bm25_pages
                if not in_sem and not in_bm25:
                    sem_miss += 1
                    bm25_miss += 1
                elif not in_sem:
                    sem_miss += 1
                elif not in_bm25:
                    bm25_miss += 1
                else:
                    rrf_issue += 1

        print(f"\n  Failure breakdown (missed pages across {len(missed_questions)} failed questions):")
        print(f"    Semantic miss:   {sem_miss}")
        print(f"    BM25 miss:       {bm25_miss}")
        print(f"    RRF ranking:     {rrf_issue} (in both sources but ranked too low)")

    # Near-miss analysis
    near_misses = 0
    for r in results:
        for mp in r["missed"]:
            diag = r["diag"]
            all_rrf_pages = [page for page, _ in diag["rrf_ranked"]]
            if mp in all_rrf_pages:
                actual_rank = all_rrf_pages.index(mp) + 1
                if actual_rank <= final_top_k + 2:
                    near_misses += 1

    if near_misses > 0:
        print(f"\n  ⚠ Near misses (rank K+1 or K+2): {near_misses}")
        print(f"    → Consider trying FINAL_TOP_K = {final_top_k + 1}")

    # Retriever decomposition table
    print_subheader("Retriever Decomposition (rank of correct page)")
    print(f"  {'#':<4} {'Expected':>10} {'Sem Rank':>10} {'BM25 Rank':>10} {'RRF Rank':>10} {'Carrier':<16}")
    print(f"  {'─'*4} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*16}")

    sem_carried = 0
    bm25_carried = 0
    both_carried = 0
    neither_carried = 0

    for i, (r, q) in enumerate(zip(results, questions)):
        for ep in q["expected_pages"]:
            sem_r = r["sem_rank_map"].get(ep)
            bm25_r = r["bm25_rank_map"].get(ep)
            rrf_r = r["rrf_rank_map"].get(ep)

            sem_str = str(sem_r) if sem_r else "—"
            bm25_str = str(bm25_r) if bm25_r else "—"
            rrf_str = str(rrf_r) if rrf_r else "—"

            # Determine who carried: which source had it in top-3?
            sem_hit = sem_r is not None and sem_r <= final_top_k
            bm25_hit = bm25_r is not None and bm25_r <= final_top_k
            if sem_hit and bm25_hit:
                carrier = "both"
                both_carried += 1
            elif sem_hit:
                carrier = "semantic"
                sem_carried += 1
            elif bm25_hit:
                carrier = "bm25"
                bm25_carried += 1
            else:
                carrier = "neither"
                neither_carried += 1

            hit_marker = "✓" if rrf_r is not None and rrf_r <= final_top_k else "✗"
            print(
                f"  Q{i+1:<3} {f'Page {ep}':>10} {sem_str:>10} {bm25_str:>10} {rrf_str:>10} {carrier:<16} {hit_marker}"
            )

    total_pages = sem_carried + bm25_carried + both_carried + neither_carried
    print(f"\n  Carrier summary ({total_pages} expected pages across {n} questions):")
    print(f"    Semantic carried:  {sem_carried} ({sem_carried}/{total_pages})")
    print(f"    BM25 carried:      {bm25_carried} ({bm25_carried}/{total_pages})")
    print(f"    Both carried:      {both_carried} ({both_carried}/{total_pages})")
    print(f"    Neither (miss):    {neither_carried} ({neither_carried}/{total_pages})")

    if bm25_carried == 0 and both_carried == total_pages - neither_carried:
        print(f"\n  ⚠ BM25 never uniquely carried a page — semantic is doing all the work.")
        print(f"    Hybrid adds latency without improving recall. Consider semantic-only retrieval.")
    elif sem_carried == 0 and both_carried == total_pages - neither_carried:
        print(f"\n  ⚠ Semantic never uniquely carried a page — BM25 is doing all the work.")

    # Per-question summary table
    print_subheader("Per-Question Summary")
    print(f"  {'#':<4} {'Hit':>4} {'Recall':>8} {'Prec':>8} {'MRR':>8}  Retrieved → Expected")
    print(f"  {'─'*4} {'─'*4} {'─'*8} {'─'*8} {'─'*8}  {'─'*30}")
    for i, (r, q) in enumerate(zip(results, questions)):
        hit_str = "✓" if r["hit"] else "✗"
        print(
            f"  Q{i+1:<3} {hit_str:>4} {r['recall']:>8.2f} {r['precision']:>8.2f} {r['mrr']:>8.2f}"
            f"  {r['diag']['final_pages']} → {q['expected_pages']}"
        )


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=None,
        help="Path to PDF (used to find its ChromaDB collection by hash)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Override FINAL_TOP_K (default: {config.FINAL_TOP_K})",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help=f"Path to eval questions JSON (default: {EVAL_QUESTIONS_PATH})",
    )
    args = parser.parse_args()

    final_top_k = args.top_k if args.top_k else config.FINAL_TOP_K
    questions_path = Path(args.questions) if args.questions else EVAL_QUESTIONS_PATH

    # Load eval questions
    if not questions_path.exists():
        print(f"ERROR: Eval questions not found at {questions_path}")
        sys.exit(1)
    questions = load_eval_questions(questions_path)
    print(f"Loaded {len(questions)} eval questions from {questions_path}")

    # Find ChromaDB collection
    if args.pdf_path:
        pdf_hash = get_pdf_hash(args.pdf_path)
        if not collection_exists(pdf_hash):
            print(f"ERROR: No ChromaDB collection for hash '{pdf_hash}'. Ingest the PDF first.")
            sys.exit(1)
    else:
        pdf_hash = detect_collection()

    print(f"Using collection: {pdf_hash}")

    # Load cached documents
    documents = load_cached_documents(pdf_hash)
    print(f"Loaded {len(documents)} documents from ChromaDB")

    # Create retriever
    retriever = HybridRetriever(pdf_hash, documents)

    # Print config
    print_config_snapshot(retriever)

    # Run evaluation
    results = []
    for i, q in enumerate(questions):
        diag = run_diagnostic_retrieval(retriever, q["question"], final_top_k)
        result = print_question_report(i, q, diag, final_top_k)
        results.append(result)

    # Aggregate report
    print_aggregate_report(results, questions, final_top_k)

    print(f"\n{'=' * 80}")
    print("  Evaluation complete.")
    print(f"{'=' * 80}\n")

    # Exit with non-zero if any misses
    misses = sum(1 for r in results if not r["hit"])
    if misses:
        sys.exit(1)


if __name__ == "__main__":
    main()
