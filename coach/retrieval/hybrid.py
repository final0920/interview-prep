"""Hybrid retrieval: BM25 sparse + dense vector, fused via RRF.

Public API:
    def rrf_fuse(rank_lists, k=60) -> list[tuple[str, float]]
    def hybrid_search(query, evidence, embedder, store, *, top_k) -> list[RetrievalHit]
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from rank_bm25 import BM25Okapi

from coach.retrieval.embed import tokenize
from coach.schemas import EvidenceUnit, RetrievalHit, SourceScope

if TYPE_CHECKING:
    from coach.retrieval.embed import Embedder
    from coach.storage.vector import VectorStore


def rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ordered id lists.

    score(id) = sum over lists of 1 / (k + rank), rank is 1-based.
    Returns [(id, score), ...] sorted descending by score, ties broken by id.
    """
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def hybrid_search(
    query: str,
    evidence: list[EvidenceUnit],
    embedder: "Embedder",
    store: "VectorStore",
    *,
    top_k: int = 20,
    rrf_k: int = 60,
) -> list[RetrievalHit]:
    """Sparse BM25 + dense vector search fused with RRF.

    Args:
        query:    natural-language query string
        evidence: list of EvidenceUnit to search over (same corpus as store)
        embedder: Embedder instance (HashingEmbedder in tests)
        store:    VectorStore already populated with the same evidence ids
        top_k:    final result count after fusion
        rrf_k:    RRF smoothing constant (default 60)

    Returns list[RetrievalHit] sorted by score descending, length <= top_k.
    """
    if not evidence:
        return []

    # --- sparse: BM25 over tokenised (symbol + text) ---
    corpus = [tokenize(f"{e.symbol}\n{e.text}") for e in evidence]
    bm25 = BM25Okapi(corpus)
    q_toks = tokenize(query)
    sparse_scores = bm25.get_scores(q_toks)
    # rank by score descending; include only positive-scoring docs
    sparse_ranked = [
        evidence[i].id
        for i in sorted(range(len(evidence)), key=lambda i: -sparse_scores[i])
        if sparse_scores[i] > 0.0
    ]

    # --- dense: cosine via VectorStore ---
    q_vec = embedder.encode_one(query)
    dense_hits = store.search(q_vec, k=min(top_k * 5, len(evidence)))
    dense_ranked = [doc_id for doc_id, _score, _meta in dense_hits]

    # --- RRF fusion ---
    fused = rrf_fuse([sparse_ranked, dense_ranked], k=rrf_k)

    # build a lookup so we can attach EvidenceUnit objects
    by_id = {e.id: e for e in evidence}

    hits: list[RetrievalHit] = []
    for rank, (doc_id, score) in enumerate(fused[:top_k], start=1):
        ev = by_id.get(doc_id)
        if ev is None:
            continue
        hits.append(RetrievalHit(
            evidence=ev,
            score=score,
            rank=rank,
            scope=SourceScope.private,
            retriever="rrf",
        ))
    return hits
