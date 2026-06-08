"""Reranker: BGE cross-encoder with identity passthrough fallback.

Public API:
    def rerank(query, hits, cfg, *, top_k) -> list[RetrievalHit]
"""
from __future__ import annotations

from coach.config import get
from coach.schemas import RetrievalHit


def rerank(
    query: str,
    hits: list[RetrievalHit],
    cfg: dict,
    *,
    top_k: int = 8,
) -> list[RetrievalHit]:
    """Rerank hits with BGE cross-encoder, falling back to identity (RRF order).

    When use_reranker=false in config, or the cross-encoder model is unavailable,
    the function returns the top_k hits from the input unchanged (identity passthrough).
    This preserves the RRF ranking which is already a good prior.

    Args:
        query:  query string
        hits:   candidate RetrievalHit list (pre-sorted by hybrid search)
        cfg:    merged config dict
        top_k:  number of results to return

    Returns list[RetrievalHit] length <= top_k, re-ranked by cross-encoder score
    (or original order when falling back).
    """
    if not hits:
        return []

    use_reranker = get(cfg, "embeddings.use_reranker", True)
    model_name = get(cfg, "embeddings.reranker", "BAAI/bge-reranker-v2-m3")

    if use_reranker:
        try:
            return _cross_encoder_rerank(query, hits, model_name, top_k)
        except Exception:
            pass  # fall through to identity

    # identity passthrough: keep RRF order, re-assign rank numbers
    return _identity(hits, top_k)


def _cross_encoder_rerank(
    query: str,
    hits: list[RetrievalHit],
    model_name: str,
    top_k: int,
) -> list[RetrievalHit]:
    """Score (query, passage) pairs with a cross-encoder, re-sort, return top_k."""
    from sentence_transformers import CrossEncoder  # type: ignore

    model = CrossEncoder(model_name)
    pairs = [[query, h.evidence.text] for h in hits]
    scores = model.predict(pairs)

    ranked = sorted(zip(hits, scores), key=lambda x: -x[1])
    out: list[RetrievalHit] = []
    for rank, (hit, score) in enumerate(ranked[:top_k], start=1):
        out.append(hit.model_copy(update={"score": float(score), "rank": rank,
                                          "retriever": "rerank"}))
    return out


def _identity(hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
    """Return top_k hits with updated rank numbers, retriever tag unchanged."""
    out: list[RetrievalHit] = []
    for rank, hit in enumerate(hits[:top_k], start=1):
        out.append(hit.model_copy(update={"rank": rank}))
    return out
