"""Geodesic reranking via tag co-occurrence graph (feature-flag: retrieval.geodesic).

Builds a tag-pair similarity table in SQLite and uses it to boost hits whose
tags are "close" to the query tags in the co-occurrence graph.

Public API:
    def build_cooccurrence(evidence, db_path) -> None
    def geodesic_rerank(query_tags, hits, db_path) -> list[RetrievalHit]
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from coach.schemas import EvidenceUnit, RetrievalHit

# DDL for the co-occurrence table
_DDL = """
CREATE TABLE IF NOT EXISTS tag_pair_similarity (
    tag_a TEXT NOT NULL,
    tag_b TEXT NOT NULL,
    similarity REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (tag_a, tag_b)
);
"""


def build_cooccurrence(evidence: list[EvidenceUnit], db_path: str | Path) -> None:
    """Compute tag pair co-occurrence counts and persist as similarity scores.

    Similarity(a, b) = co_doc_count(a, b) / sqrt(count(a) * count(b))
    (Dice-like normalisation so frequent tags don't dominate).
    Stores both (a, b) and (b, a) rows for symmetric lookup.

    Args:
        evidence: list of EvidenceUnit, each with a .tags list
        db_path:  path to SQLite database (created if absent)
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # count individual tag frequencies and pair co-occurrences
    tag_count: dict[str, int] = {}
    pair_count: dict[tuple[str, str], int] = {}

    for unit in evidence:
        tags = list(dict.fromkeys(unit.tags))  # deduplicate, preserve order
        for t in tags:
            tag_count[t] = tag_count.get(t, 0) + 1
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                key = (a, b) if a <= b else (b, a)
                pair_count[key] = pair_count.get(key, 0) + 1

    import math

    rows: list[tuple[str, str, float]] = []
    for (a, b), co in pair_count.items():
        ca = tag_count.get(a, 1)
        cb = tag_count.get(b, 1)
        sim = co / math.sqrt(ca * cb)
        rows.append((a, b, sim))
        rows.append((b, a, sim))  # symmetric

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_DDL)
        conn.execute("DELETE FROM tag_pair_similarity")
        conn.executemany(
            "INSERT OR REPLACE INTO tag_pair_similarity(tag_a, tag_b, similarity) VALUES(?,?,?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def geodesic_rerank(
    query_tags: list[str],
    hits: list[RetrievalHit],
    db_path: str | Path,
) -> list[RetrievalHit]:
    """Boost hit scores by tag proximity to query_tags, then re-sort.

    For each hit, compute relevance = max over query_tags of
    similarity(query_tag, hit_tag) (0.0 when no pair exists).
    Final score = original_score * (1 + relevance) so that hits with no
    tag overlap are not penalised relative to their RRF score.

    Returns list[RetrievalHit] sorted by boosted score descending, with
    updated rank numbers and retriever="geodesic".
    Degrades gracefully: if the db is absent or the table is empty,
    returns hits re-ranked by original score unchanged.
    """
    if not hits or not query_tags:
        return _renumber(hits, "geodesic")

    sim = _load_similarities(query_tags, hits, db_path)

    boosted: list[tuple[RetrievalHit, float]] = []
    for hit in hits:
        rel = _max_sim(query_tags, hit.evidence.tags, sim)
        boosted.append((hit, hit.score * (1.0 + rel)))

    boosted.sort(key=lambda x: -x[1])
    out: list[RetrievalHit] = []
    for rank, (hit, score) in enumerate(boosted, start=1):
        out.append(hit.model_copy(update={"score": score, "rank": rank,
                                          "retriever": "geodesic"}))
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_similarities(
    query_tags: list[str],
    hits: list[RetrievalHit],
    db_path: str | Path,
) -> dict[tuple[str, str], float]:
    """Load similarity rows relevant to (query_tags x hit_tags) from SQLite."""
    db_path = Path(db_path)
    if not db_path.exists():
        return {}

    hit_tags: set[str] = set()
    for h in hits:
        hit_tags.update(h.evidence.tags)

    if not hit_tags or not query_tags:
        return {}

    sim: dict[tuple[str, str], float] = {}
    try:
        conn = sqlite3.connect(str(db_path))
        placeholders_q = ",".join("?" * len(query_tags))
        placeholders_h = ",".join("?" * len(hit_tags))
        rows = conn.execute(
            f"SELECT tag_a, tag_b, similarity FROM tag_pair_similarity "
            f"WHERE tag_a IN ({placeholders_q}) AND tag_b IN ({placeholders_h})",
            list(query_tags) + list(hit_tags),
        ).fetchall()
        conn.close()
        for a, b, s in rows:
            sim[(a, b)] = s
    except Exception:
        pass
    return sim


def _max_sim(
    query_tags: list[str],
    hit_tags: list[str],
    sim: dict[tuple[str, str], float],
) -> float:
    """Maximum pairwise similarity between any query tag and any hit tag."""
    best = 0.0
    for qt in query_tags:
        for ht in hit_tags:
            s = sim.get((qt, ht), 0.0)
            if s > best:
                best = s
    return best


def _renumber(hits: list[RetrievalHit], retriever: str) -> list[RetrievalHit]:
    out: list[RetrievalHit] = []
    for rank, hit in enumerate(hits, start=1):
        out.append(hit.model_copy(update={"rank": rank, "retriever": retriever}))
    return out
