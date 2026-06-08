"""Tests for coach/retrieval/{embed,hybrid,rerank,geodesic}.

All tests are offline: no model downloads, no LLM calls, no network.
HashingEmbedder is the primary embedder under test; BGEEmbedder's fallback
path is also exercised by blocking sentence_transformers import.

VectorStore is stubbed inline (T2 not yet landed) -- once coach.storage.vector
exists, the stub can be replaced with the real import.
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from coach.schemas import Channel, EvidenceUnit, RetrievalHit, SourceScope

from coach.storage.vector import VectorStore as _StubVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unit(id: str, text: str, tags: list[str] | None = None) -> EvidenceUnit:
    return EvidenceUnit(
        id=id,
        source_path=f"src/{id}.py",
        text=text,
        channel=Channel.code,
        lang="python",
        tags=tags or [],
    )


def _make_store(units: list[EvidenceUnit], embedder) -> _StubVectorStore:
    store = _StubVectorStore()
    if units:
        texts = [u.text for u in units]
        vecs = embedder.encode(texts)
        store.add([u.id for u in units], vecs, [{} for _ in units])
    return store


# ---------------------------------------------------------------------------
# embed.py: tokenizer
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_snake(self):
        from coach.retrieval.embed import tokenize
        assert "thread" in tokenize("thread_pool_executor")
        assert "pool" in tokenize("thread_pool_executor")

    def test_camel(self):
        from coach.retrieval.embed import tokenize
        toks = tokenize("connectionTimeout")
        assert "connection" in toks
        assert "timeout" in toks

    def test_chinese_bigram(self):
        from coach.retrieval.embed import tokenize
        toks = tokenize("线程池")
        assert "线程" in toks
        assert "程池" in toks

    def test_empty(self):
        from coach.retrieval.embed import tokenize
        assert tokenize("") == []


# ---------------------------------------------------------------------------
# embed.py: HashingEmbedder determinism
# ---------------------------------------------------------------------------

class TestHashingEmbedder:
    def test_shape(self):
        from coach.retrieval.embed import HashingEmbedder
        emb = HashingEmbedder(dim=64)
        out = emb.encode(["hello world", "thread pool"])
        assert out.shape == (2, 64)
        assert out.dtype == np.float32

    def test_deterministic(self):
        from coach.retrieval.embed import HashingEmbedder
        emb = HashingEmbedder(dim=128)
        a = emb.encode(["transformer attention"])
        b = emb.encode(["transformer attention"])
        np.testing.assert_array_equal(a, b)

    def test_l2_normalised(self):
        from coach.retrieval.embed import HashingEmbedder
        emb = HashingEmbedder(dim=64)
        vecs = emb.encode(["foo bar baz"])
        norms = np.linalg.norm(vecs, axis=1)
        # rows should be unit-length (or zero if all tokens zero -- not expected here)
        assert abs(float(norms[0]) - 1.0) < 1e-5

    def test_encode_one(self):
        from coach.retrieval.embed import HashingEmbedder
        emb = HashingEmbedder(dim=64)
        v = emb.encode_one("hello")
        assert v.shape == (64,)

    def test_similar_texts_closer_than_random(self):
        from coach.retrieval.embed import HashingEmbedder
        emb = HashingEmbedder(dim=256)
        a = emb.encode_one("thread pool executor concurrency")
        b = emb.encode_one("thread pool worker concurrency")
        c = emb.encode_one("database index btree query optimisation")
        sim_ab = float(a @ b)
        sim_ac = float(a @ c)
        assert sim_ab > sim_ac, "related texts should be more similar than unrelated"

    def test_idf_weighting(self):
        from coach.retrieval.embed import HashingEmbedder
        idf = {"rare": 5.0, "common": 0.1}
        emb = HashingEmbedder(dim=64, idf=idf)
        out = emb.encode(["rare common"])
        assert out.shape == (2 - 1, 64)  # 1 text -> shape (1, 64)

    def test_zero_vector_no_crash(self):
        # empty string -> all-zero raw vec -> normalise returns zeros, no division error
        from coach.retrieval.embed import HashingEmbedder
        emb = HashingEmbedder(dim=32)
        out = emb.encode([""])
        assert out.shape == (1, 32)


# ---------------------------------------------------------------------------
# embed.py: BGEEmbedder fallback path (triggered by injecting an encode error)
# ---------------------------------------------------------------------------

class TestBGEEmbedderFallback:
    def test_falls_back_to_hashing_when_load_fails(self, monkeypatch):
        """When _load sets _fallback (simulating no weights), encode still works."""
        from coach.retrieval.embed import BGEEmbedder, HashingEmbedder

        emb = BGEEmbedder(model_name="BAAI/bge-m3")

        # Inject the fallback directly -- mirrors what _load does on ImportError
        emb._fallback = HashingEmbedder(dim=64)
        emb.name = emb._fallback.name
        emb.dim = emb._fallback.dim

        out = emb.encode(["hello world"])
        assert out.shape == (1, 64)
        assert out.dtype == np.float32
        assert emb.name == "hashing-fallback"

    def test_get_embedder_returns_embedder_instance(self):
        """get_embedder() returns an Embedder subclass (no model load attempted)."""
        from coach.retrieval.embed import Embedder, get_embedder

        emb = get_embedder({"embeddings": {"model": "BAAI/bge-m3", "device": "cpu"}})
        assert isinstance(emb, Embedder)


# ---------------------------------------------------------------------------
# hybrid.py: rrf_fuse math
# ---------------------------------------------------------------------------

class TestRRFFuse:
    def test_single_list(self):
        from coach.retrieval.hybrid import rrf_fuse
        result = rrf_fuse([["a", "b", "c"]], k=60)
        ids = [r[0] for r in result]
        assert ids == ["a", "b", "c"]
        # scores descend
        scores = [r[1] for r in result]
        assert scores[0] > scores[1] > scores[2]

    def test_two_lists_boost_shared(self):
        from coach.retrieval.hybrid import rrf_fuse
        # "b" appears in both lists at rank 1 -> should outscore "a" (only list1 rank1)
        result = rrf_fuse([["a", "b"], ["b", "c"]], k=60)
        by_id = {r[0]: r[1] for r in result}
        assert by_id["b"] > by_id["a"]
        assert by_id["b"] > by_id["c"]

    def test_known_math(self):
        from coach.retrieval.hybrid import rrf_fuse
        # single list ["x"], k=60 -> score = 1/(60+1)
        result = rrf_fuse([["x"]], k=60)
        expected = 1.0 / 61.0
        assert abs(result[0][1] - expected) < 1e-9

    def test_empty_lists_ignored(self):
        from coach.retrieval.hybrid import rrf_fuse
        result = rrf_fuse([[], ["a"]], k=60)
        assert result[0][0] == "a"

    def test_tie_broken_by_id(self):
        from coach.retrieval.hybrid import rrf_fuse
        # "a" and "b" each appear once at rank 1 in separate lists -> equal score
        result = rrf_fuse([["a"], ["b"]], k=60)
        assert result[0][0] == "a"  # "a" < "b" lexicographically
        assert result[1][0] == "b"

    def test_three_lists(self):
        from coach.retrieval.hybrid import rrf_fuse
        # doc "x" ranks 1 in all three -> score = 3/(60+1)
        result = rrf_fuse([["x"], ["x"], ["x"]], k=60)
        assert abs(result[0][1] - 3.0 / 61.0) < 1e-9


# ---------------------------------------------------------------------------
# hybrid.py: hybrid_search end-to-end
# ---------------------------------------------------------------------------

class TestHybridSearch:
    def _setup(self):
        from coach.retrieval.embed import HashingEmbedder
        from coach.retrieval.hybrid import hybrid_search

        units = [
            _make_unit("u1", "thread pool executor concurrent tasks worker"),
            _make_unit("u2", "database index btree query optimisation"),
            _make_unit("u3", "kafka consumer message queue reliable delivery"),
            _make_unit("u4", "transformer attention self attention mechanism"),
        ]
        emb = HashingEmbedder(dim=256)
        store = _make_store(units, emb)
        return hybrid_search, units, emb, store

    def test_returns_retrieval_hits(self):
        hybrid_search, units, emb, store = self._setup()
        hits = hybrid_search("thread pool", units, emb, store, top_k=3)
        assert isinstance(hits, list)
        assert all(isinstance(h, RetrievalHit) for h in hits)

    def test_top_result_relevant(self):
        hybrid_search, units, emb, store = self._setup()
        hits = hybrid_search("thread pool executor", units, emb, store, top_k=4)
        top_id = hits[0].evidence.id
        assert top_id == "u1"

    def test_top_k_respected(self):
        hybrid_search, units, emb, store = self._setup()
        hits = hybrid_search("database index", units, emb, store, top_k=2)
        assert len(hits) <= 2

    def test_ranks_assigned(self):
        hybrid_search, units, emb, store = self._setup()
        hits = hybrid_search("kafka consumer", units, emb, store, top_k=4)
        for i, h in enumerate(hits, start=1):
            assert h.rank == i

    def test_scope_and_retriever(self):
        hybrid_search, units, emb, store = self._setup()
        hits = hybrid_search("attention", units, emb, store, top_k=2)
        for h in hits:
            assert h.scope == SourceScope.private
            assert h.retriever == "rrf"

    def test_empty_evidence(self):
        from coach.retrieval.hybrid import hybrid_search
        from coach.retrieval.embed import HashingEmbedder
        store = _StubVectorStore()
        hits = hybrid_search("anything", [], HashingEmbedder(), store, top_k=5)
        assert hits == []

    def test_scores_descend(self):
        hybrid_search, units, emb, store = self._setup()
        hits = hybrid_search("thread pool", units, emb, store, top_k=4)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# rerank.py: identity passthrough when use_reranker=false
# ---------------------------------------------------------------------------

class TestRerank:
    def _hits(self, n=4) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                evidence=_make_unit(f"u{i}", f"text number {i}"),
                score=1.0 / (i + 1),
                rank=i + 1,
                retriever="rrf",
            )
            for i in range(n)
        ]

    def test_identity_when_disabled(self):
        from coach.retrieval.rerank import rerank
        cfg = {"embeddings": {"use_reranker": False}}
        hits = self._hits(4)
        out = rerank("query", hits, cfg, top_k=3)
        assert len(out) <= 3
        # order preserved (identity passthrough)
        assert [h.evidence.id for h in out] == ["u0", "u1", "u2"]

    def test_top_k_truncates(self):
        from coach.retrieval.rerank import rerank
        cfg = {"embeddings": {"use_reranker": False}}
        out = rerank("q", self._hits(4), cfg, top_k=2)
        assert len(out) == 2

    def test_rank_numbers_reassigned(self):
        from coach.retrieval.rerank import rerank
        cfg = {"embeddings": {"use_reranker": False}}
        out = rerank("q", self._hits(3), cfg, top_k=3)
        assert [h.rank for h in out] == [1, 2, 3]

    def test_empty_input(self):
        from coach.retrieval.rerank import rerank
        cfg = {"embeddings": {"use_reranker": False}}
        assert rerank("q", [], cfg, top_k=5) == []

    def test_fallback_when_cross_encoder_unavailable(self, monkeypatch):
        """Even with use_reranker=True, if _cross_encoder_rerank raises we fall back."""
        from coach.retrieval import rerank as rerank_mod

        def _raise(*args, **kwargs):
            raise RuntimeError("no model available")

        monkeypatch.setattr(rerank_mod, "_cross_encoder_rerank", _raise)
        cfg = {"embeddings": {"use_reranker": True,
                              "reranker": "BAAI/bge-reranker-v2-m3"}}
        hits = self._hits(3)
        out = rerank_mod.rerank("q", hits, cfg, top_k=3)
        # fell back to identity; no crash
        assert len(out) == 3
        assert [h.rank for h in out] == [1, 2, 3]


# ---------------------------------------------------------------------------
# geodesic.py: co-occurrence build + rerank
# ---------------------------------------------------------------------------

class TestGeodesic:
    def _units_with_tags(self) -> list[EvidenceUnit]:
        return [
            _make_unit("u1", "thread pool", tags=["concurrency", "thread", "pool"]),
            _make_unit("u2", "kafka consumer", tags=["kafka", "queue", "thread"]),
            _make_unit("u3", "btree index", tags=["database", "index", "btree"]),
            _make_unit("u4", "redis cache", tags=["cache", "redis", "database"]),
        ]

    def test_build_creates_table(self, tmp_path):
        from coach.retrieval.geodesic import build_cooccurrence
        import sqlite3

        db = tmp_path / "test.db"
        build_cooccurrence(self._units_with_tags(), db)
        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT * FROM tag_pair_similarity").fetchall()
        conn.close()
        assert len(rows) > 0

    def test_cooccurrence_symmetric(self, tmp_path):
        from coach.retrieval.geodesic import build_cooccurrence
        import sqlite3

        db = tmp_path / "test.db"
        build_cooccurrence(self._units_with_tags(), db)
        conn = sqlite3.connect(str(db))
        # thread+pool co-occur in u1; both directions must exist
        row_ab = conn.execute(
            "SELECT similarity FROM tag_pair_similarity WHERE tag_a=? AND tag_b=?",
            ("thread", "pool"),
        ).fetchone()
        row_ba = conn.execute(
            "SELECT similarity FROM tag_pair_similarity WHERE tag_a=? AND tag_b=?",
            ("pool", "thread"),
        ).fetchone()
        conn.close()
        assert row_ab is not None
        assert row_ba is not None
        assert abs(row_ab[0] - row_ba[0]) < 1e-9

    def test_geodesic_rerank_boosts_matching_tags(self, tmp_path):
        from coach.retrieval.geodesic import build_cooccurrence, geodesic_rerank

        units = self._units_with_tags()
        db = tmp_path / "test.db"
        build_cooccurrence(units, db)

        # query tags relate to concurrency -- u1/u2 should outscore u3/u4
        hits = [
            RetrievalHit(evidence=u, score=0.5, rank=i + 1, retriever="rrf")
            for i, u in enumerate(units)
        ]
        out = geodesic_rerank(["concurrency", "thread"], hits, db)
        ids = [h.evidence.id for h in out]
        # u1 (has concurrency+thread) or u2 (has thread) should rank ahead of u3/u4
        assert ids[0] in {"u1", "u2"}

    def test_geodesic_rerank_no_db(self, tmp_path):
        from coach.retrieval.geodesic import geodesic_rerank

        units = self._units_with_tags()
        hits = [
            RetrievalHit(evidence=u, score=0.5, rank=i + 1, retriever="rrf")
            for i, u in enumerate(units)
        ]
        missing_db = tmp_path / "nonexistent.db"
        # should degrade gracefully, just renumber with geodesic retriever tag
        out = geodesic_rerank(["concurrency"], hits, missing_db)
        assert len(out) == len(hits)
        assert all(h.retriever == "geodesic" for h in out)

    def test_geodesic_empty_query_tags(self, tmp_path):
        from coach.retrieval.geodesic import build_cooccurrence, geodesic_rerank

        db = tmp_path / "test.db"
        build_cooccurrence(self._units_with_tags(), db)
        hits = [
            RetrievalHit(evidence=_make_unit("x", "foo"), score=1.0, rank=1, retriever="rrf")
        ]
        out = geodesic_rerank([], hits, db)
        assert len(out) == 1

    def test_build_empty_evidence(self, tmp_path):
        from coach.retrieval.geodesic import build_cooccurrence
        import sqlite3

        db = tmp_path / "test.db"
        build_cooccurrence([], db)  # must not crash
        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT COUNT(*) FROM tag_pair_similarity").fetchone()
        conn.close()
        assert rows[0] == 0

    def test_ranks_contiguous(self, tmp_path):
        from coach.retrieval.geodesic import build_cooccurrence, geodesic_rerank

        units = self._units_with_tags()
        db = tmp_path / "test.db"
        build_cooccurrence(units, db)
        hits = [
            RetrievalHit(evidence=u, score=0.5, rank=i + 1, retriever="rrf")
            for i, u in enumerate(units)
        ]
        out = geodesic_rerank(["cache", "database"], hits, db)
        assert [h.rank for h in out] == list(range(1, len(out) + 1))
