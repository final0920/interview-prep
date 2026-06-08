"""Production retriever wiring the hybrid pipeline behind the engine's interface.

``EvidenceRetriever`` is the concrete implementation of the duck-typed
``Retriever`` protocol the interview engine consumes (``.search(query, top_k)
-> list[RetrievalHit]``). It assembles the heavy parts the engine intentionally
stays decoupled from:

    evidence_units.jsonl  ->  EvidenceUnit corpus
    VectorStore           <-  loaded from ingest.vector_path, else built in-mem
    Embedder              <-  get_embedder(cfg) (BGE, hashing fallback offline)
    hybrid_search + rerank

OFFLINE-SAFE by construction: every failure path (missing evidence file, missing
index, embed/search error) degrades to an empty result rather than raising, so an
interview never crashes when the corpus is absent. This is what makes interviews
*grounded* in production — server/CLI construct one of these (guarded) instead of
passing ``retriever=None``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from coach.config import REPO_ROOT, get
from coach.retrieval.embed import Embedder, get_embedder
from coach.retrieval.hybrid import hybrid_search
from coach.retrieval.rerank import rerank
from coach.schemas import EvidenceUnit, RetrievalHit
from coach.storage.vector import VectorStore


def _resolve(path: str) -> Path:
    """Resolve a config path relative to the repo root unless already absolute."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_evidence(path: str | Path) -> list[EvidenceUnit]:
    """Load EvidenceUnits from a JSONL file; return [] if absent or unreadable."""
    p = _resolve(str(path))
    if not p.exists():
        return []
    units: list[EvidenceUnit] = []
    try:
        with p.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    units.append(EvidenceUnit.model_validate_json(line))
                except Exception:
                    continue
    except Exception:
        return []
    return units


class EvidenceRetriever:
    """Grounds interview questions in the candidate's real project evidence.

    Loads the evidence corpus + a populated VectorStore at construction; if the
    index is missing it is rebuilt in-memory from the corpus using the configured
    embedder (hashing fallback when no model/weights are available). When the
    corpus itself is absent the retriever is inert: ``.search`` returns ``[]``.

    The class never raises from ``__init__`` or ``.search`` so the engine can
    always run offline.
    """

    def __init__(
        self,
        cfg: dict,
        *,
        evidence: Optional[list[EvidenceUnit]] = None,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
    ) -> None:
        self.cfg = cfg or {}
        # corpus: explicit override (tests) or load from configured jsonl
        if evidence is not None:
            self.evidence = list(evidence)
        else:
            self.evidence = load_evidence(get(self.cfg, "ingest.evidence_path",
                                              "data/evidence_units.jsonl"))

        # embedder: explicit override (tests) or factory (offline-safe fallback)
        try:
            self.embedder: Embedder = embedder or get_embedder(self.cfg)
        except Exception:
            from coach.retrieval.embed import HashingEmbedder
            self.embedder = HashingEmbedder()

        # vector store: explicit override -> persisted index -> in-memory build
        self.store = store if store is not None else self._load_or_build_store()

    def _load_or_build_store(self) -> VectorStore:
        """Load a persisted index, else build one in-memory from the corpus."""
        vector_path = get(self.cfg, "ingest.vector_path", "data/vector_index")
        try:
            vp = _resolve(str(vector_path))
            if (vp / "vectors.npy").exists() and (vp / "meta.json").exists():
                store = VectorStore.load(vp)
                if len(store) > 0:
                    return store
        except Exception:
            pass
        return self._build_store()

    def _build_store(self) -> VectorStore:
        """Embed the in-memory corpus into a fresh VectorStore (best-effort)."""
        store = VectorStore()
        if not self.evidence:
            return store
        try:
            vecs = self.embedder.encode([e.text for e in self.evidence])
            store.add(
                [e.id for e in self.evidence],
                vecs,
                [{"ref": e.ref} for e in self.evidence],
            )
        except Exception:
            return VectorStore()
        return store

    def search(self, query: str, top_k: int = 8) -> list[RetrievalHit]:
        """Hybrid (BM25 + dense) search fused via RRF, then reranked.

        Returns up to ``top_k`` ``RetrievalHit``. Returns ``[]`` on an empty
        corpus or any failure — never raises (interviews must not crash offline).
        """
        if not self.evidence or not query:
            return []
        try:
            fused = hybrid_search(
                query,
                self.evidence,
                self.embedder,
                self.store,
                top_k=int(get(self.cfg, "retrieval.top_k", 20)),
                rrf_k=int(get(self.cfg, "retrieval.rrf_k", 60)),
            )
            return rerank(query, fused, self.cfg, top_k=top_k)
        except Exception:
            return []
