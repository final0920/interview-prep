"""Pure-numpy in-process vector store.

Persisted as two files in a directory:
  vectors.npy   - float32 matrix, shape (n, dim)
  meta.json     - {"ids": [...], "metas": [...]}

Cosine similarity is computed via normalized dot product (vectors are stored
L2-normalized at insertion time).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class VectorStore:
    """In-memory float32 vector store backed by numpy, persisted to disk."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._metas: list[dict] = []
        # shape (n, dim) or (0,) when empty
        self._vecs: np.ndarray = np.empty((0,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, ids: list[str], vecs: np.ndarray, metas: list[dict]) -> None:
        """Append *ids* / *vecs* / *metas* to the store.

        *vecs* must be shape (len(ids), dim).  Each vector is L2-normalized
        before storage so cosine similarity reduces to a dot product.
        """
        if len(ids) == 0:
            return
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[np.newaxis, :]
        # L2-normalize rows; leave zero vectors as-is to avoid NaN
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = vecs / norms

        if self._vecs.ndim == 1 and self._vecs.shape[0] == 0:
            # First insertion: initialize the matrix
            self._vecs = normed
        else:
            self._vecs = np.concatenate([self._vecs, normed], axis=0)

        self._ids.extend(ids)
        self._metas.extend(metas)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[str, float, dict]]:
        """Return top-*k* (id, cosine_score, meta) tuples, descending by score.

        *qvec* is L2-normalized internally; no model required.
        """
        if len(self._ids) == 0:
            return []
        q = np.asarray(qvec, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm
        # dot product against normalized matrix = cosine similarity
        scores: np.ndarray = self._vecs @ q          # shape (n,)
        k = min(k, len(self._ids))
        # argpartition for O(n) top-k, then sort the top slice
        if k < len(self._ids):
            top_idx = np.argpartition(scores, -k)[-k:]
        else:
            top_idx = np.arange(len(self._ids))
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [
            (self._ids[int(i)], float(scores[i]), self._metas[int(i)])
            for i in top_idx
        ]

    def __len__(self) -> int:
        return len(self._ids)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dir: str | Path) -> None:
        """Persist vectors and metadata to *dir*."""
        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        vecs = self._vecs if self._vecs.ndim == 2 else np.empty((0, 0), dtype=np.float32)
        np.save(str(d / "vectors.npy"), vecs)
        (d / "meta.json").write_text(
            json.dumps({"ids": self._ids, "metas": self._metas}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, dir: str | Path) -> "VectorStore":
        """Restore a VectorStore from a previously saved directory."""
        d = Path(dir)
        store = cls()
        vecs_path = d / "vectors.npy"
        meta_path = d / "meta.json"
        if not vecs_path.exists() or not meta_path.exists():
            return store
        vecs = np.load(str(vecs_path))
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        store._ids = payload.get("ids", [])
        store._metas = payload.get("metas", [])
        store._vecs = vecs
        return store
