"""Embeddings: BGE-M3 (sentence-transformers) with deterministic HashingEmbedder fallback.

Public API:
    class Embedder          (abstract base)
    class HashingEmbedder   (deterministic offline fallback, pure numpy)
    class BGEEmbedder       (BGE-M3 via sentence-transformers; lazy-loads)
    def tokenize(text)      camelCase/snake + Chinese 2-gram
    def get_embedder(cfg)   factory: real model or hashing fallback
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Optional

import numpy as np

from coach.config import get

FALLBACK_DIM = 1024   # matches BGE-M3 dense dim; hashing uses same width for RRF compatibility


# ---------------------------------------------------------------------------
# Tokenizer (shared with hybrid BM25 channel for consistent vocabulary)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Split English camelCase/snake identifiers + Chinese 2-gram segments."""
    toks: list[str] = []
    for m in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text):
        for p in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+", m):
            toks.append(p.lower())
    for seg in re.findall(r"[一-鿿]+", text):
        if len(seg) == 1:
            toks.append(seg)
        else:
            toks.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return toks


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Embedder:
    """Abstract embedder: input texts -> L2-normalised float32 matrix."""

    name: str = "base"
    dim: int = FALLBACK_DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return shape (len(texts), self.dim) float32, L2-normalised rows."""
        raise NotImplementedError

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# HashingEmbedder: deterministic, zero-dependency, fully offline
# ---------------------------------------------------------------------------

class HashingEmbedder(Embedder):
    """Deterministic hashing embedder for offline / test use.

    Maps each token via MD5 to a bucket in [0, dim) with a +/- sign drawn
    from the hash, accumulates IDF-weighted counts, then L2-normalises.
    Same tokens always land in same bucket across processes (hashlib, not
    built-in hash which is PYTHONHASHSEED-randomised).
    """

    name = "hashing-fallback"

    def __init__(self, dim: int = FALLBACK_DIM, idf: Optional[dict[str, float]] = None):
        self.dim = int(dim)
        self._idf: dict[str, float] = idf or {}

    @staticmethod
    def _bucket(token: str, dim: int) -> tuple[int, float]:
        h = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if (h[4] & 1) == 0 else -1.0
        return idx, sign

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, t in enumerate(texts):
            for tok in tokenize(t):
                idx, sign = self._bucket(tok, self.dim)
                w = self._idf.get(tok, 1.0)
                out[row, idx] += sign * w
            nrm = np.linalg.norm(out[row])
            if nrm > 0:
                out[row] /= nrm
        return out


# ---------------------------------------------------------------------------
# BGEEmbedder: sentence-transformers wrapper with auto-fallback
# ---------------------------------------------------------------------------

class BGEEmbedder(Embedder):
    """BGE-M3 via sentence-transformers.  Falls back to HashingEmbedder on any
    import / load / encode failure so the rest of the pipeline never breaks.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.name = f"bge:{model_name}"
        self.dim = FALLBACK_DIM
        self._model = None
        self._fallback: Optional[HashingEmbedder] = None

    # try once; on any exception delegate permanently to hashing
    def _load(self) -> None:
        if self._model is not None or self._fallback is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            m = SentenceTransformer(self.model_name, device=self.device)
            d = m.get_sentence_embedding_dimension()
            if d:
                self.dim = int(d)
            self._model = m
        except Exception:
            self._fallback = HashingEmbedder(dim=FALLBACK_DIM)
            self.name = self._fallback.name
            self.dim = self._fallback.dim

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        if self._fallback is not None:
            return self._fallback.encode(texts)
        try:
            vecs = self._model.encode(texts, normalize_embeddings=True,
                                      show_progress_bar=False)
            return np.asarray(vecs, dtype="float32")
        except Exception:
            self._fallback = HashingEmbedder(dim=self.dim)
            self.name = self._fallback.name
            return self._fallback.encode(texts)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedder(cfg: dict) -> Embedder:
    """Return the configured embedder, or HashingEmbedder if unavailable.

    Config keys read:
      embeddings.model   (default BAAI/bge-m3)
      embeddings.device  (default cpu)

    Tests should pass cfg={"embeddings": {}} and monkeypatch or just use
    HashingEmbedder directly.  BGEEmbedder will automatically fall back to
    hashing if sentence-transformers / the model weights are absent.
    """
    model = get(cfg, "embeddings.model", "BAAI/bge-m3")
    device = get(cfg, "embeddings.device", "cpu")
    return BGEEmbedder(model_name=model, device=device)
