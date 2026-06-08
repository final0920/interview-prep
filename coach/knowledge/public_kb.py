"""Public knowledge base: BM25 search over shared general-knowledge markdown files.

Public API:
    def search_public(query, cfg, *, top_k) -> list[RetrievalHit]
    def fuse_private_public(private, public, k=60) -> list[RetrievalHit]
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Optional

from coach.config import get
from coach.retrieval.embed import tokenize as _tokenize
from coach.retrieval.hybrid import rrf_fuse
from coach.schemas import EvidenceUnit, Channel, RetrievalHit, SourceScope

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PUBLIC_KB_DIR = _REPO_ROOT / "public_kb"

# Built-in seed units -- used when public_kb/ directory is absent or empty
_BUILTIN_SEEDS = [
    {
        "subject": "algorithms",
        "title": "Time complexity and Big-O analysis",
        "body": (
            "Big-O describes the asymptotic upper bound of an algorithm's growth rate "
            "relative to input size, focusing on the worst case. Common orders: "
            "O(1) < O(log n) < O(n) < O(n log n) < O(n^2) < O(2^n). "
            "Drop constants and lower-order terms; use amortised analysis for dynamic arrays."
        ),
        "framework_ver": "general",
    },
    {
        "subject": "networking",
        "title": "TCP three-way handshake and four-way close",
        "body": (
            "Three-way handshake: SYN -> SYN+ACK -> ACK. Establishes sequence numbers "
            "and confirms both directions. Four-way close: each direction closes "
            "independently with FIN+ACK; active closer enters TIME_WAIT (2*MSL) to "
            "absorb delayed segments."
        ),
        "framework_ver": "general",
    },
]


# ---------------------------------------------------------------------------
# Parsing helpers (align with salvage public_kb front-matter format)
# ---------------------------------------------------------------------------

def _parse_front_matter(text: str) -> tuple[dict, str]:
    """Parse leading YAML front-matter (--- delimited, simple key: value)."""
    meta: dict = {}
    s = text.lstrip("﻿")
    if s.startswith("---"):
        end = s.find("\n---", 3)
        if end != -1:
            for line in s[3:end].splitlines():
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"').strip("'")
            return meta, s[end + 4:].lstrip("\n")
    return meta, s


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown body on ATX headings into [(title, text), ...]."""
    sections: list[tuple[str, str]] = []
    cur_title: Optional[str] = None
    cur_lines: list[str] = []
    for line in (body or "").splitlines():
        m = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if m:
            if cur_title is not None:
                sections.append((cur_title, "\n".join(cur_lines).strip()))
            cur_title = m.group(1).strip()
            cur_lines = []
        elif cur_title is not None:
            cur_lines.append(line)
    if cur_title is not None:
        sections.append((cur_title, "\n".join(cur_lines).strip()))
    return sections


def _unit_id(subject: str, title: str) -> str:
    h = hashlib.md5(f"{subject} {title}".encode("utf-8")).hexdigest()
    return "pub_" + h[:12]


# ---------------------------------------------------------------------------
# Load public units from markdown files
# ---------------------------------------------------------------------------

def _load_public_units(kb_dir: Optional[Path] = None) -> list[EvidenceUnit]:
    """Read public_kb/*.md files into EvidenceUnit objects.

    Falls back to built-in seeds when the directory is absent or empty.
    source label is stored in EvidenceUnit.repo as "framework_ver" for
    downstream source annotation (framework_doc@version).
    """
    d = Path(kb_dir or _PUBLIC_KB_DIR)
    raw: list[dict] = []

    if d.is_dir():
        for fp in sorted(d.glob("*.md")):
            try:
                text = fp.read_text(encoding="utf-8")
            except Exception:
                continue
            meta, body = _parse_front_matter(text)
            subject = meta.get("subject") or fp.stem
            fw_ver = meta.get("framework_ver") or "general"
            for title, para in _split_sections(body):
                if not para.strip():
                    continue
                raw.append({
                    "subject": subject,
                    "title": title,
                    "body": para,
                    "framework_ver": fw_ver,
                    "file": fp.name,
                })

    if not raw:
        raw = list(_BUILTIN_SEEDS)

    units: list[EvidenceUnit] = []
    for r in raw:
        subject = str(r.get("subject", "general"))
        title = str(r.get("title", "")).strip()
        body = str(r.get("body", "")).strip()
        fw_ver = str(r.get("framework_ver", "general"))
        text = (title + "\n" + body).strip() if title else body
        if not text:
            continue
        units.append(EvidenceUnit(
            id=_unit_id(subject, title),
            source_path=r.get("file") or "builtin",
            symbol=title,
            text=text,
            channel=Channel.doc,
            lang="",
            # repo field carries the source label used for citation
            repo=f"framework_doc@{fw_ver}",
            tags=[subject],
        ))
    return units


# ---------------------------------------------------------------------------
# BM25 index over public units (built on demand, no caching needed for tests)
# ---------------------------------------------------------------------------

def _bm25_fallback(corpus: list[list[str]], query: list[str],
                   k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Pure-Python BM25 with +1 IDF smoothing -- correct for small corpora.

    BM25Okapi uses log((N-df+0.5)/(df+0.5)) which goes to 0 when every
    document contains the query term (N==df), causing all-zero scores on
    tiny corpora.  This variant adds 1.0 so rare terms always score > 0.
    """
    n = len(corpus)
    if n == 0 or not query:
        return [0.0] * n
    dl = [len(d) for d in corpus]
    avgdl = sum(dl) / n if n else 1.0
    df: dict[str, int] = {}
    for d in corpus:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    tf_list = []
    for d in corpus:
        tf: dict[str, int] = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        tf_list.append(tf)
    scores = [0.0] * n
    for t in set(query):
        nq = df.get(t, 0)
        if nq == 0:
            continue
        idf = math.log((n - nq + 0.5) / (nq + 0.5) + 1.0)
        for i in range(n):
            f = tf_list[i].get(t, 0)
            if f == 0:
                continue
            denom = f + k1 * (1 - b + b * dl[i] / avgdl)
            scores[i] += idf * f * (k1 + 1) / denom
    return scores


class _PublicIndex:
    def __init__(self, units: list[EvidenceUnit]):
        self.units = units
        self._corpus = [_tokenize(u.text) for u in units]

    def search(self, query: str, k: int) -> list[EvidenceUnit]:
        if not self.units:
            return []
        q = _tokenize(query)
        if not q:
            return []
        # Use the smoothed BM25 variant (IDF + 1.0) which scores correctly on
        # small corpora. BM25Okapi's raw IDF goes negative when df==N, giving
        # all-zero scores on a 2-document corpus -- the smoothed formula avoids
        # that without changing ranking on larger corpora.
        scores = _bm25_fallback(self._corpus, q)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        hits = []
        for i in order[:k]:
            if scores[i] > 0.0:
                hits.append(self.units[i])
        return hits


# module-level cache: rebuilt lazily when kb_dir is None (normal runtime)
_cached_index: Optional[_PublicIndex] = None


def _get_index(kb_dir: Optional[Path] = None) -> _PublicIndex:
    global _cached_index
    if kb_dir is not None:
        # explicit kb_dir (tests): always build fresh, don't pollute cache
        return _PublicIndex(_load_public_units(kb_dir))
    if _cached_index is None:
        _cached_index = _PublicIndex(_load_public_units())
    return _cached_index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_public(
    query: str,
    cfg: dict,
    *,
    top_k: int = 20,
    kb_dir: Optional[Path] = None,
) -> list[RetrievalHit]:
    """BM25 search over public knowledge base markdown files.

    Each hit has scope=public, retriever="bm25", and the evidence.repo field
    carries the source label "framework_doc@<version>" for citation.

    Args:
        query:  search query
        cfg:    merged config dict (reads retrieval.top_k as default top_k)
        top_k:  max results to return
        kb_dir: override public_kb directory (tests)

    Returns list[RetrievalHit] with scope=public.
    """
    k = get(cfg, "retrieval.top_k", top_k)
    idx = _get_index(kb_dir)
    units = idx.search(query, int(k))
    hits: list[RetrievalHit] = []
    for rank, unit in enumerate(units, start=1):
        hits.append(RetrievalHit(
            evidence=unit,
            score=float(rank),   # rank-based score; overwritten by fuse downstream
            rank=rank,
            scope=SourceScope.public,
            retriever="bm25",
        ))
    return hits


def fuse_private_public(
    private: list[RetrievalHit],
    public: list[RetrievalHit],
    k: int = 60,
) -> list[RetrievalHit]:
    """RRF-fuse private and public hits into a single ranked list.

    Private hits retain scope=private; public hits retain scope=public.
    RRF parameter k defaults to 60 (standard). Final list length is
    len(private) + len(public) (no truncation here; caller applies top_k).

    Returns list[RetrievalHit] sorted by fused score descending.
    """
    priv_ids = [f"priv::{h.evidence.id}" for h in private]
    pub_ids = [f"pub::{h.evidence.id}" for h in public]

    by_id: dict[str, RetrievalHit] = {}
    for fid, h in zip(priv_ids, private):
        by_id[fid] = h
    for fid, h in zip(pub_ids, public):
        by_id[fid] = h

    fused = rrf_fuse([priv_ids, pub_ids], k=k)

    out: list[RetrievalHit] = []
    for rank, (fid, score) in enumerate(fused, start=1):
        hit = by_id[fid]
        out.append(hit.model_copy(update={"score": score, "rank": rank}))
    return out
