"""Tidal recall: multi-timescale episodic memory retrieval.

Feature flag: memory.tidal (default OFF).

tidal_recall(store, vector_store, seed_tags, now) -> list[MemoryEpisode]

Divides the episode timeline into three tidal buckets:
    near   -- 0-7 days   (high fidelity, all returned)
    mid    -- 7-90 days  (medium fidelity, scored by tag resonance)
    abyss  -- >90 days   (low fidelity, only high-resonance survivors)

Each bucket is scored by: time_decay x resonance x relevance, then the
top entries from each bucket are fused and returned ranked by combined score.

relevance is tag-overlap resonance (no dense vector store required).
vector_store is accepted for API compatibility but ignored.
"""
from __future__ import annotations

from typing import Optional

from coach.schemas import MemoryEpisode
from coach.memory.store import MemoryStore, score_recency, tag_overlap

# ---------------------------------------------------------------------------
# Bucket configuration
# ---------------------------------------------------------------------------
_NEAR_DAYS = 7.0        # 0 .. 7 days
_MID_DAYS = 90.0        # 7 .. 90 days
# beyond _MID_DAYS -> abyss

# How many survivors to take from each bucket
_NEAR_KEEP = 10
_MID_KEEP = 5
_ABYSS_KEEP = 2

# Abyss minimum resonance threshold (tag overlap) to qualify
_ABYSS_MIN_RESONANCE = 0.15


def tidal_recall(
    store: MemoryStore,
    vector_store,               # VectorStore | None
    seed_tags: list[str],
    now: float,
) -> list[MemoryEpisode]:
    """Retrieve episodes across three temporal tidal zones.

    Args:
        store:       MemoryStore to pull episodes from.
        vector_store: Optional VectorStore for dense resonance scoring.
                      Pass None to use tag-only resonance.
        seed_tags:   Tags from the current context (seed for relevance).
        now:         Reference epoch timestamp (float).

    Returns:
        List of MemoryEpisode ordered by combined tidal score (desc).
        At most _NEAR_KEEP + _MID_KEEP + _ABYSS_KEEP entries.
    """
    # Pull all episodes; we bucket them ourselves.
    # recent_episodes with a huge k and no filter gives us everything scored.
    all_eps = store.recent_episodes(k=10_000, tags=seed_tags, now=now)

    near: list[tuple[float, MemoryEpisode]] = []
    mid: list[tuple[float, MemoryEpisode]] = []
    abyss: list[tuple[float, MemoryEpisode]] = []

    for ep in all_eps:
        age_days = max(0.0, (now - ep.ts) / 86400.0)
        score = _tidal_score(ep, seed_tags, age_days, vector_store)

        if age_days <= _NEAR_DAYS:
            near.append((score, ep))
        elif age_days <= _MID_DAYS:
            mid.append((score, ep))
        else:
            # Abyss: only keep episodes with sufficient tag resonance.
            resonance = tag_overlap(seed_tags, ep.tags) if seed_tags else 0.0
            if resonance >= _ABYSS_MIN_RESONANCE:
                abyss.append((score, ep))

    near.sort(key=lambda x: x[0], reverse=True)
    mid.sort(key=lambda x: x[0], reverse=True)
    abyss.sort(key=lambda x: x[0], reverse=True)

    fused: list[MemoryEpisode] = []
    for _, ep in near[:_NEAR_KEEP]:
        fused.append(ep)
    for _, ep in mid[:_MID_KEEP]:
        fused.append(ep)
    for _, ep in abyss[:_ABYSS_KEEP]:
        fused.append(ep)

    # Final sort by tidal score across all three buckets combined.
    bucket_map: dict[Optional[str], float] = {}
    for sc, ep in near + mid + abyss:
        if ep.id is not None:
            bucket_map[ep.id] = sc

    fused.sort(key=lambda ep: bucket_map.get(ep.id, 0.0), reverse=True)
    return fused


# ---------------------------------------------------------------------------
# Scoring internals
# ---------------------------------------------------------------------------

def _tidal_score(
    ep: MemoryEpisode,
    seed_tags: list[str],
    age_days: float,
    vector_store,
) -> float:
    """Combined tidal score: time_decay * resonance * relevance.

    resonance  -- tag overlap between seed_tags and ep.tags
    relevance  -- tag-overlap resonance (vector_store is unused)
    """
    decay = score_recency(age_days, half_life_days=14.0)   # 14-day half-life for tidal

    if seed_tags:
        resonance = 1.0 + tag_overlap(seed_tags, ep.tags)
    else:
        resonance = 1.0

    relevance = _tag_relevance(ep, seed_tags)

    return decay * resonance * relevance


def _tag_relevance(ep: MemoryEpisode, seed_tags: list[str]) -> float:
    """Return a tag-overlap resonance score in [1.0, 2.0].

    Uses seed_tags vs ep.tags Jaccard-style overlap as the relevance signal.
    Returns 1.0 (neutral) when seed_tags is empty.
    """
    if seed_tags:
        return 1.0 + tag_overlap(seed_tags, ep.tags)
    return 1.0
