"""L2 episodic + L3 semantic memory backed by sqlite.

Single-user: no tenant_id. All times are epoch seconds (float).

L2 episodes  -- event/diary/review entries; recalled by time-decay x tag overlap.
L3 semantic  -- stable key/value profile; self-edit dedup via valid_from/to
                (Zep-style temporal window + Mem0-style dedup).

Public API matches DESIGN.md sec 6.T4:
    MemoryStore.__init__(db_path)
    .add_episode(ep: MemoryEpisode) -> str
    .recent_episodes(k, tags=None, now=None) -> list[MemoryEpisode]
    .upsert_semantic(m: MemorySemantic) -> None
    .get_semantic(kind=None) -> list[MemorySemantic]
    .weakpoints() -> list[str]
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

from coach.schemas import MemoryEpisode, MemorySemantic


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------
_DDL_EPISODES = """
CREATE TABLE IF NOT EXISTS memory_episodes (
    episode_id  TEXT    PRIMARY KEY,
    ts          REAL    NOT NULL,
    kind        TEXT    NOT NULL DEFAULT 'event',
    content     TEXT    NOT NULL DEFAULT '',
    tags        TEXT    NOT NULL DEFAULT '[]',
    round       TEXT    NOT NULL DEFAULT '',
    score       INTEGER
)
"""

_DDL_SEMANTIC = """
CREATE TABLE IF NOT EXISTS memory_semantic (
    record_id   TEXT    PRIMARY KEY,
    key         TEXT    NOT NULL,
    value       TEXT    NOT NULL DEFAULT '',
    kind        TEXT    NOT NULL DEFAULT 'skill',
    confidence  REAL    NOT NULL DEFAULT 0.5,
    valid_from  REAL    NOT NULL,
    valid_to    REAL,
    updated_ts  REAL    NOT NULL
)
"""

_DDL_IDX_SEM_KEY = (
    "CREATE INDEX IF NOT EXISTS idx_sem_key ON memory_semantic (key, valid_to)"
)


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------
_LAST_TS: list[float] = [0.0]


def _monotonic_now() -> float:
    """Strictly monotonically increasing epoch seconds (safe on Windows 15 ms clock)."""
    t = time.time()
    if t <= _LAST_TS[0]:
        t = _LAST_TS[0] + 1e-6
    _LAST_TS[0] = t
    return t


def _fingerprint(*parts: object) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:16]


def _dumps(v: object) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def _loads(s: Optional[str]) -> object:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return s


def _connect(path: str | Path) -> sqlite3.Connection:
    p = str(path)
    if p != ":memory:":
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_DDL_EPISODES)
    conn.execute(_DDL_SEMANTIC)
    conn.execute(_DDL_IDX_SEM_KEY)
    conn.commit()


# ---------------------------------------------------------------------------
# Score helpers (pure, exported for tests)
# ---------------------------------------------------------------------------
def score_recency(age_days: float, half_life_days: float = 7.0) -> float:
    """Exponential decay: age_days=0 -> 1.0; halves every half_life_days."""
    age = max(0.0, float(age_days))
    if half_life_days is None or half_life_days <= 0:
        return 1.0 if age == 0.0 else 0.0
    return float(0.5 ** (age / float(half_life_days)))


def tag_overlap(tags_a: list[str], tags_b: list[str]) -> float:
    """Jaccard similarity of two tag lists."""
    a, b = set(tags_a), set(tags_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------
class MemoryStore:
    """Sqlite-backed L2 episodic + L3 semantic memory store (single-user)."""

    def __init__(self, db_path: str | Path) -> None:
        self._conn = _connect(db_path)
        _init_schema(self._conn)

    # ------------------------------------------------------------------
    # L2 episodic
    # ------------------------------------------------------------------

    def add_episode(self, ep: MemoryEpisode) -> str:
        """Persist a MemoryEpisode; return the episode_id (stable fingerprint)."""
        now = _monotonic_now()
        ts = ep.ts if ep.ts else now
        tags_json = _dumps(ep.tags)
        # Stable id so re-adding the same logical event is idempotent.
        eid = ep.id or ("ep-" + _fingerprint(f"{ts:.6f}", ep.kind, ep.content[:64]))
        self._conn.execute(
            """
            INSERT INTO memory_episodes
                (episode_id, ts, kind, content, tags, round, score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(episode_id) DO UPDATE SET
                ts=excluded.ts, kind=excluded.kind, content=excluded.content,
                tags=excluded.tags, round=excluded.round, score=excluded.score
            """,
            (eid, ts, ep.kind, ep.content, tags_json, ep.round, ep.score),
        )
        self._conn.commit()
        return eid

    def recent_episodes(
        self,
        k: int,
        tags: Optional[list[str]] = None,
        now: Optional[float] = None,
    ) -> list[MemoryEpisode]:
        """Return up to k episodes ranked by time-decay x tag overlap.

        Scoring: recency(age_days) * (1 + overlap(tags, ep.tags))
        Pure time order when tags is None.
        """
        ref = now if now is not None else _monotonic_now()
        rows = self._conn.execute(
            "SELECT * FROM memory_episodes ORDER BY ts DESC"
        ).fetchall()

        want = list(tags) if tags else []
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            ep_tags: list[str] = _loads(row["tags"]) or []
            age_days = max(0.0, (ref - row["ts"]) / 86400.0)
            rec = score_recency(age_days)
            ov = tag_overlap(want, ep_tags) if want else 0.0
            score = rec * (1.0 + ov)
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        result: list[MemoryEpisode] = []
        for _, row in scored[:k]:
            result.append(_row_to_episode(row))
        return result

    # ------------------------------------------------------------------
    # L3 semantic
    # ------------------------------------------------------------------

    def upsert_semantic(self, m: MemorySemantic) -> None:
        """Write or update a semantic memory entry with Zep/Mem0-style dedup.

        Same (key) + same value  -> noop (just refresh confidence/updated_ts).
        Same (key) + changed val -> close old record (valid_to=now), insert new.
        New key                  -> plain insert.
        """
        now = _monotonic_now()
        cur = self._current_semantic(m.key)

        if cur is None:
            # First write for this key.
            rid = "sem-" + _fingerprint(m.key, f"{now:.6f}", m.value[:32])
            self._conn.execute(
                """
                INSERT INTO memory_semantic
                    (record_id, key, value, kind, confidence, valid_from, valid_to, updated_ts)
                VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (rid, m.key, m.value, m.kind, m.confidence, now, now),
            )
        elif cur["value"] == m.value:
            # Same value: noop, but lift confidence monotonically and refresh ts.
            new_conf = max(float(cur["confidence"]), m.confidence)
            self._conn.execute(
                "UPDATE memory_semantic SET confidence=?, updated_ts=? WHERE record_id=?",
                (new_conf, now, cur["record_id"]),
            )
        else:
            # Value changed: supersede (close old, insert new version).
            self._conn.execute(
                "UPDATE memory_semantic SET valid_to=?, updated_ts=? WHERE record_id=?",
                (now, now, cur["record_id"]),
            )
            rid = "sem-" + _fingerprint(m.key, f"{now:.6f}", m.value[:32])
            self._conn.execute(
                """
                INSERT INTO memory_semantic
                    (record_id, key, value, kind, confidence, valid_from, valid_to, updated_ts)
                VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (rid, m.key, m.value, m.kind, m.confidence, now, now),
            )

        self._conn.commit()

    def get_semantic(self, kind: Optional[str] = None) -> list[MemorySemantic]:
        """Return currently valid semantic entries (valid_to IS NULL), optionally by kind."""
        if kind is not None:
            rows = self._conn.execute(
                "SELECT * FROM memory_semantic WHERE valid_to IS NULL AND kind=? ORDER BY key",
                (kind,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memory_semantic WHERE valid_to IS NULL ORDER BY key"
            ).fetchall()
        return [_row_to_semantic(r) for r in rows]

    def weakpoints(self) -> list[str]:
        """Return values of all current semantic entries with kind='weakpoint'."""
        rows = self._conn.execute(
            "SELECT value FROM memory_semantic WHERE valid_to IS NULL AND kind='weakpoint' ORDER BY updated_ts DESC"
        ).fetchall()
        return [r["value"] for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _current_semantic(self, key: str) -> Optional[sqlite3.Row]:
        """Fetch the currently valid record for a key (valid_to IS NULL)."""
        return self._conn.execute(
            "SELECT * FROM memory_semantic WHERE key=? AND valid_to IS NULL ORDER BY valid_from DESC LIMIT 1",
            (key,),
        ).fetchone()

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Row -> schema helpers
# ---------------------------------------------------------------------------

def _row_to_episode(row: sqlite3.Row) -> MemoryEpisode:
    tags = _loads(row["tags"]) or []
    return MemoryEpisode(
        id=row["episode_id"],
        ts=row["ts"],
        kind=row["kind"],
        content=row["content"],
        tags=tags,
        round=row["round"],
        score=row["score"],
    )


def _row_to_semantic(row: sqlite3.Row) -> MemorySemantic:
    return MemorySemantic(
        key=row["key"],
        value=row["value"],
        kind=row["kind"],
        confidence=row["confidence"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        updated_ts=row["updated_ts"],
    )
