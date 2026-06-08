"""Tests for coach/storage: sqlite.py, vector.py, checkpointer.py."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from coach.storage.sqlite import connect, init_schema
from coach.storage.vector import VectorStore


# ---------------------------------------------------------------------------
# sqlite helpers
# ---------------------------------------------------------------------------

SAMPLE_DDL = """
CREATE TABLE IF NOT EXISTS items (
    id   TEXT PRIMARY KEY,
    val  TEXT
);
"""


def test_connect_returns_connection(tmp_path):
    conn = connect(tmp_path / "test.db")
    assert isinstance(conn, sqlite3.Connection)
    conn.close()


def test_connect_wal_mode(tmp_path):
    conn = connect(tmp_path / "wal.db")
    row = conn.execute("PRAGMA journal_mode").fetchone()
    assert row[0] == "wal"
    conn.close()


def test_connect_row_factory(tmp_path):
    conn = connect(tmp_path / "rf.db")
    init_schema(conn, SAMPLE_DDL)
    conn.execute("INSERT INTO items VALUES ('a', 'hello')")
    conn.commit()
    row = conn.execute("SELECT id, val FROM items WHERE id='a'").fetchone()
    # sqlite3.Row supports column-name access
    assert row["id"] == "a"
    assert row["val"] == "hello"
    conn.close()


def test_init_schema_idempotent(tmp_path):
    conn = connect(tmp_path / "idem.db")
    init_schema(conn, SAMPLE_DDL)
    # Second call must not raise
    init_schema(conn, SAMPLE_DDL)
    conn.close()


def test_sqlite_upsert_idempotent(tmp_path):
    """INSERT OR REPLACE pattern is idempotent (simulates storage upsert)."""
    conn = connect(tmp_path / "upsert.db")
    init_schema(conn, SAMPLE_DDL)
    conn.execute("INSERT OR REPLACE INTO items VALUES ('k', 'v1')")
    conn.commit()
    conn.execute("INSERT OR REPLACE INTO items VALUES ('k', 'v2')")
    conn.commit()
    rows = conn.execute("SELECT val FROM items WHERE id='k'").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "v2"
    conn.close()


def test_connect_creates_parent_dirs(tmp_path):
    nested = tmp_path / "a" / "b" / "c.db"
    conn = connect(nested)
    assert nested.exists()
    conn.close()


def test_connect_memory():
    conn = connect(":memory:")
    assert isinstance(conn, sqlite3.Connection)
    conn.close()


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

def _make_vecs(n: int, dim: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v


def test_vector_store_empty_search():
    store = VectorStore()
    assert store.search(np.zeros(4, dtype=np.float32), k=5) == []


def test_vector_store_len():
    store = VectorStore()
    assert len(store) == 0
    store.add(["a", "b"], _make_vecs(2), [{}, {}])
    assert len(store) == 2


def test_vector_store_add_and_search_top1():
    """The most-similar vector should be returned first."""
    dim = 8
    store = VectorStore()
    # Create orthogonal-ish vectors
    v0 = np.zeros(dim, dtype=np.float32); v0[0] = 1.0
    v1 = np.zeros(dim, dtype=np.float32); v1[1] = 1.0
    v2 = np.zeros(dim, dtype=np.float32); v2[2] = 1.0
    store.add(["id0", "id1", "id2"], np.stack([v0, v1, v2]), [{}, {}, {}])
    # Query aligned with v1
    results = store.search(v1, k=3)
    assert results[0][0] == "id1"
    assert abs(results[0][1] - 1.0) < 1e-5  # cosine = 1.0 for same direction


def test_vector_store_cosine_ordering():
    """Scores must be descending."""
    store = VectorStore()
    vecs = _make_vecs(10, dim=16)
    ids = [f"doc{i}" for i in range(10)]
    store.add(ids, vecs, [{} for _ in ids])
    query = vecs[3]
    results = store.search(query, k=5)
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_vector_store_k_clamp():
    """Requesting more results than stored items should not raise."""
    store = VectorStore()
    store.add(["x"], _make_vecs(1), [{"tag": "x"}])
    results = store.search(_make_vecs(1)[0], k=100)
    assert len(results) == 1


def test_vector_store_meta_roundtrip():
    store = VectorStore()
    meta = {"source": "file.py", "line": 42}
    store.add(["doc1"], _make_vecs(1), [meta])
    results = store.search(_make_vecs(1)[0], k=1)
    assert results[0][2] == meta


def test_vector_store_save_load(tmp_path):
    store = VectorStore()
    dim = 6
    vecs = _make_vecs(5, dim=dim, seed=7)
    ids = [f"id{i}" for i in range(5)]
    metas = [{"i": i} for i in range(5)]
    store.add(ids, vecs, metas)

    store.save(tmp_path / "vs")
    loaded = VectorStore.load(tmp_path / "vs")

    assert len(loaded) == 5
    assert loaded._ids == ids
    # Search result on loaded store should match original
    q = vecs[2]
    r_orig = store.search(q, k=3)
    r_load = loaded.search(q, k=3)
    assert [x[0] for x in r_orig] == [x[0] for x in r_load]


def test_vector_store_load_empty_dir(tmp_path):
    """Loading from a directory without files returns an empty store."""
    store = VectorStore.load(tmp_path)
    assert len(store) == 0


def test_vector_store_save_creates_dir(tmp_path):
    store = VectorStore()
    store.add(["a"], _make_vecs(1), [{}])
    target = tmp_path / "nested" / "vs"
    store.save(target)
    assert (target / "vectors.npy").exists()
    assert (target / "meta.json").exists()


def test_vector_store_meta_json_content(tmp_path):
    store = VectorStore()
    store.add(["z"], _make_vecs(1, dim=2), [{"k": "v"}])
    store.save(tmp_path)
    payload = json.loads((tmp_path / "meta.json").read_text())
    assert payload["ids"] == ["z"]
    assert payload["metas"] == [{"k": "v"}]


# ---------------------------------------------------------------------------
# checkpointer (importorskip if langgraph absent)
# ---------------------------------------------------------------------------

def test_get_checkpointer(tmp_path):
    langgraph = pytest.importorskip("langgraph")
    from coach.storage.checkpointer import get_checkpointer

    cp = get_checkpointer(tmp_path / "cp.db")
    assert cp is not None
