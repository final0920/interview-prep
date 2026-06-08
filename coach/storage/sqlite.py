"""SQLite connection helpers for the coach storage layer.

Provides a WAL-enabled connection factory and an idempotent schema initializer.
No multi-tenant plumbing; single-user only.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (or create) a SQLite database at *path*.

    Settings applied:
      - row_factory = sqlite3.Row  (column access by name)
      - WAL journal mode           (better concurrent read performance)
      - foreign_keys = ON          (enforce FK constraints)
    """
    p = Path(path)
    if str(p) != ":memory:":
        p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection, ddl: str) -> None:
    """Execute *ddl* against *conn* and commit.

    Idempotent when the DDL uses CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS.
    Accepts a single statement or multiple statements separated by semicolons.
    """
    conn.executescript(ddl)
    conn.commit()
