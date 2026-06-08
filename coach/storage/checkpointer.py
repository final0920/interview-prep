"""LangGraph checkpointer factory.

Returns a langgraph checkpointer for persistent interview session state.
Tests that cannot import langgraph should use pytest.importorskip.
"""
from __future__ import annotations

from pathlib import Path


def get_checkpointer(path: str | Path):
    """Return a LangGraph checkpointer bound to *path*.

    Tries langgraph.checkpoint.sqlite.SqliteSaver first (persistent, file-based).
    Falls back to langgraph.checkpoint.memory.InMemorySaver when the sqlite
    extra is not installed in the current environment.

    Raises ImportError only if langgraph itself is absent (tests should guard
    with pytest.importorskip("langgraph")).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
        return SqliteSaver.from_conn_string(str(p))
    except ImportError:
        from langgraph.checkpoint.memory import InMemorySaver  # type: ignore
        return InMemorySaver()
