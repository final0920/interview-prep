"""SQL sandbox: normalize MySQL DDL to SQLite and verify SQL by execution.

Ported from interview-prep/gen_sql.py (to_sqlite_ddl + sandbox_check +
verify_answer_sql), stripped to two public functions the rest of the coach
needs. Everything runs in an in-memory ``sqlite3`` connection: no files, no
network, no new deps. SELECT/WITH statements are validated via ``EXPLAIN`` so
they pass without table data; writes/DDL execute directly.
"""
from __future__ import annotations

import re
import sqlite3


# ---------------------------------------------------------------------------
# MySQL DDL -> SQLite-compatible dialect
# ---------------------------------------------------------------------------

def normalize_mysql_to_sqlite(ddl: str) -> str:
    """Rewrite a MySQL ``CREATE TABLE`` into a SQLite-executable form.

    Strips/converts the MySQL-specific syntax SQLite rejects or interprets
    differently:
      - backtick identifier quoting -> removed
      - COMMENT '...' (table/column) -> removed
      - CHARACTER SET / COLLATE / USING BTREE|HASH -> removed
      - int family int(11)/bigint(20)/tinyint(1)/... -> INTEGER
      - datetime(0)/timestamp(0)/time/date precision parens -> stripped
      - decimal(p,s) -> NUMERIC; double/float -> REAL
      - unsigned / zerofill -> removed
      - column-level AUTO_INCREMENT -> removed (SQLite uses INTEGER PRIMARY KEY)
      - inline INDEX/KEY definition lines (non-constraint) -> removed
      - trailing ") ENGINE=... DEFAULT CHARSET=..." table options -> collapsed to ")"
      - dangling commas left by removed lines -> fixed
    Returns a string ready for ``sqlite3.execute``.
    """
    s = ddl or ""
    s = s.replace("`", "")
    # column/table COMMENT '....' (allow escaped quotes)
    s = re.sub(r"COMMENT\s+'(?:[^'\\]|\\.)*'", " ", s, flags=re.I)
    s = re.sub(r"CHARACTER\s+SET\s+\w+", " ", s, flags=re.I)
    s = re.sub(r"COLLATE\s+\w+", " ", s, flags=re.I)
    s = re.sub(r"USING\s+(BTREE|HASH)", " ", s, flags=re.I)
    # type narrowing (decimal/float before int so 'int' doesn't eat them)
    s = re.sub(r"\bdecimal\s*\([^)]*\)", "NUMERIC", s, flags=re.I)
    s = re.sub(r"\b(?:double|float)\b", "REAL", s, flags=re.I)
    s = re.sub(r"\b(?:big|small|tiny|medium)?int\s*(?:\(\s*\d+\s*\))?", "INTEGER", s, flags=re.I)
    s = re.sub(r"\b(datetime|timestamp|time|date)\s*\(\s*\d+\s*\)", r"\1", s, flags=re.I)
    s = re.sub(r"\b(?:unsigned|zerofill)\b", " ", s, flags=re.I)
    s = re.sub(r"\bAUTO_INCREMENT\b", " ", s, flags=re.I)
    # drop inline INDEX/KEY lines (incl. UNIQUE/FULLTEXT/SPATIAL prefixes);
    # keep PRIMARY KEY and column-level UNIQUE constraints.
    kept = []
    for ln in s.splitlines():
        st = ln.strip()
        if re.match(r"^(UNIQUE\s+|FULLTEXT\s+|SPATIAL\s+)?(INDEX|KEY)\b", st, re.I):
            continue
        kept.append(ln)
    s = "\n".join(kept)
    # collapse trailing table options ") ENGINE=... ;" -> ")"
    s = re.sub(r"\)\s*ENGINE\b.*$", ")", s, flags=re.I | re.S)
    # fix dangling comma left by removed lines: ", )" -> "\n)"
    s = re.sub(r",\s*\)", "\n)", s)
    # drop a trailing semicolon (single-statement execute does not need it)
    s = s.rstrip().rstrip(";")
    return s


# ---------------------------------------------------------------------------
# In-memory verification
# ---------------------------------------------------------------------------

def _split_sql(text: str) -> list[str]:
    """Split multi-statement SQL on semicolons, ignoring those inside strings."""
    if not text:
        return []
    out: list[str] = []
    buf: list[str] = []
    in_str = False
    quote = ""
    for ch in text:
        if in_str:
            buf.append(ch)
            if ch == quote:
                in_str = False
        elif ch in ("'", '"'):
            in_str = True
            quote = ch
            buf.append(ch)
        elif ch == ";":
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


def verify_sql(ddl: str, query: str) -> tuple[bool, str]:
    """Build ``ddl`` then validate ``query`` in an in-memory SQLite sandbox.

    The DDL is normalized first (callers may pass raw MySQL). Each statement in
    ``query`` is checked: SELECT/WITH via ``EXPLAIN`` (no data needed), other
    statements by direct execution. Returns ``(ok, detail)`` where ``ok`` is
    True only if the setup succeeded and every checked statement ran; ``detail``
    is "" on success or a "Type: message" diagnostic on the first failure.
    """
    setup = normalize_mysql_to_sqlite(ddl) if ddl else ""
    con = sqlite3.connect(":memory:")
    try:
        if setup.strip():
            try:
                con.executescript(setup) if ";" in setup else con.execute(setup)
            except Exception as e:  # noqa: BLE001
                return False, f"setup {type(e).__name__}: {e}"
        stmts = _split_sql(query)
        if not stmts:
            return False, "no SQL statement to verify"
        for stmt in stmts:
            head = stmt.lstrip().split(None, 1)[0].upper() if stmt.strip() else ""
            try:
                if head in ("SELECT", "WITH"):
                    con.execute("EXPLAIN " + stmt)
                else:
                    con.execute(stmt)
            except Exception as e:  # noqa: BLE001
                return False, f"{type(e).__name__}: {e}"
        return True, ""
    finally:
        con.close()
