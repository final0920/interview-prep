"""Ingest chunker: tree-sitter AST chunking with line/brace fallback.

Priority:
  1. tree-sitter AST chunker (guarded by try-import; skipped offline if unavailable)
  2. line/brace-based fallback chunker (always available, deterministic)

Both paths return list[EvidenceUnit] with file:line traceability.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

from coach.schemas import Channel, EvidenceUnit

# ---- tree-sitter availability guard -----------------------------------------
try:
    from tree_sitter_language_pack import get_parser as _ts_get_parser
    _HAS_TS = True
except Exception:
    _HAS_TS = False

# ---- AST node types to extract per language ---------------------------------
_TS_NODE_TYPES: dict[str, set[str]] = {
    "java": {
        "method_declaration", "constructor_declaration",
        "class_declaration", "interface_declaration",
    },
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "c": {"function_definition", "struct_specifier"},
    "python": {"function_definition", "class_definition"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "typescript": {
        "function_declaration", "method_definition", "class_declaration",
    },
    "javascript": {
        "function_declaration", "method_definition", "class_declaration",
    },
}

# ---- Language -> Channel mapping --------------------------------------------
_LANG_TO_CHANNEL: dict[str, Channel] = {
    "sql": Channel.sql,
    "yaml": Channel.config,
    "json": Channel.config,
    "toml": Channel.config,
    "ini": Channel.config,
    "properties": Channel.config,
    "xml": Channel.config,
    "markdown": Channel.doc,
    "text": Channel.doc,
}

# ---- Shared helpers ---------------------------------------------------------
_PARSER_CACHE: dict[str, object] = {}


def _sha(text: str, path: str, line: int) -> str:
    raw = f"{path}:{line}:{text}"
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:16]


def _channel(lang: str) -> Channel:
    return _LANG_TO_CHANNEL.get(lang, Channel.code)


def _get_parser(lang: str):
    if lang not in _PARSER_CACHE:
        _PARSER_CACHE[lang] = _ts_get_parser(lang)
    return _PARSER_CACHE[lang]


# ---- tree-sitter helpers (PyO3-style API) -----------------------------------

def _g(v):
    """Invoke if callable (method getter), else return directly (property)."""
    return v() if callable(v) else v


def _row(pos) -> int:
    r = getattr(pos, "row", None)
    return int(r) if r is not None else int(pos[0])


def _node_bytes(node, code_bytes: bytes) -> bytes:
    return code_bytes[_g(node.start_byte): _g(node.end_byte)]


def _node_text(node, code_bytes: bytes) -> str:
    return _node_bytes(node, code_bytes).decode("utf-8", "replace")


def _node_name(node, code_bytes: bytes) -> str:
    nn = node.child_by_field_name("name")
    if nn is not None:
        return _node_text(nn, code_bytes)
    for i in range(_g(node.child_count)):
        c = node.child(i)
        kind = _g(c.kind)
        if "identifier" in kind:
            return _node_text(c, code_bytes)
    return "?"


def _walk_ast(node, code_bytes: bytes, targets: set[str], out: list) -> None:
    kind = _g(node.kind)
    if kind in targets:
        name = _node_name(node, code_bytes)
        start = _row(_g(node.start_position)) + 1
        end = _row(_g(node.end_position)) + 1
        text = _node_text(node, code_bytes)
        out.append((f"{kind}:{name}", start, end, text))
    for i in range(_g(node.child_count)):
        _walk_ast(node.child(i), code_bytes, targets, out)


# ---- AST chunker ------------------------------------------------------------

def _chunk_with_ast(
    text: str, lang: str, path: str
) -> Optional[list[EvidenceUnit]]:
    """Return AST-based chunks or None if unavailable/failed."""
    if not _HAS_TS or lang not in _TS_NODE_TYPES:
        return None
    try:
        parser = _get_parser(lang)
        code_bytes = text.encode("utf-8", "replace")
        tree = parser.parse(code_bytes)
        targets = _TS_NODE_TYPES[lang]
        raw: list = []
        _walk_ast(_g(tree.root_node), code_bytes, targets, raw)
        if not raw:
            return None
        ch = _channel(lang)
        units: list[EvidenceUnit] = []
        for sym, start, end, snippet in raw:
            uid = _sha(snippet, path, start)
            units.append(EvidenceUnit(
                id=uid,
                source_path=path,
                symbol=sym,
                start_line=start,
                end_line=end,
                channel=ch,
                lang=lang,
                text=snippet[:4000],
                content_hash=uid,
            ))
        return units
    except Exception:
        return None


# ---- Line/brace fallback chunker -------------------------------------------

def _split_by_brace_blocks(lines: list[str]) -> list[tuple[int, int, str]]:
    """Split by top-level brace blocks (C/Java/Go/Rust style).

    Returns list of (start_line_1indexed, end_line_1indexed, text).
    Falls back to whole-file single block if no braces found.
    """
    depth = 0
    block_start: Optional[int] = None
    blocks: list[tuple[int, int, str]] = []

    for i, line in enumerate(lines):
        opens = line.count("{")
        closes = line.count("}")
        if depth == 0 and opens > 0:
            block_start = i
        depth += opens - closes
        if depth <= 0 and block_start is not None:
            depth = 0
            block_lines = lines[block_start: i + 1]
            blocks.append((block_start + 1, i + 1, "\n".join(block_lines)))
            block_start = None

    if not blocks:
        # no brace structure: emit whole file as one chunk
        if lines:
            blocks.append((1, len(lines), "\n".join(lines)))
    return blocks


def _chunk_by_window(
    text: str, path: str, lang: str, win: int = 80, overlap: int = 12
) -> list[EvidenceUnit]:
    """Sliding-window line chunker used for non-code or as final fallback."""
    lines = text.splitlines()
    ch = _channel(lang)
    units: list[EvidenceUnit] = []
    i = 0
    while i < len(lines):
        seg = lines[i: i + win]
        if any(s.strip() for s in seg):
            start = i + 1
            end = i + len(seg)
            snippet = "\n".join(seg)
            sym = f"lines:{start}-{end}"
            uid = _sha(snippet, path, start)
            units.append(EvidenceUnit(
                id=uid,
                source_path=path,
                symbol=sym,
                start_line=start,
                end_line=end,
                channel=ch,
                lang=lang,
                text=snippet[:4000],
                content_hash=uid,
            ))
        i += max(win - overlap, 1)
    return units


def _fallback_chunk(text: str, lang: str, path: str) -> list[EvidenceUnit]:
    """Brace-block chunker with line-window fallback."""
    lines = text.splitlines()
    ch = _channel(lang)
    # Use brace-block splitting for code-like languages
    brace_langs = {
        "java", "kotlin", "scala", "cpp", "c", "go", "rust",
        "typescript", "tsx", "javascript",
    }
    if lang in brace_langs and "{" in text:
        blocks = _split_by_brace_blocks(lines)
        units: list[EvidenceUnit] = []
        for start, end, snippet in blocks:
            sym = f"block:{start}-{end}"
            uid = _sha(snippet, path, start)
            units.append(EvidenceUnit(
                id=uid,
                source_path=path,
                symbol=sym,
                start_line=start,
                end_line=end,
                channel=ch,
                lang=lang,
                text=snippet[:4000],
                content_hash=uid,
            ))
        if units:
            return units

    # Sliding window for everything else
    return _chunk_by_window(text, path, lang)


# ---- Public API -------------------------------------------------------------

def chunk_code(text: str, lang: str, path: str) -> list[EvidenceUnit]:
    """Chunk source code text into EvidenceUnits.

    Tries tree-sitter AST first; falls back to line/brace chunker.
    Always returns at least one unit for non-empty input.
    """
    if not text.strip():
        return []

    ast_units = _chunk_with_ast(text, lang, path)
    if ast_units:
        return ast_units

    return _fallback_chunk(text, lang, path)


def chunk_pdf(pdf_path: str) -> list[EvidenceUnit]:
    """Extract text blocks from a PDF using PyMuPDF.

    Returns one EvidenceUnit per non-trivial text block (>= 25 chars).
    start_line / end_line hold the 1-indexed page number.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return []

    import re as _re
    _PII_RE = _re.compile(r"(1[3-9]\d{9}|[\w.\-]+@[\w.\-]+\.\w+)")

    units: list[EvidenceUnit] = []
    doc = fitz.open(pdf_path)
    try:
        for pno in range(len(doc)):
            for blk in doc[pno].get_text("blocks"):
                snippet = (blk[4] or "").strip()
                if len(snippet) < 25:
                    continue
                sym = f"page{pno + 1}:block{int(blk[5])}"
                uid = _sha(snippet, pdf_path, pno + 1)
                tags: list[str] = []
                if _PII_RE.search(snippet):
                    tags.append("has_pii")
                units.append(EvidenceUnit(
                    id=uid,
                    source_path=pdf_path,
                    symbol=sym,
                    start_line=pno + 1,
                    end_line=pno + 1,
                    channel=Channel.doc,
                    lang="resume",
                    text=snippet[:4000],
                    content_hash=uid,
                    tags=tags,
                ))
    finally:
        doc.close()
    return units


# ---- SQL CREATE TABLE extraction --------------------------------------------

_CREATE_TABLE_RE = re.compile(
    r"(CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\S+[^;]+;)",
    re.IGNORECASE | re.DOTALL,
)


def extract_create_tables(text: str, path: str) -> list[EvidenceUnit]:
    """Extract individual CREATE TABLE statements from SQL text."""
    units: list[EvidenceUnit] = []
    for m in _CREATE_TABLE_RE.finditer(text):
        stmt = m.group(1).strip()
        # find line numbers within original text
        start_char = m.start()
        start_line = text[:start_char].count("\n") + 1
        end_line = start_line + stmt.count("\n")
        # extract table name
        name_m = re.search(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\S+)",
            stmt,
            re.IGNORECASE,
        )
        sym = f"create_table:{name_m.group(1)}" if name_m else "create_table:?"
        uid = _sha(stmt, path, start_line)
        units.append(EvidenceUnit(
            id=uid,
            source_path=path,
            symbol=sym,
            start_line=start_line,
            end_line=end_line,
            channel=Channel.sql,
            lang="sql",
            text=stmt[:4000],
            content_hash=uid,
        ))
    return units
