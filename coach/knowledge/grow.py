"""Knowledge growth: LLM-generate a topic card and write it to public_kb/.

Public API:
    def grow_topic(topic, cfg, gateway, *, use_web=False) -> Path
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from coach.config import get

if TYPE_CHECKING:
    from coach.llm.gateway import LLMGateway

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PUBLIC_KB_DIR = _REPO_ROOT / "public_kb"

# ---------------------------------------------------------------------------
# PII gate -- block any generated content that looks like personal data
# ---------------------------------------------------------------------------

_PII_PATTERNS = [
    r"1[3-9]\d{9}",           # Chinese mobile
    r"[\w.+-]+@[\w-]+\.[\w.-]+",  # email
    r"\b\d{17}[\dXx]\b",     # Chinese ID card
    r"\b\d{16,19}\b",         # bank card / long number
]


def _looks_like_pii(text: str) -> bool:
    return any(re.search(p, text or "") for p in _PII_PATTERNS)


def _redact_pii(text: str) -> str:
    """Replace PII matches with [REDACTED] rather than discarding the whole body."""
    s = text or ""
    for p in _PII_PATTERNS:
        s = re.sub(p, "[REDACTED]", s)
    return s


# ---------------------------------------------------------------------------
# Slug: topic -> safe, stable, unique filename stem
# ---------------------------------------------------------------------------

_SLUG_MAX = 48
_HASH_LEN = 10


def _slugify(topic: str) -> str:
    """Convert topic to a safe kebab-case slug, stable across calls (idempotent)."""
    t = (topic or "").strip()
    if not t:
        return "kb-empty"

    digest = hashlib.md5(t.casefold().encode()).hexdigest()[:_HASH_LEN]

    ascii_text = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii").lower()
    kebab = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    alnum_len = len(re.sub(r"[^a-z0-9]", "", kebab))

    if alnum_len < 2:
        return f"kb-{digest}"

    orig_alnum = sum(1 for ch in t if ch.isalnum())
    lossy = orig_alnum > alnum_len

    if not lossy and len(kebab) <= _SLUG_MAX:
        return kebab
    return f"{kebab[:_SLUG_MAX].rstrip('-')}-{digest}"


def _safe_path(slug: str, kb_dir: Path) -> Path:
    """Resolve write path and enforce it stays within kb_dir (path traversal guard)."""
    base = kb_dir.resolve()
    safe_slug = re.sub(r"[^a-z0-9-]+", "", slug.lower()).strip("-") or "kb-empty"
    target = (base / f"{safe_slug}.md").resolve()
    if base not in target.parents:
        raise ValueError(f"Illegal write path outside public_kb/: {target}")
    return target


# ---------------------------------------------------------------------------
# Markdown assembly
# ---------------------------------------------------------------------------

def _sanitize_body(text: str) -> str:
    """Strip code fences and any accidental front-matter from LLM output."""
    s = (text or "").strip().lstrip("﻿")
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s).strip()
    if s.startswith("---"):
        end = s.find("\n---", 3)
        if end != -1:
            s = s[end + 4:].lstrip("\n")
    return s.strip()


def _yaml_val(v: str) -> str:
    s = str(v)
    if re.search(r'[:#"\']', s) or s != s.strip():
        return '"' + s.replace('"', '\\"') + '"'
    return s


def _build_markdown(topic: str, body: str, source: str = "ai-generated") -> str:
    """Assemble front-matter + body into a public_kb-compatible markdown string."""
    lines = [
        "---",
        f"subject: {_yaml_val(topic)}",
        "authority: ai-grown",
        "framework_ver: general",
        "license: L0-public",
        "pii: none",
        f"source: {_yaml_val(source)}",
        f"topic: {_yaml_val(topic)}",
        'generated_note: "AI generated - verify before use"',
        "---",
        "",
    ]
    return "\n".join(lines) + _sanitize_body(body) + "\n"


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a senior software engineer writing a structured interview knowledge card "
    "for a public study database. Requirements:\n"
    "1. Content must be accurate, factual, and structured for technical interview prep.\n"
    "2. Do not invent specific numbers, benchmark figures, or version strings you are "
    "not certain about; use qualitative descriptions instead.\n"
    "3. No personal data, secrets, or internal information -- only general public knowledge.\n"
    "4. Use markdown with only top-level '# Heading' sections (no ## or deeper, no front-matter).\n"
    "5. Include at least: 'Definition and background', 'Core mechanism', "
    "'Key points and pitfalls', 'Common interview follow-ups'.\n"
    "6. Output markdown body only -- no preamble, no code fences, no pleasantries."
)


def _build_messages(topic: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Topic: {topic}\n\nWrite the knowledge card."},
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_MIN_BODY = (
    "# Definition and background\n"
    "Content for this topic could not be generated. Please retry or fill in manually.\n"
)


def grow_topic(
    topic: str,
    cfg: dict,
    gateway: "LLMGateway",
    *,
    use_web: bool = False,
    kb_dir: Optional[Path] = None,
) -> Path:
    """Generate a knowledge card for *topic* and write it to public_kb/.

    Behaviour:
    - Calls gateway.cheap_complete to generate a markdown knowledge card.
    - Applies PII gate: if PII patterns are detected the content is redacted
      (not discarded) so the structural knowledge is preserved.
    - Idempotent: writing the same topic twice overwrites the existing file.
    - use_web=False (default): pure LLM generation, fully offline.
    - use_web=True: reserved for future web-augmented generation; currently
      falls back to pure LLM (no network calls in this implementation).

    Args:
        topic:   topic string (non-empty)
        cfg:     merged config dict
        gateway: LLMGateway instance (injected; use a fake in tests)
        use_web: whether to attempt web-augmented generation (ignored for now)
        kb_dir:  override public_kb directory (used by tests for isolation)

    Returns:
        Path to the written markdown file.

    Raises:
        ValueError: if topic is empty.
    """
    t = (topic or "").strip()
    if not t:
        raise ValueError("topic must be non-empty")

    target_dir = Path(kb_dir or _PUBLIC_KB_DIR)
    slug = _slugify(t)
    target = _safe_path(slug, target_dir)

    messages = _build_messages(t)
    try:
        raw = gateway.cheap_complete(messages)
        body = _sanitize_body(str(raw or ""))
        if not body:
            raise ValueError("empty response from gateway")
    except Exception:
        body = _MIN_BODY

    if _looks_like_pii(body):
        body = _redact_pii(body)

    md = _build_markdown(t, body, source="ai-generated")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(md, encoding="utf-8", newline="\n")
    return target
