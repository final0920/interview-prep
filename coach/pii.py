"""PII detection and redaction: single source of truth.

Public API:
    def redact(text) -> tuple[str, dict]   mask phone/email/ID/bank-card; report actual counts
    def looks_like_pii(text) -> bool        True if any PII pattern is present

Hardening against trivial bypass:
  - NFKC-normalise input first (collapses full-width digits / @ / dots to ASCII).
  - Numeric patterns tolerate in-number separators (single spaces / hyphens
    between digits) so "188 8850 1310" and "6222-0212-3456-7890-123" are
    caught, while distinct numbers in running text stay distinct.
  - Phone accepts an optional +86 / 0086 country prefix.
  - Email tolerates spaces around '@' and the dots.

Order matters: the 18-digit PRC ID pattern is tried before the generic
16-19 digit bank-card pattern so an ID is never mis-tagged as a bank card.
"""
from __future__ import annotations

import re
import unicodedata

# A single optional in-number separator (space or hyphen) between digits.
_SEP = r"[ \-]?"


def _normalize(text: str) -> str:
    """NFKC-normalise (full-width -> ASCII) without merging distinct tokens."""
    return unicodedata.normalize("NFKC", text or "")


# ---------------------------------------------------------------------------
# Patterns (applied to NFKC-normalised text; numeric ones are separator-tolerant)
# ---------------------------------------------------------------------------

# Email: tolerate whitespace around '@' and each '.'.
_EMAIL_RE = r"[\w.\-]+\s*@\s*[\w\-]+(?:\s*\.\s*[\w\-]+)+"

# Phone: optional +86/0086 prefix, 11-digit mainland mobile (1[3-9] + 9 digits),
# with an optional separator permitted between every digit.
_PHONE_RE = (
    r"(?<![\d+])(?:(?:\+?86|0086)" + _SEP + r")?"
    r"1" + _SEP + r"[3-9]" + _SEP + _SEP.join([r"\d"] * 9) +
    r"(?!" + _SEP + r"?\d)"
)

# PRC 18-digit ID: 6 region digits, YYYYMMDD, 3 seq digits, checksum [\dXx];
# separator-tolerant between digits.
_ID_RE = (
    r"(?<!\d)[1-9](?:" + _SEP + r"\d){5}"
    r"" + _SEP + r"(?:19|20)(?:" + _SEP + r"\d){2}"
    r"" + _SEP + r"(?:0[1-9]|1[0-2])"
    r"" + _SEP + r"(?:0[1-9]|[12]\d|3[01])"
    r"(?:" + _SEP + r"\d){3}"
    r"" + _SEP + r"[\dXx](?![\dXx])"
)

# Bank card: 16-19 digits, separator-tolerant.
_BANK_RE = r"(?<![\d.])(?:\d" + _SEP + r"){15,18}\d(?!\d)"

# (kind, compiled-pattern, placeholder) -- order is significant (ID before bank)
_PATTERNS: list[tuple[str, "re.Pattern[str]", str]] = [
    ("phone", re.compile(_PHONE_RE), "<PHONE>"),
    ("email", re.compile(_EMAIL_RE), "<EMAIL>"),
    ("id_card", re.compile(_ID_RE), "<ID_CARD>"),
    ("bank_card", re.compile(_BANK_RE), "<BANK_CARD>"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def redact(text: str) -> tuple[str, dict]:
    """Redact PII from *text*; return ``(masked_text, report)``.

    Input is NFKC-normalised before matching. Numeric patterns are
    separator-tolerant so spaced / dashed bypass attempts are still masked.
    The report carries actual per-kind counts and a real coverage ratio
    (masked / detected); coverage is 1.0 only when everything detected was
    masked, and is defined as 1.0 for the no-PII case.

    report keys:
        pii_masked : True if any PII was masked
        counts     : {kind: n, ...}
        total_pii  : total PII matches detected
        masked_pii : number actually masked (== total here; redact-on-detect)
        coverage   : masked_pii / total_pii (1.0 when total_pii == 0)
        spans      : [{type, placeholder, raw_len}, ...] for auditing
    """
    masked = _normalize(text)
    counts = {kind: 0 for kind, _, _ in _PATTERNS}
    spans: list[dict] = []

    for kind, rx, placeholder in _PATTERNS:
        def _sub(m, _k=kind, _ph=placeholder):
            counts[_k] += 1
            spans.append({"type": _k, "placeholder": _ph, "raw_len": len(m.group(0))})
            return _ph

        masked = rx.sub(_sub, masked)

    total = sum(counts.values())
    masked_count = total  # redact-on-detect: every detected span is replaced
    coverage = (masked_count / total) if total else 1.0
    return masked, {
        "pii_masked": total > 0,
        "counts": counts,
        "total_pii": total,
        "masked_pii": masked_count,
        "coverage": coverage,
        "spans": spans,
    }


def looks_like_pii(text: str) -> bool:
    """Return True if any PII pattern matches *text* (after NFKC normalisation)."""
    norm = _normalize(text)
    return any(rx.search(norm) for _, rx, _ in _PATTERNS)
