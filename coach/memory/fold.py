"""Context folding: summarise distant/low-relevance chunks via cheap LLM.

Feature flag: memory.fold (default OFF).

fold_context(chunks, gateway, *, keep_recent) -> str

The first `keep_recent` chunks are kept verbatim; the remainder are
summarised with a single cheap_complete call so the full context window
is not wasted on distant material.  If the gateway raises or chunks is
short enough, the chunks are concatenated as-is (graceful degradation).
"""
from __future__ import annotations

_SUMMARY_PROMPT = (
    "Summarise the following conversation context concisely, preserving "
    "key facts, decisions, and technical details. Output a short paragraph "
    "in the same language as the input.\n\n"
    "CONTEXT:\n{context}"
)


def fold_context(
    chunks: list[str],
    gateway,
    *,
    keep_recent: int = 3,
) -> str:
    """Summarise distant chunks via a cheap LLM call; keep the most recent verbatim.

    Args:
        chunks:      List of context strings, ordered oldest -> newest.
        gateway:     LLMGateway instance (must have .cheap_complete()).
        keep_recent: Number of tail chunks to keep verbatim (not summarised).

    Returns:
        A single string combining [summary of distant] + [verbatim recent],
        separated by a blank line.  Falls back to plain join on any error.
    """
    if not chunks:
        return ""

    keep = max(0, int(keep_recent))

    # Nothing to summarise: all chunks are in the "recent" window.
    if len(chunks) <= keep:
        return "\n\n".join(chunks)

    distant = chunks[: len(chunks) - keep] if keep > 0 else chunks
    recent = chunks[len(chunks) - keep :] if keep > 0 else []

    try:
        prompt = _SUMMARY_PROMPT.format(context="\n\n".join(distant))
        summary = gateway.cheap_complete([{"role": "user", "content": prompt}])
        summary = summary.strip()
    except Exception:
        # Graceful degradation: just concatenate everything.
        return "\n\n".join(chunks)

    parts: list[str] = []
    if summary:
        parts.append(f"[Summary of earlier context]\n{summary}")
    parts.extend(recent)
    return "\n\n".join(parts)
