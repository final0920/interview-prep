"""Placeholder rendering for prompt templates.

Supported tokens (feature flag memory.placeholders, default ON):
    {{profile}}    -- one-line summary of current semantic profile
    {{weakpoints}} -- bullet list of current weakpoints
    {{resume}}     -- basics from ResumeProfile (name, skills, projects)

Usage:
    from coach.memory.placeholders import render
    text = render(template, store, profile)
"""
from __future__ import annotations

import re
from typing import Optional

from coach.schemas import ResumeProfile
from coach.memory.store import MemoryStore

# Tokens we expand; anything else is left as-is.
_TOKEN_RE = re.compile(r"\{\{(\w+)\}\}")


def render(
    template: str,
    store: MemoryStore,
    profile: Optional[ResumeProfile] = None,
) -> str:
    """Expand {{profile}}, {{weakpoints}}, {{resume}} placeholders in *template*.

    Unknown tokens are left unchanged so callers can chain multiple renders
    or use tokens reserved for other systems.
    """
    def _replace(m: re.Match) -> str:
        token = m.group(1)
        if token == "weakpoints":
            return _render_weakpoints(store)
        if token == "profile":
            return _render_profile(store)
        if token == "resume":
            return _render_resume(profile)
        return m.group(0)   # unknown token: leave as-is

    return _TOKEN_RE.sub(_replace, template)


# ---------------------------------------------------------------------------
# Token renderers (all return str; graceful when data absent)
# ---------------------------------------------------------------------------

def _render_weakpoints(store: MemoryStore) -> str:
    wps = store.weakpoints()
    if not wps:
        return "(no weakpoints recorded)"
    return "\n".join(f"- {w}" for w in wps)


def _render_profile(store: MemoryStore) -> str:
    entries = store.get_semantic()
    if not entries:
        return "(no semantic profile yet)"
    lines: list[str] = []
    for m in entries:
        lines.append(f"{m.key}: {m.value} (confidence={m.confidence:.2f})")
    return "\n".join(lines)


def _render_resume(profile: Optional[ResumeProfile]) -> str:
    if profile is None:
        return "(no resume loaded)"
    parts: list[str] = []
    name = (profile.basics or {}).get("name", "")
    if name:
        parts.append(f"Name: {name}")
    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills[:20]))
    if profile.projects:
        proj_names = [p.name for p in profile.projects if p.name]
        if proj_names:
            parts.append("Projects: " + ", ".join(proj_names[:10]))
    if profile.experiences:
        titles = [e.get("title", "") for e in profile.experiences if e.get("title")]
        if titles:
            parts.append("Experience: " + ", ".join(titles[:5]))
    return "\n".join(parts) if parts else "(resume present but empty)"
