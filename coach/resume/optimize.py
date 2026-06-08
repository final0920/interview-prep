"""Resume optimize: skill-gap classification, STAR rewrite, unsupported-number flagging.

Public API:
    def classify_skill_gap(profile, target_role) -> list[SkillGap]
    def optimize(profile, evidence, gateway) -> dict
"""
from __future__ import annotations

import re
from typing import Optional

from coach.llm.gateway import extract_json
from coach.schemas import EvidenceUnit, ResumeProfile, SkillGap, SkillGapCategory

from coach.resume.analyze import (
    _DEFAULT_REQUIRED_SKILLS,
    _DEFAULT_TRANSFERABLE,
    _all_skills,
    _match_required,
    classify_skill_gap as _analyze_classify,
)


# ---------------------------------------------------------------------------
# Quantified-claim detection (numbers that must be evidence-backed)
# ---------------------------------------------------------------------------

_NUM_PATTERNS = [
    r"\d+(?:\.\d+)?\s*%",                           # 40%
    r"\d+(?:\.\d+)?\s*(?:倍|x|X)",                  # 3x / 10倍
    r"\d+(?:\.\d+)?\s*(?:ms|毫秒|s|秒)",            # 200ms
    r"\d+(?:\.\d+)?\s*(?:万|亿|千|w|W|k|K|m|M)",   # 5万 / 10k
    r"\d{3,}",                                       # 5000 (QPS etc.)
]
_NUM_RE = re.compile("|".join(_NUM_PATTERNS))


def _extract_numbers(text: str) -> list[str]:
    """Return unique quantified tokens found in text."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _NUM_RE.finditer(str(text or "")):
        tok = re.sub(r"\s+", "", m.group(0))
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def _number_in_evidence(num_token: str, evidence: list[EvidenceUnit]) -> bool:
    """Return True if the digit part of num_token appears in any evidence text."""
    digits = re.sub(r"[^\d.]", "", num_token)
    if not digits:
        return True  # no digits to check
    for eu in evidence:
        if digits in re.sub(r"\s+", "", eu.text):
            return True
    return False


def flag_unsupported_numbers(
    texts: list[str] | str,
    evidence: list[EvidenceUnit],
) -> list[dict]:
    """Return risk flags for quantified claims not backed by any evidence unit.

    Args:
        texts:    One string or list of strings to scan (bullet points, STAR result, etc.).
        evidence: EvidenceUnit list from retrieval — used as the backing corpus.

    Returns list of dicts: {claim, reason, severity, where}.
    """
    if isinstance(texts, str):
        texts = [texts]
    flags: list[dict] = []
    seen: set[str] = set()
    for t in texts or []:
        t = str(t or "")
        for num in _extract_numbers(t):
            if num in seen:
                continue
            if not _number_in_evidence(num, evidence):
                seen.add(num)
                flags.append({
                    "claim": num,
                    "reason": "no evidence backing this quantified claim",
                    "severity": "high",
                    "where": t.strip()[:80],
                })
    return flags


# ---------------------------------------------------------------------------
# STAR prompt builder (pure function, no LLM/network)
# ---------------------------------------------------------------------------

_RESUME_SYS = (
    "You are a senior technical resume optimization expert. "
    "Rewrite the given experience using the STAR method "
    "(Situation / Task / Action / Result). "
    "RULES: "
    "1. Never invent or exaggerate quantified metrics (KPIs, percentages, QPS, latency). "
    "   Only keep a number if it is supported by the provided evidence citations. "
    "   If unsupported, replace with qualitative language or mark as [TBD]. "
    "2. Write all four STAR sections; focus on skills relevant to the target role. "
    "3. Citations must come from the provided evidence list only; never fabricate refs. "
    "4. Return only JSON, no extra text."
)


def _fmt_evidence(evidence: list[EvidenceUnit]) -> str:
    if not evidence:
        return "(no evidence available — do not include any specific numbers)"
    lines = [f"- {eu.source_path}:{eu.start_line} | {eu.text[:120]}" for eu in evidence[:10]]
    return "\n".join(lines)


def _build_star_prompt(
    experience: dict,
    evidence: list[EvidenceUnit],
    target_role: str,
) -> str:
    title = experience.get("title", experience.get("name", ""))
    skills = ", ".join(str(s) for s in (experience.get("skills") or experience.get("tech") or []))
    bullets = "\n".join(f"- {b}" for b in (experience.get("bullets") or [])) or "(none)"
    ev_text = _fmt_evidence(evidence)
    allowed = [f"{eu.source_path}:{eu.start_line}" for eu in evidence[:10]]
    return (
        f"Target role: {target_role}\n"
        f"Experience title: {title}\n"
        f"Declared skills: {skills}\n"
        f"Original bullets:\n{bullets}\n\n"
        f"Available evidence (cite only from this list):\n{ev_text}\n"
        f"Allowed citations: {allowed}\n\n"
        "Return JSON:\n"
        '{"title":"...", '
        '"star":{"situation":"...","task":"...","action":"...","result":"..."},'
        '"rewritten_bullets":["..."],'
        '"diff":["explanation of changes"],'
        '"citations":["path:line"]}'
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_skill_gap(
    profile: ResumeProfile,
    target_role: str,
) -> list[SkillGap]:
    """Classify skills into have / transferable / missing against the target role.

    Thin wrapper around analyze.classify_skill_gap so callers can import from
    either module.
    """
    return _analyze_classify(profile, target_role)


def optimize(
    profile: ResumeProfile,
    evidence: list[EvidenceUnit],
    gateway,
) -> dict:
    """STAR rewrite + citation check + flag unsupported numbers.

    For each experience and project in the profile, calls the LLM to produce a
    STAR rewrite, checks citations are in the allowed set, and flags any
    quantified claims that have no backing in *evidence*.

    Falls back gracefully (no rewrite, diagnosis only) if the LLM call fails.

    Args:
        profile:  Parsed ResumeProfile.
        evidence: EvidenceUnit list from ingestion / retrieval (may be empty).
        gateway:  LLMGateway instance.

    Returns a summary dict with per-experience results.
    """
    target_role = profile.basics.get("target_role", "") or ""

    items: list[dict] = []
    all_exps: list[dict] = list(profile.experiences or [])
    # add projects as experience-like entries
    for proj in profile.projects or []:
        all_exps.append({
            "title": proj.name,
            "skills": proj.tech,
            "bullets": proj.description.splitlines() if proj.description else [],
        })

    for exp in all_exps:
        title = exp.get("title", exp.get("name", ""))

        # skills that are relevant to this experience
        exp_skills = exp.get("skills") or exp.get("tech") or []
        gaps = _analyze_classify(
            profile, target_role,
            required_skills=list(_DEFAULT_REQUIRED_SKILLS),
        )

        # evidence relevant to this experience (simple text filter)
        title_lower = title.lower()
        skill_tokens = {s.lower() for s in exp_skills}
        relevant_ev = [
            eu for eu in evidence
            if title_lower in eu.text.lower()
            or any(tok in eu.text.lower() for tok in skill_tokens)
        ][:10]

        rewrite: Optional[dict] = None
        llm_error: Optional[str] = None

        try:
            prompt = _build_star_prompt(exp, relevant_ev, target_role)
            raw = gateway.complete(
                [{"role": "system", "content": _RESUME_SYS},
                 {"role": "user",   "content": prompt}],
            )
            rewrite = extract_json(raw) or None
            if rewrite:
                # sanitise citations: only allow refs from relevant_ev
                allowed = {f"{eu.source_path}:{eu.start_line}" for eu in relevant_ev}
                rewrite["citations"] = [
                    c for c in (rewrite.get("citations") or [])
                    if c in allowed
                ]
        except Exception as exc:
            llm_error = f"{type(exc).__name__}: {exc}"

        # flag unsupported numbers regardless of whether rewrite succeeded
        check_texts: list[str]
        if rewrite:
            star = rewrite.get("star") or {}
            check_texts = list(star.values()) + list(rewrite.get("rewritten_bullets") or [])
        else:
            check_texts = list(exp.get("bullets") or [])
        risk_flags = flag_unsupported_numbers(check_texts, relevant_ev)

        items.append({
            "title": title,
            "gaps": [g.model_dump() for g in gaps],
            "rewrite": rewrite,
            "degraded": rewrite is None,
            "llm_error": llm_error,
            "risk_flags": risk_flags,
            "evidence_used": len(relevant_ev),
        })

    n_degraded = sum(1 for it in items if it["degraded"])
    n_flags = sum(len(it["risk_flags"]) for it in items)
    return {
        "target_role": target_role,
        "experiences": items,
        "summary": {
            "total": len(items),
            "rewritten": len(items) - n_degraded,
            "degraded": n_degraded,
            "risk_flag_count": n_flags,
        },
    }
