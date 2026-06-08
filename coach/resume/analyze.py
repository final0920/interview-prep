"""Resume analyze: health report, coverage heatmap, gap 3-class, ATS check.

Public API:
    def health_report(profile, target_role, retr) -> dict
"""
from __future__ import annotations

import re
from typing import Optional

from coach.schemas import ResumeProfile, SkillGap, SkillGapCategory


# ---------------------------------------------------------------------------
# Default role ontology (same structure as salvage; config can override)
# ---------------------------------------------------------------------------

_DEFAULT_REQUIRED_SKILLS: list[str] = [
    "Java", "Kafka", "Redis", "MySQL", "分布式", "高并发",
    "LLM", "RAG", "向量检索", "Prompt工程", "Agent",
]

_DEFAULT_TRANSFERABLE: dict[str, str] = {
    "消息队列": "Kafka",
    "RocketMQ": "Kafka",
    "缓存": "Redis",
    "Memcached": "Redis",
    "PostgreSQL": "MySQL",
    "Oracle": "MySQL",
    "微服务": "分布式",
    "Spring Cloud": "分布式",
    "检索增强": "RAG",
    "Embedding": "向量检索",
    "Faiss": "向量检索",
    "大模型": "LLM",
    "GPT": "LLM",
    "Prompt 工程": "Prompt工程",
    "Prompt工程": "Prompt工程",
}

# ATS red-flag patterns in raw text (weak action verbs / vague language)
_ATS_REDFLAG_RE = re.compile(
    r"\b(responsible for|worked on|helped with|assisted|various|etc\.?|负责过|参与了?)\b",
    re.IGNORECASE,
)

# Quantified-claim pattern (numbers/percentages) — presence is positive for ATS
_QUANT_RE = re.compile(r"\d+\s*(?:%|倍|ms|毫秒|万|亿|k|K|m|M|QPS|TPS|GB|MB)")


def _all_skills(profile: ResumeProfile) -> list[str]:
    """Collect every skill token from the profile (skill list + experience/project tech)."""
    seen: set[str] = set()
    out: list[str] = []

    def _add(s: str) -> None:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    for s in profile.skills:
        _add(s)
    for exp in profile.experiences:
        for s in exp.get("skills", []) or exp.get("tech", []) or []:
            _add(str(s))
    for proj in profile.projects:
        for s in proj.tech:
            _add(s)
    return out


def _match_required(skill: str, required_lower: dict[str, str]) -> Optional[str]:
    """Case-insensitive substring match of skill against required set. Returns canonical name."""
    sl = skill.lower()
    for rl, rname in required_lower.items():
        if rl == sl or rl in sl or sl in rl:
            return rname
    return None


def classify_skill_gap(
    profile: ResumeProfile,
    target_role: str,
    required_skills: Optional[list[str]] = None,
    transferable: Optional[dict[str, str]] = None,
) -> list[SkillGap]:
    """3-class skill gap: have / transferable / missing.

    Returns a list of SkillGap for the *missing* and *transferable* items (skills
    the role requires that the profile either lacks or only has via analogy).
    Skills the profile already has are not included (they are not gaps).
    """
    req = required_skills if required_skills is not None else list(_DEFAULT_REQUIRED_SKILLS)
    xfer = transferable if transferable is not None else dict(_DEFAULT_TRANSFERABLE)

    resume_skills = _all_skills(profile)
    req_lower = {r.lower(): r for r in req}
    covered: set[str] = set()

    for sk in resume_skills:
        hit = _match_required(sk, req_lower)
        if hit:
            covered.add(hit)
            continue
        for src, dst in xfer.items():
            if src.lower() == sk.lower() or src.lower() in sk.lower():
                covered.add(dst)
                break

    gaps: list[SkillGap] = []
    for skill in req:
        if skill in covered:
            continue
        # check if transferable path exists from something in resume
        xfer_source: Optional[str] = None
        for src, dst in xfer.items():
            if dst == skill:
                for rsk in resume_skills:
                    if src.lower() == rsk.lower() or src.lower() in rsk.lower():
                        xfer_source = rsk
                        break
            if xfer_source:
                break
        if xfer_source:
            gaps.append(SkillGap(
                skill=skill,
                category=SkillGapCategory.transferable,
                note=f"transferable from: {xfer_source}",
            ))
        else:
            gaps.append(SkillGap(
                skill=skill,
                category=SkillGapCategory.missing,
            ))
    return gaps


def _coverage_heatmap(profile: ResumeProfile, required: list[str]) -> dict[str, str]:
    """Map each required skill to 'have' / 'transferable' / 'missing'."""
    resume_skills = _all_skills(profile)
    req_lower = {r.lower(): r for r in required}
    covered_direct: set[str] = set()
    covered_xfer: set[str] = set()

    for sk in resume_skills:
        hit = _match_required(sk, req_lower)
        if hit:
            covered_direct.add(hit)
            continue
        for src, dst in _DEFAULT_TRANSFERABLE.items():
            if src.lower() == sk.lower() or src.lower() in sk.lower():
                covered_xfer.add(dst)
                break

    heatmap: dict[str, str] = {}
    for skill in required:
        if skill in covered_direct:
            heatmap[skill] = "have"
        elif skill in covered_xfer:
            heatmap[skill] = "transferable"
        else:
            heatmap[skill] = "missing"
    return heatmap


def _ats_check(raw_text: str) -> dict:
    """Lightweight ATS signal scan: red-flag phrases and quantified claims."""
    redflags = _ATS_REDFLAG_RE.findall(raw_text or "")
    quant_hits = _QUANT_RE.findall(raw_text or "")
    return {
        "redflag_phrases": list(set(f.lower() for f in redflags)),
        "quantified_claims": len(quant_hits),
        "ats_score": max(0, 100 - len(set(redflags)) * 10),  # rough heuristic
    }


def health_report(
    profile: ResumeProfile,
    target_role: str,
    retr=None,
    *,
    required_skills: Optional[list[str]] = None,
    transferable: Optional[dict[str, str]] = None,
) -> dict:
    """Compute a health report for the resume against a target role.

    Args:
        profile:     Parsed ResumeProfile.
        target_role: Job title string (e.g. "AI Application Engineer").
        retr:        Optional retrieval context (unused in heuristic path; reserved
                     for future evidence-grounded scoring).
        required_skills: Override default required skill list.
        transferable:    Override default transferable map.

    Returns a dict with:
        match_score      : 0-100 overall match
        coverage_heatmap : {skill: have|transferable|missing}
        gaps             : list[SkillGap] (missing + transferable only)
        ats              : ATS signal dict
        summary          : human-readable string
    """
    req = required_skills if required_skills is not None else list(_DEFAULT_REQUIRED_SKILLS)
    gaps = classify_skill_gap(profile, target_role, required_skills=req,
                              transferable=transferable)
    heatmap = _coverage_heatmap(profile, req)

    have_count = sum(1 for v in heatmap.values() if v == "have")
    xfer_count = sum(1 for v in heatmap.values() if v == "transferable")
    total = len(req) or 1
    match_score = round((have_count + xfer_count * 0.5) / total * 100)

    ats = _ats_check(profile.raw_text)

    missing = [g.skill for g in gaps if g.category == SkillGapCategory.missing]
    summary = (
        f"Match {match_score}% for '{target_role}': "
        f"{have_count}/{total} direct, {xfer_count} transferable, "
        f"{len(missing)} missing ({', '.join(missing[:3])}{'...' if len(missing) > 3 else ''})."
    )

    return {
        "match_score": match_score,
        "coverage_heatmap": heatmap,
        "gaps": [g.model_dump() for g in gaps],
        "ats": ats,
        "summary": summary,
    }
