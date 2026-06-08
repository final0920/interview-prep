"""Skill gap analysis: compare question coverage vs evaluation outcomes vs profile.

Public API:
    def find_gaps(questions, evals, claims, profile) -> list[SkillGap]
"""
from __future__ import annotations

from typing import Optional

from coach.schemas import (
    AnswerEvaluation,
    ClaimCheck,
    ClaimVerdict,
    Question,
    ResumeProfile,
    SkillGap,
    SkillGapCategory,
    Verdict,
)


def find_gaps(
    questions: list[Question],
    evals: list[AnswerEvaluation],
    claims: list[ClaimCheck],
    profile: Optional[ResumeProfile],
) -> list[SkillGap]:
    """Identify skill gaps from interview evidence.

    Three gap sources:

    1. missing  -- skills listed in profile.skills that have no question
                   or whose questions were not passed (verdict != pass).
    2. transferable -- skills that appeared in questions and were passed but
                       are not in profile.skills (candidate knows them but
                       hasn't claimed them on the resume).
    3. have     -- skills in profile.skills that have at least one passing
                   evaluation and at least one verified claim.

    Args:
        questions: list of Question objects from the interview session.
        evals:     list of AnswerEvaluation objects (may be shorter than questions
                   if the session is incomplete).
        claims:    list of ClaimCheck objects from claim_check.
        profile:   parsed ResumeProfile (may be None if no resume was loaded).

    Returns list[SkillGap] sorted: missing first, then transferable, then have.
    All inputs may be empty/None without raising.
    """
    questions = list(questions or [])
    evals = list(evals or [])
    claims = list(claims or [])
    profile_skills: list[str] = list((profile.skills if profile else None) or [])

    # Build lookup: question_id -> evaluation
    eval_by_qid: dict[str, AnswerEvaluation] = {e.question_id: e for e in evals}

    # Collect tags covered by passing evaluations
    passed_tags: set[str] = set()
    failed_tags: set[str] = set()
    evidence_by_tag: dict[str, list[str]] = {}

    for q in questions:
        ev = eval_by_qid.get(q.id)
        passed = ev is not None and ev.verdict == Verdict.passed
        for tag in q.tags:
            tag_l = tag.lower()
            if passed:
                passed_tags.add(tag_l)
            else:
                failed_tags.add(tag_l)
            evidence_by_tag.setdefault(tag_l, [])
            evidence_by_tag[tag_l].extend(q.linked_evidence)

    # Collect verified claims by evidence_id for cross-referencing
    verified_evidence: set[str] = set()
    for c in claims:
        if c.verdict == ClaimVerdict.verified:
            verified_evidence.update(c.evidence_ids)

    profile_skill_lower = {s.lower(): s for s in profile_skills}

    gaps: list[SkillGap] = []

    # --- missing: in profile but not confidently passed ---
    for skill_l, skill in profile_skill_lower.items():
        if skill_l in passed_tags:
            # Check whether any linked evidence is also claim-verified
            ev_ids = evidence_by_tag.get(skill_l, [])
            has_verified = any(e in verified_evidence for e in ev_ids)
            if has_verified:
                # Well-evidenced pass -> "have"
                gaps.append(SkillGap(
                    skill=skill,
                    category=SkillGapCategory.have,
                    evidence_ids=ev_ids[:5],
                    note="Demonstrated with verified evidence.",
                ))
            else:
                # Passed but no verified claim -> still "have" (softer)
                gaps.append(SkillGap(
                    skill=skill,
                    category=SkillGapCategory.have,
                    evidence_ids=ev_ids[:5],
                    note="Passed in evaluation.",
                ))
        else:
            gaps.append(SkillGap(
                skill=skill,
                category=SkillGapCategory.missing,
                evidence_ids=[],
                note=(
                    "Not demonstrated or failed in evaluation."
                    if skill_l in failed_tags
                    else "No question covered this skill."
                ),
            ))

    # --- transferable: passed in interview but not in profile skills ---
    for tag_l in passed_tags:
        if tag_l not in profile_skill_lower:
            ev_ids = evidence_by_tag.get(tag_l, [])
            gaps.append(SkillGap(
                skill=tag_l,
                category=SkillGapCategory.transferable,
                evidence_ids=ev_ids[:5],
                note="Demonstrated in interview but not listed in resume.",
            ))

    # Stable sort: missing first, transferable second, have last
    _ORDER = {
        SkillGapCategory.missing: 0,
        SkillGapCategory.transferable: 1,
        SkillGapCategory.have: 2,
    }
    gaps.sort(key=lambda g: (_ORDER[g.category], g.skill.lower()))
    return gaps
