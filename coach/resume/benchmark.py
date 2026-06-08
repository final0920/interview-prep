"""Resume benchmark: parse a directory of competitor PDFs/text files,
build per-file ResumeProfiles, and compare against the user profile.

Public API:
    def benchmark_competitors(resume_dir, cfg) -> dict
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from coach.schemas import ResumeProfile

from coach.resume.parse import parse_resume
from coach.resume.analyze import (
    health_report,
    _DEFAULT_REQUIRED_SKILLS,
    _DEFAULT_TRANSFERABLE,
    _all_skills,
)
from coach.config import get


def _load_user_profile(cfg: dict) -> Optional[ResumeProfile]:
    """Try to load the user's own profile from the configured path."""
    import json
    profile_path = get(cfg, "paths.resume_profile")
    if not profile_path:
        return None
    p = Path(profile_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return ResumeProfile(**data)
    except Exception:
        return None


def _score_profile(profile: ResumeProfile, target_role: str, cfg: dict) -> dict:
    """Run health_report on a single profile and return a compact summary."""
    req = get(cfg, "resume.required_skills", _DEFAULT_REQUIRED_SKILLS)
    xfer = get(cfg, "resume.transferable", _DEFAULT_TRANSFERABLE)
    report = health_report(profile, target_role,
                           required_skills=req, transferable=xfer)
    skills = _all_skills(profile)
    return {
        "match_score": report["match_score"],
        "coverage_heatmap": report["coverage_heatmap"],
        "skill_count": len(skills),
        "skills": skills[:20],          # first 20 for display
        "experience_count": len(profile.experiences),
        "project_count": len(profile.projects),
        "ats": report["ats"],
        "gaps": report["gaps"],
    }


def benchmark_competitors(resume_dir: str, cfg: dict) -> dict:
    """Parse every resume in resume_dir, score them, and compare with the user.

    Supports .pdf files (via PyMuPDF) and plain-text files (.txt / no extension).
    Directories and unsupported files are skipped silently.

    Args:
        resume_dir: Path to a directory containing competitor resume files.
        cfg:        Loaded config dict.

    Returns a dict:
        target_role      : role from cfg or empty string
        competitors      : [{name, match_score, skill_count, ...}, ...]
        user             : the user's own scores (or None if no profile configured)
        gap_vs_best      : skills the user is missing vs the top-scoring competitor
        summary          : human-readable string
    """
    target_role: str = get(cfg, "run.target_role", "")

    src = Path(resume_dir)
    if not src.is_dir():
        return {
            "target_role": target_role,
            "competitors": [],
            "user": None,
            "gap_vs_best": [],
            "summary": f"resume_dir not found: {resume_dir}",
        }

    # Collect candidate files
    _SUPPORTED_SUFFIXES = {".pdf", ".txt", ""}
    candidates: list[Path] = [
        f for f in src.iterdir()
        if f.is_file() and f.suffix.lower() in _SUPPORTED_SUFFIXES
    ]

    competitors: list[dict] = []
    for fpath in sorted(candidates):
        try:
            profile = parse_resume(str(fpath))
            scores = _score_profile(profile, target_role, cfg)
            scores["name"] = fpath.stem
            competitors.append(scores)
        except Exception as exc:
            # skip files that fail to parse
            competitors.append({
                "name": fpath.stem,
                "match_score": 0,
                "error": f"{type(exc).__name__}: {exc}",
            })

    # Sort competitors by match_score descending
    competitors.sort(key=lambda c: c.get("match_score", 0), reverse=True)

    # User profile comparison
    user_profile = _load_user_profile(cfg)
    user_scores: Optional[dict] = None
    gap_vs_best: list[str] = []

    if user_profile is not None:
        user_scores = _score_profile(user_profile, target_role, cfg)
        user_scores["name"] = "user"

        if competitors:
            best = competitors[0]
            best_skills = set(best.get("skills", []))
            user_skills = set(_all_skills(user_profile))
            gap_vs_best = sorted(best_skills - user_skills)

    n = len(competitors)
    top_score = competitors[0]["match_score"] if competitors else 0
    user_score = user_scores["match_score"] if user_scores else None
    summary = (
        f"Benchmarked {n} competitor(s) for '{target_role}'. "
        f"Top competitor score: {top_score}%. "
        + (f"Your score: {user_score}%." if user_score is not None else "No user profile found.")
    )

    return {
        "target_role": target_role,
        "competitors": competitors,
        "user": user_scores,
        "gap_vs_best": gap_vs_best,
        "summary": summary,
    }
