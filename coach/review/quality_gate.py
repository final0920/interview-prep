"""Quality report with redline thresholds.

Public API:
    def quality_report(questions, evals, claims, cfg) -> dict
"""
from __future__ import annotations

from coach.config import get
from coach.schemas import (
    AnswerEvaluation,
    ClaimCheck,
    ClaimVerdict,
    Question,
    QuestionType,
    Verdict,
)

# Default thresholds -- overridden by config keys under review.thresholds.*
_DEFAULTS = {
    "pass_rate": 0.6,
    "grounding_rate": 0.6,
    "min_questions": 1,
}


def quality_report(
    questions: list[Question],
    evals: list[AnswerEvaluation],
    claims: list[ClaimCheck],
    cfg: dict,
) -> dict:
    """Compute quality metrics and flag redlines vs configured thresholds.

    Metrics computed:
        pass_rate       fraction of evaluated questions with verdict=pass
        grounding_rate  fraction of claims with verdict=verified
        type_dist       {QuestionType -> count} distribution over questions
        n_questions     total question count
        n_evaluated     number of questions that have an evaluation
        n_claims        total claim count
        redlines        list of threshold violations with name/value/threshold

    Args:
        questions: list of Question (may be empty).
        evals:     list of AnswerEvaluation (may be shorter than questions).
        claims:    list of ClaimCheck (may be empty).
        cfg:       merged config dict; reads review.thresholds.{pass_rate,
                   grounding_rate, min_questions}.

    Returns plain dict (JSON-serialisable).  Never raises on empty inputs.
    """
    questions = list(questions or [])
    evals = list(evals or [])
    claims = list(claims or [])

    # Thresholds from config, falling back to defaults
    th_pass = float(get(cfg, "review.thresholds.pass_rate", _DEFAULTS["pass_rate"]))
    th_ground = float(get(cfg, "review.thresholds.grounding_rate", _DEFAULTS["grounding_rate"]))
    th_min_q = int(get(cfg, "review.thresholds.min_questions", _DEFAULTS["min_questions"]))

    # --- pass rate ---
    n_evaluated = len(evals)
    n_passed = sum(1 for e in evals if e.verdict == Verdict.passed)
    pass_rate = n_passed / n_evaluated if n_evaluated else 0.0

    # --- grounding rate ---
    n_claims = len(claims)
    n_verified = sum(1 for c in claims if c.verdict == ClaimVerdict.verified)
    grounding_rate = n_verified / n_claims if n_claims else 0.0

    # --- type distribution ---
    type_dist: dict[str, int] = {}
    for q in questions:
        key = q.type.value if isinstance(q.type, QuestionType) else str(q.type)
        type_dist[key] = type_dist.get(key, 0) + 1

    # --- redlines ---
    redlines: list[dict] = []

    if len(questions) < th_min_q:
        redlines.append({
            "name": "min_questions",
            "value": len(questions),
            "threshold": th_min_q,
            "message": f"Only {len(questions)} question(s); minimum is {th_min_q}.",
        })

    if n_evaluated > 0 and pass_rate < th_pass:
        redlines.append({
            "name": "pass_rate",
            "value": round(pass_rate, 4),
            "threshold": th_pass,
            "message": (
                f"Pass rate {pass_rate:.1%} is below threshold {th_pass:.1%}. "
                f"{n_passed}/{n_evaluated} questions passed."
            ),
        })

    if n_claims > 0 and grounding_rate < th_ground:
        redlines.append({
            "name": "grounding_rate",
            "value": round(grounding_rate, 4),
            "threshold": th_ground,
            "message": (
                f"Grounding rate {grounding_rate:.1%} is below threshold {th_ground:.1%}. "
                f"{n_verified}/{n_claims} claims verified."
            ),
        })

    return {
        "n_questions": len(questions),
        "n_evaluated": n_evaluated,
        "n_claims": n_claims,
        "pass_rate": round(pass_rate, 4),
        "grounding_rate": round(grounding_rate, 4),
        "type_dist": type_dist,
        "thresholds": {
            "pass_rate": th_pass,
            "grounding_rate": th_ground,
            "min_questions": th_min_q,
        },
        "redlines": redlines,
        "ok": len(redlines) == 0,
    }
