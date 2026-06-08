"""Tests for coach/review/{sm2,gap,quality_gate}.

All tests are offline: no LLM, no network, no disk I/O.
SM-2 math is verified against known vectors from the SM-2 spec.
"""
from __future__ import annotations

import math
import time

import pytest

from coach.schemas import (
    AnswerEvaluation,
    ClaimCheck,
    ClaimVerdict,
    Question,
    QuestionType,
    ResumeProfile,
    SkillGap,
    SkillGapCategory,
    Verdict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _q(id: str, tags: list[str] | None = None, qtype: QuestionType = QuestionType.tech_basics,
       linked: list[str] | None = None) -> Question:
    return Question(
        id=id,
        type=qtype,
        prompt=f"Question {id}",
        tags=tags or [],
        linked_evidence=linked or [],
        key_points=["kp1", "kp2"],
        followups=["f1", "f2", "f3"],
    )


def _eval(qid: str, passed: bool) -> AnswerEvaluation:
    return AnswerEvaluation(
        question_id=qid,
        verdict=Verdict.passed if passed else Verdict.needs_fix,
        score=80 if passed else 30,
    )


def _claim(verdict: ClaimVerdict, evidence_ids: list[str] | None = None) -> ClaimCheck:
    return ClaimCheck(
        claim="some claim",
        verdict=verdict,
        evidence_ids=evidence_ids or [],
        score=1.0 if verdict == ClaimVerdict.verified else 0.0,
    )


def _profile(skills: list[str]) -> ResumeProfile:
    return ResumeProfile(skills=skills)


# ---------------------------------------------------------------------------
# sm2.py: SM-2 algorithm math
# ---------------------------------------------------------------------------

class TestSM2:
    """Verify SM-2 math against the SuperMemo-2 specification vectors."""

    def test_new_card_quality5_ef(self):
        """New card, quality=5: EF should increase from 2.5."""
        from coach.review.sm2 import sm2_schedule
        r = sm2_schedule(5, None)
        # EF' = 2.5 + 0.1 - 0*(0.08+0*0.02) = 2.6
        assert abs(r["ef"] - 2.6) < 1e-5

    def test_new_card_quality5_interval(self):
        """New card, quality=5: first interval = 1 day (reps=1)."""
        from coach.review.sm2 import sm2_schedule
        r = sm2_schedule(5, None)
        assert r["interval"] == 1
        assert r["reps"] == 1

    def test_second_correct_interval(self):
        """After first pass (reps=1, interval=1), quality=5 -> interval=6."""
        from coach.review.sm2 import sm2_schedule
        prev = sm2_schedule(5, None)
        r = sm2_schedule(5, prev)
        assert r["interval"] == 6
        assert r["reps"] == 2

    def test_third_correct_interval(self):
        """After two passes (reps=2, interval=6), quality=5 -> interval=round(6*EF)."""
        from coach.review.sm2 import sm2_schedule
        s1 = sm2_schedule(5, None)
        s2 = sm2_schedule(5, s1)
        s3 = sm2_schedule(5, s2)
        assert s3["reps"] == 3
        expected = max(1, round(6 * s3["ef"]))
        assert s3["interval"] == expected

    def test_fail_resets_reps_and_interval(self):
        """quality < 3 resets reps to 0 and interval to 1."""
        from coach.review.sm2 import sm2_schedule
        # First build some history
        s1 = sm2_schedule(5, None)
        s2 = sm2_schedule(5, s1)
        # Now fail
        fail = sm2_schedule(1, s2)
        assert fail["reps"] == 0
        assert fail["interval"] == 1

    def test_ef_decreases_on_low_quality(self):
        """Lower quality scores decrease EF."""
        from coach.review.sm2 import sm2_schedule
        r_q5 = sm2_schedule(5, None)
        r_q3 = sm2_schedule(3, None)
        assert r_q3["ef"] < r_q5["ef"]

    def test_ef_clamped_at_minimum(self):
        """EF never drops below 1.3 regardless of how many failures occur."""
        from coach.review.sm2 import sm2_schedule
        state = None
        for _ in range(20):
            state = sm2_schedule(0, state)
        assert state["ef"] >= 1.3

    def test_quality_3_is_pass_boundary(self):
        """quality=3 is the pass threshold: reps should increment."""
        from coach.review.sm2 import sm2_schedule
        r = sm2_schedule(3, None)
        assert r["reps"] == 1
        assert r["interval"] == 1

    def test_quality_2_is_fail(self):
        """quality=2 is below pass threshold: reps reset to 0."""
        from coach.review.sm2 import sm2_schedule
        r = sm2_schedule(2, None)
        assert r["reps"] == 0
        assert r["interval"] == 1

    def test_due_ts_in_future(self):
        """due_ts should be approximately now + interval*86400."""
        from coach.review.sm2 import sm2_schedule
        before = time.time()
        r = sm2_schedule(4, None)
        after = time.time()
        expected_low = before + r["interval"] * 86400 - 1
        expected_high = after + r["interval"] * 86400 + 1
        assert expected_low <= r["due_ts"] <= expected_high

    def test_quality_clamped_0_to_5(self):
        """Out-of-range quality values are clamped to [0, 5]."""
        from coach.review.sm2 import sm2_schedule
        r_neg = sm2_schedule(-5, None)
        r_high = sm2_schedule(99, None)
        r_0 = sm2_schedule(0, None)
        r_5 = sm2_schedule(5, None)
        assert r_neg["ef"] == r_0["ef"]
        assert r_high["ef"] == r_5["ef"]

    def test_known_vector_q4(self):
        """Verify EF calculation for quality=4 from SM-2 spec.

        EF' = 2.5 + 0.1 - (5-4)*(0.08 + (5-4)*0.02) = 2.5 + 0.1 - 0.10 = 2.5
        """
        from coach.review.sm2 import sm2_schedule
        r = sm2_schedule(4, None)
        assert abs(r["ef"] - 2.5) < 1e-5

    def test_known_vector_q3(self):
        """quality=3: EF' = 2.5 + 0.1 - 2*(0.08+2*0.02) = 2.5 + 0.1 - 0.24 = 2.36."""
        from coach.review.sm2 import sm2_schedule
        r = sm2_schedule(3, None)
        assert abs(r["ef"] - 2.36) < 1e-5

    def test_prev_none_same_as_empty_dict(self):
        from coach.review.sm2 import sm2_schedule
        r1 = sm2_schedule(4, None)
        r2 = sm2_schedule(4, {})
        assert r1["ef"] == r2["ef"]
        assert r1["interval"] == r2["interval"]
        assert r1["reps"] == r2["reps"]

    def test_result_keys_present(self):
        from coach.review.sm2 import sm2_schedule
        r = sm2_schedule(4, None)
        assert set(r.keys()) >= {"ef", "interval", "reps", "due_ts"}


# ---------------------------------------------------------------------------
# gap.py: find_gaps
# ---------------------------------------------------------------------------

class TestFindGaps:
    def test_empty_inputs_return_empty(self):
        from coach.review.gap import find_gaps
        assert find_gaps([], [], [], None) == []

    def test_profile_skill_not_covered_is_missing(self):
        from coach.review.gap import find_gaps
        profile = _profile(["kubernetes"])
        # No questions about kubernetes
        gaps = find_gaps([], [], [], profile)
        assert len(gaps) == 1
        assert gaps[0].category == SkillGapCategory.missing
        assert gaps[0].skill == "kubernetes"

    def test_passed_skill_in_profile_is_have(self):
        from coach.review.gap import find_gaps
        profile = _profile(["python"])
        q = _q("q1", tags=["python"])
        ev = _eval("q1", passed=True)
        gaps = find_gaps([q], [ev], [], profile)
        have = [g for g in gaps if g.category == SkillGapCategory.have]
        assert any(g.skill.lower() == "python" for g in have)

    def test_failed_skill_in_profile_is_missing(self):
        from coach.review.gap import find_gaps
        profile = _profile(["sql"])
        q = _q("q1", tags=["sql"])
        ev = _eval("q1", passed=False)
        gaps = find_gaps([q], [ev], [], profile)
        missing = [g for g in gaps if g.category == SkillGapCategory.missing]
        assert any(g.skill.lower() == "sql" for g in missing)

    def test_passed_not_in_profile_is_transferable(self):
        from coach.review.gap import find_gaps
        profile = _profile(["python"])
        q = _q("q1", tags=["python", "pandas"])
        ev = _eval("q1", passed=True)
        gaps = find_gaps([q], [ev], [], profile)
        transferable = [g for g in gaps if g.category == SkillGapCategory.transferable]
        assert any(g.skill.lower() == "pandas" for g in transferable)

    def test_sort_order_missing_first(self):
        from coach.review.gap import find_gaps
        profile = _profile(["python", "docker"])
        # python question passed, docker not covered
        q = _q("q1", tags=["python"])
        ev = _eval("q1", passed=True)
        gaps = find_gaps([q], [ev], [], profile)
        # missing should come before have
        categories = [g.category for g in gaps]
        missing_idx = categories.index(SkillGapCategory.missing)
        have_idx = categories.index(SkillGapCategory.have)
        assert missing_idx < have_idx

    def test_no_profile_no_profile_skill_gaps(self):
        from coach.review.gap import find_gaps
        q = _q("q1", tags=["python"])
        ev = _eval("q1", passed=True)
        # No profile: no missing/have from profile; only transferable from passed
        gaps = find_gaps([q], [ev], [], None)
        categories = {g.category for g in gaps}
        assert SkillGapCategory.missing not in categories

    def test_evidence_ids_populated(self):
        from coach.review.gap import find_gaps
        profile = _profile(["python"])
        q = _q("q1", tags=["python"], linked=["src/main.py:10", "src/util.py:5"])
        ev = _eval("q1", passed=True)
        gaps = find_gaps([q], [ev], [], profile)
        have = [g for g in gaps if g.category == SkillGapCategory.have][0]
        assert len(have.evidence_ids) > 0

    def test_verified_claim_noted(self):
        from coach.review.gap import find_gaps
        profile = _profile(["python"])
        q = _q("q1", tags=["python"], linked=["eid1"])
        ev = _eval("q1", passed=True)
        c = _claim(ClaimVerdict.verified, evidence_ids=["eid1"])
        gaps = find_gaps([q], [ev], [c], profile)
        have = [g for g in gaps if g.category == SkillGapCategory.have][0]
        assert "verified" in have.note.lower()

    def test_multiple_skills_mixed(self):
        from coach.review.gap import find_gaps
        profile = _profile(["python", "kafka", "docker"])
        # python: passed; kafka: failed; docker: not covered
        q_py = _q("q1", tags=["python"])
        q_kf = _q("q2", tags=["kafka"])
        gaps = find_gaps([q_py, q_kf],
                         [_eval("q1", True), _eval("q2", False)],
                         [], profile)
        cats = {g.skill.lower(): g.category for g in gaps}
        assert cats["python"] == SkillGapCategory.have
        assert cats["kafka"] == SkillGapCategory.missing
        assert cats["docker"] == SkillGapCategory.missing


# ---------------------------------------------------------------------------
# quality_gate.py: quality_report
# ---------------------------------------------------------------------------

class TestQualityReport:
    def _cfg(self, pass_rate=0.6, grounding_rate=0.6, min_q=1) -> dict:
        return {"review": {"thresholds": {
            "pass_rate": pass_rate,
            "grounding_rate": grounding_rate,
            "min_questions": min_q,
        }}}

    def test_empty_inputs_ok(self):
        from coach.review.quality_gate import quality_report
        r = quality_report([], [], [], {})
        assert isinstance(r, dict)
        assert "pass_rate" in r
        assert "redlines" in r

    def test_pass_rate_computed(self):
        from coach.review.quality_gate import quality_report
        qs = [_q("q1"), _q("q2")]
        evs = [_eval("q1", True), _eval("q2", False)]
        r = quality_report(qs, evs, [], self._cfg(pass_rate=0.0))
        assert r["pass_rate"] == 0.5

    def test_grounding_rate_computed(self):
        from coach.review.quality_gate import quality_report
        claims = [
            _claim(ClaimVerdict.verified),
            _claim(ClaimVerdict.verified),
            _claim(ClaimVerdict.needs_evidence),
        ]
        r = quality_report([], [], claims, self._cfg(grounding_rate=0.0))
        assert abs(r["grounding_rate"] - 2 / 3) < 1e-4

    def test_type_distribution(self):
        from coach.review.quality_gate import quality_report
        qs = [
            _q("q1", qtype=QuestionType.tech_basics),
            _q("q2", qtype=QuestionType.tech_basics),
            _q("q3", qtype=QuestionType.sql),
        ]
        r = quality_report(qs, [], [], {})
        assert r["type_dist"]["tech_basics"] == 2
        assert r["type_dist"]["sql"] == 1

    def test_redline_pass_rate(self):
        from coach.review.quality_gate import quality_report
        qs = [_q("q1"), _q("q2")]
        evs = [_eval("q1", False), _eval("q2", False)]  # 0% pass rate
        r = quality_report(qs, evs, [], self._cfg(pass_rate=0.6))
        redline_names = [rd["name"] for rd in r["redlines"]]
        assert "pass_rate" in redline_names
        assert r["ok"] is False

    def test_redline_grounding_rate(self):
        from coach.review.quality_gate import quality_report
        claims = [_claim(ClaimVerdict.needs_evidence)] * 5  # 0% grounding
        r = quality_report([], [], claims, self._cfg(grounding_rate=0.6))
        redline_names = [rd["name"] for rd in r["redlines"]]
        assert "grounding_rate" in redline_names

    def test_redline_min_questions(self):
        from coach.review.quality_gate import quality_report
        r = quality_report([], [], [], self._cfg(min_q=3))
        redline_names = [rd["name"] for rd in r["redlines"]]
        assert "min_questions" in redline_names

    def test_no_redlines_when_thresholds_met(self):
        from coach.review.quality_gate import quality_report
        qs = [_q("q1")]
        evs = [_eval("q1", True)]  # 100% pass
        claims = [_claim(ClaimVerdict.verified)] * 5  # 100% grounding
        r = quality_report(qs, evs, claims, self._cfg(pass_rate=0.6, grounding_rate=0.6, min_q=1))
        assert r["redlines"] == []
        assert r["ok"] is True

    def test_config_threshold_override(self):
        from coach.review.quality_gate import quality_report
        qs = [_q("q1")]
        evs = [_eval("q1", True)]
        # Set pass_rate threshold to 1.0 (impossible to meet with 1/1=100% actually ok,
        # use 0/1 = 0% pass to trigger it)
        evs_fail = [_eval("q1", False)]
        r = quality_report(qs, evs_fail, [], self._cfg(pass_rate=0.5))
        assert r["thresholds"]["pass_rate"] == 0.5
        assert "pass_rate" in [rd["name"] for rd in r["redlines"]]

    def test_no_eval_pass_rate_zero(self):
        from coach.review.quality_gate import quality_report
        qs = [_q("q1")]
        r = quality_report(qs, [], [], self._cfg(pass_rate=0.0))
        assert r["pass_rate"] == 0.0
        assert r["n_evaluated"] == 0

    def test_redline_contains_message(self):
        from coach.review.quality_gate import quality_report
        evs = [_eval("q1", False), _eval("q2", False)]
        qs = [_q("q1"), _q("q2")]
        r = quality_report(qs, evs, [], self._cfg(pass_rate=0.9))
        rd = next(x for x in r["redlines"] if x["name"] == "pass_rate")
        assert isinstance(rd["message"], str) and len(rd["message"]) > 0
