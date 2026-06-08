"""Offline tests for coach/review/export.py.

Covers:
- study_book.md generated from sample Question/AnswerEvaluation
- anki.csv generated with RFC4180 quote-escaping
- graceful degradation when evals is None or missing entries
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from coach.schemas import (
    AnswerEvaluation,
    Question,
    QuestionType,
    Verdict,
)
from coach.review.export import export_study_book, export_anki


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _q(qid: str, prompt: str, ref: str = "ref answer", **kw) -> Question:
    defaults = dict(
        id=qid,
        type=QuestionType.tech_basics,
        prompt=prompt,
        reference_answer=ref,
        key_points=["point A", "point B"],
        followups=["followup 1", "followup 2"],
        difficulty="mid",
    )
    defaults.update(kw)
    return Question(**defaults)


def _ev(qid: str, score: int = 80, verdict: Verdict = Verdict.passed) -> AnswerEvaluation:
    return AnswerEvaluation(
        question_id=qid,
        user_answer="user said something",
        score=score,
        verdict=verdict,
        key_points_hit=["point A"],
        issues=["missed point B"],
        fabrication_flags=[],
        followup="Can you elaborate?",
        grounding_rate=0.75,
    )


@pytest.fixture
def sample_questions():
    return [
        _q("q1", "Explain Python GIL", ref="The GIL is a mutex..."),
        _q("q2", "What is a deadlock?", ref="Circular wait condition...",
           key_points=["hold and wait", "circular wait"],
           followups=["How to prevent?"],
           type=QuestionType.scenario),
    ]


@pytest.fixture
def sample_evals():
    return [_ev("q1", score=85), _ev("q2", score=60, verdict=Verdict.needs_fix)]


# ---------------------------------------------------------------------------
# export_study_book
# ---------------------------------------------------------------------------

class TestExportStudyBook:
    def test_creates_file(self, tmp_path, sample_questions, sample_evals):
        p = export_study_book(sample_questions, sample_evals, tmp_path)
        assert p.exists()
        assert p.name == "study_book.md"

    def test_contains_question_prompts(self, tmp_path, sample_questions, sample_evals):
        p = export_study_book(sample_questions, sample_evals, tmp_path)
        text = p.read_text(encoding="utf-8")
        assert "Explain Python GIL" in text
        assert "What is a deadlock?" in text

    def test_contains_reference_answers(self, tmp_path, sample_questions, sample_evals):
        p = export_study_book(sample_questions, sample_evals, tmp_path)
        text = p.read_text(encoding="utf-8")
        assert "The GIL is a mutex" in text
        assert "Circular wait condition" in text

    def test_contains_key_points(self, tmp_path, sample_questions, sample_evals):
        p = export_study_book(sample_questions, sample_evals, tmp_path)
        text = p.read_text(encoding="utf-8")
        assert "point A" in text
        assert "point B" in text
        assert "hold and wait" in text

    def test_contains_followups(self, tmp_path, sample_questions, sample_evals):
        p = export_study_book(sample_questions, sample_evals, tmp_path)
        text = p.read_text(encoding="utf-8")
        assert "followup 1" in text
        assert "How to prevent?" in text

    def test_contains_verdict(self, tmp_path, sample_questions, sample_evals):
        p = export_study_book(sample_questions, sample_evals, tmp_path)
        text = p.read_text(encoding="utf-8")
        assert "pass" in text
        assert "needs_fix" in text

    def test_contains_score(self, tmp_path, sample_questions, sample_evals):
        p = export_study_book(sample_questions, sample_evals, tmp_path)
        text = p.read_text(encoding="utf-8")
        assert "85" in text
        assert "60" in text

    def test_contains_issues(self, tmp_path, sample_questions, sample_evals):
        p = export_study_book(sample_questions, sample_evals, tmp_path)
        text = p.read_text(encoding="utf-8")
        assert "missed point B" in text

    def test_graceful_no_evals(self, tmp_path, sample_questions):
        p = export_study_book(sample_questions, None, tmp_path)
        assert p.exists()
        text = p.read_text(encoding="utf-8")
        # Questions still present
        assert "Explain Python GIL" in text
        # No eval section header
        assert "Evaluation" not in text

    def test_graceful_partial_evals(self, tmp_path, sample_questions):
        # Only eval for q1, not q2
        evals = [_ev("q1")]
        p = export_study_book(sample_questions, evals, tmp_path)
        text = p.read_text(encoding="utf-8")
        # q1 has eval, q2 does not raise
        assert "What is a deadlock?" in text

    def test_header_shows_question_count(self, tmp_path, sample_questions, sample_evals):
        p = export_study_book(sample_questions, sample_evals, tmp_path)
        text = p.read_text(encoding="utf-8")
        assert "2 question" in text

    def test_header_no_evals_note(self, tmp_path, sample_questions):
        p = export_study_book(sample_questions, None, tmp_path)
        text = p.read_text(encoding="utf-8")
        assert "no evaluations" in text

    def test_question_type_in_output(self, tmp_path, sample_questions, sample_evals):
        p = export_study_book(sample_questions, sample_evals, tmp_path)
        text = p.read_text(encoding="utf-8")
        assert "tech_basics" in text
        assert "scenario" in text

    def test_out_dir_created_if_missing(self, tmp_path, sample_questions):
        new_dir = tmp_path / "subdir" / "nested"
        p = export_study_book(sample_questions, None, new_dir)
        assert p.exists()

    def test_empty_questions_list(self, tmp_path):
        p = export_study_book([], None, tmp_path)
        assert p.exists()
        text = p.read_text(encoding="utf-8")
        assert "0 question" in text

    def test_question_with_no_key_points(self, tmp_path):
        q = _q("q1", "Simple question", key_points=[], followups=[])
        p = export_study_book([q], None, tmp_path)
        text = p.read_text(encoding="utf-8")
        assert "(none)" in text


# ---------------------------------------------------------------------------
# export_anki
# ---------------------------------------------------------------------------

class TestExportAnki:
    def _read_csv(self, path: Path) -> list[dict]:
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def test_creates_file(self, tmp_path, sample_questions, sample_evals):
        p = export_anki(sample_questions, sample_evals, tmp_path)
        assert p.exists()
        assert p.name == "anki.csv"

    def test_has_header_row(self, tmp_path, sample_questions, sample_evals):
        p = export_anki(sample_questions, sample_evals, tmp_path)
        with p.open(encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["front", "back", "tags"]

    def test_row_count_matches_questions(self, tmp_path, sample_questions, sample_evals):
        p = export_anki(sample_questions, sample_evals, tmp_path)
        rows = self._read_csv(p)
        assert len(rows) == len(sample_questions)

    def test_front_is_prompt(self, tmp_path, sample_questions, sample_evals):
        p = export_anki(sample_questions, sample_evals, tmp_path)
        rows = self._read_csv(p)
        assert rows[0]["front"] == "Explain Python GIL"
        assert rows[1]["front"] == "What is a deadlock?"

    def test_back_contains_reference_answer(self, tmp_path, sample_questions, sample_evals):
        p = export_anki(sample_questions, sample_evals, tmp_path)
        rows = self._read_csv(p)
        assert "The GIL is a mutex" in rows[0]["back"]

    def test_back_contains_key_points(self, tmp_path, sample_questions, sample_evals):
        p = export_anki(sample_questions, sample_evals, tmp_path)
        rows = self._read_csv(p)
        assert "point A" in rows[0]["back"]

    def test_back_contains_followups(self, tmp_path, sample_questions, sample_evals):
        p = export_anki(sample_questions, sample_evals, tmp_path)
        rows = self._read_csv(p)
        assert "followup 1" in rows[0]["back"]

    def test_back_contains_verdict_when_eval_present(self, tmp_path, sample_questions, sample_evals):
        p = export_anki(sample_questions, sample_evals, tmp_path)
        rows = self._read_csv(p)
        assert "pass" in rows[0]["back"]
        assert "needs_fix" in rows[1]["back"]

    def test_tags_contain_type(self, tmp_path, sample_questions, sample_evals):
        p = export_anki(sample_questions, sample_evals, tmp_path)
        rows = self._read_csv(p)
        assert "type:tech_basics" in rows[0]["tags"]
        assert "type:scenario" in rows[1]["tags"]

    def test_tags_contain_verdict(self, tmp_path, sample_questions, sample_evals):
        p = export_anki(sample_questions, sample_evals, tmp_path)
        rows = self._read_csv(p)
        assert "verdict:pass" in rows[0]["tags"]
        assert "verdict:needs_fix" in rows[1]["tags"]

    def test_tags_no_spaces_inside(self, tmp_path, sample_questions, sample_evals):
        """Each individual tag must not contain spaces (Anki requirement)."""
        p = export_anki(sample_questions, sample_evals, tmp_path)
        rows = self._read_csv(p)
        for row in rows:
            for tag in row["tags"].split():
                assert " " not in tag

    def test_graceful_no_evals(self, tmp_path, sample_questions):
        p = export_anki(sample_questions, None, tmp_path)
        rows = self._read_csv(p)
        assert len(rows) == len(sample_questions)
        # No verdict tags without evals
        for row in rows:
            assert "verdict:" not in row["tags"]

    def test_rfc4180_comma_in_field_is_quoted(self, tmp_path):
        """Fields containing commas must be double-quoted (RFC4180)."""
        q = _q("q1", "Explain A, B, and C", ref="Answer with, commas, inside")
        p = export_anki([q], None, tmp_path)
        raw = p.read_text(encoding="utf-8")
        # csv module wraps fields with commas in double quotes
        assert '"' in raw

    def test_rfc4180_newline_in_field_is_quoted(self, tmp_path):
        """Fields containing newlines must be double-quoted (RFC4180)."""
        q = _q("q1", "Question\nwith newline", ref="Answer\nwith newline")
        p = export_anki([q], None, tmp_path)
        raw = p.read_text(encoding="utf-8")
        assert '"' in raw
        # Parsed back correctly via csv reader
        rows = self._read_csv(p)
        assert "Question\nwith newline" in rows[0]["front"]

    def test_rfc4180_double_quote_escaped(self, tmp_path):
        """Embedded double-quotes must be escaped as two double-quotes (RFC4180)."""
        q = _q("q1", 'He said "hello"', ref='Answer with "quotes"')
        p = export_anki([q], None, tmp_path)
        rows = self._read_csv(p)
        assert 'He said "hello"' in rows[0]["front"]
        assert 'Answer with "quotes"' in rows[0]["back"]

    def test_out_dir_created_if_missing(self, tmp_path, sample_questions):
        new_dir = tmp_path / "deep" / "nested"
        p = export_anki(sample_questions, None, new_dir)
        assert p.exists()

    def test_empty_questions_list(self, tmp_path):
        p = export_anki([], None, tmp_path)
        rows = self._read_csv(p)
        assert rows == []
