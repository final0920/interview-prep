"""Export study materials: Markdown study book + Anki-importable CSV.

Public API (per DESIGN.md sec 6.T7):
    export_study_book(questions, evals, out_dir) -> Path   # study_book.md
    export_anki(questions, evals, out_dir) -> Path         # anki.csv  (RFC4180)

Both functions degrade gracefully when evals is None or incomplete:
missing evaluations are simply omitted rather than raising.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from coach.schemas import AnswerEvaluation, Question


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_list(v: object) -> list[str]:
    """Coerce a field value to a clean list of non-empty strings."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    return [s] if s else []


def _eval_index(evals: Optional[list[AnswerEvaluation]]) -> dict[str, AnswerEvaluation]:
    """Build {question_id -> AnswerEvaluation} lookup; returns empty dict for None."""
    if not evals:
        return {}
    return {e.question_id: e for e in evals}


# ---------------------------------------------------------------------------
# Markdown study book
# ---------------------------------------------------------------------------

def _question_block(num: int, q: Question, ev: Optional[AnswerEvaluation]) -> str:
    """Render one question as a Markdown section string."""
    lines: list[str] = []

    # Section header: number, prompt, type badge
    header = f"## {num}. {q.prompt.strip()}"
    if q.type:
        header += f"  `{q.type.value}`"
    if q.difficulty:
        header += f"  `{q.difficulty}`"
    lines.append(header)
    lines.append("")

    # Reference answer
    lines.append("### Reference Answer")
    lines.append(q.reference_answer.strip() if q.reference_answer else "(none)")
    lines.append("")

    # Key points
    kps = _as_list(q.key_points)
    lines.append("### Key Points")
    if kps:
        lines.extend(f"- {x}" for x in kps)
    else:
        lines.append("(none)")
    lines.append("")

    # Follow-ups
    fus = _as_list(q.followups)
    lines.append("### Follow-up Questions")
    if fus:
        lines.extend(f"{i}. {x}" for i, x in enumerate(fus, 1))
    else:
        lines.append("(none)")
    lines.append("")

    # Evaluation section (omitted when no eval available)
    if ev is not None:
        lines.append("### Evaluation")
        lines.append(f"- **verdict**: {ev.verdict.value}  score: {ev.score}/100")
        if ev.grounding_rate:
            lines.append(f"- grounding rate: {ev.grounding_rate:.0%}")
        kph = _as_list(ev.key_points_hit)
        if kph:
            lines.append("- key points hit:")
            lines.extend(f"  - {x}" for x in kph)
        issues = _as_list(ev.issues)
        if issues:
            lines.append("- issues:")
            lines.extend(f"  - {x}" for x in issues)
        flags = _as_list(ev.fabrication_flags)
        if flags:
            lines.append("- fabrication flags:")
            lines.extend(f"  - {x}" for x in flags)
        if ev.followup:
            lines.append(f"- suggested followup: {ev.followup}")
        lines.append("")

    return "\n".join(lines)


def export_study_book(
    questions: list[Question],
    evals: Optional[list[AnswerEvaluation]],
    out_dir: str | Path,
) -> Path:
    """Write study_book.md to out_dir; return the Path.

    Degrades gracefully when evals is None or shorter than questions.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "study_book.md"

    ev_idx = _eval_index(evals)
    n = len(questions)
    n_eval = sum(1 for q in questions if q.id in ev_idx)

    header_lines: list[str] = [
        "# Interview Study Book",
        "",
        f"> {n} question(s)" + (f" · {n_eval} with evaluation" if ev_idx else " · no evaluations attached"),
        "",
        "---",
        "",
    ]

    blocks: list[str] = []
    for i, q in enumerate(questions, 1):
        ev = ev_idx.get(q.id)
        blocks.append(_question_block(i, q, ev))
        blocks.append("---")
        blocks.append("")

    content = "\n".join(header_lines) + "\n".join(blocks)
    md_path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# Anki CSV (RFC4180)
# ---------------------------------------------------------------------------

def _anki_back(q: Question, ev: Optional[AnswerEvaluation]) -> str:
    """Build the back-of-card text: answer + key points + followups + verdict."""
    segs: list[str] = []

    ans = (q.reference_answer or "").strip()
    if ans:
        segs.append("[Answer]\n" + ans)

    kps = _as_list(q.key_points)
    if kps:
        segs.append("[Key Points]\n" + "\n".join(f"- {x}" for x in kps))

    fus = _as_list(q.followups)
    if fus:
        segs.append("[Follow-ups]\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(fus, 1)))

    if ev is not None:
        segs.append(f"[Verdict] {ev.verdict.value}  score:{ev.score}/100")

    return "\n\n".join(segs)


def _anki_tags(q: Question, ev: Optional[AnswerEvaluation]) -> str:
    """Build Anki tag string (space-separated; no spaces inside a tag)."""
    tags: list[str] = []
    if q.type:
        tags.append("type:" + q.type.value.replace(" ", "_"))
    if q.difficulty:
        tags.append("diff:" + q.difficulty.replace(" ", "_"))
    if ev is not None:
        tags.append("verdict:" + ev.verdict.value.replace(" ", "_"))
    return " ".join(tags)


def export_anki(
    questions: list[Question],
    evals: Optional[list[AnswerEvaluation]],
    out_dir: str | Path,
) -> Path:
    """Write anki.csv (front/back/tags, RFC4180) to out_dir; return the Path.

    csv.writer with QUOTE_MINIMAL handles comma/newline/quote escaping
    automatically; newline="" lets the csv module control line endings.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "anki.csv"

    ev_idx = _eval_index(evals)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["front", "back", "tags"])
        for q in questions:
            ev = ev_idx.get(q.id)
            front = q.prompt.strip()
            back = _anki_back(q, ev)
            tags = _anki_tags(q, ev)
            w.writerow([front, back, tags])

    return csv_path
