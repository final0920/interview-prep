"""Adversarial follow-up generation.

After an answer is scored, the engine may probe deeper. ``make_followup``
produces one sharp, grounded follow-up question: it prefers the evaluator's own
suggested follow-up (already adversarial and context-aware), otherwise asks the
gateway to generate one anchored in the candidate's real project evidence so the
probe cannot be deflected with hand-waving. Falls back to a generic-but-pointed
probe when no gateway/evidence is available.
"""
from __future__ import annotations

from coach.schemas import Question, RetrievalHit

_PROBE_SYS = (
    "你是极其挑剔的资深技术面试官。针对候选人刚才的回答, 提出 1 个最尖锐、最能暴露其薄弱处的追问。\n"
    "要求:\n"
    "1. 追问必须层层递进(设计动机 -> 取舍 -> 极端场景 -> 优化演进), 紧扣其回答的薄弱点。\n"
    "2. 尽量结合候选人项目里的真实技术细节, 让其无法用空泛套话蒙混。\n"
    "3. 只输出追问本身一句话, 不要解释、不要 JSON、不要多余文字。"
)


def _evidence_block(evidence: list[RetrievalHit], limit: int = 5) -> str:
    lines = []
    for h in evidence[:limit]:
        u = h.evidence
        first = (u.text or "").strip().splitlines()
        head = first[0].strip()[:80] if first else ""
        lines.append(f"- {u.ref} | {head}")
    return "\n".join(lines) or "(无可用项目证据)"


def make_followup(
    question: Question,
    answer: str,
    evidence: list[RetrievalHit],
    gateway,
) -> str:
    """Return one adversarial follow-up question grounded in real evidence.

    Uses the gateway to generate a probe tied to ``question``, the candidate's
    ``answer``, and the retrieved ``evidence``. Returns a stripped single line.
    On any failure returns a generic-but-pointed probe so the engine can still
    advance. Never raises.
    """
    if gateway is None:
        return _generic_probe(question)

    user = (
        f"面试题:\n{question.prompt}\n\n"
        f"候选人的回答:\n{answer or '(空)'}\n\n"
        f"候选人项目中的相关真实证据(追问尽量结合这些细节):\n{_evidence_block(evidence)}\n\n"
        "请给出 1 个最尖锐的追问(一句话):"
    )
    try:
        out = gateway.complete([
            {"role": "system", "content": _PROBE_SYS},
            {"role": "user", "content": user},
        ])
    except Exception:
        return _generic_probe(question)

    text = (out or "").strip()
    # tolerate models that wrap the line in quotes or a leading bullet/label
    text = text.strip('"“”').lstrip("-* ").strip()
    if text.startswith("追问"):
        # drop a leading "追问:" / "追问1:" label if present
        idx = text.find(":")
        idx2 = text.find("：")
        cut = max(idx, idx2)
        if 0 <= cut <= 6:
            text = text[cut + 1:].strip()
    return text or _generic_probe(question)


def _generic_probe(question: Question) -> str:
    """A pointed fallback probe keyed off the question type."""
    qt = question.type.value
    if qt == "sql":
        return "如果这张表的数据量增长到上亿行, 你这条 SQL 会遇到什么瓶颈, 如何优化?"
    if qt == "hr":
        return "能具体讲讲那次经历中你个人做出的关键决策, 以及如果重来你会怎么做?"
    return "在更极端的并发或故障场景下, 你刚才的方案会在哪里先崩溃? 你会如何演进它?"
