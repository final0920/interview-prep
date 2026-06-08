"""Round definitions + question selection for the mock interview.

``select_question`` is the bridge between the dialog engine and the rest of the
coach: it pulls the candidate's weakpoints from memory, retrieves real project
evidence for the current round's topic, and asks the LLM (via the gateway) to
produce one grounded, design-altitude question. Every path degrades to a
deterministic fallback question so the interview never stalls offline.

The ``retr`` argument is a duck-typed retriever exposing
``search(query, top_k) -> list[RetrievalHit]`` (see coach.interview.deps).
This keeps the engine decoupled from the heavy hybrid_search + embedder + vector
store wiring, which the server/CLI assemble behind that interface.
"""
from __future__ import annotations

import uuid
from typing import Optional, Protocol

from pydantic import BaseModel, Field

from coach.schemas import (
    EvidenceUnit,
    InterviewState,
    Question,
    QuestionType,
    RetrievalHit,
)

# Canonical round order for a full mock interview.
ROUNDS: list[str] = ["tech_basics", "project_deep_dive", "sql", "scenario", "hr"]


class Retriever(Protocol):
    """Minimal retrieval surface the interview engine needs."""

    def search(self, query: str, top_k: int = 8) -> list[RetrievalHit]:
        ...


class MemoryLike(Protocol):
    """Minimal memory surface the interview engine needs."""

    def weakpoints(self) -> list[str]:
        ...


_SELECT_SYS = (
    "你是资深技术面试官兼面试辅导专家。基于候选人真实项目中的技术证据出题, 但必须站在 "
    "'系统设计 / 架构与技术选型 / 高并发·高可用·一致性·容灾等场景应对 / 中间件与框架底层原理(八股) / "
    "目标岗位方向能力' 的高度——这才是真实面试会问的。\n"
    "铁律:\n"
    "1. 禁止问 '这段代码第几行 / 这个方法做了哪几步' 这类实现细节。\n"
    "2. 题目要能引出候选人讲: 整体怎么设计、为什么这么选、做过哪些权衡、踩过什么坑、如何演进优化。\n"
    "3. 若候选人有薄弱项, 优先围绕薄弱项出题以查漏补缺。\n"
    "4. key_points 写该题的得分要点; 只输出 JSON, 不要多余文字。"
)

# Default prompts per round so a missing/failed LLM still yields a sane question.
_FALLBACK_PROMPTS: dict[str, str] = {
    "tech_basics": "讲讲你最熟悉的一个中间件(如 Redis/Kafka/MySQL)的核心原理与你在项目中的取舍。",
    "project_deep_dive": "挑一个你主导的项目, 讲整体架构设计、关键技术选型与你做过的权衡和踩过的坑。",
    "sql": "给定一个高频查询场景, 说说你会如何设计表结构与索引, 并写出核心查询及优化思路。",
    "scenario": "如果线上服务突发流量翻十倍并开始超时, 你会如何定位、止血和长期治理?",
    "hr": "讲讲你最有成就感的一段经历, 以及你为什么选择我们这个目标岗位方向。",
}

_DIFFICULTY_BY_ROUND: dict[str, str] = {
    "tech_basics": "mid",
    "project_deep_dive": "hard",
    "sql": "mid",
    "scenario": "hard",
    "hr": "easy",
}


class _QuestionDraft(BaseModel):
    """Schema the LLM fills for one interview question; mapped to Question."""
    prompt: str = ""
    key_points: list[str] = Field(default_factory=list)
    difficulty: str = "mid"
    tags: list[str] = Field(default_factory=list)


def _round_of(state: InterviewState) -> str:
    rnd = state.round or (state.rounds_remaining[0] if state.rounds_remaining else ROUNDS[0])
    return rnd if rnd in QuestionType.__members__ else "tech_basics"


def _query_for(round_name: str, weakpoints: list[str]) -> str:
    """Build a retrieval query biased toward the round topic + weakpoints."""
    base = _FALLBACK_PROMPTS.get(round_name, round_name)
    if weakpoints:
        return f"{round_name} {' '.join(weakpoints[:3])} {base}"
    return f"{round_name} {base}"


def _evidence_block(hits: list[RetrievalHit], limit: int = 6) -> str:
    lines = []
    for h in hits[:limit]:
        u = h.evidence
        first = (u.text or "").strip().splitlines()
        head = first[0].strip()[:80] if first else ""
        lines.append(f"- {u.ref} <{u.lang or u.channel.value}> {u.symbol} | {head}")
    return "\n".join(lines) or "(候选人暂无与该主题直接相关的项目证据; 作为目标岗位需准备方向出题)"


def _build_user_prompt(round_name: str, weakpoints: list[str], hits: list[RetrievalHit]) -> str:
    wp = ", ".join(weakpoints[:5]) if weakpoints else "(暂无记录)"
    return (
        f"面试轮次: {round_name}\n候选人已知薄弱项: {wp}\n"
        f"候选人项目中与该轮次相关的真实技术证据:\n{_evidence_block(hits)}\n\n"
        "请出 1 道该轮次下真实面试官会问的高质量题, 严格输出 JSON:\n"
        '{"prompt":"题干(设计/架构/场景/原理 高度)", '
        '"key_points":["得分要点1","得分要点2"], '
        '"difficulty":"easy|mid|hard", "tags":["标签"]}'
    )


def _fallback_question(round_name: str, hits: list[RetrievalHit]) -> Question:
    return Question(
        id=f"q-{round_name}-{uuid.uuid4().hex[:8]}",
        type=QuestionType(round_name),
        prompt=_FALLBACK_PROMPTS.get(round_name, f"请围绕『{round_name}』展开你的设计思路与权衡。"),
        difficulty=_DIFFICULTY_BY_ROUND.get(round_name, "mid"),
        linked_evidence=[h.evidence.ref for h in hits[:6]],
        key_points=[],
        tags=[round_name],
        verified=True,
    )


def select_question(
    state: InterviewState,
    retr: Optional[Retriever],
    memory: Optional[MemoryLike],
    gateway,
) -> Question:
    """Pick the next question for the current round, grounded in real evidence.

    Flow: read weakpoints from memory -> retrieve evidence for the round topic
    (biased by weakpoints) -> ask the gateway for a grounded question -> map to
    ``Question`` with linked evidence refs. Any failure (no retriever, no
    gateway, bad JSON) degrades to a deterministic per-round fallback question.
    """
    round_name = _round_of(state)
    weakpoints: list[str] = []
    if memory is not None:
        try:
            weakpoints = list(memory.weakpoints() or [])
        except Exception:
            weakpoints = list(state.weakpoints or [])
    else:
        weakpoints = list(state.weakpoints or [])

    hits: list[RetrievalHit] = []
    if retr is not None:
        try:
            hits = list(retr.search(_query_for(round_name, weakpoints), top_k=6) or [])
        except Exception:
            hits = []

    if gateway is None:
        return _fallback_question(round_name, hits)

    messages = [
        {"role": "system", "content": _SELECT_SYS},
        {"role": "user", "content": _build_user_prompt(round_name, weakpoints, hits)},
    ]
    try:
        draft = gateway.structured(messages, _QuestionDraft)
    except Exception:
        return _fallback_question(round_name, hits)

    prompt = (draft.prompt or "").strip()
    if not prompt:
        return _fallback_question(round_name, hits)
    difficulty = draft.difficulty if draft.difficulty in ("easy", "mid", "hard") else \
        _DIFFICULTY_BY_ROUND.get(round_name, "mid")
    return Question(
        id=f"q-{round_name}-{uuid.uuid4().hex[:8]}",
        type=QuestionType(round_name),
        prompt=prompt,
        difficulty=difficulty,
        linked_evidence=[h.evidence.ref for h in hits[:6]],
        key_points=list(draft.key_points),
        tags=list(draft.tags) or [round_name],
        verified=True,
    )
