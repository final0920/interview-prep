"""Offline tests for the interview engine (coach.interview).

No network, no real LLM: a FakeGateway returns canned structured judgements and
canned followups; a FakeRetriever returns fixed evidence; an in-memory fake
memory supplies weakpoints + records review episodes. The langgraph checkpointer
is InMemorySaver (SqliteSaver not installed). Covers: graph builds, question
selection grounded in evidence, a full turn yields a scored AnswerEvaluation +
grounded followup, followup budgeting, full multi-round run, and checkpoint
resume after a simulated restart (new compiled graph, same saver + thread_id).
"""
from __future__ import annotations

from typing import Any, Optional

import pytest
from pydantic import BaseModel

from coach.interview import graph as gmod
from coach.interview.graph import Deps, build_graph, start_session, step
from coach.interview.prober import make_followup
from coach.interview.rounds import ROUNDS, select_question
from coach.schemas import (
    AnswerEvaluation,
    EvidenceUnit,
    InterviewState,
    MemoryEpisode,
    Question,
    QuestionType,
    RetrievalHit,
    Verdict,
)


# --- fakes ------------------------------------------------------------------

def _ev(eid: str, symbol: str, text: str) -> EvidenceUnit:
    return EvidenceUnit(id=eid, source_path="svc.py", symbol=symbol, start_line=10, text=text)


class FakeRetriever:
    def __init__(self):
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 8) -> list[RetrievalHit]:
        self.queries.append(query)
        return [
            RetrievalHit(evidence=_ev("e1", "function_definition:consume",
                                      "Kafka 幂等 消费 offset 手动 提交 重试")),
            RetrievalHit(evidence=_ev("e2", "class_definition:RateLimiter",
                                      "限流 降级 熔断 令牌桶 滑动窗口")),
        ][:top_k]


class FakeMemory:
    def __init__(self, weak: Optional[list[str]] = None):
        self._weak = weak or ["分布式一致性", "索引优化"]
        self.episodes: list[MemoryEpisode] = []

    def weakpoints(self) -> list[str]:
        return list(self._weak)

    def add_episode(self, ep: MemoryEpisode) -> str:
        self.episodes.append(ep)
        return "ep-1"


class FakeGateway:
    """structured() -> canned judgement; complete() -> canned followup."""

    def __init__(self, *, score: int = 75, verdict: str = "pass",
                 question_prompt: str = "讲讲你的消息队列高可用设计与权衡。"):
        self.score = score
        self.verdict = verdict
        self.question_prompt = question_prompt
        self.structured_calls = 0
        self.complete_calls = 0

    def structured(self, messages, schema: type[BaseModel], **kw):
        self.structured_calls += 1
        # rounds._QuestionDraft has a 'prompt'; calibrate._JudgeOut has 'verdict'
        fields = set(schema.model_fields.keys())
        if "prompt" in fields and "verdict" not in fields:
            return schema.model_validate({
                "prompt": self.question_prompt,
                "key_points": ["副本机制", "幂等消费"],
                "difficulty": "hard",
                "tags": ["kafka"],
            })
        return schema.model_validate({
            "score": self.score, "verdict": self.verdict,
            "key_points_hit": ["幂等消费"], "issues": ["未提副本数"],
            "fabrication_flags": [], "followup": "如何保证 exactly-once 语义?",
        })

    def complete(self, messages, **kw):
        self.complete_calls += 1
        return "结合你项目里 offset 的提交时机, 说说消费者宕机重放时如何避免重复消费?"


def _deps(gw=None, retr=None, mem=None, max_followups=2, saver=None) -> Deps:
    return Deps(gateway=gw or FakeGateway(), retriever=retr or FakeRetriever(),
                memory=mem or FakeMemory(), checkpointer=saver, max_followups=max_followups)


# --- rounds.select_question -------------------------------------------------

def test_select_question_grounded_in_evidence():
    gw, retr, mem = FakeGateway(), FakeRetriever(), FakeMemory()
    st = InterviewState(round="project_deep_dive", rounds_remaining=["project_deep_dive"])
    q = select_question(st, retr, mem, gw)
    assert isinstance(q, Question)
    assert q.type == QuestionType.project_deep_dive
    assert q.prompt
    assert q.linked_evidence  # refs attached from retrieved evidence
    assert "svc.py" in q.linked_evidence[0]
    # weakpoints biased the retrieval query
    assert any("分布式一致性" in qq for qq in retr.queries)


def test_select_question_fallback_without_gateway():
    q = select_question(InterviewState(round="sql", rounds_remaining=["sql"]),
                        FakeRetriever(), FakeMemory(), gateway=None)
    assert q.type == QuestionType.sql
    assert q.prompt  # deterministic fallback prompt


def test_select_question_degrades_on_gateway_error():
    class BoomGW:
        def structured(self, *a, **k):
            raise RuntimeError("down")
    q = select_question(InterviewState(round="hr", rounds_remaining=["hr"]),
                        FakeRetriever(), FakeMemory(), BoomGW())
    assert q.type == QuestionType.hr and q.prompt


# --- prober.make_followup ---------------------------------------------------

def test_make_followup_uses_gateway():
    q = Question(id="q1", type=QuestionType.tech_basics, prompt="Redis 持久化")
    out = make_followup(q, "我用了 RDB", [], FakeGateway())
    assert out and "重复消费" in out


def test_make_followup_strips_label_and_quotes():
    class GW:
        def complete(self, m, **k):
            return '"追问1: 你如何处理主从延迟?"'
    out = make_followup(Question(id="q", type=QuestionType.scenario, prompt="x"), "a", [], GW())
    assert out == "你如何处理主从延迟?"


def test_make_followup_fallback_without_gateway():
    out = make_followup(Question(id="q", type=QuestionType.sql, prompt="x"), "a", [], None)
    assert "上亿行" in out  # sql-typed generic probe


# --- graph build ------------------------------------------------------------

def test_build_graph_compiles():
    app = build_graph(_deps())
    assert app is not None
    # graph has the expected nodes (use the Pregel node registry directly;
    # get_graph() would trigger a draw-time dry run incompatible with a bare
    # dict state schema).
    nodes = set(app.nodes.keys())
    for n in ("route", "ask", "await_answer", "score", "probe", "decide", "review", "done"):
        assert n in nodes


# --- one full turn ----------------------------------------------------------

def _cfg(tmp_label: str = ":mem-test:") -> dict:
    return {"interview": {"rounds": ROUNDS, "max_followups": 2, "checkpoint_db": tmp_label}}


def test_first_question_then_scored_turn_and_followup():
    gw = FakeGateway(score=40, verdict="needs_fix")   # needs_fix -> triggers a followup
    deps = _deps(gw=gw, max_followups=2)
    st = start_session("AI 应用工程师", _cfg("t-turn"), deps=deps)
    # paused at the first question
    assert st.phase == "await_answer"
    assert st.current_question is not None
    assert st.round == "tech_basics"
    first_qid = st.current_question.id

    # answer it -> should score and then come back paused on a FOLLOWUP
    st2 = step(st.session_id, "我们用 Kafka 幂等消费保证不重复。", _cfg("t-turn"), deps=deps)
    assert st2.phase == "await_answer"
    # a turn was recorded with a scored evaluation
    assert len(st2.turns) == 1
    ev = st2.turns[0].evaluation
    assert isinstance(ev, AnswerEvaluation)
    assert ev.score == 40 and ev.verdict == Verdict.needs_fix
    assert 0.0 <= ev.grounding_rate <= 1.0
    # the new pending question is a grounded followup (different id, derived from first)
    assert st2.current_question is not None
    assert st2.current_question.id != first_qid
    assert st2.current_question.id.startswith(first_qid)


def test_followup_budget_caps_then_advances():
    gw = FakeGateway(score=30, verdict="needs_fix")
    deps = _deps(gw=gw, max_followups=2)
    cfg = _cfg("t-budget")
    st = start_session("role", cfg, deps=deps)
    rounds_seen = {st.round}
    interrupts = 0
    for _ in range(40):
        if st.finished:
            break
        st = step(st.session_id, "一个回答。", cfg, deps=deps)
        interrupts += 1
        rounds_seen.add(st.round)
    assert st.finished
    # every round must have been visited despite always-needs_fix answers
    assert rounds_seen == set(ROUNDS)
    # first round: 1 question + 2 followups = 3 answers before advancing;
    # total answers = sum over 5 rounds of (1 + min(budget, ...)) = 5 * 3 = 15
    assert interrupts == 15


def test_full_run_passes_advance_without_followups():
    gw = FakeGateway(score=85, verdict="pass")   # pass -> no followups
    deps = _deps(gw=gw, max_followups=2)
    cfg = _cfg("t-pass")
    st = start_session("role", cfg, deps=deps)
    n = 0
    while not st.finished and n < 20:
        st = step(st.session_id, "扎实的回答。", cfg, deps=deps)
        n += 1
    assert st.finished
    # one answer per round, no followups
    assert n == len(ROUNDS)
    assert len(st.turns) == len(ROUNDS)
    assert all(t.evaluation.verdict == Verdict.passed for t in st.turns)
    # review wrote a summary episode to memory
    assert any(ep.kind == "review" for ep in deps.memory.episodes)


# --- checkpoint resume after simulated restart ------------------------------

def test_checkpoint_resume_after_restart():
    from langgraph.checkpoint.memory import InMemorySaver
    saver = InMemorySaver()
    gw = FakeGateway(score=80, verdict="pass")
    retr, mem = FakeRetriever(), FakeMemory()

    # session created with graph instance #1
    deps1 = Deps(gateway=gw, retriever=retr, memory=mem, checkpointer=saver, max_followups=1)
    app1 = build_graph(deps1)
    sid = "resume-sess"
    gcfg = {"configurable": {"thread_id": sid}}
    init = InterviewState(session_id=sid, target_role="role",
                          rounds_remaining=list(ROUNDS), phase="route")
    res1 = app1.invoke(init.model_dump(mode="json"), config=gcfg)
    assert "__interrupt__" in res1   # paused at first question
    pending_q = res1["__interrupt__"][0].value["question"]["id"]

    # simulate process restart: brand-new compiled graph, SAME saver + thread_id
    deps2 = Deps(gateway=gw, retriever=retr, memory=mem, checkpointer=saver, max_followups=1)
    app2 = build_graph(deps2)
    res2 = app2.invoke(gmod.Command(resume="我的详细回答, 涉及副本与幂等。"), config=gcfg)

    # state persisted across the restart: the answered turn is present
    values = app2.get_state(gcfg).values
    st = InterviewState.model_validate({k: v for k, v in values.items() if k != "_pending_answer"})
    assert len(st.turns) >= 1
    assert st.turns[0].question.id == pending_q
    assert st.turns[0].answer.startswith("我的详细回答")
    assert st.turns[0].evaluation is not None


def test_step_resumes_via_session_helpers_across_engine_rebuild():
    # start_session + step share a per-checkpoint-path cached saver, so step()
    # resumes even though it rebuilds Deps internally when none is passed.
    gw = FakeGateway(score=90, verdict="pass")
    # inject deps explicitly to keep fakes; same cfg key reused by step()
    cfg = _cfg("t-helper")
    deps = Deps(gateway=gw, retriever=FakeRetriever(), memory=FakeMemory(),
                checkpointer=None, max_followups=1)
    st = start_session("role", cfg, deps=deps)
    assert st.phase == "await_answer"
    st2 = step(st.session_id, "回答内容。", cfg, deps=deps)
    assert len(st2.turns) == 1
    assert st2.turns[0].evaluation.verdict == Verdict.passed
