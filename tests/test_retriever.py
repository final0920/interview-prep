"""Tests for coach.retrieval.retriever.EvidenceRetriever.

Fully offline: a HashingEmbedder (no model download, no network) is injected so
the dense channel is deterministic, and the corpus is a handful of in-memory
EvidenceUnits. Covers the offline-safe contract (missing corpus/index -> []),
the hybrid+rerank search path, and a real integration test that drives the
interview engine (start_session + step) with a *real* EvidenceRetriever, proving
the produced question/turn is grounded in the retrieved evidence.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from coach.interview.graph import Deps, start_session, step
from coach.interview.rounds import ROUNDS, select_question
from coach.retrieval.embed import HashingEmbedder
from coach.retrieval.retriever import EvidenceRetriever, load_evidence
from coach.schemas import (
    AnswerEvaluation,
    Channel,
    EvidenceUnit,
    InterviewState,
    Question,
    Verdict,
)


# --- tiny in-memory corpus --------------------------------------------------

def _unit(uid: str, symbol: str, text: str) -> EvidenceUnit:
    return EvidenceUnit(
        id=uid,
        source_path=f"src/{uid}.py",
        symbol=symbol,
        start_line=10,
        channel=Channel.code,
        lang="python",
        text=text,
    )


def _corpus() -> list[EvidenceUnit]:
    return [
        _unit("u1", "consume", "kafka consumer offset commit idempotent retry 幂等 消费"),
        _unit("u2", "RateLimiter", "rate limiter token bucket sliding window 限流 降级 熔断"),
        _unit("u3", "btree", "database index btree query optimisation 索引 优化"),
        _unit("u4", "cache", "redis cache eviction lru ttl 缓存 淘汰"),
    ]


def _retriever() -> EvidenceRetriever:
    # inject the hashing embedder + corpus directly: no file IO, no model load.
    # use_reranker=False -> identity passthrough (no cross-encoder download).
    return EvidenceRetriever(
        {"embeddings": {"use_reranker": False},
         "retrieval": {"top_k": 20, "rrf_k": 60}},
        evidence=_corpus(),
        embedder=HashingEmbedder(dim=256),
    )


# --- fakes (mirror the interview test doubles) ------------------------------

class FakeMemory:
    def __init__(self, weak: Optional[list[str]] = None):
        self._weak = weak or ["分布式一致性"]

    def weakpoints(self) -> list[str]:
        return list(self._weak)

    def add_episode(self, ep):
        return "ep-1"


class FakeGateway:
    """structured() -> question draft / judgement; complete() -> followup."""

    def __init__(self, *, score: int = 80, verdict: str = "pass"):
        self.score = score
        self.verdict = verdict

    def structured(self, messages, schema: type[BaseModel], **kw):
        fields = set(schema.model_fields.keys())
        if "prompt" in fields and "verdict" not in fields:
            return schema.model_validate({
                "prompt": "讲讲你的消息队列高可用设计与幂等消费取舍。",
                "key_points": ["副本机制", "幂等消费"],
                "difficulty": "hard",
                "tags": ["kafka"],
            })
        return schema.model_validate({
            "score": self.score, "verdict": self.verdict,
            "key_points_hit": ["幂等消费"], "issues": [],
            "fabrication_flags": [], "followup": "如何保证 exactly-once?",
        })

    def complete(self, messages, **kw):
        return "结合 offset 提交时机, 宕机重放时如何避免重复消费?"


# --- offline-safe contract --------------------------------------------------

def test_search_empty_corpus_returns_empty():
    r = EvidenceRetriever({"embeddings": {}}, evidence=[], embedder=HashingEmbedder(dim=64))
    assert r.search("anything", top_k=5) == []


def test_search_empty_query_returns_empty():
    assert _retriever().search("", top_k=5) == []


def test_missing_evidence_file_returns_empty(tmp_path):
    # evidence_path points at a non-existent file -> inert retriever, no crash
    cfg = {"ingest": {"evidence_path": str(tmp_path / "nope.jsonl")},
           "embeddings": {}}
    r = EvidenceRetriever(cfg, embedder=HashingEmbedder(dim=64))
    assert r.evidence == []
    assert r.search("kafka", top_k=5) == []


def test_load_evidence_missing_returns_empty(tmp_path):
    assert load_evidence(tmp_path / "absent.jsonl") == []


def test_load_evidence_roundtrip(tmp_path):
    p = tmp_path / "evidence_units.jsonl"
    units = _corpus()
    p.write_text("\n".join(u.model_dump_json() for u in units) + "\n", encoding="utf-8")
    loaded = load_evidence(p)
    assert [u.id for u in loaded] == [u.id for u in units]


def test_load_evidence_from_configured_path(tmp_path):
    p = tmp_path / "evidence_units.jsonl"
    p.write_text("\n".join(u.model_dump_json() for u in _corpus()) + "\n", encoding="utf-8")
    cfg = {"ingest": {"evidence_path": str(p)}, "embeddings": {}}
    r = EvidenceRetriever(cfg, embedder=HashingEmbedder(dim=256))
    assert len(r.evidence) == 4
    hits = r.search("kafka 幂等 消费 offset", top_k=3)
    assert hits and hits[0].evidence.id == "u1"


def test_load_evidence_skips_bad_lines(tmp_path):
    p = tmp_path / "evidence_units.jsonl"
    good = _corpus()[0].model_dump_json()
    p.write_text(good + "\n" + "{not json}\n" + "\n", encoding="utf-8")
    loaded = load_evidence(p)
    assert len(loaded) == 1


# --- hybrid + rerank search path --------------------------------------------

def test_search_returns_relevant_hits():
    hits = _retriever().search("kafka 幂等 消费 offset commit", top_k=3)
    assert hits
    assert all(isinstance(h.evidence, EvidenceUnit) for h in hits)
    # the kafka/offset unit must surface as the top hit
    assert hits[0].evidence.id == "u1"


def test_search_respects_top_k():
    hits = _retriever().search("database index btree", top_k=2)
    assert len(hits) <= 2


def test_search_ranks_contiguous():
    hits = _retriever().search("redis cache eviction", top_k=4)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_builds_in_memory_store_when_no_index():
    # no ingest.vector_path on disk -> store built in-memory from the corpus
    r = _retriever()
    assert len(r.store) == 4


# --- select_question grounded in a REAL retriever ---------------------------

def test_select_question_grounded_with_real_retriever():
    r = _retriever()
    st = InterviewState(round="project_deep_dive", rounds_remaining=["project_deep_dive"])
    q = select_question(st, r, FakeMemory(), FakeGateway())
    assert isinstance(q, Question)
    assert q.linked_evidence  # refs attached from REAL retrieved evidence
    # every linked ref must point back into the real corpus
    corpus_refs = {u.ref for u in _corpus()}
    assert all(ref in corpus_refs for ref in q.linked_evidence)


# --- full integration: start_session + step with a real retriever -----------

def _cfg() -> dict:
    return {"interview": {"rounds": ROUNDS, "max_followups": 1,
                          "checkpoint_db": ":retriever-int-mem:"}}


def test_start_session_and_step_are_grounded():
    """Drive the engine end-to-end with a REAL EvidenceRetriever (hashing
    embedder, no model download) and assert the first question + scored turn are
    grounded in the in-memory corpus."""
    retr = _retriever()
    deps = Deps(gateway=FakeGateway(score=80, verdict="pass"),
                retriever=retr, memory=FakeMemory(), checkpointer=None,
                max_followups=1)

    st = start_session("AI 应用工程师", _cfg(), deps=deps)
    assert st.phase == "await_answer"
    assert st.current_question is not None
    # first question is grounded: linked_evidence drawn from the real corpus
    corpus_refs = {u.ref for u in _corpus()}
    assert st.current_question.linked_evidence
    assert all(ref in corpus_refs for ref in st.current_question.linked_evidence)

    st2 = step(st.session_id, "我们用 Kafka 幂等消费, offset 手动提交保证不重复。",
               _cfg(), deps=deps)
    assert len(st2.turns) == 1
    ev = st2.turns[0].evaluation
    assert isinstance(ev, AnswerEvaluation)
    assert ev.verdict == Verdict.passed
    # the scored turn's question carries the grounding evidence refs
    assert st2.turns[0].question.linked_evidence
    assert all(ref in corpus_refs for ref in st2.turns[0].question.linked_evidence)


def test_full_run_finishes_with_real_retriever():
    retr = _retriever()
    deps = Deps(gateway=FakeGateway(score=85, verdict="pass"),
                retriever=retr, memory=FakeMemory(), checkpointer=None,
                max_followups=1)
    cfg = {"interview": {"rounds": ROUNDS, "max_followups": 1,
                         "checkpoint_db": ":retriever-int-run:"}}
    st = start_session("role", cfg, deps=deps)
    n = 0
    while not st.finished and n < 20:
        st = step(st.session_id, "扎实的回答, 涉及副本与幂等。", cfg, deps=deps)
        n += 1
    assert st.finished
    assert len(st.turns) == len(ROUNDS)
