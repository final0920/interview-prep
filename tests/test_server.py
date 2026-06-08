"""Offline tests for coach.server (FastAPI TestClient).

No network, no real LLM: we inject a fake Deps (fake gateway + fake retriever +
fake memory + InMemorySaver) via server.set_deps_factory so the interview engine
runs fully offline. Read endpoints are checked for graceful degradation
(ok:true + empty payload) with no data configured.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from coach import server
from coach.interview.graph import Deps
from coach.schemas import EvidenceUnit, RetrievalHit


# --- fakes (mirror the interview test doubles) ------------------------------

def _ev(eid, symbol, text):
    return EvidenceUnit(id=eid, source_path="svc.py", symbol=symbol, start_line=1, text=text)


class FakeRetriever:
    def search(self, query, top_k=8):
        return [RetrievalHit(evidence=_ev("e1", "function_definition:f", "Kafka 幂等 消费 offset"))][:top_k]


class FakeMemory:
    def weakpoints(self):
        return ["分布式一致性"]

    def add_episode(self, ep):
        return "ep-1"


class FakeGateway:
    def __init__(self, verdict="pass", score=80):
        self.verdict = verdict
        self.score = score

    def structured(self, messages, schema: type[BaseModel], **kw):
        fields = set(schema.model_fields.keys())
        if "prompt" in fields and "verdict" not in fields:
            return schema.model_validate({"prompt": "讲讲消息队列高可用设计与权衡。",
                                          "key_points": ["副本", "幂等"], "difficulty": "hard",
                                          "tags": ["kafka"]})
        return schema.model_validate({"score": self.score, "verdict": self.verdict,
                                      "key_points_hit": ["幂等"], "issues": [],
                                      "fabrication_flags": [], "followup": "如何保证 exactly-once?"})

    def complete(self, messages, **kw):
        return "结合 offset 提交时机, 宕机重放如何不重复消费?"


@pytest.fixture
def client(tmp_path):
    # unique checkpoint path per test so engine cache + saver don't leak
    cfg = {"interview": {"rounds": ["tech_basics", "project_deep_dive", "sql", "scenario", "hr"],
                         "max_followups": 1,
                         "checkpoint_db": str(tmp_path / "cp.sqlite")},
           "target_role": {"name": "AI 应用工程师"}}
    server.set_cfg(cfg)
    server.set_deps_factory(lambda c: Deps(gateway=FakeGateway(), retriever=FakeRetriever(),
                                           memory=FakeMemory(), checkpointer=None, max_followups=1))
    c = TestClient(server.app)
    yield c
    server.set_deps_factory(None)
    server.set_cfg({})


# --- read endpoints: graceful degradation -----------------------------------

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_memory_graceful_empty(client):
    # no memory.db_path in cfg -> ok:true with empty lists
    r = client.get("/api/memory")
    body = r.json()
    assert body["ok"] is True
    assert body["data"] == {"episodes": [], "semantic": [], "weakpoints": []}


def test_resume_graceful_empty(client):
    r = client.get("/api/resume")
    body = r.json()
    assert body["ok"] is True
    assert body["data"]["profile"] is None


def test_review_graceful(client):
    r = client.get("/api/review")
    body = r.json()
    assert body["ok"] is True
    assert "quality_report" in body["data"]
    assert body["data"]["quality_report"]["n_questions"] == 0


def test_all_responses_have_envelope(client):
    for path in ("/api/health", "/api/memory", "/api/resume", "/api/review"):
        body = client.get(path).json()
        assert set(body.keys()) == {"ok", "data", "message"}


# --- interview HTTP turn loop -----------------------------------------------

def test_interview_start_then_answer(client):
    r = client.post("/api/interview/start", json={"target_role": "AI 应用工程师"})
    body = r.json()
    assert body["ok"] is True
    sid = body["data"]["session_id"]
    assert sid
    assert body["data"]["question"] is not None
    assert body["data"]["round"] == "tech_basics"

    r2 = client.post("/api/interview/answer", json={"session_id": sid, "answer": "我们用 Kafka 幂等消费。"})
    b2 = r2.json()
    assert b2["ok"] is True
    assert b2["data"]["evaluation"] is not None
    assert b2["data"]["evaluation"]["score"] == 80
    # pass verdict -> advances to next question (not finished after 1 of 5 rounds)
    assert b2["data"]["next"]["kind"] in ("question", "followup")
    assert b2["data"]["finished"] is False


def test_interview_full_run_finishes(client):
    sid = client.post("/api/interview/start", json={}).json()["data"]["session_id"]
    finished = False
    for _ in range(20):
        b = client.post("/api/interview/answer", json={"session_id": sid, "answer": "扎实回答。"}).json()
        finished = b["data"]["finished"]
        if finished:
            break
    assert finished


def test_interview_get_state(client):
    sid = client.post("/api/interview/start", json={}).json()["data"]["session_id"]
    r = client.get(f"/api/interview/{sid}")
    body = r.json()
    assert body["ok"] is True
    assert body["data"]["session_id"] == sid


def test_interview_get_unknown_session(client):
    r = client.get("/api/interview/does-not-exist")
    body = r.json()
    assert body["ok"] is True
    assert body["data"] is None


def test_answer_unknown_session_is_handled(client):
    # resuming a non-existent thread should not 500; engine returns an empty/edge state
    r = client.post("/api/interview/answer", json={"session_id": "nope", "answer": "x"})
    assert r.status_code == 200
    assert "ok" in r.json()


# --- WebSocket smoke --------------------------------------------------------

def test_ws_turn_loop(client):
    with client.websocket_connect("/interview/ws") as ws:
        ws.send_text(json.dumps({"type": "start", "target_role": "AI 应用工程师"}))
        msg = ws.receive_json()
        assert msg["type"] == "question"
        sid = msg["payload"]["session_id"]
        assert msg["payload"]["question"] is not None

        ws.send_text(json.dumps({"type": "answer", "answer": "我们用 Kafka 幂等消费保证不丢不重。"}))
        score = ws.receive_json()
        assert score["type"] == "score"
        assert "score" in score["payload"]
        nxt = ws.receive_json()
        assert nxt["type"] in ("question", "followup", "done")
        assert nxt["payload"]["session_id"] == sid


def test_ws_invalid_json(client):
    with client.websocket_connect("/interview/ws") as ws:
        ws.send_text("not json")
        msg = ws.receive_json()
        assert msg["type"] == "error"


# --- SSE streaming ----------------------------------------------------------

def test_sse_streams_question_tokens(client):
    sid = client.post("/api/interview/start", json={}).json()["data"]["session_id"]
    with client.stream("GET", f"/api/interview/{sid}/stream") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk for chunk in r.iter_text())
    # at least one token event and a terminal done event
    assert "\"type\": \"token\"" in body or '"type":"token"' in body
    assert "done" in body


def test_sse_unknown_session_done(client):
    with client.stream("GET", "/api/interview/unknown/stream") as r:
        body = "".join(chunk for chunk in r.iter_text())
    assert "done" in body
