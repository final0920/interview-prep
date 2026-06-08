"""FastAPI server: mock-interview turn loop (HTTP + WebSocket), SSE token
streaming, and graceful read endpoints for memory / resume / review.

Contract (DESIGN.md sec 10): every response is ``{ok, data, message}``; read
endpoints degrade gracefully (empty payload + ok:true) when their data source is
absent. SSE uses FastAPI ``StreamingResponse`` (no sse-starlette). The interview
engine (coach.interview.graph) is driven through ``start_session`` / ``step``.

Dependency injection for tests: the engine's collaborators (gateway / retriever /
memory) are built by ``_deps_factory(cfg)``. Tests call ``set_deps_factory(fn)``
to inject fakes so the whole server runs offline with no network or real LLM.
The default factory is fully lazy and offline-safe: it only constructs a real
LLM gateway when an API key is configured, otherwise the engine runs on its
deterministic fallbacks.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from coach.config import get, load_config
from coach.interview.graph import Deps, get_session_state, start_session, step
from coach.schemas import InterviewState

logger = logging.getLogger("coach.server")

app = FastAPI(title="interview-coach", version="0.1.0")

# Cap on candidate answer length to bound prompt size / abuse (chars).
MAX_ANSWER_LEN = 20000


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------

def _ok(data: Any = None, message: str = "") -> dict:
    return {"ok": True, "data": data, "message": message}


def _err(message: str, data: Any = None) -> dict:
    return {"ok": False, "data": data, "message": message}


# ---------------------------------------------------------------------------
# Config + dependency wiring (injectable for offline tests)
# ---------------------------------------------------------------------------

_CFG: Optional[dict] = None
_DEPS: Optional[Deps] = None
_DEPS_FACTORY: Optional[Callable[[dict], Deps]] = None


def get_cfg() -> dict:
    global _CFG
    if _CFG is None:
        try:
            _CFG = load_config()
        except Exception:
            _CFG = {}
    return _CFG


def set_cfg(cfg: dict) -> None:
    """Override config (tests)."""
    global _CFG
    _CFG = cfg


def set_deps_factory(factory: Optional[Callable[[dict], Deps]]) -> None:
    """Inject a Deps factory (tests). Pass None to restore the default."""
    global _DEPS_FACTORY, _DEPS
    _DEPS_FACTORY = factory
    _DEPS = None  # force rebuild


def _default_deps(cfg: dict) -> Deps:
    """Build engine deps from config, offline-safe.

    A real LLM gateway is only constructed when an API key is present; without
    it the interview engine uses its deterministic fallbacks (no network). The
    retriever is the real ``EvidenceRetriever`` (offline-safe: empty results when
    the evidence corpus/index is absent) so production interviews are grounded;
    memory is a sqlite store when a path is configured.
    """
    gateway = None
    api_key = get(cfg, "llm.api_key")
    if api_key and api_key != "REPLACE_ME":
        try:
            from coach.llm.gateway import LLMGateway
            gateway = LLMGateway(cfg)
        except Exception:
            gateway = None

    retriever = None
    try:
        from coach.retrieval.retriever import EvidenceRetriever
        retriever = EvidenceRetriever(cfg)
    except Exception:
        retriever = None

    memory = None
    db_path = get(cfg, "memory.db_path")
    if db_path:
        try:
            from coach.memory.store import MemoryStore
            memory = MemoryStore(db_path)
        except Exception:
            memory = None

    return Deps(
        gateway=gateway,
        retriever=retriever,
        memory=memory,
        checkpointer=None,
        max_followups=int(get(cfg, "interview.max_followups", 3)),
    )


def get_deps() -> Deps:
    global _DEPS
    if _DEPS is None:
        factory = _DEPS_FACTORY or _default_deps
        _DEPS = factory(get_cfg())
    return _DEPS


# ---------------------------------------------------------------------------
# Interview HTTP endpoints
# ---------------------------------------------------------------------------

class StartReq(BaseModel):
    target_role: str = ""


class AnswerReq(BaseModel):
    session_id: str
    answer: str = Field(default="", max_length=MAX_ANSWER_LEN)


def _question_payload(state: InterviewState) -> Optional[dict]:
    q = state.current_question
    return q.model_dump(mode="json") if q else None


def _next_descriptor(state: InterviewState) -> dict:
    """Describe what the client should do next: ask a question/followup or stop."""
    if state.finished:
        return {"kind": "done", "question": None}
    kind = "followup" if state.followup_count > 0 else "question"
    return {"kind": kind, "question": _question_payload(state)}


@app.post("/api/interview/start")
def interview_start(req: StartReq) -> dict:
    cfg = get_cfg()
    role = req.target_role or get(cfg, "target_role.name", "")
    try:
        state = start_session(role, cfg, deps=get_deps())
    except Exception:  # noqa: BLE001
        logger.exception("interview_start failed")
        return _err("failed to start session")
    return _ok({
        "session_id": state.session_id,
        "question": _question_payload(state),
        "round": state.round,
    })


@app.post("/api/interview/answer")
def interview_answer(req: AnswerReq) -> dict:
    cfg = get_cfg()
    try:
        state = step(req.session_id, req.answer, cfg, deps=get_deps())
    except Exception:  # noqa: BLE001
        logger.exception("interview_answer failed")
        return _err("failed to submit answer")
    last = state.turns[-1] if state.turns else None
    evaluation = (last.evaluation.model_dump(mode="json")
                  if last and last.evaluation else None)
    return _ok({
        "evaluation": evaluation,
        "next": _next_descriptor(state),
        "finished": state.finished,
    })


@app.get("/api/interview/{session_id}")
def interview_get(session_id: str) -> dict:
    cfg = get_cfg()
    try:
        # resume-less read: re-hydrate state from the checkpointer
        state = get_session_state(session_id, cfg, get_deps())
        if state is None:
            return _ok(None, "unknown session")
        return _ok(state.model_dump(mode="json"))
    except Exception:  # noqa: BLE001
        logger.exception("interview_get failed")
        return _err("failed to read session")


# ---------------------------------------------------------------------------
# WebSocket turn loop
# ---------------------------------------------------------------------------

@app.websocket("/interview/ws")
async def interview_ws(ws: WebSocket) -> None:
    """Turn loop. Client sends JSON; server replies with typed messages.

    Protocol:
      client -> {"type":"start","target_role":"..."}      (or first message)
      server -> {"type":"question","payload":{...}}
      client -> {"type":"answer","answer":"..."}
      server -> {"type":"score","payload":{...}}
                then {"type":"question"|"followup"|"done","payload":{...}}
    """
    await ws.accept()
    cfg = get_cfg()
    deps = get_deps()
    session_id: Optional[str] = None
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await ws.send_json({"type": "error", "payload": {"message": "invalid JSON"}})
                continue
            mtype = msg.get("type", "answer" if session_id else "start")

            if mtype == "start" or session_id is None:
                role = msg.get("target_role", "") or get(cfg, "target_role.name", "")
                state = await asyncio.to_thread(start_session, role, cfg, deps=deps)
                session_id = state.session_id
                await ws.send_json({
                    "type": "question",
                    "payload": {
                        "session_id": session_id,
                        "round": state.round,
                        "question": _question_payload(state),
                    },
                })
                continue

            if mtype == "answer":
                answer = msg.get("answer", "")
                state = await asyncio.to_thread(step, session_id, answer, cfg, deps=deps)
                last = state.turns[-1] if state.turns else None
                if last and last.evaluation:
                    await ws.send_json({
                        "type": "score",
                        "payload": last.evaluation.model_dump(mode="json"),
                    })
                nxt = _next_descriptor(state)
                await ws.send_json({"type": nxt["kind"], "payload": {
                    "session_id": session_id,
                    "round": state.round,
                    "question": nxt["question"],
                    "finished": state.finished,
                }})
                continue

            await ws.send_json({"type": "error", "payload": {"message": f"unknown type {mtype}"}})
    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001
        logger.exception("interview_ws failed")
        try:
            await ws.send_json({"type": "error", "payload": {"message": "internal error"}})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SSE token streaming (StreamingResponse; no sse-starlette)
# ---------------------------------------------------------------------------

def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/interview/{session_id}/stream")
def interview_stream(session_id: str) -> StreamingResponse:
    """Stream the current pending question token-by-token as SSE.

    Useful for the chat UI to render the interviewer's question progressively.
    Degrades to a single 'done' event when there is no pending question.
    """
    cfg = get_cfg()

    def gen():
        text = ""
        try:
            state = get_session_state(session_id, cfg, get_deps())
            q = state.current_question if state else None
            text = q.prompt if q else ""
        except Exception:
            text = ""
        if not text:
            yield _sse_event({"type": "done", "text": ""})
            return
        # chunk by a few characters to simulate token streaming
        step_n = 12
        for i in range(0, len(text), step_n):
            yield _sse_event({"type": "token", "text": text[i:i + step_n]})
        yield _sse_event({"type": "done", "text": text})

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Read endpoints (graceful degradation)
# ---------------------------------------------------------------------------

@app.get("/api/memory")
def read_memory() -> dict:
    cfg = get_cfg()
    try:
        from coach.memory.store import MemoryStore
        db_path = get(cfg, "memory.db_path")
        if not db_path:
            return _ok({"episodes": [], "semantic": [], "weakpoints": []}, "no memory db configured")
        store = MemoryStore(db_path)
        episodes = [e.model_dump(mode="json") for e in store.recent_episodes(20)]
        semantic = [s.model_dump(mode="json") for s in store.get_semantic()]
        return _ok({
            "episodes": episodes,
            "semantic": semantic,
            "weakpoints": store.weakpoints(),
        })
    except Exception:  # noqa: BLE001
        logger.exception("read_memory degraded")
        return _ok({"episodes": [], "semantic": [], "weakpoints": []}, "memory unavailable")


@app.get("/api/resume")
def read_resume() -> dict:
    cfg = get_cfg()
    try:
        from pathlib import Path
        from coach.config import data_dir
        from coach.resume.parse import parse_resume
        from coach.resume.analyze import health_report
        # look for a stored resume text/json under data_dir
        d = data_dir(cfg)
        candidates = list(Path(d).glob("resume.*"))
        if not candidates:
            return _ok({"profile": None, "health_report": {}}, "no resume on file")
        profile = parse_resume(str(candidates[0]), llm=None)
        role = get(cfg, "target_role.name", "")
        try:
            report = health_report(profile, role, None)
        except Exception:
            report = {}
        return _ok({"profile": profile.model_dump(mode="json"), "health_report": report})
    except Exception:  # noqa: BLE001
        logger.exception("read_resume degraded")
        return _ok({"profile": None, "health_report": {}}, "resume unavailable")


@app.get("/api/review")
def read_review() -> dict:
    cfg = get_cfg()
    try:
        from coach.review.quality_gate import quality_report
        report = quality_report([], [], [], cfg)
        # schedule is empty until cards exist; keep the shape stable
        return _ok({"schedule": [], "quality_report": report})
    except Exception:  # noqa: BLE001
        logger.exception("read_review degraded")
        return _ok({"schedule": [], "quality_report": {}}, "review unavailable")


@app.get("/api/health")
def health() -> dict:
    return _ok({"status": "up"})
