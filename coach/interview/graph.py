"""LangGraph mock-interview engine (the centerpiece).

State machine:
    route -> ask -> await_answer (interrupt) -> score -> probe
                         ^                                  |
                         |  (budgeted followup, verdict!=pass)
                         +----------------------------------+
    probe -> decide -> (route | review)
    review -> done

- ``await_answer`` calls ``interrupt({...})`` to hand control back to the human;
  the resume value (``Command(resume=answer)``) becomes the candidate's answer.
- ``score`` runs the evaluator (``evaluate.calibrate.judge``) against retrieved
  evidence, producing an ``AnswerEvaluation`` recorded on the turn.
- ``probe`` emits up to ``cfg interview.max_followups`` adversarial follow-ups
  per question before advancing.
- ``review`` writes a session summary + refreshed weakpoints into memory.

Dependencies (gateway / retriever / memory / checkpointer) are injected via a
``Deps`` object so the engine is fully testable offline. ``build_graph(deps)``
returns a compiled graph; ``start_session`` / ``step`` are the thin
session-oriented helpers used by the server and CLI.

Checkpointer: ``deps.checkpointer`` defaults to langgraph ``InMemorySaver``
(``SqliteSaver`` is not installed in this env). The same saver instance is
reused per checkpoint path within a process so ``step`` can resume the session.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from coach.config import get
from coach.evaluate.calibrate import judge
from coach.interview.prober import make_followup
from coach.interview.rounds import ROUNDS, MemoryLike, Retriever, select_question
from coach.schemas import (
    AnswerEvaluation,
    InterviewState,
    InterviewTurn,
    MemoryEpisode,
    Question,
    RetrievalHit,
    Verdict,
)


# ---------------------------------------------------------------------------
# Dependency bundle
# ---------------------------------------------------------------------------

@dataclass
class Deps:
    """Injected collaborators for the interview engine.

    gateway      : LLMGateway-like (complete/structured). Required for live runs;
                   may be None in degraded/offline scenarios (fallbacks kick in).
    retriever    : duck-typed Retriever (.search(query, top_k)) or None.
    memory       : MemoryLike (.weakpoints(); optional .add_episode()) or None.
    checkpointer : a langgraph checkpointer; defaults to InMemorySaver.
    max_followups: budget of adversarial follow-ups per question.
    """
    gateway: Any = None
    retriever: Optional[Retriever] = None
    memory: Optional[MemoryLike] = None
    checkpointer: Any = None
    max_followups: int = 3

    def __post_init__(self) -> None:
        if self.checkpointer is None:
            from langgraph.checkpoint.memory import InMemorySaver
            self.checkpointer = InMemorySaver()


# ---------------------------------------------------------------------------
# Graph state (TypedDict-free: we use a plain dict mirror of InterviewState)
# ---------------------------------------------------------------------------
# LangGraph state is a dict. We serialize Pydantic models with mode="json" so
# the checkpointer's msgpack serde stores plain strings (no custom enum types).


def _state_to_dict(st: InterviewState) -> dict:
    return st.model_dump(mode="json")


def _dict_to_state(d: dict) -> InterviewState:
    return InterviewState.model_validate(d)


def _retr_hits(deps: Deps, query: str, top_k: int = 6) -> list[RetrievalHit]:
    if deps.retriever is None:
        return []
    try:
        return list(deps.retriever.search(query, top_k=top_k) or [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Node factories (closures over deps)
# ---------------------------------------------------------------------------

def _make_nodes(deps: Deps):

    def route(state: dict) -> dict:
        st = _dict_to_state(state)
        if not st.rounds_remaining and not st.turns:
            st.rounds_remaining = list(ROUNDS)
        if not st.rounds_remaining:
            # nothing left to ask -> go straight to review
            st.phase = "review"
            return _state_to_dict(st)
        st.round = st.rounds_remaining[0]
        st.followup_count = 0
        st.phase = "ask"
        # refresh weakpoints from memory each round so question selection adapts
        if deps.memory is not None:
            try:
                st.weakpoints = list(deps.memory.weakpoints() or [])
            except Exception:
                pass
        return _state_to_dict(st)

    def ask(state: dict) -> dict:
        st = _dict_to_state(state)
        q = select_question(st, deps.retriever, deps.memory, deps.gateway)
        st.current_question = q
        st.phase = "await_answer"
        return _state_to_dict(st)

    def await_answer(state: dict) -> dict:
        st = _dict_to_state(state)
        q = st.current_question
        payload = {
            "type": "question" if st.followup_count == 0 else "followup",
            "round": st.round,
            "followup_count": st.followup_count,
            "question": q.model_dump(mode="json") if q else None,
        }
        answer = interrupt(payload)        # <- human-in-the-loop pause
        st.phase = "score"
        # stash the just-received answer on a transient field via a turn-in-progress
        st_dict = _state_to_dict(st)
        st_dict["_pending_answer"] = answer if isinstance(answer, str) else str(answer)
        return st_dict

    def score(state: dict) -> dict:
        answer = state.get("_pending_answer", "")
        st = _dict_to_state({k: v for k, v in state.items() if k != "_pending_answer"})
        q: Optional[Question] = st.current_question
        if q is None:
            st.phase = "decide"
            return _state_to_dict(st)
        hits = _retr_hits(deps, q.prompt, top_k=6)
        try:
            evaluation = judge(q, answer, hits, deps.gateway)
        except Exception:
            evaluation = AnswerEvaluation(
                question_id=q.id, user_answer=answer, score=0,
                verdict=Verdict.needs_fix, issues=["scoring failed"])
        st.turns.append(InterviewTurn(question=q, answer=answer, evaluation=evaluation))
        st.phase = "probe"
        return _state_to_dict(st)

    def probe(state: dict) -> dict:
        st = _dict_to_state(state)
        last = st.turns[-1] if st.turns else None
        ev = last.evaluation if last else None
        budget_left = st.followup_count < deps.max_followups
        needs_probe = bool(ev) and ev.verdict != Verdict.passed
        if last is not None and budget_left and needs_probe:
            q = last.question
            hits = _retr_hits(deps, q.prompt, top_k=5)
            # prefer the evaluator's own sharp followup; else generate one
            text = (ev.followup or "").strip() if ev else ""
            if not text:
                text = make_followup(q, last.answer, hits, deps.gateway)
            # re-ask: the followup becomes the current question
            fq = q.model_copy(update={
                "id": f"{q.id}-f{st.followup_count + 1}",
                "prompt": text,
                "followups": [],
            })
            st.current_question = fq
            st.followup_count += 1
            st.phase = "await_answer"
            return _state_to_dict(st)
        st.phase = "decide"
        return _state_to_dict(st)

    def decide(state: dict) -> dict:
        st = _dict_to_state(state)
        # current round done: drop it; if more remain, loop; else review
        if st.rounds_remaining:
            st.rounds_remaining = st.rounds_remaining[1:]
        if st.rounds_remaining:
            st.followup_count = 0
            st.current_question = None
            st.phase = "route"
        else:
            st.phase = "review"
        return _state_to_dict(st)

    def review(state: dict) -> dict:
        st = _dict_to_state(state)
        # persist a short session summary + weakpoints to memory (best-effort)
        if deps.memory is not None and hasattr(deps.memory, "add_episode"):
            try:
                scored = [t for t in st.turns if t.evaluation is not None]
                avg = (sum(t.evaluation.score for t in scored) / len(scored)) if scored else 0
                weak = sorted({
                    iss
                    for t in scored
                    for iss in (t.evaluation.issues or [])
                })[:10]
                deps.memory.add_episode(MemoryEpisode(
                    kind="review",
                    content=(f"模拟面试结束: {len(st.turns)} 轮问答, 平均分 {avg:.0f}. "
                             f"待加强: {', '.join(weak) if weak else '无明显短板'}"),
                    tags=["interview", st.target_role] if st.target_role else ["interview"],
                    score=int(avg),
                ))
            except Exception:
                pass
        st.phase = "done"
        st.finished = True
        st.current_question = None
        return _state_to_dict(st)

    return {
        "route": route, "ask": ask, "await_answer": await_answer,
        "score": score, "probe": probe, "decide": decide, "review": review,
    }


def _route_after_probe(state: dict) -> str:
    return "await_answer" if state.get("phase") == "await_answer" else "decide"


def _route_after_decide(state: dict) -> str:
    return "review" if state.get("phase") == "review" else "route"


def _route_after_route(state: dict) -> str:
    return "review" if state.get("phase") == "review" else "ask"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(deps: Deps):
    """Build and compile the interview StateGraph with an injected checkpointer."""
    nodes = _make_nodes(deps)
    g = StateGraph(dict)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", _route_after_route, {"ask": "ask", "review": "review"})
    g.add_edge("ask", "await_answer")
    g.add_edge("await_answer", "score")
    g.add_edge("score", "probe")
    g.add_conditional_edges("probe", _route_after_probe,
                            {"await_answer": "await_answer", "decide": "decide"})
    g.add_conditional_edges("decide", _route_after_decide, {"route": "route", "review": "review"})
    g.add_edge("review", "done")
    g.add_edge("done", END)
    # a terminal no-op node so "done" is a real node we can route into
    g.add_node("done", lambda state: state)
    return g.compile(checkpointer=deps.checkpointer)


# ---------------------------------------------------------------------------
# Session helpers (build deps from cfg)
# ---------------------------------------------------------------------------

def build_deps(cfg: dict, *, gateway=None, retriever=None, memory=None, checkpointer=None) -> Deps:
    """Construct a Deps bundle from config, with optional explicit overrides.

    Live wiring (gateway/retriever/memory) is supplied by the caller (server/
    CLI) because it pulls in heavy modules; when omitted we construct the real
    ``EvidenceRetriever`` (offline-safe: it yields no hits when the corpus is
    absent) so interviews are grounded by default, while the engine still runs
    with deterministic fallbacks for everything else.
    """
    if retriever is None:
        try:
            from coach.retrieval.retriever import EvidenceRetriever
            retriever = EvidenceRetriever(cfg)
        except Exception:
            retriever = None
    return Deps(
        gateway=gateway,
        retriever=retriever,
        memory=memory,
        checkpointer=checkpointer,
        max_followups=int(get(cfg, "interview.max_followups", 3)),
    )


def _engine(cfg: dict, deps: Optional[Deps] = None):
    """Return (compiled_graph, deps) for cfg.

    The graph is recompiled per call (cheap: it's just node/edge wiring). Session
    continuity comes from ``deps.checkpointer`` keyed on ``thread_id`` — callers
    (server/CLI) reuse one ``Deps`` instance across ``start_session``/``step`` so
    the same saver resumes the session. No process-global cache (it provided no
    benefit on the server path, where deps are always passed, and was a
    multi-session footgun).
    """
    d = deps if deps is not None else build_deps(cfg)
    return build_graph(d), d


def _interrupt_payload(result: dict) -> Optional[dict]:
    """Extract the interrupt payload from an invoke result, if paused."""
    intr = result.get("__interrupt__") if isinstance(result, dict) else None
    if not intr:
        return None
    first = intr[0]
    return getattr(first, "value", first)


def start_session(target_role: str, cfg: dict, *, deps: Optional[Deps] = None,
                  session_id: Optional[str] = None) -> InterviewState:
    """Begin a session: run until the first interrupt (first question pending).

    Returns the InterviewState with ``current_question`` set and
    ``phase == 'await_answer'``. Pass the returned ``session_id`` to ``step``.
    """
    app, _d = _engine(cfg, deps)
    sid = session_id or uuid.uuid4().hex
    init = InterviewState(
        session_id=sid,
        target_role=target_role or get(cfg, "target_role.name", ""),
        rounds_remaining=list(get(cfg, "interview.rounds", ROUNDS)),
        phase="route",
    )
    gcfg = {"configurable": {"thread_id": sid}}
    result = app.invoke(_state_to_dict(init), config=gcfg)
    return _read_state(app, gcfg, result, sid)


def step(session_id: str, user_answer: str, cfg: dict, *, deps: Optional[Deps] = None) -> InterviewState:
    """Resume a paused session with the human's answer; run to next pause/end.

    Returns the updated InterviewState: either paused at the next question
    (``phase == 'await_answer'``) or finished (``finished == True``).
    """
    app, _d = _engine(cfg, deps)
    gcfg = {"configurable": {"thread_id": session_id}}
    result = app.invoke(Command(resume=user_answer), config=gcfg)
    return _read_state(app, gcfg, result, session_id)


def get_session_state(session_id: str, cfg: dict, deps: Optional[Deps] = None) -> Optional[InterviewState]:
    """Re-hydrate a session's persisted state from the checkpointer (read-only).

    Returns the ``InterviewState`` for ``session_id`` without resuming the graph,
    or ``None`` when the session is unknown / the checkpointer has no state for
    that thread. Never raises (callers degrade gracefully).
    """
    try:
        app, _d = _engine(cfg, deps)
        gcfg = {"configurable": {"thread_id": session_id}}
        snap = app.get_state(gcfg)
        values = {k: v for k, v in (snap.values or {}).items() if k != "_pending_answer"}
        if not values:
            return None
        values.setdefault("session_id", session_id)
        return _dict_to_state(values)
    except Exception:
        return None


def _read_state(app, gcfg: dict, result: dict, sid: str) -> InterviewState:
    """Build the InterviewState from the graph's persisted state + interrupt info."""
    try:
        values = app.get_state(gcfg).values
    except Exception:
        values = result if isinstance(result, dict) else {}
    values = {k: v for k, v in (values or {}).items() if k != "_pending_answer"}
    if not values:
        values = {"session_id": sid}
    values.setdefault("session_id", sid)
    st = _dict_to_state(values)
    # if paused at an interrupt, surface the pending question + phase
    payload = _interrupt_payload(result)
    if payload is not None:
        st.phase = "await_answer"
        st.finished = False
        q = payload.get("question") if isinstance(payload, dict) else None
        if q:
            st.current_question = Question.model_validate(q)
    return st
