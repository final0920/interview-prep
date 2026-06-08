"""Offline tests for coach.llm.gateway.

No network, no real openai client: a FakeClient records the kwargs each call
receives and returns canned content. We assert structured() validates against
the pydantic schema across the json_schema / json_object / plain fallback
ladder, that extract_json tolerates messy output, and that reasoning_effort is
threaded via extra_body with graceful degradation.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import pytest
from pydantic import BaseModel

from coach.llm.gateway import LLMGateway, extract_json


# --- test schema ------------------------------------------------------------

class Toy(BaseModel):
    verdict: str
    score: int
    issues: list[str] = []


# --- fake openai client -----------------------------------------------------

class FakeCompletions:
    def __init__(self, responder):
        self._responder = responder
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self._responder(kwargs)
        if isinstance(text, Exception):
            raise text
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )


class FakeClient:
    """Mimics openai.OpenAI: .chat.completions.create(**kwargs)."""

    def __init__(self, responder, *, accept_extra_body: bool = True):
        self._accept_extra_body = accept_extra_body
        self.chat = SimpleNamespace(completions=FakeCompletions(self._wrap(responder)))

    def _wrap(self, responder):
        def inner(kwargs):
            if not self._accept_extra_body and "extra_body" in kwargs:
                # emulate an SDK that doesn't accept the extra_body kwarg
                raise TypeError("unexpected keyword argument 'extra_body'")
            return responder(kwargs)
        return inner

    @property
    def calls(self):
        return self.chat.completions.calls


def make_gateway(responder, *, cfg: Optional[dict] = None, accept_extra_body: bool = True):
    base = {"llm": {"model": "gpt-5", "cheap_model": "gpt-5-mini", "reasoning_effort": "high"}}
    if cfg:
        base["llm"].update(cfg.get("llm", {}))
    gw = LLMGateway(base)
    client = FakeClient(responder, accept_extra_body=accept_extra_body)
    gw._client = client  # inject, bypass _build_client (no network)
    return gw, client


# --- extract_json -----------------------------------------------------------

def test_extract_json_plain():
    assert extract_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_extract_json_fenced():
    text = 'sure, here:\n```json\n{"a": 1}\n```\nhope that helps'
    assert extract_json(text) == {"a": 1}


def test_extract_json_prose_wrapped():
    text = 'The answer is {"verdict": "pass", "score": 90} as shown.'
    assert extract_json(text) == {"verdict": "pass", "score": 90}


def test_extract_json_nested_braces():
    text = 'x {"a": {"b": [1, 2]}, "c": "}"} y'
    assert extract_json(text) == {"a": {"b": [1, 2]}, "c": "}"}


def test_extract_json_truncated_inner_falls_back():
    # outer object closes but an inner structure was cut; last-resort
    # first..last brace slice recovers the outer object.
    text = '{"a": 1, "note": "see {x}"} trailing chatter cut off here {'
    assert extract_json(text) == {"a": 1, "note": "see {x}"}


def test_extract_json_no_closing_brace_returns_empty():
    # genuinely unrecoverable (never closed): return {} rather than guess
    assert extract_json('{"a": 1, "b": 2') == {}


def test_extract_json_empty_and_garbage():
    assert extract_json("") == {}
    assert extract_json("no json here") == {}


# --- complete / cheap_complete ---------------------------------------------

def test_complete_returns_content_and_uses_model():
    gw, client = make_gateway(lambda kw: "hello")
    out = gw.complete([{"role": "user", "content": "hi"}])
    assert out == "hello"
    assert client.calls[0]["model"] == "gpt-5"


def test_cheap_complete_uses_cheap_model():
    gw, client = make_gateway(lambda kw: "cheap")
    out = gw.cheap_complete([{"role": "user", "content": "hi"}])
    assert out == "cheap"
    assert client.calls[0]["model"] == "gpt-5-mini"


def test_reasoning_effort_threaded_via_extra_body():
    gw, client = make_gateway(lambda kw: "ok")
    gw.complete([{"role": "user", "content": "hi"}])
    assert client.calls[0].get("extra_body", {}).get("reasoning_effort") == "high"
    # reasoning models must not also send temperature
    assert "temperature" not in client.calls[0]


def test_temperature_used_when_no_reasoning_effort():
    gw, client = make_gateway(lambda kw: "ok", cfg={"llm": {"reasoning_effort": None}})
    gw.complete([{"role": "user", "content": "hi"}])
    assert "extra_body" not in client.calls[0]
    assert client.calls[0]["temperature"] == pytest.approx(0.3)


def test_extra_body_rejected_retries_plain():
    # client raises TypeError on extra_body; gateway must retry without it
    gw, client = make_gateway(lambda kw: "ok", accept_extra_body=False)
    out = gw.complete([{"role": "user", "content": "hi"}])
    assert out == "ok"
    # two attempts: first with extra_body (TypeError), then plain
    assert len(client.calls) == 2
    assert "extra_body" not in client.calls[1]


# --- structured: the ladder -------------------------------------------------

VALID = '{"verdict": "pass", "score": 88, "issues": ["minor"]}'


def test_structured_native_json_schema():
    gw, client = make_gateway(lambda kw: VALID)
    out = gw.structured([{"role": "user", "content": "judge"}], Toy)
    assert isinstance(out, Toy)
    assert out.verdict == "pass" and out.score == 88 and out.issues == ["minor"]
    # first attempt should use the json_schema response_format
    rf = client.calls[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "Toy"


def test_structured_falls_back_to_json_object():
    # json_schema rejected -> json_object accepted
    def responder(kw):
        rf = kw.get("response_format")
        if rf and rf.get("type") == "json_schema":
            raise RuntimeError("json_schema unsupported by gateway")
        return VALID
    gw, client = make_gateway(responder)
    out = gw.structured([{"role": "user", "content": "judge"}], Toy)
    assert out.score == 88
    types = [c.get("response_format", {}).get("type") for c in client.calls]
    assert "json_schema" in types and "json_object" in types


def test_structured_falls_back_to_plain_and_extracts():
    # both structured formats rejected; plain reply wraps JSON in prose+fence
    messy = 'Here is the verdict:\n```json\n' + VALID + '\n```\n'

    def responder(kw):
        rf = kw.get("response_format")
        if rf is not None:
            raise RuntimeError("structured formats unsupported")
        return messy
    gw, client = make_gateway(responder)
    out = gw.structured([{"role": "user", "content": "judge"}], Toy)
    assert out.verdict == "pass"
    # last successful call had no response_format
    assert client.calls[-1].get("response_format") is None


def test_structured_skips_unparseable_then_succeeds():
    # json_schema returns junk (no JSON) -> skip; json_object returns valid
    def responder(kw):
        rf = kw.get("response_format")
        if rf and rf.get("type") == "json_schema":
            return "I cannot comply."
        return VALID
    gw, _ = make_gateway(responder)
    out = gw.structured([{"role": "user", "content": "judge"}], Toy)
    assert out.score == 88


def test_structured_raises_when_never_valid():
    gw, _ = make_gateway(lambda kw: "never any json")
    with pytest.raises(ValueError):
        gw.structured([{"role": "user", "content": "judge"}], Toy)


def test_structured_reasoning_effort_present_on_attempts():
    gw, client = make_gateway(lambda kw: VALID)
    gw.structured([{"role": "user", "content": "judge"}], Toy)
    assert client.calls[0].get("extra_body", {}).get("reasoning_effort") == "high"


# --- lazy client (no network at import / construct) -------------------------

def test_client_is_lazy(monkeypatch):
    built = {"n": 0}

    class Dummy:
        chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
                )
            )
        )

    def fake_build(self):
        built["n"] += 1
        return Dummy()

    monkeypatch.setattr(LLMGateway, "_build_client", fake_build)
    gw = LLMGateway({"llm": {"model": "gpt-5"}})
    assert built["n"] == 0  # not built on construct
    gw.complete([{"role": "user", "content": "hi"}])
    assert built["n"] == 1  # built on first use
    gw.complete([{"role": "user", "content": "hi"}])
    assert built["n"] == 1  # reused
