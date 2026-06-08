"""LLM gateway over the openai SDK to an OpenAI-compatible gpt-5 endpoint.

This is the single choke point for every model call in the project. LangGraph
nodes, the evaluator, the resume optimizer, etc. all go through ``LLMGateway``
so behaviour (model selection, reasoning effort, structured-output parsing,
robust JSON recovery) lives in one place.

Design notes:
- Structured output tries native ``response_format={"type": "json_schema", ...}``
  first; if the gateway rejects it (older / proxy backends), it degrades to
  JSON-mode (``{"type": "json_object"}``) and finally to a plain completion,
  always finishing with a tolerant ``extract_json`` parse before validating
  against the requested pydantic model.
- ``reasoning_effort`` is passed via ``extra_body`` so gateways that do not
  understand it can ignore it; we also retry without it if the call is rejected.
- No global state, no network at import time. Tests monkeypatch the underlying
  client (see ``_build_client``) so nothing here ever hits the wire offline.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional, TypeVar

from pydantic import BaseModel

from coach.config import get

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Tolerant JSON extraction (ported / hardened from interview-prep/calibrate.py)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict:
    """Best-effort recovery of a single JSON object from messy model output.

    Handles fenced blocks (```json ... ```), leading/trailing prose, and
    truncated tails by scanning for the first balanced ``{...}`` span. Returns
    an empty dict when nothing parseable is found (callers treat that as
    "model produced no usable structure" rather than crashing).
    """
    if not text:
        return {}

    candidates: list[str] = []
    for m in _FENCE_RE.finditer(text):
        inner = m.group(1).strip()
        if inner:
            candidates.append(inner)
    candidates.append(text)

    for cand in candidates:
        # 1) direct parse (cand may already be clean JSON)
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # 2) first-brace .. matching-brace scan, string/escape aware
        obj = _scan_balanced_object(cand)
        if obj is not None:
            return obj
    return {}


def _scan_balanced_object(text: str) -> Optional[dict]:
    """Find the first ``{`` and return the JSON object it opens, if balanced."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : i + 1]
                try:
                    obj = json.loads(snippet)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    # unbalanced (truncated): last-resort first..last brace slice
    end = text.rfind("}")
    if end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------

class LLMGateway:
    """Thin, defensible wrapper around the openai chat-completions API.

    Parameters are read from the merged config (``coach.config``); nothing is
    hardcoded. The OpenAI client is constructed lazily on first use so importing
    this module never requires network or credentials (important for tests).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.base_url: Optional[str] = get(self.cfg, "llm.base_url")
        self.api_key: str = get(self.cfg, "llm.api_key", "") or ""
        self.model: str = get(self.cfg, "llm.model", "gpt-5")
        self.cheap_model: str = get(self.cfg, "llm.cheap_model", self.model)
        self.reasoning_effort: Optional[str] = get(self.cfg, "llm.reasoning_effort")
        self.temperature: float = get(self.cfg, "llm.temperature", 0.3)
        self.timeout_s: float = get(self.cfg, "llm.timeout_s", 120)
        self.max_retries: int = get(self.cfg, "llm.max_retries", 3)
        self.user_agent: Optional[str] = get(self.cfg, "llm.user_agent")
        self._client: Any = None

    # -- client lifecycle ---------------------------------------------------

    def _build_client(self) -> Any:
        """Construct the OpenAI client. Overridden/patched in tests."""
        from openai import OpenAI

        headers = {"User-Agent": self.user_agent} if self.user_agent else None
        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "sk-nokey",
            default_headers=headers,
            timeout=self.timeout_s,
            max_retries=self.max_retries,
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    # -- raw completion -----------------------------------------------------

    def _create(self, messages: list[dict], model: str, **kw: Any) -> Any:
        """Call chat.completions.create, threading reasoning_effort via extra_body.

        Retries once without the reasoning param if the gateway rejects it, so a
        backend that does not understand ``reasoning_effort`` still works.
        """
        params: dict[str, Any] = {"model": model, "messages": messages}
        extra_body: dict[str, Any] = dict(kw.pop("extra_body", {}) or {})
        response_format = kw.pop("response_format", None)
        if response_format is not None:
            params["response_format"] = response_format

        effort = kw.pop("reasoning_effort", self.reasoning_effort)
        if effort:
            extra_body["reasoning_effort"] = effort
        else:
            # only gpt-5-style reasoning models drop temperature; keep it when
            # no reasoning effort is configured so non-reasoning models behave.
            params.setdefault("temperature", kw.pop("temperature", self.temperature))
        params.update(kw)

        if extra_body:
            try:
                return self.client.chat.completions.create(extra_body=extra_body, **params)
            except TypeError:
                # client signature without extra_body (e.g. a test double)
                pass
            except Exception:
                # gateway rejected the reasoning/extra params: retry plain
                pass
        return self.client.chat.completions.create(**params)

    @staticmethod
    def _content(resp: Any) -> str:
        try:
            content = resp.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError):
            return ""
        return content or ""

    # -- public API ---------------------------------------------------------

    def complete(self, messages: list[dict], *, model: Optional[str] = None, **kw: Any) -> str:
        """Return the assistant text for ``messages`` (default model)."""
        resp = self._create(messages, model or self.model, **kw)
        return self._content(resp)

    def cheap_complete(self, messages: list[dict], **kw: Any) -> str:
        """Like :meth:`complete` but on the configured cheap model.

        Used for folding / summarize / classify paths where the strong model is
        overkill. Falls back to the main model if no cheap model is configured.
        """
        return self.complete(messages, model=self.cheap_model, **kw)

    def structured(
        self,
        messages: list[dict],
        schema: type[T],
        *,
        model: Optional[str] = None,
        **kw: Any,
    ) -> T:
        """Return a validated ``schema`` instance from the model's reply.

        Strategy, most-capable first:
        1. native ``response_format={"type": "json_schema", ...}``
        2. JSON-mode ``response_format={"type": "json_object"}``
        3. plain completion
        Every branch ends in :func:`extract_json` + ``schema.model_validate`` so
        a slightly-off reply (prose around JSON, fences, trailing junk) still
        validates. Raises ``ValueError`` only if no branch yields a valid model.
        """
        mdl = model or self.model
        errors: list[str] = []

        for rf in (self._json_schema_format(schema), {"type": "json_object"}, None):
            try:
                resp = self._create(list(messages), mdl, response_format=rf, **kw)
            except Exception as exc:  # noqa: BLE001 - try the next, less strict mode
                errors.append(f"{_rf_label(rf)}: {exc!r}")
                continue
            text = self._content(resp)
            data = extract_json(text)
            if not data:
                errors.append(f"{_rf_label(rf)}: no JSON in reply")
                continue
            try:
                return schema.model_validate(data)
            except Exception as exc:  # noqa: BLE001 - JSON shape mismatch
                errors.append(f"{_rf_label(rf)}: validate failed {exc!r}")
                continue

        raise ValueError(
            f"structured() could not produce a valid {schema.__name__}: "
            + " | ".join(errors)
        )

    @staticmethod
    def _json_schema_format(schema: type[BaseModel]) -> dict:
        """Build an OpenAI ``json_schema`` response_format from a pydantic model."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": False,
            },
        }


def _rf_label(rf: Optional[dict]) -> str:
    if rf is None:
        return "plain"
    return rf.get("type", "unknown")
