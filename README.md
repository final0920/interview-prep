# interview-coach

Evidence-grounded **mock-interview coach**. A greenfield rebuild of the earlier `interview-prep`:
single-user (parameterizable), centered on realistic multi-round text interviews, powered by a
**LangGraph** dialog engine + **local RAG** + **VCP-inspired layered memory**. Target role is
configurable (default: AI Application / LLM Engineering).

> Core principle - **evidence grounding**: every question, follow-up, and score traces back to
> *your own* real code (`file:symbol:line`) or confirmed resume facts, not the model's generic memory.
> The differentiator: the interviewer can probe details only *you* have worked on.

## What it does

- **Mock interview** across rounds (`tech_basics`, `project_deep_dive`, `sql`, `scenario`, `hr`):
  asks questions grounded in your real resume + code, adversarially scores answers, generates
  follow-ups from real code detail, writes weak points to memory, and schedules review (SM-2).
- **Resume** analysis (diagnose-only health report), optimization (STAR rewrite with citation
  check; unsupported numbers flagged), and competitor benchmarking.
- **RAG** over your private code/docs plus a separate public knowledge base (self-growing).

## Architecture (`coach/` package)

| Area | Module | Responsibility |
|---|---|---|
| LLM | `llm/gateway.py` | gpt-5 via OpenAI-compatible gateway; structured output (json_schema -> json_object -> extract_json fallback) |
| Ingest | `ingest/` | safe unzip + noise filter; tree-sitter AST chunking (line/brace fallback); PDF (PyMuPDF); SQL `CREATE TABLE`; -> `evidence_units.jsonl` with `file:symbol:line` |
| Retrieval | `retrieval/` | BGE-M3 hybrid (dense + BM25 + RRF) + reranker, deterministic hashing fallback; numpy vector store; geodesic rerank (flag) |
| Memory | `memory/` | sqlite L2 episodic + L3 semantic (self-edit dedup, time-decay); placeholders; context folding (flag); tidal three-timeline recall (flag) |
| Evaluate | `evaluate/` | adversarial `calibrate`, NLI `claim_check` (grounding rate), `sql_sandbox` (in-memory verify) |
| Interview | `interview/` | LangGraph state machine: route -> ask -> answer(interrupt) -> score -> probe -> decide -> review -> done |
| Resume | `resume/` | parse (PII redaction) / analyze / optimize / benchmark |
| Knowledge | `knowledge/` | public KB search + self-grow (PII-gated, idempotent) |
| Review | `review/` | SM-2 scheduling, gap discovery, quality gate, export (study book + Anki) |
| Surfaces | `server.py`, `web/`, `cli.py` | FastAPI (WS + SSE), Vue3 chat UI, `coach` CLI |

Shared data contracts live in `coach/schemas.py`; configuration in `coach/config.py`.

## Setup

Python 3.12. Reuse an environment that already has the stack, or install fresh:

```bash
pip install -r requirements.txt
# optional, for local GPU embeddings + reranker:
pip install sentence-transformers torch
# optional, for cross-process interview checkpoint persistence:
pip install langgraph-checkpoint-sqlite
```

Config: copy `config.example.yaml` to `config.local.yaml` (gitignored) and set `llm.api_key` and
`llm.base_url` for your gpt-5 gateway. If the gateway rejects the `/v1` suffix, remove it.

## Usage

```bash
coach ingest <zip-or-path> ...   # build the evidence index from your code / resume
coach interview                  # start a text mock interview (CLI)
coach serve                      # FastAPI server on :8000
coach resume ...                 # resume analyze / optimize / benchmark
coach review                     # SM-2 schedule + gap report
coach export                     # study_book.md + anki.csv
coach kb ...                     # public knowledge base search / grow
```

Web UI: `coach serve` then, in `web/`, `npm install && npm run dev` (Vite proxies `/api` and
`/interview/ws` to `:8000`).

## Feature flags (config)

`retrieval.geodesic`, `memory.fold`, `memory.tidal` default **off**; `memory.placeholders` **on**.
Flagged code is fully implemented and unit-tested but never required for the base interview loop.

## Offline behaviour / fallbacks

- Embeddings: BGE-M3 (cuda -> cpu) **else** a deterministic hashing embedder.
- Reranker: BGE cross-encoder **else** identity passthrough.
- Chunking: tree-sitter AST **else** a line/brace chunker.
- Checkpointer: `SqliteSaver` if installed **else** `InMemorySaver`.
- The test suite is fully offline (LLM mocked, no model downloads):
  ```bash
  python -m pytest tests/ -q
  ```

## Notes

- Single-user by design: switch target role / resume / materials via config. No multi-tenant plumbing.
- Secrets live only in `config.local.yaml` (gitignored). Never commit keys.
