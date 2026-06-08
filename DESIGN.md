# interview-coach — Build Contract (read this first)

Greenfield rebuild of `interview-prep`. Single-user (parameterizable), evidence-grounded,
**mock-interview-centric** coach. LangGraph dialog engine + local RAG + VCP-inspired memory.
Target role: AI Application / LLM Engineering.

This document is the **contract** every worker codes against. Do not invent interfaces that
diverge from `coach/schemas.py` and the signatures below.

---

## 1. Environment & how to run (IMPORTANT)

- **Python / pytest MUST use the `iprep` conda env**, which already has the full stack
  (langgraph 1.2.4, langchain-core, openai 2.41, torch 2.12, sentence-transformers 5.5,
  rank-bm25, tree-sitter + tree-sitter-language-pack, pymupdf, fastapi, uvicorn, numpy 2.4, pytest 9):
  ```
  & 'D:\anaconda3\envs\iprep\python.exe' -m pytest tests/ -q
  ```
  Run pytest from the repo root `D:\ms\interview-prep-v2`.
- **No new pip installs.** All needed deps are present. Do NOT add chromadb / langmem /
  langchain-openai / sse-starlette — we deliberately avoid them (see section 5).
- **No network from this machine to github / model hubs.** So: tests must be fully offline
  (no model downloads, no LLM calls, no internet). The lead handles the final git push.
- Salvage source (proven algorithms to port) lives at `D:\ms\interview-prep\*.py` — READ-ONLY.
  Reuse the algorithm, drop all `tenant_id` / multi-tenant plumbing, re-organize into this layout.

## 2. Layout (flat package)

```
interview-prep-v2/
  pyproject.toml  requirements.txt  config.example.yaml  config.local.yaml(gitignored)
  coach/
    __init__.py  schemas.py  config.py        # DONE by lead — import from these
    llm/gateway.py
    storage/{sqlite.py, vector.py, checkpointer.py}
    ingest/{extract.py, chunk.py, run.py}
    retrieval/{embed.py, hybrid.py, rerank.py, geodesic.py}
    memory/{store.py, placeholders.py, fold.py, tidal.py}
    evaluate/{calibrate.py, claim_check.py, sql_sandbox.py}
    interview/{rounds.py, prober.py, graph.py}
    resume/{parse.py, analyze.py, optimize.py, benchmark.py}
    knowledge/{public_kb.py, grow.py}
    review/{sm2.py, gap.py, quality_gate.py, export.py}
    cli.py  server.py
  tests/                                       # tests/conftest.py DONE by lead
  web/                                         # Vue3 + Vite chat UI
```

## 3. Shared contract (already written — use as-is)

- `coach/schemas.py` — all Pydantic v2 models: `EvidenceUnit, RetrievalHit, Question,
  AnswerEvaluation, ClaimCheck, MemoryEpisode, MemorySemantic, SkillGap, ResumeProfile,
  InterviewState, InterviewTurn`, plus enums. Import models from here; extend only by adding
  optional fields (never break existing field names/types).
- `coach/config.py` — `load_config() -> dict`, `get(cfg, "a.b.c", default)`, `data_dir(cfg)`.
  Read all tunables/flags from config; never hardcode paths or the API key.

## 4. Offline & degradation rules (NON-NEGOTIABLE)

1. **Tests never touch the network or the LLM.** Mock the gateway (monkeypatch) in tests.
   Heavy/optional model paths use `pytest.importorskip(...)` or are exercised via the fallback.
2. **Every external capability has a deterministic offline fallback:**
   - Embeddings: BGE-M3 (sentence-transformers, cuda->cpu) **else** a deterministic hashing
     embedder (pure numpy). Same API either way.
   - Reranker: BGE cross-encoder **else** identity passthrough (keep RRF order).
   - tree-sitter AST chunking **else** a line/brace-based chunker.
   - LLM: real gpt-5 gateway at runtime; in tests, injected fake returning canned structured output.
3. Functions should be **pure where possible** (input -> output, no global state), to stay unit-testable.
4. Large evidence files are read **streaming / line-by-line**; never load multi-GB into memory.

## 5. Deliberate tech choices (do not deviate)

- **Vector store = pure numpy in-process** (`coach/storage/vector.py`): a float32 matrix + id/meta
  lists, cosine via normalized dot product, persisted as `.npy` + json sidecar. No chromadb.
- **Memory = sqlite directly** (`coach/memory/store.py`). No langmem.
- **LLM = `openai` SDK directly** (`coach/llm/gateway.py`). No langchain-openai. (LangGraph is used
  only for the interview state machine; call the gateway inside nodes.)
- **SSE = FastAPI `StreamingResponse`** (`coach/server.py`). No sse-starlette.
- **Checkpointer**: `coach/storage/checkpointer.py` tries langgraph `SqliteSaver`, falls back to
  `InMemorySaver` (langgraph-checkpoint-sqlite is NOT installed in iprep; no new pip installs here).
  `build_graph(deps)` accepts an injected checkpointer. For true cross-process persistence later,
  `pip install langgraph-checkpoint-sqlite`. `interrupt()/Command(resume=)` import from `langgraph.types`.

## 6. Module ownership, salvage map & key signatures

> Salvage paths are under `D:\ms\interview-prep\`. Port the algorithm, strip tenant_id.

### T1 llm  ->  coach/llm/gateway.py   (owner: worker-1)
Salvage: `gen_questions.py` (call_llm, extract_json).
```
class LLMGateway:
    def __init__(self, cfg: dict): ...
    def complete(self, messages: list[dict], *, model: str|None=None, **kw) -> str
    def structured(self, messages: list[dict], schema: type[BaseModel], *, model=None) -> BaseModel
        # try response_format=json_schema / tool-calling; fall back to JSON-mode + robust extract_json
    def cheap_complete(self, messages: list[dict], **kw) -> str   # uses cfg llm.cheap_model
def extract_json(text: str) -> dict   # tolerant fenced/loose JSON extraction
```
Reasoning param: pass `reasoning_effort` from cfg via `extra_body`; tolerate gateways that reject it.
Tests: monkeypatch the openai client; assert `structured` parses + fallback path works. No network.

### T2 storage  ->  coach/storage/{sqlite.py, vector.py, checkpointer.py}   (owner: worker-2)
Salvage: `persist.py` (sqlite patterns, minus tenant), `vector_index.py` (.npy cache).
```
# sqlite.py
def connect(path: str|Path) -> sqlite3.Connection            # row_factory=Row, WAL
def init_schema(conn, ddl: str) -> None
# vector.py
class VectorStore:
    def add(self, ids: list[str], vecs: np.ndarray, metas: list[dict]) -> None
    def search(self, qvec: np.ndarray, k: int) -> list[tuple[str, float, dict]]   # cosine desc
    def save(self, dir: str|Path) -> None
    @classmethod
    def load(cls, dir: str|Path) -> "VectorStore"
# checkpointer.py
def get_checkpointer(path: str|Path):   # returns langgraph SqliteSaver (importorskip in tests)
```
Tests: vector search correctness on toy vectors; sqlite upsert/idempotency.

### T3 ingest  ->  coach/ingest/{extract.py, chunk.py, run.py}   (owner: worker-4)
Salvage: `ingest_mvp.py` (zip safety, noise filter, tree-sitter AST chunk, ref linkback, hash dedup),
`ingest_docs.py` (pymupdf). NOTE: tree-sitter-language-pack API is PyO3-style
(`node.kind`, `node.start_point`, `node.child(i)`, `parser.parse(bytes)`); getters are methods.
```
# extract.py
def safe_unzip(zip_path, dest, noise_globs: list[str]) -> list[Path]   # zip-slip/bomb guard + noise filter
def is_noise(path: str, noise_globs: list[str]) -> bool
# chunk.py
def chunk_code(text: str, lang: str, path: str) -> list[EvidenceUnit]   # tree-sitter else line/brace fallback
def chunk_pdf(pdf_path: str) -> list[EvidenceUnit]
def extract_create_tables(text: str, path: str) -> list[EvidenceUnit]   # SQL CREATE TABLE
# run.py
def ingest(paths: list[str], cfg: dict) -> str   # -> writes data/evidence_units.jsonl, returns path
```
Tests: zip-slip rejected; noise filter; fallback chunker splits a sample C++/py string into units with file:line.

### T4 memory  ->  coach/memory/{store.py, placeholders.py, fold.py, tidal.py}   (owner: worker-5)
Salvage: `memory.py` (L2/L3, self-edit dedup, temporal window, decay recall).
```
# store.py
class MemoryStore:
    def __init__(self, db_path): ...
    def add_episode(self, ep: MemoryEpisode) -> str
    def recent_episodes(self, k: int, tags: list[str]|None=None, now: float|None=None) -> list[MemoryEpisode]  # time-decay x tag overlap
    def upsert_semantic(self, m: MemorySemantic) -> None   # same (key) + changed value => close old (valid_to) + insert new
    def get_semantic(self, kind: str|None=None) -> list[MemorySemantic]
    def weakpoints(self) -> list[str]
# placeholders.py
def render(template: str, store: MemoryStore, profile: ResumeProfile|None=None) -> str  # {{profile}}/{{weakpoints}}/{{resume}}
# fold.py  (feature-flag memory.fold)
def fold_context(chunks: list[str], gateway, *, keep_recent: int) -> str   # summarize distant low-relevance via cheap model
# tidal.py (feature-flag memory.tidal)
def tidal_recall(store: MemoryStore, vector_store, seed_tags: list[str], now: float) -> list[MemoryEpisode]
    # near(0-7d)/mid(7-90d)/abyss(>90d) buckets; fuse time_decay x resonance x relevance
```
Tests: self-edit dedup closes old record; decay recall ordering; tidal bucketing on toy timestamps (no LLM); fold with a fake gateway.

### T5 retrieval  ->  coach/retrieval/{embed.py, hybrid.py, rerank.py, geodesic.py}   (owner: worker-3, blockedBy T2)
Salvage: `vector_index.py` (BGE-M3 dense+sparse, RRF, hashing fallback), `retrieve_bm25.py` (tokenizer),
`public_kb.py` (RRF fuse).
```
# embed.py
class Embedder:
    def __init__(self, cfg): ...   # BGE-M3 on cuda/cpu, else HashingEmbedder
    def encode(self, texts: list[str]) -> np.ndarray
def tokenize(text: str) -> list[str]   # camelCase/snake split + Chinese 2-gram
# hybrid.py
def rrf_fuse(rank_lists: list[list[str]], k: int=60) -> list[tuple[str, float]]
def hybrid_search(query: str, evidence: list[EvidenceUnit], embedder, store, *, top_k) -> list[RetrievalHit]
# rerank.py
def rerank(query: str, hits: list[RetrievalHit], cfg, *, top_k) -> list[RetrievalHit]   # BGE cross-encoder else identity
# geodesic.py (feature-flag retrieval.geodesic)
def build_cooccurrence(evidence: list[EvidenceUnit], db_path) -> None   # sqlite tag_pair_similarity
def geodesic_rerank(query_tags: list[str], hits: list[RetrievalHit], db_path) -> list[RetrievalHit]
```
Tests: RRF math; tokenizer; HashingEmbedder determinism; geodesic rerank on toy co-occurrence (no model).

### T6 evaluate  ->  coach/evaluate/{calibrate.py, claim_check.py, sql_sandbox.py}   (owner: worker-1, blockedBy T1)
Salvage: `calibrate.py`, `claim_check.py`, `gen_sql.py`.
```
# calibrate.py
def judge(question: Question, answer: str, evidence: list[RetrievalHit], gateway) -> AnswerEvaluation
# claim_check.py
def split_claims(text: str) -> list[str]
def check_claims(claims: list[str], evidence: list[EvidenceUnit], gateway=None) -> list[ClaimCheck]  # L1 recall + L2 NLI heuristic offline; optional LLM
def grounding_rate(checks: list[ClaimCheck]) -> float
# sql_sandbox.py
def normalize_mysql_to_sqlite(ddl: str) -> str
def verify_sql(ddl: str, query: str) -> tuple[bool, str]   # in-memory sqlite3 execute/EXPLAIN
```
Tests: offline heuristic NLI verdicts; sql normalize + execute a toy CREATE/SELECT; judge parse with fake gateway.

### T7 export  ->  coach/review/export.py   (owner: worker-2)
Salvage: `export.py`. `export_study_book(questions, evals, out_dir) -> Path` (study_book.md),
`export_anki(questions, evals, out_dir) -> Path` (anki.csv, RFC4180). Graceful when evals missing.
Tests: md/csv generated from sample Question/AnswerEvaluation; csv quote-escaping.

### T8 review  ->  coach/review/{sm2.py, gap.py, quality_gate.py}   (owner: worker-2, blockedBy T6)
Salvage: `reviewer.py` (SM-2, gap), `eval_quality.py` (metrics + redline).
```
def sm2_schedule(quality: int, prev: dict|None) -> dict   # {ef, interval, reps, due_ts}
def find_gaps(questions, evals, claims, profile) -> list[SkillGap]
def quality_report(questions, evals, claims, cfg) -> dict   # pass rate / grounding / type dist; redlines
```
Tests: SM-2 math vs known vectors; quality thresholds + redline flags.

### T9 knowledge  ->  coach/knowledge/{public_kb.py, grow.py}   (owner: worker-3, blockedBy T5,T1)
Salvage: `public_kb.py`, `knowledge_grow.py`.
```
def search_public(query, cfg, *, top_k) -> list[RetrievalHit]   # scope=public, source label framework@version
def fuse_private_public(private: list[RetrievalHit], public: list[RetrievalHit], k=60) -> list[RetrievalHit]
def grow_topic(topic: str, cfg, gateway, *, use_web=False) -> Path   # LLM-generate -> public_kb/{slug}.md; PII gate; idempotent
```
Tests: scope labeling + RRF fuse; PII gate blocks; idempotent overwrite; mock gateway.

### T10 resume  ->  coach/resume/{parse.py, analyze.py, optimize.py, benchmark.py}   (owner: worker-4, blockedBy T1,T5)
Salvage: `resume_parser.py` (PII redaction, ResumeProfile), `resume_optimize.py` (STAR, redflag).
```
# parse.py
def redact_pii(text: str) -> tuple[str, dict]            # phone/email/id/bank -> masked + coverage
def parse_resume(pdf_or_text, *, llm=None) -> ResumeProfile
# analyze.py
def health_report(profile, target_role, retr) -> dict    # match score, coverage heatmap, gap 3-class, ATS
# optimize.py
def classify_skill_gap(profile, target_role) -> list[SkillGap]
def optimize(profile, evidence, gateway) -> dict         # STAR rewrite + citation check + flag unsupported numbers
# benchmark.py
def benchmark_competitors(resume_dir, cfg) -> dict       # parse competitor pack -> target-role profile + gap vs user
```
Tests: PII redaction on text fixture (no real PDF); unsupported-number redflag; gap 3-class. Mock LLM.

### T11 interview (CENTERPIECE)  ->  coach/interview/{rounds.py, prober.py, graph.py}   (owner: worker-1, blockedBy T1,T5,T4,T6)
Salvage: `graph.py` (LangGraph usage), `gen_questions.py` (question gen), `interview.py` (session loop).
```
# rounds.py
ROUNDS = ["tech_basics","project_deep_dive","sql","scenario","hr"]
def select_question(state: InterviewState, retr, memory, gateway) -> Question   # uses weakpoints + real evidence
# prober.py
def make_followup(question, answer, evidence: list[RetrievalHit], gateway) -> str   # adversarial, grounded in real code
# graph.py
def build_graph(deps) -> CompiledGraph
    # nodes: route -> ask -> (interrupt: await_answer) -> score -> probe -> decide -> review -> done
    # SqliteSaver checkpoint; interrupt()/Command(resume=) for human answer; budgeted followups (cfg max_followups)
def start_session(target_role, cfg) -> InterviewState
def step(session_id, user_answer, cfg) -> InterviewState
```
Tests: graph builds; one full turn with fake gateway + fake retrieval returns scored eval + a grounded followup;
checkpoint resume after simulated restart.

### T12 server  ->  coach/server.py   (owner: worker-1, blockedBy T11)
FastAPI. WS `/interview/ws` (turn loop), SSE via `StreamingResponse` for streamed tokens,
read endpoints `/api/questions|/api/memory|/api/resume|/api/review`. See section 10 contract.
Tests: TestClient on read endpoints (offline); ws smoke.

### T13 cli  ->  coach/cli.py   (owner: worker-3, blockedBy T3,T11,T10,T8)
argparse subcommands (lazy imports): `ingest, interview, resume, review, export, kb, serve`.
`def main(argv=None) -> int`. Tests: dispatch + `--help` for each subcommand.

### T14 frontend  ->  web/   (owner: worker-6 designer, build against section 10 contract)
Vue3 + Vite + Pinia chat UI: interview conversation stream, score card side panel, weakpoint heatmap,
memory timeline, resume panel. Write source files only (do NOT run npm — no network; user runs
`npm install && npm run dev` later). `vite.config.js` proxies `/api` and `/interview/ws` to :8000.

### T15 verify   (owner: verifier — assigned later by lead)
Run full `pytest`, smoke-import every module with the iprep python, report failures.

## 7. Conventions
- Code + comments + docstrings in **ASCII/English**. User-facing strings and LLM prompts may be Chinese.
- **No `tenant_id`, no multi-tenant tables, no Role Profile ontology.** Single user; switch target role via config.
- Type hints everywhere; prefer pure functions; small files.
- Absolute imports: `from coach.schemas import ...`, `from coach.llm.gateway import LLMGateway`.

## 8. Testing
- One `tests/test_<module>.py` per module. Run: `& 'D:\anaconda3\envs\iprep\python.exe' -m pytest tests/ -q`.
- Offline only. Use `tmp_path` for files. Mock the gateway. `pytest.importorskip` for tree-sitter/torch paths
  if you cannot exercise them deterministically — but PREFER testing the fallback path so coverage stays green here.

## 9. Feature flags (config-driven, default OFF except placeholders)
`retrieval.geodesic`, `memory.fold`, `memory.tidal` default **false**; `memory.placeholders` default **true**.
Flagged code must be importable and unit-tested, but never required for the base interview loop.

## 10. API contract (server <-> frontend)
- `POST /api/interview/start {target_role} -> {session_id, question}`
- `POST /api/interview/answer {session_id, answer} -> {evaluation, next: question|followup|done}`
- `GET  /api/interview/{session_id} -> InterviewState`
- `WS   /interview/ws` -> messages `{type: question|followup|score|done, payload}`
- `GET  /api/memory -> {episodes, semantic, weakpoints}`
- `GET  /api/resume -> {profile, health_report}`
- `GET  /api/review -> {schedule, quality_report}`
All responses JSON `{ok: bool, data, message}`. Read endpoints must degrade gracefully (empty + ok:true).
