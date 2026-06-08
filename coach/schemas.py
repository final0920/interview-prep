"""Shared data contracts for interview-coach.

Every module imports its models from here so interfaces stay consistent across
the codebase. Pydantic v2.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Evidence / retrieval ---------------------------------------------------

class Channel(str, Enum):
    code = "code"
    sql = "sql"
    doc = "doc"
    config = "config"


class SourceScope(str, Enum):
    private = "private"   # the user's own resume / code
    public = "public"     # shared general knowledge base (L0, no PII)


class EvidenceUnit(BaseModel):
    """One retrievable chunk of evidence, traceable to file:symbol:line."""
    id: str
    source_path: str = ""
    symbol: str = ""
    start_line: int = 0
    end_line: int = 0
    channel: Channel = Channel.code
    lang: str = ""
    text: str = ""
    content_hash: str = ""
    repo: str = ""
    tags: list[str] = Field(default_factory=list)

    @property
    def ref(self) -> str:
        if self.symbol:
            return f"{self.source_path}:{self.symbol}:{self.start_line}"
        return f"{self.source_path}:{self.start_line}"


class RetrievalHit(BaseModel):
    evidence: EvidenceUnit
    score: float = 0.0
    rank: int = 0
    scope: SourceScope = SourceScope.private
    retriever: str = ""   # bm25 | dense | rrf | rerank | geodesic


# --- Questions / answers ----------------------------------------------------

class QuestionType(str, Enum):
    tech_basics = "tech_basics"
    project_deep_dive = "project_deep_dive"
    sql = "sql"
    scenario = "scenario"
    hr = "hr"


class Question(BaseModel):
    id: str
    type: QuestionType
    prompt: str
    difficulty: str = "mid"            # easy | mid | hard
    linked_evidence: list[str] = Field(default_factory=list)  # evidence ids or file:line refs
    key_points: list[str] = Field(default_factory=list)
    followups: list[str] = Field(default_factory=list)
    reference_answer: str = ""
    tags: list[str] = Field(default_factory=list)
    verified: bool = True             # sql sandbox verification flag


class Verdict(str, Enum):
    passed = "pass"
    needs_fix = "needs_fix"


class AnswerEvaluation(BaseModel):
    question_id: str
    user_answer: str = ""
    score: int = 0                    # 0-100
    verdict: Verdict = Verdict.needs_fix
    key_points_hit: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    fabrication_flags: list[str] = Field(default_factory=list)
    followup: str = ""                # next adversarial probe
    grounding_rate: float = 0.0


class ClaimVerdict(str, Enum):
    verified = "verified"
    needs_evidence = "needs_evidence"
    rejected = "rejected"


class ClaimCheck(BaseModel):
    claim: str
    verdict: ClaimVerdict = ClaimVerdict.needs_evidence
    evidence_ids: list[str] = Field(default_factory=list)
    score: float = 0.0


# --- Memory (L2 episodic / L3 semantic) ------------------------------------

class MemoryEpisode(BaseModel):
    id: Optional[str] = None
    ts: float = 0.0
    kind: str = "event"               # event | diary | review
    content: str = ""
    tags: list[str] = Field(default_factory=list)
    round: str = ""
    score: Optional[int] = None


class MemorySemantic(BaseModel):
    key: str
    value: str
    kind: str = "skill"               # skill | weakpoint | preference | profile
    confidence: float = 0.5
    valid_from: float = 0.0
    valid_to: Optional[float] = None  # None => currently valid (Zep-style temporal window)
    updated_ts: float = 0.0


# --- Resume -----------------------------------------------------------------

class SkillGapCategory(str, Enum):
    have = "have"
    transferable = "transferable"
    missing = "missing"


class SkillGap(BaseModel):
    skill: str
    category: SkillGapCategory
    evidence_ids: list[str] = Field(default_factory=list)
    note: str = ""


class ResumeProject(BaseModel):
    name: str = ""
    role: str = ""
    description: str = ""
    tech: list[str] = Field(default_factory=list)
    offset: int = 0                   # char offset back into raw resume text


class ResumeProfile(BaseModel):
    basics: dict = Field(default_factory=dict)
    education: list[dict] = Field(default_factory=list)
    experiences: list[dict] = Field(default_factory=list)
    projects: list[ResumeProject] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    raw_text: str = ""


# --- Interview session (LangGraph state) -----------------------------------

class InterviewTurn(BaseModel):
    question: Question
    answer: str = ""
    evaluation: Optional[AnswerEvaluation] = None


class InterviewState(BaseModel):
    """Canonical mock-interview session state.

    The LangGraph graph may mirror this as a TypedDict internally, but this is
    the serialized contract used across the engine, server, and storage.
    """
    session_id: str = ""
    target_role: str = ""
    round: str = "tech_basics"
    rounds_remaining: list[str] = Field(default_factory=list)
    current_question: Optional[Question] = None
    turns: list[InterviewTurn] = Field(default_factory=list)
    followup_count: int = 0
    weakpoints: list[str] = Field(default_factory=list)
    phase: str = "route"              # route | ask | await_answer | score | probe | decide | review | done
    finished: bool = False
