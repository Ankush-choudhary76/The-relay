"""Pydantic schemas for The Relay pipeline, retrieval, and telemetry."""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class IntentCategory(str, Enum):
    FACTUAL_LOOKUP = "factual_lookup"
    ARCHITECTURAL_DETAILS = "architectural_details"
    COMPARATIVE = "comparative"
    DEFINITION = "definition"
    MULTIPART = "multipart"
    UNANSWERABLE = "unanswerable"


class IntentResult(BaseModel):
    category: IntentCategory
    confidence: float = Field(ge=0.0, le=1.0)
    sub_queries: List[str] = Field(default_factory=list)
    reasoning: str = ""


class Citation(BaseModel):
    paper_title: str
    arxiv_id: str
    version: str
    page_number: int
    section: Optional[str] = None
    supporting_passage: str


class RetrievedPassage(BaseModel):
    passage_id: str
    paper_id: str
    paper_title: str
    arxiv_id: str
    version: str
    page_number: int
    section: Optional[str] = None
    text: str
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


class EvidenceEvaluation(BaseModel):
    is_sufficient: bool
    score: float
    covered_aspects: List[str] = Field(default_factory=list)
    missing_aspects: List[str] = Field(default_factory=list)
    needs_refinement: bool = False
    refined_query: Optional[str] = None
    reason: str = ""


class CitationVerificationResult(BaseModel):
    is_valid: bool
    verified_citations: List[Citation] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    rejection_reason: Optional[str] = None


class ProviderResponse(BaseModel):
    text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    first_token_ms: float = 0.0
    total_ms: float = 0.0
    status_code: int = 200
    headers: Dict[str, str] = Field(default_factory=dict)
    retries_attempted: int = 0
    is_error: bool = False
    error_message: Optional[str] = None
    provider: str = ""
    model: str = ""
    is_simulated: bool = False


class RelayRequest(BaseModel):
    question: str
    request_id: Optional[str] = None
    stream: bool = False
    scenario: str = "ad_hoc"
    retrieval_setting: str = "setting_b"
    timeout_seconds: Optional[float] = None
    provider: Optional[str] = Field(
        default=None,
        description="Target model provider: 'standin' (safe simulator) or 'groq' (real Groq API).",
    )

    @field_validator("provider", mode="before")
    @classmethod
    def validate_provider(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        norm = str(v).lower().strip()
        if norm in {"standin", "stand-in", "simulator"}:
            return "standin"
        if norm in {"groq", "real"}:
            return "groq"
        raise ValueError(
            f"Invalid model provider '{v}'. Permitted values are 'standin' or 'groq'."
        )


class PipelineStageTimings(BaseModel):
    queue_wait_ms: float = 0.0
    validation_ms: float = 0.0
    intent_ms: float = 0.0
    retrieval_ms: float = 0.0
    refinement_ms: float = 0.0
    evidence_eval_ms: float = 0.0
    generation_ms: float = 0.0
    verification_ms: float = 0.0
    retry_wait_ms: float = 0.0
    total_ms: float = 0.0


class RelayResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    is_refusal: bool = False
    refusal_reason: Optional[str] = None
    status: Literal["ok", "rate_limited", "timeout", "error"] = "ok"

    # Provider and model identification
    provider: str = "standin"
    model: str = "standin-calibrated-lpu"
    is_simulated: bool = True
    source: str = "stand-in"

    # Token telemetry
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    # Latency telemetry
    latency_ms: float = 0.0
    first_token_latency_ms: Optional[float] = None
    first_token_ms: float = 0.0
    total_ms: float = 0.0

    retries: int = 0
    stage_timings: PipelineStageTimings = Field(default_factory=PipelineStageTimings)


# Schema strictly conforming to required results.csv columns
class ResultCSVRow(BaseModel):
    request_id: str
    scenario: str
    source: Literal["real", "stand-in"]
    start_time: str
    first_token_ms: float
    total_ms: float
    status: Literal["ok", "rate_limited", "timeout", "error"]
    retries: int
    tokens_in: int
    tokens_out: int
    retrieval_setting: str
    answer_grade: Optional[Literal["correct", "wrong", "refused_correctly", "refused_wrongly", ""]] = ""

    # Optional diagnostics columns
    queue_wait_ms: Optional[float] = None
    provider_working_ms: Optional[float] = None
    retry_wait_ms: Optional[float] = None
    peak_queue_len: Optional[int] = None

