from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator

from eirel.groups import FamilyId, ensure_family_id


def utcnow() -> datetime:
    return datetime.now(UTC)


# -- Score primitives -----------------------------------------------------


class ScoreComponent(BaseModel):
    name: str
    weight: float = Field(ge=0.0)
    score: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None


class JudgeResult(BaseModel):
    model: str
    rubric_name: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str
    latency_seconds: float = Field(ge=0.0)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    constraint_flags: list[str] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoreResult(BaseModel):
    family_id: FamilyId
    miner_hotkey: str
    overall_score: float = Field(ge=0.0, le=1.0)
    components: list[ScoreComponent] = Field(default_factory=list)
    judge: JudgeResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_score_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))


# -- Benchmark task primitives -------------------------------------------


class BenchmarkTask(BaseModel):
    task_id: str
    family_id: FamilyId
    prompt: str
    task_mode: str | None = None
    execution_mode: str | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    retrieval_constraints: dict[str, Any] = Field(default_factory=dict)
    expected_output: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_task_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))


class SpecialistBenchmarkTask(BaseModel):
    task_id: str
    family_id: FamilyId
    prompt: str
    category: str
    difficulty: Literal["standard", "hard", "expert"] = "standard"
    benchmark_version: str | None = None
    rubric_version: str | None = None
    risk_tags: list[str] = Field(default_factory=list)
    task_family: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    expected_output: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_specialist_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))


class FamilyEvaluationTask(BaseModel):
    task_id: str
    family_id: FamilyId
    prompt: str
    task_mode: str | None = None
    execution_mode: str | None = None
    mode: Literal["instant", "thinking"] | None = None
    domain: str | None = None
    category: str = ""
    difficulty: Literal["standard", "hard", "expert"] = "standard"
    # Whether the task's prompt is expected to need live web search. Mirrors
    # the end-user toggle; the baseline reads this directly. Without the
    # explicit field Pydantic would drop it on bundle validation, so the
    # dashboard would always render "no web".
    web_search: bool = False
    risk_tags: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    retrieval_constraints: dict[str, Any] = Field(default_factory=dict)
    expected_output: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_evaluation_task_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))


class FamilyEvaluationBundle(BaseModel):
    kind: Literal["family_evaluation_bundle"] = "family_evaluation_bundle"
    run_id: str | None = None
    family_id: FamilyId
    benchmark_version: str
    rubric_version: str
    tasks: list[FamilyEvaluationTask] = Field(default_factory=list)
    retrieval_environment: dict[str, Any] | None = None
    allowed_tool_policy: dict[str, Any] | None = None
    judge_config: dict[str, Any] | None = None
    policy_version: str | None = None
    source_artifacts: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_bundle_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))


class BenchmarkRunContext(BaseModel):
    run_id: str = Field(validation_alias=AliasChoices("run_id", "epoch_id"))
    family_id: FamilyId
    rubric_version: str
    benchmark_version: str = "v1"
    retrieval_environment: dict[str, Any] | None = None
    policy_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def epoch_id(self) -> str:
        return self.run_id

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_context_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))


class MinerBenchmarkTarget(BaseModel):
    hotkey: str
    endpoint: str
    stake: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FamilyEpochScore(BaseModel):
    run_id: str = Field(validation_alias=AliasChoices("run_id", "epoch_id"))
    family_id: FamilyId
    evaluation_plane: Literal["family_protocol"] = "family_protocol"
    benchmark_version: str
    rubric_version: str
    miner_scores: dict[str, float] = Field(default_factory=dict)
    normalized_weights: dict[str, float] = Field(default_factory=dict)
    judge_outputs: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    miner_responses: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def epoch_id(self) -> str:
        return self.run_id

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_epoch_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))


class BenchmarkTaskRun(BaseModel):
    task_id: str
    family_id: FamilyId
    prompt: str
    expected_output: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_task_run_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))


# -- general_chat 4D evaluation models -----------------------------------


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    metadata: dict[str, Any] | None = None


class TraceEntry(BaseModel):
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result_digest: str = ""
    # First ~2 KB of the tool response body, lowercased, used by the scoring
    # pipeline to verify that a cited URL's surrounding sentence actually
    # overlaps with the fetched content. Stored per-entry rather than via
    # sha256 digest so semantic grounding can be checked cheaply.
    result_body_excerpt: str = Field(default="", max_length=2048)
    latency_ms: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationTrace(BaseModel):
    conversation_id: str = ""
    entries: list[TraceEntry] = Field(default_factory=list)

    def total_latency_ms(self) -> int:
        return sum(int(entry.latency_ms) for entry in self.entries)

    def total_cost_usd(self) -> float:
        return float(sum(float(entry.cost_usd) for entry in self.entries))

    def tool_names(self) -> list[str]:
        return [entry.tool_name for entry in self.entries]


class ModeBudget(BaseModel):
    mode: Literal["instant", "thinking"]
    web_search: bool = False
    latency_seconds: float = Field(gt=0.0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    cost_usd: float = Field(gt=0.0, default=0.10)

    @property
    def latency_ms(self) -> int:
        return int(self.latency_seconds * 1000)


class RunBudget(BaseModel):
    max_usd: float = Field(gt=0.0, default=30.0)


class ConversationScore(BaseModel):
    quality: float = Field(ge=0.0, le=1.0)
    latency: float = Field(ge=0.0, le=1.0)
    cost: float = Field(ge=0.0, le=1.0)
    trace_gate: float = Field(ge=0.0, le=1.0)
    total: float = Field(ge=0.0, le=1.0)
    per_dimension: dict[str, float] = Field(default_factory=dict)
    mode: Literal["instant", "thinking"] = "instant"
    web_search: bool = False
    # USD penalty the scoring manager should charge against the run budget
    # when this conversation's trace integrity gate failed. 0.0 when the
    # gate passed or when no penalty was configured.
    trace_gate_penalty_usd: float = Field(ge=0.0, default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MinerGeneralChatScore(BaseModel):
    miner_hotkey: str
    instant_mean: float = Field(ge=0.0, le=1.0, default=0.0)
    thinking_mean: float = Field(ge=0.0, le=1.0, default=0.0)
    blended: float = Field(ge=0.0, le=1.0, default=0.0)
    cost_efficiency: float = Field(ge=0.0, le=1.0, default=0.0)
    run_cost_usd_used: float = Field(ge=0.0, default=0.0)
    run_budget_usd: float = Field(gt=0.0, default=30.0)
    llm_cost_usd: float = Field(ge=0.0, default=0.0)
    tool_cost_usd: float = Field(ge=0.0, default=0.0)
    cost_rejection_count: int = Field(ge=0, default=0)
    # Bad-actor flag: True iff any conversation in this run cited an
    # active honeytoken URL. When True, ``blended`` is forced to 0.0
    # regardless of quality — fabricated citations zero the miner.
    honeytoken_cited: bool = False
    conversation_scores: list[ConversationScore] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# -- Outcome-only agreement evaluation primitives ------------------------


# Ground-truth agreement verdicts produced by the eiretes agreement judge,
# plus "error" for rows where the miner failed or the judge call itself
# failed. "error" never comes from the judge — it's inserted by the
# validator or control plane to mark a missing/failed judgment.
AgreementVerdict = Literal[
    "matches", "partially_matches", "contradicts", "not_applicable", "error",
    "latency_violation",
]


# Scalar mapping used by the aggregation layer. Kept in sync with
# eiretes.models.VERDICT_SCORES; "error" maps to 0 and does not contribute
# to the denominator of the final aggregation (see aggregate_miner_score).
# "latency_violation" is the SLA-failure verdict the validator stamps when a
# miner's response exceeds the mode-specific latency budget — counts as a
# loss (0.0) but unlike "error" it WILL contribute to the denominator so
# slow miners actually drag their average down.
VERDICT_SCORES: dict[str, float] = {
    "matches": 1.0,
    "partially_matches": 0.6,
    "not_applicable": 0.7,
    "contradicts": 0.0,
    "error": 0.0,
    "latency_violation": 0.0,
}


class BaselineResponse(BaseModel):
    """Normalized OpenAI Responses API baseline for a task.

    Filled by the validator's openai_baseline client and stored on the
    TaskEvaluation row so reruns of the same task (e.g. after a validator
    crash and re-claim) don't repeat the baseline call. Citations are
    preserved for dashboard display only — they do not participate in
    scoring.
    """

    response_text: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    raw_output: list[dict[str, Any]] = Field(default_factory=list)
    latency_seconds: float = Field(ge=0.0, default=0.0)
    cost_usd: float = Field(ge=0.0, default=0.0)
    model: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgreementJudgeOutput(BaseModel):
    """Caller-space agreement judge output (post-swap un-randomization).

    Matches the shape returned by eiretes' ``/v1/judge/agreement`` endpoint
    plus the "error" verdict which is synthesized by the validator when
    the judge call itself fails. Citations are NOT present here — they
    are stripped before the judge is called.
    """

    verdict: AgreementVerdict
    agreement_score: float = Field(ge=0.0, le=1.0, default=0.0)
    rationale: str = ""
    swap_applied: bool = False
    model: str = ""
    rubric_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgreementMinerResult(BaseModel):
    """One miner's agreement result for a task — payload shape the validator
    sends to owner-api's submit-result endpoint."""

    miner_hotkey: str
    miner_response: dict[str, Any] = Field(default_factory=dict)
    # Extracted from the miner_response for dashboard display; NOT used in
    # scoring. Empty list if the miner didn't emit citations.
    miner_citations: list[dict[str, Any]] = Field(default_factory=list)
    judge_output: AgreementJudgeOutput | None = None
    verdict: AgreementVerdict
    agreement_score: float = Field(ge=0.0, le=1.0, default=0.0)
    latency_seconds: float = Field(ge=0.0, default=0.0)
