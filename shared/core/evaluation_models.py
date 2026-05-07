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


class EvaluationConversationTurn(BaseModel):
    """One turn in a multi-turn evaluation fixture.

    ``user`` is always the user message for that turn. ``assistant`` is
    optional:
      * **None (default)** — *live* mode. The validator calls the miner
        and the baseline for that turn; whatever each one replies is
        appended to its own private history before the next user turn.
      * **set** — *scripted* mode. Both miner and baseline see the
        identical canned assistant message in their history; neither is
        called for that turn. Used to set up specific multi-turn probes
        (clarification, contradiction, reference-resolution, etc.).

    The final turn is always live — it produces the assistant answer the
    pairwise judge scores.

    Distinct from the runtime ``ConversationTurn`` (defined later in
    this module) which uses ``role`` / ``content`` for in-flight
    history; this fixture model is the on-disk schema for a multi-turn
    benchmark task and uses ``user`` / ``assistant`` to make scripted
    vs live turns visually obvious.
    """

    user: str
    assistant: str | None = None


OracleSource = Literal["three_oracle", "deterministic"]

# Legacy ``oracle_source`` values published by older bundles describing
# the pool's gold-provenance shape (live_lookup baked from a structured
# endpoint, sandbox computed deterministically, etc.). All map to
# ``deterministic`` under the new architecture — the validator's
# 3-oracle enrichment is opt-in via ``oracle_source="three_oracle"``,
# which legacy bundles never set.
_LEGACY_DETERMINISTIC_ORACLE_SOURCES: frozenset[str] = frozenset({
    "live_endpoint",
    "live_endpoint_composed",
    "deterministic_grader",
    "gpt5_oracle",  # eirel-eval-pool's prior single-GPT-5 path
    "planted_fact",
    "document_span",
    "sandbox_reference",
    "url_fetch_cache",
})


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
    # Multi-turn evaluation fixture. When set, the validator replays the
    # ``turns`` script against the miner and the baseline (each building
    # its own history) and judges the *final* assistant answer only.
    # Single-turn tasks leave this None and use ``prompt``.
    turns: list[EvaluationConversationTurn] | None = None
    risk_tags: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    retrieval_constraints: dict[str, Any] = Field(default_factory=dict)
    expected_output: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Tells the validator whether to run 3-oracle enrichment at
    # task-claim time. ``three_oracle`` → fan out OpenAI/Gemini/Grok +
    # Chutes reconciler, produce ``expected_claims`` in-memory,
    # consumed by ``_judge_miner`` for every miner that judges this
    # task. ``deterministic`` → skip enrichment; the pool's built-in
    # grader (live_endpoint / sandbox_python / span F1 / regex) is the
    # truth. None defaults to ``deterministic`` for back-compat with
    # bundles published before this field existed.
    oracle_source: OracleSource | None = None

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_evaluation_task_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))

    @field_validator("oracle_source", mode="before")
    @classmethod
    def normalize_oracle_source(cls, value: Any) -> str | None:
        """Accept both new enum values and legacy gold-provenance tags.

        Legacy bundles publish ``oracle_source`` as a free-form string
        describing the pool's grader shape (``live_endpoint`` /
        ``deterministic_grader`` / etc.). Map those to ``deterministic``
        so existing pool deployments keep loading without coordinating
        a lockstep cutover. New bundles emit ``three_oracle`` /
        ``deterministic`` directly.
        """
        if value is None:
            return None
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned in ("three_oracle", "deterministic"):
            return cleaned
        if cleaned in _LEGACY_DETERMINISTIC_ORACLE_SOURCES:
            return "deterministic"
        # Unknown value — drop to None rather than crash the load. The
        # validator treats None as "deterministic, no enrichment" so a
        # mistagged item still gets judged; operators see the
        # unfamiliar value preserved nowhere on the loaded task.
        return None


class RagBundleDocument(BaseModel):
    """One document inside a bundle-level RAG corpus."""

    doc_id: str
    content: str
    title: str | None = None


class RagBundleCorpus(BaseModel):
    """A corpus the validator must index into the rag-tool-service
    before tasks fan out. Every ``rag_required`` task references one
    of these by ``corpus_id``."""

    corpus_id: str
    documents: list[RagBundleDocument] = Field(default_factory=list)


class FamilyEvaluationBundle(BaseModel):
    kind: Literal["family_evaluation_bundle"] = "family_evaluation_bundle"
    run_id: str | None = None
    family_id: FamilyId
    benchmark_version: str
    rubric_version: str
    tasks: list[FamilyEvaluationTask] = Field(default_factory=list)
    # RAG eval — corpora the validator must POST to the rag-tool-service
    # at run-start so ``rag.retrieve`` calls have an indexed source.
    # Empty / missing on bundles without rag_required tasks (back-compat).
    corpora: list[RagBundleCorpus] = Field(default_factory=list)
    retrieval_environment: dict[str, Any] | None = None
    allowed_tool_policy: dict[str, Any] | None = None
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
    total: float = Field(ge=0.0, le=1.0)
    per_dimension: dict[str, float] = Field(default_factory=dict)
    mode: Literal["instant", "thinking"] = "instant"
    web_search: bool = False
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
    """Pairwise-comparator reference for a task.

    Filled by the validator from the chosen oracle's cached answer
    (no separate per-task model call) and stored on the TaskEvaluation
    row so reruns of the same task don't re-run reconciliation.
    Citations are preserved for dashboard display only — they do not
    participate in scoring.
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
