from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, Field, field_validator

from eirel.groups import FamilyId, ensure_family_id


def utcnow() -> datetime:
    return datetime.now(UTC)


class TaskConstraints(BaseModel):
    max_latency_ms: int | None = Field(default=8000, ge=1)
    modalities_allowed: list[str] = Field(default_factory=lambda: ["text"])
    quality_tier: Literal["standard", "premium"] = "standard"
    families_excluded: list[FamilyId] = Field(default_factory=list)

    @field_validator("families_excluded", mode="before")
    @classmethod
    def normalize_families_excluded(cls, value: Any) -> list[str]:
        items = value if isinstance(value, list) else []
        return [ensure_family_id(str(item)) for item in items]


class TaskObject(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    raw_input: str
    structured_input: dict[str, Any] | None = None
    mode: Literal["sync", "async"] = "sync"
    user_id: str = "anonymous"
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    context_history: list[dict[str, Any]] = Field(default_factory=list)
    constraints: TaskConstraints = Field(default_factory=TaskConstraints)
    billing: dict[str, Any] = Field(default_factory=dict)
    status: Literal["received", "classified", "executing", "complete", "failed"] = "received"
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RequiredFamily(BaseModel):
    node_id: str
    family_id: FamilyId
    order: int = Field(ge=1)
    parallel_with: list[str] = Field(default_factory=list)
    input_node_ids: list[str] = Field(default_factory=list)
    subtask: str
    required: bool = True

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))

    @field_validator("parallel_with", "input_node_ids", mode="before")
    @classmethod
    def normalize_node_id_lists(cls, value: Any) -> list[str]:
        items = value if isinstance(value, list) else []
        return [str(item) for item in items]


class RoutingPlan(BaseModel):
    primary_goal: str
    task_type: Literal["conversational", "creative", "analytical", "agentic"]
    required_families: list[RequiredFamily]
    mode: Literal["sync", "async"]
    classifier_version: str = "eirel-managed-routing"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionNode(BaseModel):
    node_id: str
    family_id: FamilyId | None = None
    subtask: str
    input_node_ids: list[str] = Field(default_factory=list)
    parallel_with: list[str] = Field(default_factory=list)
    stage_index: int = Field(default=1, ge=1)
    composition_role: Literal["specialist", "synthesis", "control_plane"] = "specialist"
    execution_owner: Literal["family_worker", "control_plane"] = "family_worker"
    node_type: str = "family_worker"
    condition: str | None = None
    max_iterations: int = Field(default=1, ge=1)
    quality_gate: dict[str, Any] = Field(default_factory=dict)
    loop_back_to: str | None = None
    miner_hotkey: str | None = None
    miner_endpoint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_execution_family_id(cls, value: Any) -> str | None:
        if value is None:
            return None
        return ensure_family_id(str(value))

    @field_validator("input_node_ids", "parallel_with", mode="before")
    @classmethod
    def normalize_execution_lists(cls, value: Any) -> list[str]:
        items = value if isinstance(value, list) else []
        return [str(item) for item in items]


class ExecutionDAG(BaseModel):
    task_id: str
    mode: Literal["sync", "async"]
    nodes: list[ExecutionNode]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class ExecutionNodeResult(BaseModel):
    node_id: str
    family_id: FamilyId | None = None
    status: Literal["planned", "completed", "failed", "skipped"] = "planned"
    output: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = Field(default=0, ge=0)
    miner_hotkey: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_result_family_id(cls, value: Any) -> str | None:
        if value is None:
            return None
        return ensure_family_id(str(value))


class ExecutionResult(BaseModel):
    task_id: str
    status: Literal["planned", "completed", "failed"] = "planned"
    nodes: list[ExecutionNodeResult]
    final_output: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    completed_at: datetime | None = None


class AttributionContribution(BaseModel):
    family_id: FamilyId
    miner_hotkey: str
    depth: int = Field(ge=1)
    contribution_weight: float = Field(ge=0.0)
    latency_ms: int = Field(default=0, ge=0)

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_contribution_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))


class AttributionRecord(BaseModel):
    task_id: str
    pipeline_depth: int = Field(ge=1)
    contributions: list[AttributionContribution]
    query_volume_families: list[FamilyId] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("query_volume_families", mode="before")
    @classmethod
    def normalize_query_volume_families(cls, value: Any) -> list[str]:
        items = value if isinstance(value, list) else []
        return [ensure_family_id(str(item)) for item in items]


class ConsumerSession(BaseModel):
    session_id: str
    user_id: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class MinerRegistryEntry(BaseModel):
    hotkey: str
    uid: int | None = None
    family_id: FamilyId
    endpoint: str
    stake: float = Field(default=0.0, ge=0.0)
    latency_score: float = Field(default=1.0, ge=0.0)
    quality_score: float = Field(default=0.0, ge=0.0)
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_registry_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))


class FamilyScoreSnapshot(BaseModel):
    run_id: str = Field(validation_alias=AliasChoices("run_id", "epoch_id"))
    family_id: FamilyId
    evaluation_plane: Literal["family_protocol"] = "family_protocol"
    miner_scores: dict[str, float] = Field(default_factory=dict)
    query_volume_share: float = Field(default=0.0, ge=0.0)
    rubric_version: str

    # Per-miner query volume share (hotkey -> fraction)
    miner_query_volume_shares: dict[str, float] = Field(default_factory=dict)

    # Per-miner score breakdown summary (hotkey -> {component: score})
    miner_score_breakdowns: dict[str, dict[str, float]] = Field(default_factory=dict)

    # Robustness scores per miner if available
    miner_robustness_scores: dict[str, float] = Field(default_factory=dict)

    # Anti-gaming flags per miner if available
    miner_anti_gaming_flags: dict[str, list[str]] = Field(default_factory=dict)

    # Evaluation metadata
    task_count: int = 0
    judge_model: str | None = None
    evaluation_timestamp: str | None = None

    @property
    def epoch_id(self) -> str:
        return self.run_id

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_snapshot_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))


class BenchmarkRunRecord(BaseModel):
    run_id: str = Field(validation_alias=AliasChoices("run_id", "epoch_id"))
    family_id: FamilyId
    evaluation_plane: Literal["family_protocol"] = "family_protocol"
    benchmark_version: str
    rubric_version: str
    miner_responses: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    judge_outputs: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    score_snapshot: FamilyScoreSnapshot
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def epoch_id(self) -> str:
        return self.run_id

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_run_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))


class WorkflowSpecNode(BaseModel):
    node_id: str
    role_id: str
    family_id: FamilyId
    description: str
    input_node_ids: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    expected_deliverables: dict[str, Any] = Field(default_factory=dict)
    scoring_hooks: dict[str, Any] = Field(default_factory=dict)
    condition: str | None = None
    max_iterations: int = Field(default=1, ge=1)
    quality_gate: dict[str, Any] = Field(default_factory=dict)
    loop_back_to: str | None = None
    terminal: bool = False

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_spec_node_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))

    @field_validator("input_node_ids", "allowed_tools", mode="before")
    @classmethod
    def normalize_spec_node_lists(cls, value: Any) -> list[str]:
        items = value if isinstance(value, list) else []
        return [str(item) for item in items]


class WorkflowSpec(BaseModel):
    workflow_spec_id: str
    workflow_version: str
    workflow_class: str
    description: str
    nodes: list[WorkflowSpecNode] = Field(default_factory=list)
    allowed_tool_policy: dict[str, Any] = Field(default_factory=dict)
    episode_budget: dict[str, Any] = Field(default_factory=dict)
    expected_deliverables: dict[str, Any] = Field(default_factory=dict)
    scoring_hooks: dict[str, Any] = Field(default_factory=dict)
    credit_assignment_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowEpisodeNode(BaseModel):
    node_id: str
    role_id: str
    family_id: FamilyId
    input_node_ids: list[str] = Field(default_factory=list)
    miner_hotkey: str | None = None
    endpoint: str | None = None
    episode_input: dict[str, Any] = Field(default_factory=dict)
    checkpoint_state: dict[str, Any] = Field(default_factory=dict)
    tool_budget: dict[str, Any] = Field(default_factory=dict)
    latency_budget_ms: int | None = Field(default=None, ge=1)
    artifact_requirements: dict[str, Any] = Field(default_factory=dict)
    deadline: datetime | None = None
    resume_token: str | None = None
    trace_policy: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_episode_node_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))

    @field_validator("input_node_ids", mode="before")
    @classmethod
    def normalize_episode_node_inputs(cls, value: Any) -> list[str]:
        items = value if isinstance(value, list) else []
        return [str(item) for item in items]


class WorkflowEpisode(BaseModel):
    episode_id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_spec_id: str
    workflow_version: str
    workflow_class: str
    run_id: str | None = None
    task_prompt: str
    nodes: list[WorkflowEpisodeNode] = Field(default_factory=list)
    initial_context: dict[str, Any] = Field(default_factory=dict)
    hidden_slice: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class NodeTrace(BaseModel):
    episode_id: str
    node_id: str
    role_id: str
    family_id: FamilyId
    miner_hotkey: str | None = None
    status: Literal["completed", "failed", "skipped"] = "completed"
    input_digest: str | None = None
    output_digest: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    artifact_refs: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: int = Field(default=0, ge=0)
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost_tao: float = Field(default=0.0, ge=0.0)
    retry_count: int = Field(default=0, ge=0)
    runtime_contract_mode: str | None = None
    runtime_state_patch: dict[str, Any] = Field(default_factory=dict)
    checkpoint_events: list[dict[str, Any]] = Field(default_factory=list)
    handoff_payload: dict[str, Any] = Field(default_factory=dict)
    local_role_score_hint: float | None = Field(default=None, ge=0.0, le=1.0)
    reliability_score_hint: float | None = Field(default=None, ge=0.0, le=1.0)
    recovery_score_hint: float | None = Field(default=None, ge=0.0, le=1.0)
    counterfactual_final_outcome_score: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_node_trace_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))


class HandoffTrace(BaseModel):
    episode_id: str
    edge_id: str
    from_node_id: str
    to_node_id: str
    usefulness_score: float = Field(ge=0.0, le=1.0)
    payload_digest: str | None = None
    accepted: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContributionScoreRecord(BaseModel):
    episode_id: str
    node_id: str
    role_id: str
    family_id: FamilyId
    miner_hotkey: str | None = None
    local_role_score: float = Field(ge=0.0, le=1.0)
    handoff_score: float = Field(ge=0.0, le=1.0)
    marginal_lift_score: float = Field(ge=0.0, le=1.0)
    recovery_score: float = Field(ge=0.0, le=1.0)
    reliability_score: float = Field(ge=0.0, le=1.0)
    contribution_score: float = Field(ge=0.0, le=1.0)
    credit_assignment_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_contribution_record_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))


class WorkflowEpisodeResult(BaseModel):
    episode_id: str
    workflow_spec_id: str
    workflow_version: str
    workflow_class: str
    status: Literal["completed", "failed"] = "completed"
    final_output: dict[str, Any] = Field(default_factory=dict)
    final_outcome_score: float = Field(default=0.0, ge=0.0, le=1.0)
    node_traces: list[NodeTrace] = Field(default_factory=list)
    handoff_traces: list[HandoffTrace] = Field(default_factory=list)
    contribution_records: list[ContributionScoreRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Orchestrator models
# ---------------------------------------------------------------------------


class PlatformToolInvocation(BaseModel):
    """Record of a platform tool (Tier 1) invocation by the orchestrator."""
    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)


class FamilyDecision(BaseModel):
    """Record of the orchestrator's decision to route to a specialist family."""
    family_id: FamilyId
    subtask: str
    miner_hotkey: str | None = None
    miner_endpoint: str | None = None
    status: Literal["planned", "completed", "failed", "skipped"] = "planned"
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_decision_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))


class OrchestratorComposition(BaseModel):
    """Full trace of an orchestrator request — routing, tools, and specialist calls."""
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str = "anonymous"
    prompt: str = ""
    route_type: Literal["direct", "platform_tool", "specialist", "composite"] = "direct"
    workflow_template: str = "direct_response"
    task_type: Literal["conversational", "creative", "analytical", "agentic"] = "conversational"
    platform_tool_invocations: list[PlatformToolInvocation] = Field(default_factory=list)
    family_decisions: list[FamilyDecision] = Field(default_factory=list)
    final_output: dict[str, Any] = Field(default_factory=dict)
    status: Literal["completed", "partial", "failed"] = "completed"
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    routing_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)

