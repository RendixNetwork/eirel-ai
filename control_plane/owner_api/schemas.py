from __future__ import annotations

"""Request/response Pydantic models for the owner API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from shared.contracts.models import (
    WorkflowEpisode,
    WorkflowEpisodeResult,
)


class RunTargetResponse(BaseModel):
    run_id: str
    family_id: str
    benchmark_version: str
    rubric_version: str
    judge_model: str
    status: str
    members: list[dict[str, Any]] = Field(default_factory=list)
    evaluation_bundle: dict[str, Any] | None = None
    evaluation_bundle_artifact: dict[str, Any] | None = None
    judge_config: dict[str, Any] | None = None
    retrieval_environment: dict[str, Any] | None = None
    allowed_tool_policy: dict[str, Any] | None = None
    policy_version: str | None = None


class AggregateRunScoreResponse(BaseModel):
    run_id: str
    family_id: str
    validator_submission_count: int
    required_quorum: int
    quorum_reached: bool
    validators: list[str] = Field(default_factory=list)
    late_submission_count: int = 0
    status: str
    aggregate_snapshot: dict[str, Any] | None = None
    official_scoring_version: str | None = None
    scoring_policy_version: str | None = None
    scored_miner_count: int = 0
    eligible_scored_miner_count: int = 0
    top_official_family_score: float | None = None
    winner_submission_id: str | None = None
    gate_passed_candidate_count: int = 0
    consistency_passed_candidate_count: int = 0
    top_candidate_gate_status: str | None = None
    top_candidate_consistency_status: str | None = None
    winner_gate_status: str | None = None
    winner_gate_failures: list[dict[str, Any]] = Field(default_factory=list)
    winner_consistency_status: str | None = None
    analyst_variant_scores: dict[str, Any] = Field(default_factory=dict)
    analyst_variant_gate_status: dict[str, Any] = Field(default_factory=dict)
    analyst_track_scores: dict[str, Any] = Field(default_factory=dict)
    analyst_track_gate_status: dict[str, Any] = Field(default_factory=dict)
    analyst_protocol_status: dict[str, Any] = Field(default_factory=dict)
    analyst_truth_metrics: dict[str, Any] = Field(default_factory=dict)
    analyst_trace_metrics: dict[str, Any] = Field(default_factory=dict)
    analyst_anti_gaming_metrics: dict[str, Any] = Field(default_factory=dict)
    analyst_failure_taxonomy: dict[str, Any] = Field(default_factory=dict)


class RolloutFreezeRequest(BaseModel):
    rollout_frozen: bool
    reason: str | None = None


class InternalArtifactUpload(BaseModel):
    family_id: str
    artifact_kind: str
    content_base64: str
    mime_type: str | None = None
    deployment_id: str | None = None
    submission_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    retained_for_run_id: str | None = None


class ServingReleaseTriggerRequest(BaseModel):
    force: bool = False
    candidate_overrides: dict[str, str] = Field(default_factory=dict)


class WorkflowEpisodeUpload(BaseModel):
    episode: WorkflowEpisode
    result: WorkflowEpisodeResult


class WorkflowEpisodeRegisterPayload(BaseModel):
    episode: WorkflowEpisode


class WorkflowEpisodeLeaseRequest(BaseModel):
    worker_id: str
    lease_seconds: int = Field(default=60, ge=1)
    active_node_id: str | None = None
    active_role_id: str | None = None


class WorkflowEpisodeHeartbeatRequest(BaseModel):
    worker_id: str
    lease_seconds: int = Field(default=60, ge=1)
    queue_state: str = "executing"
    active_node_id: str | None = None
    active_role_id: str | None = None
    checkpoint_state: dict[str, Any] = Field(default_factory=dict)
    runtime_state_patch: dict[str, Any] = Field(default_factory=dict)
    resume_tokens: dict[str, str] = Field(default_factory=dict)
    deferred_node_ids: list[str] = Field(default_factory=list)
    metadata_patch: dict[str, Any] = Field(default_factory=dict)


class WorkflowEpisodeCompleteRequest(BaseModel):
    worker_id: str | None = None
    status: str = "completed"
    error_text: str | None = None
    final_outcome_score: float | None = None


class WorkflowEpisodeRequeueRequest(BaseModel):
    reason: str | None = None


class WorkflowEpisodeDeadLetterRequest(BaseModel):
    reason: str | None = None


class WorkflowEpisodeAdminFinalizeRequest(BaseModel):
    status: str = "cancelled"
    error_text: str | None = None
    final_outcome_score: float | None = None


class WorkflowEpisodeCancelRequest(BaseModel):
    reason: str | None = None
    requested_by: str = "operator"
    cancellation_source: str = "operator"


class WorkflowEpisodeSelectionUpdateRequest(BaseModel):
    episode: WorkflowEpisode
    metadata_patch: dict[str, Any] = Field(default_factory=dict)


class WorkflowIncidentRemediationRequest(BaseModel):
    action: str
    dry_run: bool = False
    reason: str | None = None
    episode_ids: list[str] = Field(default_factory=list)
    incident_state: str | None = None
    workflow_spec_id: str | None = None
    task_id: str | None = None
    lease_owner: str | None = None
    run_id: str | None = None
    limit: int = Field(default=50, ge=0)
    offset: int = Field(default=0, ge=0)


class RuntimeRemediationRequest(WorkflowIncidentRemediationRequest):
    trigger_worker_recover: bool = True
    trigger_worker_run_once: bool = True
    run_once_non_blocking: bool = True


class RuntimeRemediationPolicyRequest(BaseModel):
    dry_run: bool = False
    reason: str | None = None
    run_id: str | None = None
    workflow_spec_id: str | None = None
    cooldown_seconds: int | None = Field(default=None, ge=0)
    max_actions: int | None = Field(default=None, ge=0)
    trigger_worker_recover: bool = True
    trigger_worker_run_once: bool = True
    run_once_non_blocking: bool = True


class RuntimeRemediationSuppressionRequest(BaseModel):
    target_kind: str
    target_value: str
    reason: str
    created_by: str = "operator"
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Distributed evaluation task schemas
# ---------------------------------------------------------------------------


class TaskClaimRequest(BaseModel):
    run_id: str | None = None
    batch_size: int = Field(default=1, ge=1, le=40)


class TaskClaimMiner(BaseModel):
    """One miner target in the fan-out list the validator must call for a task."""

    hotkey: str
    endpoint: str
    auth_headers: dict[str, str] = Field(default_factory=dict)


class TaskClaimItem(BaseModel):
    task_evaluation_id: str
    run_id: str
    family_id: str
    task_id: str
    task_index: int
    task_payload: dict[str, Any]
    miners: list[TaskClaimMiner] = Field(default_factory=list)
    claim_expires_at: str
    judge_config: dict[str, Any] | None = None
    rubric_version: str | None = None
    benchmark_version: str | None = None


class TaskClaimResponse(BaseModel):
    tasks: list[TaskClaimItem] = Field(default_factory=list)
    remaining_task_count: int = 0


class TaskResultSubmission(BaseModel):
    baseline_response: dict[str, Any] | None = None
    miner_results: list[dict[str, Any]] = Field(default_factory=list)
    validator_hotkey: str | None = None
    judge_model: str | None = None


class TaskResultResponse(BaseModel):
    task_evaluation_id: str
    status: str
    family_evaluation_complete: bool = False
    remaining_task_count: int = 0


class MinerEvaluationProgress(BaseModel):
    miner_hotkey: str
    judgments_received: int = 0


class EvaluationStatusResponse(BaseModel):
    run_id: str
    family_id: str
    total_tasks: int
    pending_tasks: int
    claimed_tasks: int
    evaluated_tasks: int
    miners: list[MinerEvaluationProgress] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Weight serving schemas (validators poll this to set on-chain weights)
# ---------------------------------------------------------------------------


class FamilyWinnerWeight(BaseModel):
    family_id: str
    winner_hotkey: str | None = None
    family_weight: float = 0.0
    official_family_score: float = 0.0


class WeightsResponse(BaseModel):
    run_id: str | None = None
    weights: dict[str, float] = Field(default_factory=dict)
    family_winners: list[FamilyWinnerWeight] = Field(default_factory=list)
    ready: bool = False


# ---------------------------------------------------------------------------
# Owner dataset binding schemas (Phase 4 — owner-signed admin endpoints)
# ---------------------------------------------------------------------------


class DatasetBindingResponse(BaseModel):
    id: str
    family_id: str
    run_id: str
    bundle_uri: str
    bundle_sha256: str
    generator_version: str
    generated_by: str
    signature_hex: str
    generator_provider: str
    generator_model: str
    status: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    activated_at: datetime | None = None


class DatasetBindingListResponse(BaseModel):
    family_id: str
    bindings: list[DatasetBindingResponse] = Field(default_factory=list)
