from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, LargeBinary, String, Text, UniqueConstraint, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class ValidatorRecord(Base):
    """Operator-managed whitelist of validator hotkeys.

    Not synced from the metagraph — validators are added manually (e.g. via
    SQL INSERT or the admin CLI). Only hotkeys in this table can claim
    evaluation tasks and submit scores. `is_active=False` temporarily
    disables a validator without removing the row.
    """

    __tablename__ = "validators"

    hotkey: Mapped[str] = mapped_column(String(128), primary_key=True)
    uid: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stake: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )


class RegisteredNeuron(Base):
    """Hotkeys currently registered on the metagraph (uid:hotkey pairs).

    Populated by the metagraph listener. Rows are deleted when a hotkey
    drops off the metagraph. Presence in this table = registered on chain =
    allowed to submit.
    """

    __tablename__ = "registered_neurons"

    hotkey: Mapped[str] = mapped_column(String(128), primary_key=True)
    uid: Mapped[int] = mapped_column(Integer, nullable=False)
    last_synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )


class SubmissionArtifact(Base):
    __tablename__ = "submission_artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    archive_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    manifest_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )


class ManagedMinerSubmission(Base):
    __tablename__ = "managed_miner_submissions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    miner_hotkey: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    submission_seq: Mapped[int] = mapped_column(Integer, nullable=False)
    family_id: Mapped[str] = mapped_column("family_id", String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    artifact_id: Mapped[str] = mapped_column(
        ForeignKey("submission_artifacts.id"), nullable=False, index=True
    )
    manifest_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    archive_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    submission_block: Mapped[int] = mapped_column(BigInteger, nullable=False)
    introduced_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    extrinsic_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )


class ManagedDeployment(Base):
    __tablename__ = "managed_deployments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    submission_id: Mapped[str] = mapped_column(
        ForeignKey("managed_miner_submissions.id"), nullable=False, index=True
    )
    miner_hotkey: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    family_id: Mapped[str] = mapped_column("family_id", String(64), nullable=False, index=True)
    deployment_revision: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    image_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    health_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    health_details_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    pending_runtime_stop: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false"),
    )
    requested_cpu_millis: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    requested_memory_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    placement_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="pending")
    assigned_node_name: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    assigned_cpu_millis: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    assigned_memory_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    placement_error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms_p50: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    active_set_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    benchmark_version: Mapped[str] = mapped_column(String(64), nullable=False, default="hivemind_v1")
    rubric_version: Mapped[str] = mapped_column(String(64), nullable=False, default="governance-v1")
    judge_model: Mapped[str] = mapped_column(String(128), nullable=False, default="local-rubric-judge")
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )


class DeploymentScoreRecord(Base):
    __tablename__ = "deployment_score_records"
    __table_args__ = (
        Index("idx_score_run_family", "epoch_id", "family_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column("epoch_id", String(128), nullable=False, index=True)
    family_id: Mapped[str] = mapped_column("family_id", String(64), nullable=False, index=True)
    deployment_id: Mapped[str] = mapped_column(
        ForeignKey("managed_deployments.id"), nullable=False, index=True
    )
    submission_id: Mapped[str] = mapped_column(
        ForeignKey("managed_miner_submissions.id"), nullable=False, index=True
    )
    miner_hotkey: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    deployment_revision: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    raw_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    normalized_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_eligible: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    run_budget_usd: Mapped[float] = mapped_column(Float, nullable=False, default=30.0)
    run_cost_usd_used: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    llm_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    tool_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cost_rejection_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )

    @property
    def epoch_id(self) -> str:
        return self.run_id


class EpochTargetSnapshot(Base):
    __tablename__ = "epoch_target_snapshots"
    __table_args__ = (
        UniqueConstraint("epoch_id", "family_id", name="uq_snapshot_run_family"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column("epoch_id", String(128), nullable=False, index=True)
    family_id: Mapped[str] = mapped_column("family_id", String(64), nullable=False, index=True)
    benchmark_version: Mapped[str] = mapped_column(String(64), nullable=False)
    rubric_version: Mapped[str] = mapped_column(String(64), nullable=False)
    judge_model: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open", index=True)
    scored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    scored_by_validator_hotkey: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True
    )
    frozen_validator_stakes_json: Mapped[dict[str, int]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    members_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )

    @property
    def epoch_id(self) -> str:
        return self.run_id


class ValidatorScoreSubmission(Base):
    __tablename__ = "validator_score_submissions"
    __table_args__ = (
        UniqueConstraint(
            "epoch_id",
            "family_id",
            "validator_hotkey",
            name="uq_validator_score_submissions_epoch_family_validator",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column("epoch_id", String(128), nullable=False, index=True)
    family_id: Mapped[str] = mapped_column("family_id", String(64), nullable=False, index=True)
    validator_hotkey: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="accepted", index=True)
    effective_stake: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    miner_responses_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    judge_outputs_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )

    @property
    def epoch_id(self) -> str:
        return self.run_id


class AggregateFamilyScoreSnapshot(Base):
    __tablename__ = "aggregate_family_score_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "epoch_id",
            "family_id",
            name="uq_aggregate_family_score_snapshots_epoch_family",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column("epoch_id", String(128), nullable=False, index=True)
    family_id: Mapped[str] = mapped_column("family_id", String(64), nullable=False, index=True)
    snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    validator_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validator_hotkeys_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    validator_weights_json: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False, default=dict)
    consensus_method: Mapped[str] = mapped_column(
        String(64), nullable=False, default="distributed_task_evaluation"
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )

    @property
    def epoch_id(self) -> str:
        return self.run_id


class MetagraphSyncSnapshot(Base):
    __tablename__ = "metagraph_sync_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    netuid: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    network: Mapped[str] = mapped_column(String(64), nullable=False)
    validator_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    miner_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="success")
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )


class FamilyRolloutState(Base):
    __tablename__ = "family_rollout_states"

    family_id: Mapped[str] = mapped_column("family_id", String(64), primary_key=True)
    rollout_frozen: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    freeze_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="open")
    benchmark_version: Mapped[str] = mapped_column(String(64), nullable=False)
    rubric_version: Mapped[str] = mapped_column(String(64), nullable=False)
    judge_model: Mapped[str] = mapped_column(String(128), nullable=False)
    min_scores_json: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )


class RunFamilyResult(Base):
    __tablename__ = "run_family_results"
    __table_args__ = (
        UniqueConstraint("run_id", "family_id", name="uq_run_family_results_run_family"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_runs.id"), nullable=False, index=True
    )
    family_id: Mapped[str] = mapped_column("family_id", String(64), nullable=False, index=True)
    winner_deployment_id: Mapped[str | None] = mapped_column(
        ForeignKey("managed_deployments.id"), nullable=True, index=True
    )
    winner_submission_id: Mapped[str | None] = mapped_column(
        ForeignKey("managed_miner_submissions.id"), nullable=True, index=True
    )
    winner_hotkey: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    best_raw_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    min_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    has_winner: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    top_deployment_ids_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )


class ServingRelease(Base):
    __tablename__ = "serving_releases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    trigger_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="pending")
    scheduled_for: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )


class ServingDeployment(Base):
    __tablename__ = "serving_deployments"
    __table_args__ = (
        UniqueConstraint("release_id", "family_id", name="uq_serving_deployments_release_family"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    release_id: Mapped[str] = mapped_column(
        ForeignKey("serving_releases.id"), nullable=False, index=True
    )
    family_id: Mapped[str] = mapped_column("family_id", String(64), nullable=False, index=True)
    source_deployment_id: Mapped[str] = mapped_column(
        ForeignKey("managed_deployments.id"), nullable=False, index=True
    )
    source_submission_id: Mapped[str] = mapped_column(
        ForeignKey("managed_miner_submissions.id"), nullable=False, index=True
    )
    miner_hotkey: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    source_deployment_revision: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    endpoint: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="pending")
    health_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="starting")
    health_details_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    requested_cpu_millis: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    requested_memory_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    placement_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="pending")
    assigned_node_name: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    assigned_cpu_millis: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    assigned_memory_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    placement_error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    drain_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )


class DeploymentHealthEvent(Base):
    __tablename__ = "deployment_health_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    deployment_id: Mapped[str] = mapped_column(
        ForeignKey("managed_deployments.id"), nullable=False, index=True
    )
    family_id: Mapped[str] = mapped_column("family_id", String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )


class RuntimeNodeSnapshot(Base):
    __tablename__ = "runtime_node_snapshots"

    node_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    pool_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True, default="miner")
    labels_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    ready: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    schedulable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    allocatable_cpu_millis: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    allocatable_memory_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    allocatable_pod_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    derived_pod_capacity: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    verification_error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    last_verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=utcnow)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )


class ManagedArtifact(Base):
    __tablename__ = "managed_artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    deployment_id: Mapped[str | None] = mapped_column(
        ForeignKey("managed_deployments.id"), nullable=True, index=True
    )
    submission_id: Mapped[str | None] = mapped_column(
        ForeignKey("managed_miner_submissions.id"), nullable=True, index=True
    )
    family_id: Mapped[str] = mapped_column("family_id", String(64), nullable=False, index=True)
    artifact_kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    storage_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    storage_uri: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="stored", index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    retained_for_run_id: Mapped[str | None] = mapped_column("retained_for_epoch_id", String(128), nullable=True, index=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )

    @property
    def retained_for_epoch_id(self) -> str | None:
        return self.retained_for_run_id


class ConsumerSessionState(Base):
    __tablename__ = "consumer_sessions"

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    latest_task_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    messages_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    # Per-session user toggles, owned by the orchestrator and applied to
    # every turn dispatched to a family agent. Persisted so the
    # frontend can render the current state on reconnect, and so the
    # toggles survive page reloads. Defaults match the family-side
    # defaults (instant + no search).
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="instant")
    web_search: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )


class TaskRequestRecord(Base):
    __tablename__ = "task_requests"

    task_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    raw_input: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="queued")
    queue_state: Mapped[str] = mapped_column(
        String(32), nullable=False, index=True, default="queued"
    )
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=False), nullable=True, index=True
    )
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    constraints_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    routing_plan_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    execution_dag_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    execution_result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    context_package_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    attribution_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    queued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )


class DagExecutionRecord(Base):
    __tablename__ = "dag_executions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    task_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="queued")
    queue_state: Mapped[str] = mapped_column(
        String(32), nullable=False, index=True, default="queued"
    )
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    errors_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    final_output_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )


class DagNodeExecutionRecord(Base):
    __tablename__ = "dag_node_executions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    execution_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    node_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    family_id: Mapped[str | None] = mapped_column("family_id", String(64), nullable=True, index=True)
    attempt_index: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    miner_hotkey: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    miner_endpoint: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)


class WorkflowEpisodeRecord(Base):
    __tablename__ = "workflow_episode_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    episode_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    run_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    workflow_spec_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    workflow_version: Mapped[str] = mapped_column(String(64), nullable=False)
    workflow_class: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed", index=True)
    queue_state: Mapped[str] = mapped_column(String(32), nullable=False, default="queued", index=True)
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True, index=True)
    active_node_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    active_role_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    last_worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    last_node_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    last_role_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    checkpoint_state_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    runtime_state_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    resume_tokens_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    deferred_node_ids_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    next_eligible_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True, index=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_failure_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    dead_lettered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True, index=True)
    dead_letter_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True, index=True)
    cancel_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_requested_by: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    cancellation_source: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    task_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    episode_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    final_outcome_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=utcnow, nullable=False)
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    last_checkpoint_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)


class TaskEvaluation(Base):
    """Task-level evaluation tracking.

    Each row represents one task in a given run/family. A validator claims a
    task, fans out to all miners with that task, calls OpenAI for the baseline
    response, and judges each miner response pairwise against the baseline.
    Per-miner results are stored in `TaskMinerResult`.
    """

    __tablename__ = "task_evaluations"
    __table_args__ = (
        UniqueConstraint(
            "epoch_id", "family_id", "task_id",
            name="uq_task_eval_run_family_task",
        ),
        Index("idx_te_claimable", "epoch_id", "family_id", "status", "claim_expires_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column("epoch_id", String(128), nullable=False, index=True)
    family_id: Mapped[str] = mapped_column("family_id", String(64), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    task_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Status: pending -> claimed -> evaluated | baseline_failed
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)

    # Claim tracking
    claimed_by_validator: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    claim_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True, index=True)
    claim_attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Baseline response populated by validator from OpenAI Responses API
    baseline_response_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=utcnow, nullable=False)

    @property
    def epoch_id(self) -> str:
        return self.run_id


class TaskMinerResult(Base):
    """Per-miner pairwise result for a given TaskEvaluation.

    One row per (task, miner). Stores the miner's response, the pairwise judge
    output vs the baseline, and the raw overall score (no anti-gaming modifiers
    applied — the pairwise verdict is the primary signal).
    """

    __tablename__ = "task_miner_results"
    __table_args__ = (
        UniqueConstraint(
            "task_evaluation_id", "miner_hotkey",
            name="uq_task_miner_result_task_miner",
        ),
        Index("idx_tmr_miner", "epoch_id", "family_id", "miner_hotkey"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    task_evaluation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("task_evaluations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Denormalized for aggregation queries
    run_id: Mapped[str] = mapped_column("epoch_id", String(128), nullable=False, index=True)
    family_id: Mapped[str] = mapped_column("family_id", String(64), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    miner_hotkey: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    miner_response_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    # Miner's cited URLs extracted from miner_response_json. Stored
    # separately for dashboard display — NOT used in scoring.
    miner_citations_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list,
    )
    judge_output_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # matches | partially_matches | contradicts | not_applicable | error |
    # latency_violation (mode-specific budget exceeded; counts as a loss).
    agreement_verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    # Scalar derived from the verdict (VERDICT_SCORES) or 0 on error.
    agreement_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Wall-clock time the miner took to return its response (seconds). Used
    # for leaderboard display and for the latency-violation scoring gate.
    miner_latency_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Wall-clock time the agreement judge took (seconds). Diagnostic only —
    # historically named `latency_seconds`; the attr name is preserved for
    # back-compat with existing seed/migration code.
    latency_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Per-task LLM cost the miner incurred against the subnet
    # provider-proxy, in USD. Sourced server-side from the
    # provider-proxy ledger via owner-api injection — never trusts
    # miner self-report. Zero when the miner made no proxied LLM calls
    # or when the cost lookup failed.
    proxy_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Per-task judge LLM cost in USD. Reported by eiretes-judge in its
    # response metadata; the validator passes it through verbatim.
    # Always >= 0.
    judge_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Graph-runtime: link an eval result back to a graph checkpoint
    # thread + the specific checkpoint id the miner emitted. Both
    # nullable for backwards compat with BaseAgent-style miners.
    thread_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True,
    )
    checkpoint_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True,
    )

    # Multi-metric per-task scoring. Each dimension is computed
    # independently (pairwise from /v1/judge/pairwise; grounded /
    # retrieval / safety from /v1/judge/multi; tool_routing /
    # latency_cost / computation_correctness deterministically by the
    # validator). All nullable so a metric marked N/A for the task type
    # re-normalizes out cleanly. ``final_task_score`` is the weighted
    # sum after re-normalization. ``applied_weights_json`` records what
    # weights actually applied; ``applicable_metrics_json`` is the set
    # of dimensions that had a real score (non-N/A).
    pairwise_preference_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    grounded_correctness: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieval_quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    tool_routing: Mapped[float | None] = mapped_column(Float, nullable=True)
    instruction_safety: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    computation_correctness: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_task_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    applied_weights_json: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)
    applicable_metrics_json: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    task_type: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=utcnow, nullable=False)

    @property
    def epoch_id(self) -> str:
        return self.run_id


class GraphCheckpoint(Base):
    """One persisted graph-runtime checkpoint.

    The miner SDK's ``PostgresCheckpointer`` writes here over HTTP
    via ``/v1/internal/checkpoints/{thread_id}`` so the miner pod
    never holds DB credentials.
    """

    __tablename__ = "graph_checkpoints"
    __table_args__ = (
        UniqueConstraint(
            "thread_id", "checkpoint_id",
            name="uq_graph_checkpoint_thread_checkpoint",
        ),
        Index(
            "idx_chk_deployment_thread",
            "deployment_id", "thread_id",
        ),
        Index(
            "idx_chk_thread_created",
            "thread_id", "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()),
    )
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    checkpoint_id: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_checkpoint_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    deployment_id: Mapped[str] = mapped_column(
        ForeignKey("managed_deployments.id"), nullable=False, index=True,
    )
    family_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    checkpoint_namespace: Mapped[str | None] = mapped_column(String(128), nullable=True)
    node: Mapped[str | None] = mapped_column(String(128), nullable=True)
    state_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    pending_writes_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    blob_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False, index=True,
    )


class ConversationThread(Base):
    """Long-lived conversation thread anchor, mirroring ConsumerSessionState.

    Lets the orchestrator pin a multi-turn conversation to a single
    ``thread_id`` (and therefore a single miner deployment) so
    checkpoint resume works without per-turn pod affinity.
    """

    __tablename__ = "conversation_threads"
    __table_args__ = (
        Index("idx_thread_user", "user_id"),
        Index("idx_thread_deployment", "deployment_id"),
    )

    thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    deployment_id: Mapped[str] = mapped_column(
        ForeignKey("managed_deployments.id"), nullable=False,
    )
    family_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    last_checkpoint_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )


class MinerEvaluationSummary(Base):
    """Aggregated per-miner score computed from individual task results."""
    __tablename__ = "miner_evaluation_summaries"
    __table_args__ = (
        UniqueConstraint(
            "epoch_id", "family_id", "miner_hotkey",
            name="uq_miner_eval_summary_run_family_miner",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column("epoch_id", String(128), nullable=False, index=True)
    family_id: Mapped[str] = mapped_column("family_id", String(64), nullable=False, index=True)
    miner_hotkey: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    total_tasks: Mapped[int] = mapped_column(Integer, nullable=False)
    completed_tasks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_tasks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    family_capability_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    robustness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    anti_gaming_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    official_family_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    protocol_gate_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # pending -> scoring -> scored -> error
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)
    rollout_metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=utcnow, nullable=False)

    @property
    def epoch_id(self) -> str:
        return self.run_id


class WorkflowRuntimeSuppressionRecord(Base):
    __tablename__ = "workflow_runtime_suppressions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    target_kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    target_value: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False, default="operator")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True, index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=utcnow, nullable=False)


class WorkflowRuntimePolicyStateRecord(Base):
    __tablename__ = "workflow_runtime_policy_state"

    state_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=utcnow, nullable=False)


# ---------------------------------------------------------------------------
# Product runtime tables
#
# These power the consumer-facing path. Miner pods never see real user data;
# the product orchestrator loads from these tables on every turn, builds an
# AgentInvocationRequest, and forwards to a ServingDeployment (the promoted
# eval-winner's image). User state survives across promotions because it
# lives here, not in any miner deployment.
# ---------------------------------------------------------------------------


class ConsumerUser(Base):
    """One real user of the product layer.

    ``auth_subject`` is whatever the consumer-api authorizer surfaces (today
    an API-key principal; OAuth subject later). Distinct from the miner
    ``hotkey`` namespace.
    """

    __tablename__ = "consumer_users"

    user_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()),
    )
    auth_subject: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )


class ConsumerProject(Base):
    """A 'chat project' container — like ChatGPT projects.

    Custom instructions inject into ``metadata.project_context`` on every
    turn. Project memory (vector recall over user-uploaded docs) is keyed
    by ``project_id`` in :class:`ConsumerProjectMemory`.
    """

    __tablename__ = "consumer_projects"
    __table_args__ = (
        Index("idx_consumer_project_user", "user_id"),
    )

    project_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("consumer_users.user_id"), nullable=False, index=True,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    custom_instructions: Mapped[str | None] = mapped_column(Text, nullable=True)
    default_family_id: Mapped[str] = mapped_column(
        String(64), nullable=False, default="general_chat",
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )


class ConsumerConversation(Base):
    """One user-visible thread.

    Distinct from the SDK's graph ``thread_id`` — the orchestrator
    generates a fresh graph thread per turn and passes the user's
    history via ``AgentInvocationRequest.history``.
    """

    __tablename__ = "consumer_conversations"
    __table_args__ = (
        Index("idx_consumer_conv_user", "user_id"),
        Index("idx_consumer_conv_project", "project_id"),
        Index("idx_consumer_conv_user_updated", "user_id", "last_message_at"),
    )

    conversation_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("consumer_users.user_id"), nullable=False, index=True,
    )
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("consumer_projects.project_id"), nullable=True, index=True,
    )
    family_id: Mapped[str] = mapped_column(
        String(64), nullable=False, default="general_chat", index=True,
    )
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_message_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=False), nullable=True, index=True,
    )
    # Rolling summary of the head of the conversation. Populated by
    # ConversationSummarizer when the verbatim tail accumulates past a
    # configurable threshold; injected as a system message into
    # request.history so the agent sees compressed context for the head
    # plus verbatim recent turns. NULL until first summarization.
    rolling_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_summarized_message_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )


class ConsumerMessage(Base):
    """One user or assistant turn within a conversation.

    On every turn, the orchestrator loads the last N messages (default 20)
    and folds them into ``request.history`` so the agent code never has to
    persist user-visible state itself.

    ``served_by_*`` fields are an audit trail: which serving release /
    deployment answered. Useful for cost reconciliation and for pulling a
    user back to the same code if a promotion regresses.
    """

    __tablename__ = "consumer_messages"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id", "turn_idx",
            name="uq_consumer_message_conversation_turn",
        ),
        Index("idx_consumer_msg_conversation", "conversation_id", "turn_idx"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()),
    )
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("consumer_conversations.conversation_id"),
        nullable=False, index=True,
    )
    turn_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user|assistant|system
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    citations_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list,
    )
    tool_calls_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list,
    )
    served_by_deployment_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    served_by_release_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )


class ConsumerPreference(Base):
    """User preference (tone, language, persona, default tools).

    Loaded into ``metadata.user_preferences`` on every turn. ``scope``
    distinguishes a global default from a per-project override.
    """

    __tablename__ = "consumer_preferences"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "scope", "project_id", "key",
            name="uq_consumer_pref_user_scope_proj_key",
        ),
        Index("idx_consumer_pref_user_scope", "user_id", "scope"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("consumer_users.user_id"), nullable=False, index=True,
    )
    scope: Mapped[str] = mapped_column(String(16), nullable=False, default="global")  # global|project
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("consumer_projects.project_id"), nullable=True, index=True,
    )
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    value_json: Mapped[Any] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )


class ConsumerProjectMemory(Base):
    """Per-project vector store for long-lived user docs / recall.

    Loaded as top-K into ``metadata.recalled_memory``. Ingestion (embedding
    job over uploaded docs and over assistant outputs) is a follow-up; the
    schema is wired now so the orchestrator can read from it as soon as
    rows exist.
    """

    __tablename__ = "consumer_project_memory"
    __table_args__ = (
        Index("idx_consumer_memory_project", "project_id"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()),
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("consumer_projects.project_id"), nullable=False, index=True,
    )
    vector_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_message_id: Mapped[str | None] = mapped_column(
        ForeignKey("consumer_messages.id"), nullable=True,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )


class ConsumerUserMemory(Base):
    """Per-user vector store for stable facts and preferences.

    Distinct from :class:`ConsumerProjectMemory`. Project memory is
    scoped to one project's history; user memory is the place for
    durable facts about the user themselves ("works in Python", "prefers
    concise answers", "lives in Tokyo") that should surface across
    every project they touch.

    Populated by :class:`UserMemoryWriter` (gated regex pre-filter →
    extractor LLM call → embedding). Loaded as top-K into
    ``metadata.user_facts`` on every turn.

    The ``kind`` column is informational, not load-bearing — useful for
    UI grouping and for retrieval policies that want to weight different
    kinds of facts differently.
    """

    __tablename__ = "consumer_user_memory"
    __table_args__ = (
        Index("idx_consumer_user_memory_user", "user_id"),
        Index("idx_consumer_user_memory_user_vector", "user_id", "vector_id"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("consumer_users.user_id"), nullable=False, index=True,
    )
    vector_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    kind: Mapped[str] = mapped_column(
        String(32), nullable=False, default="fact",  # fact | preference | skill
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    source_conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("consumer_conversations.conversation_id"), nullable=True,
    )
    source_message_id: Mapped[str | None] = mapped_column(
        ForeignKey("consumer_messages.id"), nullable=True,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )


class ServingPromotion(Base):
    """Audit trail: which eval-winner became today's serving agent.

    One row per (family_id, run_id). Idempotency for the promotion job
    is keyed on the unique constraint below.
    """

    __tablename__ = "serving_promotions"
    __table_args__ = (
        UniqueConstraint(
            "family_id", "run_id",
            name="uq_serving_promotion_family_run",
        ),
        Index("idx_serving_promotion_release", "serving_release_id"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()),
    )
    family_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    source_deployment_id: Mapped[str] = mapped_column(
        ForeignKey("managed_deployments.id"), nullable=False, index=True,
    )
    serving_release_id: Mapped[str] = mapped_column(
        ForeignKey("serving_releases.id"), nullable=False, index=True,
    )
    promoted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class ConsumerAttachment(Base):
    """One uploaded file (PDF / DOCX / CSV / TXT / etc.) for product chat.

    Matches the ChatGPT / Claude pattern: the orchestrator preprocesses
    uploads into extracted text, the agent never sees a "file upload
    tool" — it just sees ``metadata.attached_files`` on the next chat
    turn. This row is the canonical record of what was uploaded and
    what came out of extraction.

    ``blob_ref`` is opaque storage location (S3 key, local path, etc.)
    so we don't dump the raw bytes into Postgres. ``extracted_text``
    is the LLM-ready content; ``extraction_metadata_json`` carries
    page count / row count / etc. for downstream debugging.
    """

    __tablename__ = "consumer_attachments"
    __table_args__ = (
        Index("idx_consumer_attachment_user", "user_id"),
        Index("idx_consumer_attachment_conversation", "conversation_id"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("consumer_users.user_id"), nullable=False, index=True,
    )
    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("consumer_conversations.conversation_id"),
        nullable=True, index=True,
    )
    message_id: Mapped[str | None] = mapped_column(
        ForeignKey("consumer_messages.id"), nullable=True,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    blob_ref: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    extracted_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    extraction_metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    extraction_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="ok",
    )  # ok | unsupported | failed | truncated
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )


class McpIntegration(Base):
    """Operator-curated MCP server in the catalog.

    Consumers connect to integrations *from this catalog*, never to
    arbitrary URLs. The subnet operator vets each integration once
    (auth scopes, capability surface, abuse posture) before adding it;
    consumers see only the active ones.

    ``capabilities_hash`` is sha256 over the canonicalized list of
    tools the integration declared at registration time. The relay
    service rejects calls when an integration's live capability set has
    drifted from the stored hash — operator must run reprobe to refresh
    before consumers can use the new surface.
    """

    __tablename__ = "mcp_integrations"
    __table_args__ = (
        Index("idx_mcp_integration_status", "status"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()),
    )
    slug: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True,
    )
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    vendor: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    base_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    transport: Mapped[str] = mapped_column(
        String(32), nullable=False, default="http",
    )  # http | sse | stdio_via_relay
    capabilities_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list,
    )
    capabilities_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, default="",
    )
    oauth_provider: Mapped[str | None] = mapped_column(
        String(64), nullable=True,
    )
    oauth_authorize_url: Mapped[str | None] = mapped_column(
        String(1024), nullable=True,
    )
    oauth_token_url: Mapped[str | None] = mapped_column(
        String(1024), nullable=True,
    )
    oauth_scopes_json: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list,
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="active", index=True,
    )  # active | disabled
    created_by_admin_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )


class ConsumerMcpConnection(Base):
    """One user's connection to a catalog integration.

    Stores encrypted OAuth tokens keyed on (user_id, integration_id).
    Decryption happens only inside the orchestrator process; tokens
    never leave the orchestrator boundary. The miner pod sees zero
    references to these rows — the orchestrator runs MCP tool calls on
    the user's behalf and injects the results into envelope metadata.
    """

    __tablename__ = "consumer_mcp_connections"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "integration_id",
            name="uq_consumer_mcp_user_integration",
        ),
        Index("idx_consumer_mcp_user", "user_id"),
        Index("idx_consumer_mcp_integration", "integration_id"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("consumer_users.user_id"), nullable=False, index=True,
    )
    integration_id: Mapped[str] = mapped_column(
        ForeignKey("mcp_integrations.id"), nullable=False, index=True,
    )
    oauth_access_token_encrypted: Mapped[bytes | None] = mapped_column(
        LargeBinary, nullable=True,
    )
    oauth_refresh_token_encrypted: Mapped[bytes | None] = mapped_column(
        LargeBinary, nullable=True,
    )
    oauth_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=False), nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="active", index=True,
    )  # active | expired | revoked
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=False), nullable=True,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )


class ConsumerMcpToolCall(Base):
    """Audit log of every MCP tool call brokered for a consumer.

    Stores per-call cost so :class:`RunCostTracker` can attribute spend.
    Result is digested (truncated string) — full result lives only in
    transit; we don't persist user-visible payloads in case they
    contain PII the orchestrator's safety pipeline didn't catch.
    """

    __tablename__ = "consumer_mcp_tool_calls"
    __table_args__ = (
        Index("idx_consumer_mcp_call_conversation", "conversation_id"),
        Index("idx_consumer_mcp_call_connection", "connection_id"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()),
    )
    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("consumer_conversations.conversation_id"),
        nullable=True, index=True,
    )
    message_id: Mapped[str | None] = mapped_column(
        ForeignKey("consumer_messages.id"), nullable=True,
    )
    connection_id: Mapped[str] = mapped_column(
        ForeignKey("consumer_mcp_connections.id"), nullable=False, index=True,
    )
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    args_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    result_digest: Mapped[str] = mapped_column(Text, nullable=False, default="")
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )


class OrchestratorToolCallLog(Base):
    """Server-attested ledger of every tool call brokered by the subnet.

    Each row is written by the tool service that actually executed the
    call (web_search, url_fetch, sandbox; the MCP relay writes a row in
    addition to its consumer-MCP audit row). The ledger is the
    authoritative source the eval pipeline reads to compute tool-use
    KPIs — miner-emitted trace frames carry zero scoring weight.

    Indexed by ``job_id`` so an eval run can fetch every tool call its
    items triggered. Per-row write latency must stay low; tool services
    use a fire-and-forget write-behind buffer rather than blocking the
    response on the DB write.
    """

    __tablename__ = "orchestrator_tool_call_log"
    __table_args__ = (
        Index("idx_otcl_job_id", "job_id"),
        Index("idx_otcl_job_tool", "job_id", "tool_name"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()),
    )
    job_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    args_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    args_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    result_digest: Mapped[str] = mapped_column(
        Text, nullable=False, default="",
    )
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ok")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )


class EvalFeedback(Base):
    """Per-(run, miner, task) eval outcome row.

    The validator writes one row per ``_judge_miner`` invocation
    after computing the EvalJudge outcome + composite. Eiretes'
    ``GET /v1/eval/feedback`` (hotkey-signed, exposed to miners)
    aggregates rows for ``(run_id, miner_hotkey)`` into an
    ``EvalFeedbackDoc`` with categorical guidance per item — the
    miner sees ``failure_mode`` + ``guidance`` per task plus the
    largest-gap summary, but never the verbatim expected_claims.

    Indexed on ``(miner_hotkey, run_id)`` for the per-miner read
    path and ``(run_id, task_id)`` for cross-miner per-task drill-in
    on the operator dashboard.
    """

    __tablename__ = "eval_feedback"
    __table_args__ = (
        Index("idx_eval_feedback_miner_run", "miner_hotkey", "run_id"),
        Index("idx_eval_feedback_run_task", "run_id", "task_id"),
        UniqueConstraint(
            "run_id", "miner_hotkey", "task_id",
            name="uq_eval_feedback_run_miner_task",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()),
    )
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    miner_hotkey: Mapped[str] = mapped_column(String(64), nullable=False)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # EvalJudge output: outcome ∈ {correct, partial, wrong, hallucinated,
    # refused, disputed}; failure_mode is one of the categorical buckets
    # in eiretes/eval/models.py:FailureMode (or null when outcome=correct).
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    failure_mode: Mapped[str | None] = mapped_column(
        String(64), nullable=True,
    )
    # Categorical hint (≤200 chars) the miner can act on — never
    # quotes ``expected_claims`` verbatim.
    guidance: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # Substring previews surfaced in the per-miner doc; full prompt /
    # response stay on TaskMinerResult. ``prompt_excerpt`` ≤200 chars,
    # ``response_excerpt`` ≤500 chars by convention (validator
    # truncates before the POST).
    prompt_excerpt: Mapped[str] = mapped_column(
        Text, nullable=False, default="",
    )
    response_excerpt: Mapped[str] = mapped_column(
        Text, nullable=False, default="",
    )
    composite_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
    )
    # Why composite was zeroed (if it was) — e.g.
    # "tool_attestation_factor=0 because required_tool=web_search but
    # ledger had no calls". JSON list so the dashboard can render
    # one-per-line; empty for happy-path rows.
    knockout_reasons_json: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list,
    )
    # Validator-side reconciler verdict on the 3 oracles
    # (consensus / majority / disputed) or "deterministic" for non-
    # three_oracle items. Lets the miner-facing doc surface
    # "this item was disputed by oracles" so they don't chase a
    # phantom failure.
    oracle_status: Mapped[str | None] = mapped_column(
        String(32), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False,
    )
