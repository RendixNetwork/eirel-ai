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

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=utcnow, nullable=False)

    @property
    def epoch_id(self) -> str:
        return self.run_id


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


class OwnerDatasetBinding(Base):
    __tablename__ = "owner_dataset_bindings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    family_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    bundle_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    bundle_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    generator_version: Mapped[str] = mapped_column(String(128), nullable=False)
    generated_by: Mapped[str] = mapped_column(String(128), nullable=False)
    signature_hex: Mapped[str] = mapped_column(String(256), nullable=False)
    generator_provider: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    generator_model: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)
    provenance_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )
    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=False), nullable=True
    )

    __table_args__ = (
        UniqueConstraint("family_id", "run_id", name="uq_owner_dataset_bindings_family_run"),
        Index("idx_owner_dataset_bindings_family_status", "family_id", "status"),
    )
