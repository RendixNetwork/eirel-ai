from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from shared.common.enums import AssignmentStatus, CanonicalResultStatus, EvaluationStatus, SubmissionStatus
from shared.common.manifest import SubmissionManifest


class SubmissionResponse(BaseModel):
    id: str
    miner_hotkey: str
    submission_seq: int
    status: SubmissionStatus
    submission_block: int
    manifest: SubmissionManifest
    created_at: datetime
    cancel_requested_at: datetime | None = None


class ClaimResponse(BaseModel):
    assignment_id: str
    submission_task_id: str
    capability: str
    task_id: str
    task_sequence: int
    window_membership_version: str
    lease_expires_at: datetime
    artifact_download_url: str
    submission: SubmissionResponse


class HeartbeatResponse(BaseModel):
    continue_running: bool = Field(alias="continue")
    cancel_requested: bool
    lease_expires_at: datetime
    submission_status: SubmissionStatus


class EvaluationUpload(BaseModel):
    assignment_id: str
    submission_task_id: str
    submission_id: str
    capability: str
    task_id: str
    task_sequence: int
    window_membership_version: str
    validator_hotkey: str
    status: EvaluationStatus
    overall_score: float | None = None
    category_scores: dict[str, float] | None = None
    raw_result: dict[str, Any]
    started_at: datetime
    finished_at: datetime | None = None
    agent_image_digest: str | None = None
    failure_reason: str | None = None


class CanonicalResultResponse(BaseModel):
    submission_id: str
    status: CanonicalResultStatus
    validator_count: int
    overall_score: float | None
    aggregation_method: str | None = None
    summary: dict[str, Any]


class ValidatorResponse(BaseModel):
    hotkey: str
    uid: int
    stake: int
    is_active: bool
    last_synced_at: datetime


class AssignmentRelease(BaseModel):
    status: AssignmentStatus
    reason: str | None = None


class EnvWindowTask(BaseModel):
    task_id: str
    task_sequence: int
    score_bearing: bool
    window_membership_version: str
    weight: float = 1.0


class EnvWindowConfig(BaseModel):
    capability: str
    enabled: bool
    min_completeness: float
    weight: float
    evaluation_split: str | None = None
    epoch_id: str | None = None
    latest_window_membership_version: str | None = None
    tasks: list[EnvWindowTask] = Field(default_factory=list)


class SubmissionPoolEntry(BaseModel):
    submission: SubmissionResponse
    artifact_download_url: str
    archive_sha256: str
    env_capabilities: list[str]


class ValidatorScoreReportUpload(BaseModel):
    submission_id: str
    submission_seq: int
    miner_hotkey: str
    validator_hotkey: str
    env_completeness: dict[str, float]
    env_scores: dict[str, float | None]
    env_eligibility: dict[str, bool]
    env_window_versions: dict[str, str]
    globally_eligible: bool
    weighted_score: float | None = None
    dominance_state: str | None = None
    soft_terminate_recommended: bool = False
    soft_terminate_after: datetime | None = None
    raw_report: dict[str, Any] = Field(default_factory=dict)


class TaskEventUpload(BaseModel):
    submission_id: str
    submission_seq: int
    miner_hotkey: str
    validator_hotkey: str
    capability: str
    task_id: str
    task_sequence: int
    window_membership_version: str
    status: EvaluationStatus
    score: float | None = None
    env_completeness: float | None = None
    env_eligible: bool | None = None
    failure_reason: str | None = None
    raw_event: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class AggregatedRankingResponse(BaseModel):
    submission_id: str
    miner_hotkey: str
    eligible_report_count: int
    report_count: int
    participating_stake: int
    weighted_score: float | None
    env_scores: dict[str, float]
    env_completeness: dict[str, float]
    eligible_envs: dict[str, float]
    summary: dict[str, Any]
