from __future__ import annotations

"""Workflow episode lifecycle, incident management, and runtime remediation policy.

Extracted from ``ManagedOwnerServices`` to reduce the size of the god-object.
Each public method here has a thin delegation wrapper in ``ManagedOwnerServices``
for backward compatibility.
"""

from datetime import UTC, datetime, timedelta
from typing import Any, TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.common.exceptions import WorkflowEpisodeCancelledError, WorkflowEpisodeLeaseFencedError
from shared.common.models import (
    ConsumerSessionState,
    TaskRequestRecord,
    WorkflowEpisodeRecord,
    WorkflowRuntimePolicyStateRecord,
    WorkflowRuntimeSuppressionRecord,
)
from shared.contracts.models import (
    WorkflowEpisode,
    WorkflowEpisodeResult,
    WorkflowSpec,
)
from shared.workflow_specs import (
    get_workflow_spec,
    list_workflow_specs,
    workflow_corpus_public_metadata,
)

if TYPE_CHECKING:
    from control_plane.owner_api.managed import ManagedOwnerServices


WORKFLOW_EPISODE_DEFAULT_MAX_ATTEMPTS = 3
WORKFLOW_EPISODE_RETRY_BASE_SECONDS = 30
WORKFLOW_EPISODE_RETRY_MAX_SECONDS = 300
WORKFLOW_RUNTIME_REMEDIATION_AUDIT_LIMIT = 20
WORKFLOW_RUNTIME_POLICY_STATE_KEY = "runtime_remediation_policy"
WORKFLOW_RUNTIME_SUPPRESSION_TARGET_KINDS = {
    "episode_id",
    "workflow_spec_id",
    "incident_state",
    "task_id",
}


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class WorkflowManager:
    """Handles workflow episode lifecycle, incident management, and runtime remediation policy."""

    def __init__(self, owner: ManagedOwnerServices) -> None:
        self._owner = owner

    @property
    def db(self):
        return self._owner.db

    @property
    def settings(self):
        return self._owner.settings

    def list_workflow_specs(self) -> list[WorkflowSpec]:
        return list_workflow_specs()

    def get_workflow_spec(self, workflow_spec_id: str) -> WorkflowSpec:
        try:
            return get_workflow_spec(workflow_spec_id)
        except KeyError as exc:
            raise ValueError("workflow spec not found") from exc

    def workflow_spec_payload(self, workflow_spec: WorkflowSpec) -> dict[str, Any]:
        return workflow_spec.model_dump(mode="json")

    def workflow_corpus_public_payload(self) -> dict[str, Any]:
        return workflow_corpus_public_metadata(corpus_root=self.settings.workflow_corpus_root_path)

    def _workflow_episode_lease_expiry(self, *, lease_seconds: int) -> datetime:
        return datetime.fromtimestamp(
            utcnow().timestamp() + max(1, int(lease_seconds)),
            UTC,
        ).replace(tzinfo=None)

    def _workflow_episode_max_attempts(self, *, metadata: dict[str, Any] | None = None) -> int:
        configured = metadata.get("max_attempts") if isinstance(metadata, dict) else None
        if isinstance(configured, int) and configured >= 1:
            return configured
        return WORKFLOW_EPISODE_DEFAULT_MAX_ATTEMPTS

    def _workflow_episode_retry_delay_seconds(self, *, attempt_count: int) -> int:
        exponent = max(0, int(attempt_count) - 1)
        return min(
            WORKFLOW_EPISODE_RETRY_MAX_SECONDS,
            WORKFLOW_EPISODE_RETRY_BASE_SECONDS * (2**exponent),
        )

    def _workflow_episode_next_eligible_at(self, *, attempt_count: int) -> datetime:
        delay_seconds = self._workflow_episode_retry_delay_seconds(attempt_count=attempt_count)
        return datetime.fromtimestamp(
            utcnow().timestamp() + delay_seconds,
            UTC,
        ).replace(tzinfo=None)

    def _workflow_episode_task_execution_payload(self, record: WorkflowEpisodeRecord) -> dict[str, Any] | None:
        metadata_json = dict(record.metadata_json or {})
        task_id = str(metadata_json.get("task_id") or "").strip()
        source = str(metadata_json.get("source") or "").strip()
        if not task_id or source != "task_execution":
            return None
        payload: dict[str, Any] = {
            "task_id": task_id,
            "session_id": str(metadata_json.get("session_id") or "").strip() or None,
            "source": source,
            "task_status_url": f"/v1/tasks/{task_id}",
        }
        user_id = str(metadata_json.get("user_id") or "").strip()
        if user_id:
            payload["user_id"] = user_id
        return payload

    def _workflow_episode_task_execution_context(
        self,
        session: Session,
        *,
        record: WorkflowEpisodeRecord,
    ) -> dict[str, Any] | None:
        payload = self._workflow_episode_task_execution_payload(record)
        if payload is None:
            return None
        task = session.get(TaskRequestRecord, payload["task_id"])
        session_record = (
            session.get(ConsumerSessionState, payload["session_id"])
            if payload.get("session_id")
            else None
        )
        return {
            **payload,
            "task_status": task.status if task is not None else None,
            "session_status": session_record.status if session_record is not None else None,
            "task_created_at": (
                task.queued_at.isoformat() if task is not None and task.queued_at is not None else None
            ),
            "task_completed_at": (
                task.completed_at.isoformat()
                if task is not None and task.completed_at is not None
                else None
            ),
            "session_url": (
                f"/v1/sessions/{session_record.session_id}"
                if session_record is not None
                else (
                    f"/v1/sessions/{payload['session_id']}"
                    if payload.get("session_id")
                    else None
                )
            ),
        }

    def _workflow_episode_is_dead_lettered(self, record: WorkflowEpisodeRecord) -> bool:
        return bool(record.dead_lettered_at or record.queue_state == "dead_lettered" or record.status == "dead_lettered")

    def _workflow_episode_is_cancel_requested(self, record: WorkflowEpisodeRecord) -> bool:
        return bool(record.cancel_requested_at is not None and record.status != "cancelled")

    def _workflow_episode_is_retry_wait(self, record: WorkflowEpisodeRecord, *, now: datetime | None = None) -> bool:
        if self._workflow_episode_is_dead_lettered(record):
            return False
        if record.next_eligible_at is None:
            return False
        return record.next_eligible_at > (now or utcnow())

    def _workflow_episode_is_retryable(self, record: WorkflowEpisodeRecord, *, now: datetime | None = None) -> bool:
        if self._workflow_episode_is_dead_lettered(record):
            return False
        if record.status in {"completed", "cancelled"} or self._workflow_episode_is_cancel_requested(record):
            return False
        if int(record.attempt_count or 0) >= max(1, int(record.max_attempts or WORKFLOW_EPISODE_DEFAULT_MAX_ATTEMPTS)):
            return False
        return not self._workflow_episode_is_retry_wait(record, now=now)

    def _workflow_episode_is_stale(self, record: WorkflowEpisodeRecord, *, now: datetime | None = None) -> bool:
        current = now or utcnow()
        return bool(
            record.queue_state in {"leased", "executing", "deferred"}
            and record.lease_expires_at is not None
            and record.lease_expires_at < current
        )

    def _workflow_episode_incident_state(
        self,
        record: WorkflowEpisodeRecord,
        *,
        now: datetime | None = None,
    ) -> str | None:
        current = now or utcnow()
        if self._workflow_episode_is_dead_lettered(record):
            return "dead_lettered"
        if self._workflow_episode_is_cancel_requested(record):
            return "failed"
        if self._workflow_episode_is_stale(record, now=current):
            return "stale"
        if record.queue_state == "failed" or record.status == "failed":
            return "failed"
        if self._workflow_episode_is_retryable(record, now=current) and int(record.attempt_count or 0) > 0:
            return "retryable"
        if self._workflow_episode_is_retry_wait(record, now=current):
            return "retry_wait"
        return None

    def _workflow_episode_incident_sort_key(
        self,
        record: WorkflowEpisodeRecord,
        *,
        now: datetime,
    ) -> tuple[int, float, str]:
        state = self._workflow_episode_incident_state(record, now=now)
        priority = {
            "dead_lettered": 0,
            "stale": 1,
            "failed": 2,
            "retryable": 3,
            "retry_wait": 4,
        }.get(state or "", 99)
        reference = (
            record.dead_lettered_at
            or record.last_failure_at
            or record.lease_expires_at
            or record.next_eligible_at
            or record.updated_at
        )
        age = 0.0
        if reference is not None:
            age = -(now - reference).total_seconds()
        return (priority, age, record.episode_id)

    def _workflow_episode_mark_retry(
        self,
        *,
        record: WorkflowEpisodeRecord,
        error_text: str | None,
        reason: str,
        now: datetime,
    ) -> None:
        attempt_count = int(record.attempt_count or 0) + 1
        record.attempt_count = attempt_count
        record.retry_count = attempt_count
        record.last_failure_at = now
        record.last_error = error_text or reason
        record.cancel_requested_at = None
        record.cancel_reason = None
        record.cancel_requested_by = None
        record.cancellation_source = None
        if attempt_count < max(1, int(record.max_attempts or WORKFLOW_EPISODE_DEFAULT_MAX_ATTEMPTS)):
            record.status = "queued"
            record.queue_state = "queued"
            record.next_eligible_at = self._workflow_episode_next_eligible_at(
                attempt_count=attempt_count,
            )
            record.dead_lettered_at = None
            record.dead_letter_reason = None
            record.completed_at = None
        else:
            record.status = "dead_lettered"
            record.queue_state = "dead_lettered"
            record.dead_lettered_at = now
            record.dead_letter_reason = reason
            record.next_eligible_at = None
            record.completed_at = now
        record.lease_owner = None
        record.lease_expires_at = None
        record.active_node_id = None
        record.active_role_id = None
        record.updated_at = now

    def _workflow_episode_finalize_terminal(
        self,
        *,
        record: WorkflowEpisodeRecord,
        status: str,
        now: datetime,
        error_text: str | None = None,
        final_outcome_score: float | None = None,
    ) -> None:
        record.status = str(status)
        record.queue_state = str(status)
        record.lease_owner = None
        record.lease_expires_at = None
        record.deferred_node_ids_json = []
        record.resume_tokens_json = {}
        record.next_eligible_at = None
        if status != "dead_lettered":
            record.dead_lettered_at = None
            record.dead_letter_reason = None
        if status == "cancelled":
            if record.cancel_requested_at is None:
                record.cancel_requested_at = now
            if not record.cancel_reason:
                record.cancel_reason = error_text or "cancelled"
            if not record.cancel_requested_by:
                record.cancel_requested_by = "runtime"
            if not record.cancellation_source:
                record.cancellation_source = "runtime"
        if final_outcome_score is not None:
            record.final_outcome_score = float(final_outcome_score)
        if error_text:
            record.last_error = error_text
            record.metadata_json = {
                **dict(record.metadata_json or {}),
                "error_text": error_text,
            }
        elif status == "completed":
            record.last_error = None
        record.completed_at = now
        record.updated_at = now

    def register_workflow_episode(
        self,
        session: Session,
        *,
        episode: WorkflowEpisode,
    ) -> WorkflowEpisodeRecord:
        record = session.execute(
            select(WorkflowEpisodeRecord).where(WorkflowEpisodeRecord.episode_id == episode.episode_id)
        ).scalar_one_or_none()
        if record is None:
            record = WorkflowEpisodeRecord(
                episode_id=episode.episode_id,
                run_id=episode.run_id,
                workflow_spec_id=episode.workflow_spec_id,
                workflow_version=episode.workflow_version,
                workflow_class=episode.workflow_class,
                status="queued",
                queue_state="queued",
                max_attempts=self._workflow_episode_max_attempts(metadata=dict(episode.metadata or {})),
                task_prompt=episode.task_prompt,
                episode_json=episode.model_dump(mode="json"),
                metadata_json=dict(episode.metadata or {}),
            )
            session.add(record)
        else:
            record.run_id = episode.run_id
            record.workflow_spec_id = episode.workflow_spec_id
            record.workflow_version = episode.workflow_version
            record.workflow_class = episode.workflow_class
            record.task_prompt = episode.task_prompt
            record.episode_json = episode.model_dump(mode="json")
            record.metadata_json = {
                **dict(record.metadata_json or {}),
                **dict(episode.metadata or {}),
            }
            if record.status not in {"completed", "failed", "dead_lettered", "cancelled"}:
                record.status = "queued"
            if record.queue_state not in {"leased", "executing", "deferred", "completed", "failed", "dead_lettered", "cancelled"}:
                record.queue_state = "queued"
        record.max_attempts = self._workflow_episode_max_attempts(metadata=dict(record.metadata_json or {}))
        record.attempt_count = 0
        record.retry_count = 0
        record.next_eligible_at = None
        record.last_error = None
        record.last_failure_at = None
        record.dead_lettered_at = None
        record.dead_letter_reason = None
        record.cancel_requested_at = None
        record.cancel_reason = None
        record.cancel_requested_by = None
        record.cancellation_source = None
        record.lease_owner = None
        record.lease_expires_at = None
        record.last_worker_id = None
        record.last_node_id = None
        record.last_role_id = None
        record.active_node_id = None
        record.active_role_id = None
        record.completed_at = None
        record.updated_at = utcnow()
        self._clear_workflow_episode_policy_suppression(record)
        session.flush()
        return record

    def lease_workflow_episode(
        self,
        session: Session,
        *,
        episode_id: str,
        worker_id: str,
        lease_seconds: int,
        active_node_id: str | None = None,
        active_role_id: str | None = None,
    ) -> WorkflowEpisodeRecord:
        record = self.workflow_episode_record(session, episode_id=episode_id)
        if record is None:
            raise ValueError("workflow episode not found")
        now = utcnow()
        if self._workflow_episode_is_cancel_requested(record):
            raise WorkflowEpisodeCancelledError("workflow episode cancelled")
        if record.status in {"completed", "failed", "dead_lettered", "cancelled"}:
            raise ValueError("workflow episode already finalized")
        if record.next_eligible_at is not None and record.next_eligible_at > now:
            raise ValueError("workflow episode not eligible yet")
        if (
            record.lease_expires_at is not None
            and record.lease_expires_at > now
            and record.lease_owner not in {None, worker_id}
        ):
            raise WorkflowEpisodeLeaseFencedError("workflow episode lease fenced")
        record.status = "running"
        record.queue_state = "leased"
        record.lease_owner = worker_id
        record.lease_expires_at = self._workflow_episode_lease_expiry(lease_seconds=lease_seconds)
        if active_node_id is not None:
            record.active_node_id = active_node_id
            record.last_node_id = active_node_id
        if active_role_id is not None:
            record.active_role_id = active_role_id
            record.last_role_id = active_role_id
        record.last_worker_id = worker_id
        record.last_heartbeat_at = now
        record.updated_at = now
        session.flush()
        return record

    def heartbeat_workflow_episode(
        self,
        session: Session,
        *,
        episode_id: str,
        worker_id: str,
        lease_seconds: int,
        queue_state: str | None = None,
        active_node_id: str | None = None,
        active_role_id: str | None = None,
        checkpoint_state: dict[str, Any] | None = None,
        runtime_state_patch: dict[str, Any] | None = None,
        resume_tokens: dict[str, str] | None = None,
        deferred_node_ids: list[str] | None = None,
        metadata_patch: dict[str, Any] | None = None,
    ) -> WorkflowEpisodeRecord:
        record = self.workflow_episode_record(session, episode_id=episode_id)
        if record is None:
            raise ValueError("workflow episode not found")
        if self._workflow_episode_is_cancel_requested(record):
            raise WorkflowEpisodeCancelledError("workflow episode cancelled")
        if record.status in {"completed", "failed", "dead_lettered", "cancelled"}:
            raise ValueError("workflow episode already finalized")
        if record.lease_owner not in {None, worker_id}:
            raise WorkflowEpisodeLeaseFencedError("workflow episode lease fenced")
        now = utcnow()
        record.status = "running"
        record.queue_state = str(queue_state or "executing")
        record.lease_owner = worker_id
        record.last_worker_id = worker_id
        record.lease_expires_at = self._workflow_episode_lease_expiry(lease_seconds=lease_seconds)
        if active_node_id is not None:
            record.active_node_id = active_node_id
            record.last_node_id = active_node_id
        if active_role_id is not None:
            record.active_role_id = active_role_id
            record.last_role_id = active_role_id
        if checkpoint_state is not None:
            record.checkpoint_state_json = dict(checkpoint_state)
            record.last_checkpoint_at = now
        if runtime_state_patch is not None:
            record.runtime_state_json = {
                **dict(record.runtime_state_json or {}),
                **dict(runtime_state_patch),
            }
        if resume_tokens is not None:
            record.resume_tokens_json = {
                **dict(record.resume_tokens_json or {}),
                **dict(resume_tokens),
            }
        if deferred_node_ids is not None:
            record.deferred_node_ids_json = [str(item) for item in deferred_node_ids]
        if metadata_patch:
            record.metadata_json = {
                **dict(record.metadata_json or {}),
                **dict(metadata_patch),
            }
        record.last_heartbeat_at = now
        record.updated_at = now
        session.flush()
        return record

    def finalize_workflow_episode(
        self,
        session: Session,
        *,
        episode_id: str,
        status: str,
        worker_id: str | None = None,
        error_text: str | None = None,
        final_outcome_score: float | None = None,
    ) -> WorkflowEpisodeRecord:
        record = self.workflow_episode_record(session, episode_id=episode_id)
        if record is None:
            raise ValueError("workflow episode not found")
        if worker_id is not None and record.lease_owner not in {None, worker_id}:
            raise WorkflowEpisodeLeaseFencedError("workflow episode lease fenced")
        now = utcnow()
        normalized_status = str(status)
        if self._workflow_episode_is_cancel_requested(record) and normalized_status != "cancelled":
            raise WorkflowEpisodeCancelledError("workflow episode cancelled")
        if normalized_status == "failed":
            self._workflow_episode_mark_retry(
                record=record,
                error_text=error_text,
                reason="episode_failed",
                now=now,
            )
            if final_outcome_score is not None:
                record.final_outcome_score = float(final_outcome_score)
        else:
            self._workflow_episode_finalize_terminal(
                record=record,
                status=normalized_status,
                now=now,
                error_text=error_text,
                final_outcome_score=final_outcome_score,
            )
        session.flush()
        return record

    def recover_expired_workflow_episode_leases(self, session: Session) -> dict[str, list[str]]:
        now = utcnow()
        records = list(
            session.execute(
                select(WorkflowEpisodeRecord).where(
                    WorkflowEpisodeRecord.queue_state.in_(("leased", "executing", "deferred")),
                    WorkflowEpisodeRecord.status.not_in(("completed", "failed", "dead_lettered", "cancelled")),
                    WorkflowEpisodeRecord.lease_expires_at.is_not(None),
                    WorkflowEpisodeRecord.lease_expires_at < now,
                )
            ).scalars()
        )
        recovered: list[str] = []
        retried: list[str] = []
        dead_lettered: list[str] = []
        for record in records:
            if self._workflow_episode_is_cancel_requested(record):
                self._workflow_episode_finalize_terminal(
                    record=record,
                    status="cancelled",
                    now=now,
                    error_text=record.cancel_reason or "workflow_cancelled",
                )
                recovered.append(record.episode_id)
                continue
            self._workflow_episode_mark_retry(
                record=record,
                error_text=record.last_error or "workflow episode lease expired",
                reason="lease_expired",
                now=now,
            )
            recovered.append(record.episode_id)
            if self._workflow_episode_is_dead_lettered(record):
                dead_lettered.append(record.episode_id)
            else:
                retried.append(record.episode_id)
        session.flush()
        return {
            "recovered_episode_ids": recovered,
            "retried_episode_ids": retried,
            "dead_lettered_episode_ids": dead_lettered,
        }

    def store_workflow_episode(
        self,
        session: Session,
        *,
        episode: WorkflowEpisode,
        result: WorkflowEpisodeResult,
    ) -> WorkflowEpisodeRecord:
        existing = session.execute(
            select(WorkflowEpisodeRecord).where(WorkflowEpisodeRecord.episode_id == episode.episode_id)
        ).scalar_one_or_none()
        merged_metadata = {
            **(dict(existing.metadata_json or {}) if existing is not None else {}),
            **dict(episode.metadata or {}),
            **dict(result.metadata or {}),
        }
        if existing is None:
            record = WorkflowEpisodeRecord(
                episode_id=episode.episode_id,
                run_id=episode.run_id,
                workflow_spec_id=episode.workflow_spec_id,
                workflow_version=episode.workflow_version,
                workflow_class=episode.workflow_class,
                status=result.status,
                queue_state=result.status,
                task_prompt=episode.task_prompt,
                episode_json=episode.model_dump(mode="json"),
                result_json=result.model_dump(mode="json"),
                final_outcome_score=result.final_outcome_score,
                metadata_json=merged_metadata,
                checkpoint_state_json=dict(episode.initial_context or {}),
                completed_at=result.completed_at,
            )
            session.add(record)
        else:
            record = existing
            record.run_id = episode.run_id
            record.workflow_spec_id = episode.workflow_spec_id
            record.workflow_version = episode.workflow_version
            record.workflow_class = episode.workflow_class
            record.status = result.status
            record.queue_state = result.status
            record.task_prompt = episode.task_prompt
            record.episode_json = episode.model_dump(mode="json")
            record.result_json = result.model_dump(mode="json")
            record.final_outcome_score = result.final_outcome_score
            record.metadata_json = merged_metadata
            record.completed_at = result.completed_at
        record.max_attempts = self._workflow_episode_max_attempts(metadata=merged_metadata)
        record.lease_owner = None
        record.lease_expires_at = None
        record.active_node_id = None
        record.active_role_id = None
        record.deferred_node_ids_json = []
        record.resume_tokens_json = {}
        record.next_eligible_at = None
        if result.status == "completed":
            record.dead_lettered_at = None
            record.dead_letter_reason = None
            record.last_error = None
        record.last_checkpoint_at = utcnow()
        record.last_heartbeat_at = utcnow()
        session.flush()
        return record

    def workflow_episode_payload(self, record: WorkflowEpisodeRecord) -> dict[str, Any]:
        result_json = dict(record.result_json or {})
        metadata_json = dict(record.metadata_json or {})
        now = utcnow()
        runtime_contract_modes: dict[str, int] = {}
        for trace in list(result_json.get("node_traces", []) or []):
            if not isinstance(trace, dict):
                continue
            mode = str(trace.get("runtime_contract_mode") or (trace.get("metadata") or {}).get("runtime_contract_mode") or "")
            if not mode:
                continue
            runtime_contract_modes[mode] = runtime_contract_modes.get(mode, 0) + 1
        return {
            "episode_id": record.episode_id,
            "run_id": record.run_id,
            "workflow_spec_id": record.workflow_spec_id,
            "workflow_version": record.workflow_version,
            "workflow_class": record.workflow_class,
            "status": record.status,
            "queue_state": record.queue_state,
            "lease_owner": record.lease_owner,
            "lease_expires_at": (
                record.lease_expires_at.isoformat() if record.lease_expires_at else None
            ),
            "active_node_id": record.active_node_id,
            "active_role_id": record.active_role_id,
            "last_worker_id": record.last_worker_id,
            "last_node_id": record.last_node_id,
            "last_role_id": record.last_role_id,
            "checkpoint_state": dict(record.checkpoint_state_json or {}),
            "runtime_state": dict(record.runtime_state_json or {}),
            "resume_tokens": dict(record.resume_tokens_json or {}),
            "deferred_node_ids": list(record.deferred_node_ids_json or []),
            "retry_count": int(record.retry_count or 0),
            "attempt_count": int(record.attempt_count or 0),
            "max_attempts": int(record.max_attempts or WORKFLOW_EPISODE_DEFAULT_MAX_ATTEMPTS),
            "next_eligible_at": (
                record.next_eligible_at.isoformat() if record.next_eligible_at else None
            ),
            "last_error": record.last_error,
            "last_failure_at": (
                record.last_failure_at.isoformat() if record.last_failure_at else None
            ),
            "cancel_requested_at": (
                record.cancel_requested_at.isoformat() if record.cancel_requested_at else None
            ),
            "cancel_reason": record.cancel_reason,
            "cancel_requested_by": record.cancel_requested_by,
            "cancellation_source": record.cancellation_source,
            "dead_lettered_at": (
                record.dead_lettered_at.isoformat() if record.dead_lettered_at else None
            ),
            "dead_letter_reason": record.dead_letter_reason,
            "retryable": self._workflow_episode_is_retryable(record, now=now),
            "backoff_delayed": self._workflow_episode_is_retry_wait(record, now=now),
            "dead_lettered": self._workflow_episode_is_dead_lettered(record),
            "cancel_requested": bool(record.cancel_requested_at),
            "task_prompt": record.task_prompt,
            "task_execution": self._workflow_episode_task_execution_payload(record),
            "final_outcome_score": float(record.final_outcome_score),
            "metadata": metadata_json,
            "runtime_remediation_audit": self._workflow_episode_remediation_audit(record),
            "last_runtime_remediation_at": metadata_json.get("last_runtime_remediation_at"),
            "last_runtime_remediation_action": metadata_json.get("last_runtime_remediation_action"),
            "last_runtime_remediation_reason": metadata_json.get("last_runtime_remediation_reason"),
            "last_runtime_remediation_outcome": metadata_json.get("last_runtime_remediation_outcome"),
            "runtime_policy_suppressed": bool(metadata_json.get("runtime_policy_suppressed", False)),
            "runtime_policy_suppressed_at": metadata_json.get("runtime_policy_suppressed_at"),
            "runtime_policy_suppression_reason": metadata_json.get("runtime_policy_suppression_reason"),
            "runtime_policy_escalation_reason": metadata_json.get("runtime_policy_escalation_reason"),
            "corpus_version": metadata_json.get("corpus_version"),
            "corpus_manifest_digest": metadata_json.get("corpus_manifest_digest"),
            "workflow_slice_id": metadata_json.get("workflow_slice_id"),
            "hidden_slice": bool(metadata_json.get("hidden_slice", False)),
            "replay_baseline_id": metadata_json.get("replay_baseline_id"),
            "workflow_composition_source": metadata_json.get("workflow_composition_source"),
            "workflow_composition_revision": metadata_json.get("workflow_composition_revision"),
            "workflow_composition_registry_url": metadata_json.get("workflow_composition_registry_url"),
            "workflow_composition_reason": metadata_json.get("workflow_composition_reason"),
            "selected_deployment_ids": list(metadata_json.get("selected_deployment_ids") or []),
            "runtime_contract_modes": runtime_contract_modes,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "last_heartbeat_at": (
                record.last_heartbeat_at.isoformat() if record.last_heartbeat_at else None
            ),
            "last_checkpoint_at": (
                record.last_checkpoint_at.isoformat() if record.last_checkpoint_at else None
            ),
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
        }

    def workflow_episode_trace_payload(self, record: WorkflowEpisodeRecord) -> dict[str, Any]:
        return {
            **self.workflow_episode_payload(record),
            "episode": dict(record.episode_json or {}),
            "result": dict(record.result_json or {}),
        }

    def workflow_incident_payload(
        self,
        session: Session,
        *,
        record: WorkflowEpisodeRecord,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        payload = self.workflow_episode_payload(record)
        current = now or utcnow()
        incident_state = self._workflow_episode_incident_state(record, now=current)
        recommended_action = self._workflow_incident_recommended_action(
            record,
            incident_state=incident_state,
            now=current,
        )
        active_suppressions = self._workflow_episode_matching_suppressions(
            session,
            record=record,
            incident_state=incident_state,
            now=current,
        )
        return {
            "episode_id": payload["episode_id"],
            "run_id": payload["run_id"],
            "workflow_spec_id": payload["workflow_spec_id"],
            "workflow_class": payload["workflow_class"],
            "queue_state": payload["queue_state"],
            "status": payload["status"],
            "incident_state": incident_state,
            "attempt_count": payload["attempt_count"],
            "max_attempts": payload["max_attempts"],
            "next_eligible_at": payload["next_eligible_at"],
            "last_error": payload["last_error"],
            "last_failure_at": payload["last_failure_at"],
            "dead_lettered_at": payload["dead_lettered_at"],
            "dead_letter_reason": payload["dead_letter_reason"],
            "lease_owner": payload["lease_owner"],
            "active_node_id": payload["active_node_id"],
            "active_role_id": payload["active_role_id"],
            "runtime_contract_modes": payload["runtime_contract_modes"],
            "workflow_trace_url": payload["workflow_trace_url"] if "workflow_trace_url" in payload else f"/v1/internal/workflow-episodes/{payload['episode_id']}/trace",
            "task_execution": self._workflow_episode_task_execution_context(
                session,
                record=record,
            ),
            "active_suppressions": active_suppressions,
            "suppressed": bool(active_suppressions),
            "recommended_action": recommended_action,
            "actionable": recommended_action in {"requeue", "dead_letter", "cancel"},
            "last_runtime_remediation_at": payload.get("last_runtime_remediation_at"),
            "last_runtime_remediation_action": payload.get("last_runtime_remediation_action"),
            "last_runtime_remediation_outcome": payload.get("last_runtime_remediation_outcome"),
            "runtime_policy_suppressed": payload.get("runtime_policy_suppressed"),
            "runtime_policy_suppression_reason": payload.get("runtime_policy_suppression_reason"),
            "runtime_policy_escalation_reason": payload.get("runtime_policy_escalation_reason"),
            "updated_at": payload["updated_at"],
        }

    def _workflow_incident_recommended_action(
        self,
        record: WorkflowEpisodeRecord,
        *,
        incident_state: str | None = None,
        now: datetime | None = None,
    ) -> str | None:
        state = incident_state or self._workflow_episode_incident_state(record, now=now)
        if state in {"stale", "retryable", "failed"}:
            return "requeue"
        if state == "dead_lettered":
            return None
        if state == "retry_wait":
            return None
        return None

    def _workflow_episode_remediation_audit(
        self,
        record: WorkflowEpisodeRecord,
    ) -> list[dict[str, Any]]:
        metadata = dict(record.metadata_json or {})
        entries = metadata.get("runtime_remediation_audit")
        if not isinstance(entries, list):
            return []
        return [dict(item) for item in entries if isinstance(item, dict)]

    def _workflow_episode_last_remediation_at(
        self,
        record: WorkflowEpisodeRecord,
    ) -> datetime | None:
        metadata = dict(record.metadata_json or {})
        value = metadata.get("last_runtime_remediation_at")
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _state_record(
        self,
        session: Session,
        *,
        state_key: str,
        create: bool = False,
    ) -> WorkflowRuntimePolicyStateRecord | None:
        record = session.get(WorkflowRuntimePolicyStateRecord, state_key)
        if record is None and create:
            record = WorkflowRuntimePolicyStateRecord(
                state_key=state_key,
                value_json={},
            )
            session.add(record)
            session.flush()
        return record

    def runtime_remediation_policy_state_payload(self, session: Session) -> dict[str, Any]:
        record = self._state_record(session, state_key=WORKFLOW_RUNTIME_POLICY_STATE_KEY)
        payload = dict(record.value_json or {}) if record is not None else {}
        active_suppressions = self.list_runtime_remediation_suppressions(
            session,
            active_only=True,
        )
        return {
            "worker_action_failure_count": int(payload.get("worker_action_failure_count", 0) or 0),
            "last_worker_action_failure": payload.get("last_worker_action_failure"),
            "last_worker_action_failure_at": payload.get("last_worker_action_failure_at"),
            "next_worker_action_retry_at": payload.get("next_worker_action_retry_at"),
            "worker_action_backoff_active": bool(
                self._runtime_remediation_worker_backoff_active(session)
            ),
            "active_suppression_count": len(active_suppressions),
            "active_suppressions": active_suppressions,
        }

    def _update_runtime_policy_state(
        self,
        session: Session,
        *,
        values: dict[str, Any],
        now: datetime | None = None,
    ) -> dict[str, Any]:
        record = self._state_record(
            session,
            state_key=WORKFLOW_RUNTIME_POLICY_STATE_KEY,
            create=True,
        )
        assert record is not None
        payload = {
            **dict(record.value_json or {}),
            **values,
        }
        record.value_json = payload
        record.updated_at = now or utcnow()
        session.flush()
        return payload

    def _clear_runtime_policy_worker_failure(
        self,
        session: Session,
        *,
        now: datetime,
    ) -> dict[str, Any]:
        return self._update_runtime_policy_state(
            session,
            now=now,
            values={
                "worker_action_failure_count": 0,
                "last_worker_action_failure": None,
                "last_worker_action_failure_at": None,
                "next_worker_action_retry_at": None,
            },
        )

    def record_runtime_policy_worker_failure(
        self,
        session: Session,
        *,
        error_text: str,
        now: datetime,
    ) -> dict[str, Any]:
        record = self._state_record(
            session,
            state_key=WORKFLOW_RUNTIME_POLICY_STATE_KEY,
            create=True,
        )
        current = dict(record.value_json or {}) if record is not None else {}
        failure_count = int(current.get("worker_action_failure_count", 0) or 0) + 1
        base_backoff = max(
            1,
            int(self.settings.workflow_runtime_auto_remediation_worker_failure_backoff_seconds),
        )
        backoff_seconds = min(
            base_backoff * max(1, failure_count),
            base_backoff * 4,
        )
        next_retry_at = now + timedelta(seconds=backoff_seconds)
        return self._update_runtime_policy_state(
            session,
            now=now,
            values={
                "worker_action_failure_count": failure_count,
                "last_worker_action_failure": error_text,
                "last_worker_action_failure_at": now.isoformat(),
                "next_worker_action_retry_at": next_retry_at.isoformat(),
            },
        )

    def _runtime_remediation_worker_backoff_active(
        self,
        session: Session,
        *,
        now: datetime | None = None,
    ) -> bool:
        record = self._state_record(session, state_key=WORKFLOW_RUNTIME_POLICY_STATE_KEY)
        if record is None:
            return False
        value = dict(record.value_json or {}).get("next_worker_action_retry_at")
        if not isinstance(value, str) or not value.strip():
            return False
        try:
            retry_at = datetime.fromisoformat(value)
        except ValueError:
            return False
        return retry_at > (now or utcnow())

    def list_runtime_remediation_suppressions(
        self,
        session: Session,
        *,
        active_only: bool = False,
    ) -> list[dict[str, Any]]:
        now = utcnow()
        rows = list(
            session.execute(
                select(WorkflowRuntimeSuppressionRecord).order_by(
                    WorkflowRuntimeSuppressionRecord.created_at.desc()
                )
            ).scalars()
        )
        payloads: list[dict[str, Any]] = []
        for row in rows:
            expired = bool(row.expires_at is not None and row.expires_at <= now)
            if active_only and expired:
                continue
            payloads.append(
                {
                    "suppression_id": row.id,
                    "target_kind": row.target_kind,
                    "target_value": row.target_value,
                    "reason": row.reason,
                    "created_by": row.created_by,
                    "metadata": dict(row.metadata_json or {}),
                    "created_at": row.created_at.isoformat(),
                    "updated_at": row.updated_at.isoformat(),
                    "expires_at": row.expires_at.isoformat() if row.expires_at else None,
                    "active": not expired,
                }
            )
        return payloads

    def create_runtime_remediation_suppression(
        self,
        session: Session,
        *,
        target_kind: str,
        target_value: str,
        reason: str,
        created_by: str,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_kind = str(target_kind or "").strip()
        if normalized_kind not in WORKFLOW_RUNTIME_SUPPRESSION_TARGET_KINDS:
            raise ValueError("unsupported suppression target_kind")
        normalized_value = str(target_value or "").strip()
        if not normalized_value:
            raise ValueError("suppression target_value is required")
        row = WorkflowRuntimeSuppressionRecord(
            target_kind=normalized_kind,
            target_value=normalized_value,
            reason=str(reason or "").strip() or "operator_suppression",
            created_by=str(created_by or "").strip() or "operator",
            expires_at=expires_at,
            metadata_json=dict(metadata or {}),
        )
        session.add(row)
        session.flush()
        return self.list_runtime_remediation_suppressions(session)[0]

    def delete_runtime_remediation_suppression(
        self,
        session: Session,
        *,
        suppression_id: str,
    ) -> bool:
        row = session.get(WorkflowRuntimeSuppressionRecord, suppression_id)
        if row is None:
            return False
        session.delete(row)
        session.flush()
        return True

    def _workflow_episode_recent_requeue_applied_count(
        self,
        record: WorkflowEpisodeRecord,
        *,
        now: datetime,
        window_seconds: int,
    ) -> int:
        count = 0
        for entry in self._workflow_episode_remediation_audit(record):
            if entry.get("actor") != "runtime_policy":
                continue
            if entry.get("outcome") != "applied":
                continue
            if entry.get("action") != "requeue":
                continue
            at = entry.get("at")
            if not isinstance(at, str) or not at.strip():
                continue
            try:
                applied_at = datetime.fromisoformat(at)
            except ValueError:
                continue
            if (now - applied_at).total_seconds() <= max(1, window_seconds):
                count += 1
        return count

    def _clear_workflow_episode_policy_suppression(
        self,
        record: WorkflowEpisodeRecord,
    ) -> None:
        metadata = dict(record.metadata_json or {})
        for key in (
            "runtime_policy_suppressed",
            "runtime_policy_suppressed_at",
            "runtime_policy_suppression_reason",
            "runtime_policy_escalation_reason",
        ):
            metadata.pop(key, None)
        record.metadata_json = metadata

    def _mark_workflow_episode_policy_suppressed(
        self,
        *,
        record: WorkflowEpisodeRecord,
        reason: str,
        now: datetime,
    ) -> None:
        metadata = dict(record.metadata_json or {})
        metadata["runtime_policy_suppressed"] = True
        metadata["runtime_policy_suppressed_at"] = now.isoformat()
        metadata["runtime_policy_suppression_reason"] = reason
        metadata["runtime_policy_escalation_reason"] = reason
        record.metadata_json = metadata

    def _workflow_episode_matching_suppressions(
        self,
        session: Session,
        *,
        record: WorkflowEpisodeRecord,
        incident_state: str | None,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        current = now or utcnow()
        metadata = dict(record.metadata_json or {})
        task_id = str(metadata.get("task_id") or "").strip()
        matches: list[dict[str, Any]] = []
        for item in self.list_runtime_remediation_suppressions(session, active_only=True):
            target_kind = str(item.get("target_kind") or "")
            target_value = str(item.get("target_value") or "")
            if target_kind == "episode_id" and target_value == record.episode_id:
                matches.append(item)
            elif target_kind == "workflow_spec_id" and target_value == record.workflow_spec_id:
                matches.append(item)
            elif target_kind == "incident_state" and target_value == str(incident_state or ""):
                matches.append(item)
            elif target_kind == "task_id" and task_id and target_value == task_id:
                matches.append(item)
        if bool(metadata.get("runtime_policy_suppressed")):
            matches.append(
                {
                    "suppression_id": None,
                    "target_kind": "episode_id",
                    "target_value": record.episode_id,
                    "reason": metadata.get("runtime_policy_suppression_reason") or "policy_escalation",
                    "created_by": "runtime_policy",
                    "metadata": {
                        "policy_generated": True,
                    },
                    "created_at": metadata.get("runtime_policy_suppressed_at") or current.isoformat(),
                    "updated_at": metadata.get("runtime_policy_suppressed_at") or current.isoformat(),
                    "expires_at": None,
                    "active": True,
                }
            )
        return matches

    def _append_workflow_remediation_audit(
        self,
        *,
        record: WorkflowEpisodeRecord,
        actor: str,
        incident_state: str | None,
        action: str | None,
        reason: str | None,
        outcome: str,
        now: datetime,
        skipped_reason: str | None = None,
    ) -> None:
        metadata = dict(record.metadata_json or {})
        audit = self._workflow_episode_remediation_audit(record)
        entry = {
            "at": now.isoformat(),
            "actor": actor,
            "incident_state": incident_state,
            "action": action,
            "reason": reason,
            "outcome": outcome,
        }
        if skipped_reason:
            entry["skipped_reason"] = skipped_reason
        audit.append(entry)
        audit = audit[-WORKFLOW_RUNTIME_REMEDIATION_AUDIT_LIMIT:]
        metadata["runtime_remediation_audit"] = audit
        metadata["last_runtime_remediation_attempt_at"] = now.isoformat()
        metadata["last_runtime_remediation_action"] = action
        metadata["last_runtime_remediation_reason"] = reason
        metadata["last_runtime_remediation_outcome"] = outcome
        if incident_state:
            metadata["last_runtime_remediation_incident_state"] = incident_state
        if outcome == "applied":
            metadata["last_runtime_remediation_at"] = now.isoformat()
        record.metadata_json = metadata

    def auto_remediate_workflow_incidents(
        self,
        session: Session,
        *,
        dry_run: bool = False,
        cooldown_seconds: int,
        max_actions: int,
        reason: str | None = None,
        run_id: str | None = None,
        workflow_spec_id: str | None = None,
        incident_states: tuple[str, ...] = ("stale", "failed", "retryable"),
    ) -> dict[str, Any]:
        rows = self.list_workflow_episodes(
            session,
            run_id=run_id,
            workflow_spec_id=workflow_spec_id,
        )
        now = utcnow()
        candidates: list[WorkflowEpisodeRecord] = []
        for record in rows:
            state = self._workflow_episode_incident_state(record, now=now)
            if state is None or state not in incident_states:
                continue
            candidates.append(record)
        candidates.sort(key=lambda item: self._workflow_episode_incident_sort_key(item, now=now))

        results: list[dict[str, Any]] = []
        applied_count = 0
        skipped_count = 0
        eligible_count = 0
        budget = max(0, int(max_actions))
        cooldown = max(0, int(cooldown_seconds))
        escalation_window_seconds = max(
            1,
            int(self.settings.workflow_runtime_auto_remediation_escalation_window_seconds),
        )
        requeue_limit = max(
            1,
            int(self.settings.workflow_runtime_auto_remediation_requeue_limit),
        )

        for record in candidates:
            state = self._workflow_episode_incident_state(record, now=now)
            action = self._workflow_incident_recommended_action(
                record,
                incident_state=state,
                now=now,
            )
            active_suppressions = self._workflow_episode_matching_suppressions(
                session,
                record=record,
                incident_state=state,
                now=now,
            )
            if active_suppressions:
                skipped_count += 1
                results.append(
                    {
                        "episode_id": record.episode_id,
                        "incident_state": state,
                        "recommended_action": action,
                        "applied_action": None,
                        "skipped_reason": "suppressed",
                        "active_suppressions": active_suppressions,
                    }
                )
                if not dry_run:
                    self._append_workflow_remediation_audit(
                        record=record,
                        actor="runtime_policy",
                        incident_state=state,
                        action=action,
                        reason=reason or "auto_policy",
                        outcome="skipped",
                        skipped_reason="suppressed",
                        now=now,
                    )
                continue
            if not action:
                skipped_count += 1
                results.append(
                    {
                        "episode_id": record.episode_id,
                        "incident_state": state,
                        "recommended_action": None,
                        "applied_action": None,
                        "skipped_reason": "no_policy_action",
                    }
                )
                continue
            last_remediation_at = self._workflow_episode_last_remediation_at(record)
            if (
                last_remediation_at is not None
                and cooldown > 0
                and (now - last_remediation_at).total_seconds() < cooldown
            ):
                skipped_count += 1
                results.append(
                    {
                        "episode_id": record.episode_id,
                        "incident_state": state,
                        "recommended_action": action,
                        "applied_action": None,
                        "skipped_reason": "cooldown_active",
                    }
                )
                if not dry_run:
                    self._append_workflow_remediation_audit(
                        record=record,
                        actor="runtime_policy",
                        incident_state=state,
                        action=action,
                        reason=reason or "auto_policy",
                        outcome="skipped",
                        skipped_reason="cooldown_active",
                        now=now,
                )
                continue
            if (
                action == "requeue"
                and self._workflow_episode_recent_requeue_applied_count(
                    record,
                    now=now,
                    window_seconds=escalation_window_seconds,
                )
                >= requeue_limit
            ):
                if not dry_run:
                    self._mark_workflow_episode_policy_suppressed(
                        record=record,
                        reason="requeue_limit_exceeded",
                        now=now,
                    )
                    self._append_workflow_remediation_audit(
                        record=record,
                        actor="runtime_policy",
                        incident_state=state,
                        action=action,
                        reason=reason or "auto_policy",
                        outcome="escalated",
                        skipped_reason="policy_escalated",
                        now=now,
                    )
                skipped_count += 1
                results.append(
                    {
                        "episode_id": record.episode_id,
                        "incident_state": state,
                        "recommended_action": action,
                        "applied_action": None,
                        "skipped_reason": "policy_escalated",
                        "policy_escalation_reason": "requeue_limit_exceeded",
                    }
                )
                continue
            eligible_count += 1
            if applied_count >= budget:
                skipped_count += 1
                results.append(
                    {
                        "episode_id": record.episode_id,
                        "incident_state": state,
                        "recommended_action": action,
                        "applied_action": None,
                        "skipped_reason": "budget_exhausted",
                    }
                )
                continue
            if dry_run:
                results.append(
                    {
                        "episode_id": record.episode_id,
                        "incident_state": state,
                        "recommended_action": action,
                        "applied_action": action,
                        "dry_run": True,
                    }
                )
                continue
            if action == "requeue":
                updated = self.requeue_workflow_episode(
                    session,
                    episode_id=record.episode_id,
                    reason=reason or f"auto_policy_{state}",
                )
            elif action == "dead_letter":
                updated = self.dead_letter_workflow_episode(
                    session,
                    episode_id=record.episode_id,
                    reason=reason or f"auto_policy_{state}",
                )
            elif action == "cancel":
                updated = self.admin_finalize_workflow_episode(
                    session,
                    episode_id=record.episode_id,
                    status="cancelled",
                    error_text=reason or f"auto_policy_cancelled:{state}",
                )
            else:
                skipped_count += 1
                results.append(
                    {
                        "episode_id": record.episode_id,
                        "incident_state": state,
                        "recommended_action": action,
                        "applied_action": None,
                        "skipped_reason": "unsupported_policy_action",
                    }
                )
                continue
            self._append_workflow_remediation_audit(
                record=updated,
                actor="runtime_policy",
                incident_state=state,
                action=action,
                reason=reason or "auto_policy",
                outcome="applied",
                now=now,
            )
            applied_count += 1
            results.append(
                {
                    "episode_id": updated.episode_id,
                    "incident_state": state,
                    "recommended_action": action,
                    "applied_action": action,
                    "resulting_queue_state": updated.queue_state,
                    "resulting_status": updated.status,
                    "workflow_trace_url": f"/v1/internal/workflow-episodes/{updated.episode_id}/trace",
                }
            )
        return {
            "policy": "workflow_runtime_auto_remediation_v1",
            "dry_run": dry_run,
            "cooldown_seconds": cooldown,
            "max_actions": budget,
            "matched_count": len(candidates),
            "eligible_count": eligible_count,
            "applied_count": applied_count,
            "skipped_count": skipped_count,
            "results": results,
        }

    def workflow_incidents(
        self,
        session: Session,
        *,
        incident_state: str | None = None,
        workflow_spec_id: str | None = None,
        task_id: str | None = None,
        lease_owner: str | None = None,
        run_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        rows = self.list_workflow_episodes(
            session,
            run_id=run_id,
            workflow_spec_id=workflow_spec_id,
            lease_owner=lease_owner,
            task_id=task_id,
        )
        now = utcnow()
        incidents = []
        for record in rows:
            state = self._workflow_episode_incident_state(record, now=now)
            if state is None:
                continue
            if incident_state is not None and state != incident_state:
                continue
            incidents.append(record)
        incidents.sort(key=lambda item: self._workflow_episode_incident_sort_key(item, now=now))
        sliced = incidents[max(0, offset): max(0, offset) + max(0, limit)]
        return [
            self.workflow_incident_payload(session, record=item, now=now)
            for item in sliced
        ]

    def remediate_workflow_incidents(
        self,
        session: Session,
        *,
        action: str,
        dry_run: bool = False,
        reason: str | None = None,
        episode_ids: list[str] | None = None,
        incident_state: str | None = None,
        workflow_spec_id: str | None = None,
        task_id: str | None = None,
        lease_owner: str | None = None,
        run_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        valid_actions = {"requeue", "dead_letter", "cancel", "auto"}
        if action not in valid_actions:
            raise ValueError("unsupported remediation action")
        rows = self.list_workflow_episodes(
            session,
            run_id=run_id,
            workflow_spec_id=workflow_spec_id,
            lease_owner=lease_owner,
            task_id=task_id,
        )
        now = utcnow()
        incident_records: list[WorkflowEpisodeRecord] = []
        explicit_ids = [str(item).strip() for item in list(episode_ids or []) if str(item).strip()]
        if explicit_ids:
            by_id = {record.episode_id: record for record in rows}
            incident_records = [by_id[item] for item in explicit_ids if item in by_id]
        else:
            for record in rows:
                state = self._workflow_episode_incident_state(record, now=now)
                if state is None:
                    continue
                if incident_state is not None and state != incident_state:
                    continue
                incident_records.append(record)
            incident_records.sort(key=lambda item: self._workflow_episode_incident_sort_key(item, now=now))
            incident_records = incident_records[max(0, offset): max(0, offset) + max(0, limit)]

        results: list[dict[str, Any]] = []
        applied_count = 0
        skipped_count = 0
        for record in incident_records:
            state = self._workflow_episode_incident_state(record, now=now)
            if state is None:
                skipped_count += 1
                results.append(
                    {
                        "episode_id": record.episode_id,
                        "incident_state": None,
                        "applied_action": None,
                        "recommended_action": None,
                        "skipped_reason": "not_incident",
                    }
                )
                continue
            if incident_state is not None and state != incident_state:
                skipped_count += 1
                results.append(
                    {
                        "episode_id": record.episode_id,
                        "incident_state": state,
                        "applied_action": None,
                        "recommended_action": self._workflow_incident_recommended_action(record, incident_state=state, now=now),
                        "skipped_reason": "incident_state_mismatch",
                    }
                )
                continue
            resolved_action = action
            if action == "auto":
                resolved_action = self._workflow_incident_recommended_action(
                    record,
                    incident_state=state,
                    now=now,
                ) or ""
            if not resolved_action:
                skipped_count += 1
                results.append(
                    {
                        "episode_id": record.episode_id,
                        "incident_state": state,
                        "applied_action": None,
                        "recommended_action": self._workflow_incident_recommended_action(record, incident_state=state, now=now),
                        "skipped_reason": "no_auto_action",
                    }
                )
                continue
            if dry_run:
                results.append(
                    {
                        "episode_id": record.episode_id,
                        "incident_state": state,
                        "applied_action": resolved_action,
                        "recommended_action": self._workflow_incident_recommended_action(record, incident_state=state, now=now),
                        "dry_run": True,
                    }
                )
                continue
            if resolved_action == "requeue":
                updated = self.requeue_workflow_episode(session, episode_id=record.episode_id, reason=reason or f"operator_{state}")
            elif resolved_action == "dead_letter":
                updated = self.dead_letter_workflow_episode(session, episode_id=record.episode_id, reason=reason or f"operator_{state}")
            elif resolved_action == "cancel":
                updated = self.admin_finalize_workflow_episode(
                    session,
                    episode_id=record.episode_id,
                    status="cancelled",
                    error_text=reason or f"operator_cancelled:{state}",
                )
            else:
                skipped_count += 1
                results.append(
                    {
                        "episode_id": record.episode_id,
                        "incident_state": state,
                        "applied_action": None,
                        "recommended_action": self._workflow_incident_recommended_action(record, incident_state=state, now=now),
                        "skipped_reason": "unsupported_action",
                    }
                )
                continue
            applied_count += 1
            updated_payload = self.workflow_incident_payload(session, record=updated)
            results.append(
                {
                    "episode_id": updated.episode_id,
                    "incident_state": state,
                    "applied_action": resolved_action,
                    "recommended_action": self._workflow_incident_recommended_action(record, incident_state=state, now=now),
                    "resulting_queue_state": updated.queue_state,
                    "resulting_status": updated.status,
                    "workflow_trace_url": updated_payload["workflow_trace_url"],
                }
            )
        return {
            "action": action,
            "dry_run": dry_run,
            "matched_count": len(incident_records),
            "applied_count": applied_count,
            "skipped_count": skipped_count,
            "results": results,
        }

    def list_workflow_episodes(
        self,
        session: Session,
        *,
        run_id: str | None = None,
        workflow_spec_id: str | None = None,
        queue_state: str | None = None,
        retryable_only: bool = False,
        dead_lettered_only: bool = False,
        stale_only: bool = False,
        lease_owner: str | None = None,
        task_id: str | None = None,
    ) -> list[WorkflowEpisodeRecord]:
        statement = select(WorkflowEpisodeRecord).order_by(WorkflowEpisodeRecord.created_at.desc())
        if run_id is not None:
            statement = statement.where(WorkflowEpisodeRecord.run_id == run_id)
        if workflow_spec_id is not None:
            statement = statement.where(WorkflowEpisodeRecord.workflow_spec_id == workflow_spec_id)
        if queue_state is not None:
            statement = statement.where(WorkflowEpisodeRecord.queue_state == queue_state)
        if lease_owner is not None:
            statement = statement.where(WorkflowEpisodeRecord.lease_owner == lease_owner)
        rows = list(session.execute(statement).scalars())
        now = utcnow()
        filtered: list[WorkflowEpisodeRecord] = []
        for record in rows:
            metadata_json = dict(record.metadata_json or {})
            if task_id is not None and str(metadata_json.get("task_id") or "").strip() != task_id:
                continue
            if retryable_only and not self._workflow_episode_is_retryable(record, now=now):
                continue
            if dead_lettered_only and not self._workflow_episode_is_dead_lettered(record):
                continue
            if stale_only:
                if record.queue_state not in {"leased", "executing", "deferred"}:
                    continue
                if record.lease_expires_at is None or record.lease_expires_at >= now:
                    continue
            filtered.append(record)
        return filtered

    def requeue_workflow_episode(
        self,
        session: Session,
        *,
        episode_id: str,
        reason: str | None = None,
    ) -> WorkflowEpisodeRecord:
        record = self.workflow_episode_record(session, episode_id=episode_id)
        if record is None:
            raise ValueError("workflow episode not found")
        now = utcnow()
        record.status = "queued"
        record.queue_state = "queued"
        record.lease_owner = None
        record.lease_expires_at = None
        record.active_node_id = None
        record.active_role_id = None
        record.attempt_count = 0
        record.retry_count = 0
        record.next_eligible_at = None
        record.dead_lettered_at = None
        record.dead_letter_reason = None
        record.cancel_requested_at = None
        record.cancel_reason = None
        record.cancel_requested_by = None
        record.cancellation_source = None
        record.completed_at = None
        if reason:
            record.metadata_json = {
                **dict(record.metadata_json or {}),
                "operator_requeue_reason": reason,
            }
        self._clear_workflow_episode_policy_suppression(record)
        record.updated_at = now
        session.flush()
        return record

    def dead_letter_workflow_episode(
        self,
        session: Session,
        *,
        episode_id: str,
        reason: str | None = None,
    ) -> WorkflowEpisodeRecord:
        record = self.workflow_episode_record(session, episode_id=episode_id)
        if record is None:
            raise ValueError("workflow episode not found")
        now = utcnow()
        record.status = "dead_lettered"
        record.queue_state = "dead_lettered"
        record.lease_owner = None
        record.lease_expires_at = None
        record.next_eligible_at = None
        record.dead_lettered_at = now
        record.dead_letter_reason = reason or "manual_dead_letter"
        record.completed_at = now
        record.updated_at = now
        session.flush()
        return record

    def cancel_workflow_episode(
        self,
        session: Session,
        *,
        episode_id: str,
        reason: str | None = None,
        requested_by: str = "operator",
        cancellation_source: str = "operator",
    ) -> WorkflowEpisodeRecord:
        record = self.workflow_episode_record(session, episode_id=episode_id)
        if record is None:
            raise ValueError("workflow episode not found")
        now = utcnow()
        record.cancel_requested_at = now
        record.cancel_reason = reason or "workflow_cancelled"
        record.cancel_requested_by = requested_by
        record.cancellation_source = cancellation_source
        record.last_error = record.cancel_reason
        if record.status in {"queued"} and record.queue_state in {"queued", "retry_wait", "retry_waiting", "failed"}:
            self._workflow_episode_finalize_terminal(
                record=record,
                status="cancelled",
                now=now,
                error_text=record.cancel_reason,
            )
        elif record.status not in {"completed", "failed", "dead_lettered", "cancelled"}:
            record.queue_state = "cancel_requested"
            record.updated_at = now
        session.flush()
        return record

    def update_workflow_episode_selection(
        self,
        session: Session,
        *,
        episode_id: str,
        episode: WorkflowEpisode,
        metadata_patch: dict[str, Any] | None = None,
    ) -> WorkflowEpisodeRecord:
        record = self.workflow_episode_record(session, episode_id=episode_id)
        if record is None:
            raise ValueError("workflow episode not found")
        if record.status in {"completed", "failed", "dead_lettered", "cancelled"}:
            raise ValueError("workflow episode already finalized")
        record.workflow_spec_id = episode.workflow_spec_id
        record.workflow_version = episode.workflow_version
        record.workflow_class = episode.workflow_class
        record.task_prompt = episode.task_prompt
        record.episode_json = episode.model_dump(mode="json")
        if metadata_patch:
            record.metadata_json = {
                **dict(record.metadata_json or {}),
                **dict(metadata_patch),
            }
        record.updated_at = utcnow()
        session.flush()
        return record

    def admin_finalize_workflow_episode(
        self,
        session: Session,
        *,
        episode_id: str,
        status: str,
        error_text: str | None = None,
        final_outcome_score: float | None = None,
    ) -> WorkflowEpisodeRecord:
        record = self.workflow_episode_record(session, episode_id=episode_id)
        if record is None:
            raise ValueError("workflow episode not found")
        self._workflow_episode_finalize_terminal(
            record=record,
            status=status,
            now=utcnow(),
            error_text=error_text,
            final_outcome_score=final_outcome_score,
        )
        session.flush()
        return record

    def workflow_episode_record(
        self,
        session: Session,
        *,
        episode_id: str,
    ) -> WorkflowEpisodeRecord | None:
        return session.execute(
            select(WorkflowEpisodeRecord).where(WorkflowEpisodeRecord.episode_id == episode_id)
        ).scalar_one_or_none()

    def _workflow_runtime_health_summary(
        self,
        session: Session,
        *,
        workflow_episode_records: list[WorkflowEpisodeRecord] | None = None,
    ) -> dict[str, Any]:
        records = workflow_episode_records
        if records is None:
            records = list(session.execute(select(WorkflowEpisodeRecord)).scalars())
        now = utcnow()
        lifecycle_counts = {
            "queued": 0,
            "executing": 0,
            "deferred": 0,
            "completed": 0,
            "retry_wait": 0,
            "failed": 0,
            "dead_lettered": 0,
        }
        stale_episode_count = 0
        retryable_episode_count = 0
        retry_wait_episode_count = 0
        task_backed_count = 0
        internal_episode_count = 0
        oldest_stale_episode_age_seconds = 0.0
        oldest_retry_wait_age_seconds = 0.0
        for item in records:
            if self._workflow_episode_task_execution_payload(item) is not None:
                task_backed_count += 1
            else:
                internal_episode_count += 1
            if item.queue_state == "completed":
                lifecycle_counts["completed"] += 1
            elif item.queue_state == "dead_lettered" or item.status == "dead_lettered":
                lifecycle_counts["dead_lettered"] += 1
            elif item.queue_state == "failed" or item.status == "failed":
                lifecycle_counts["failed"] += 1
            elif item.queue_state == "deferred":
                lifecycle_counts["deferred"] += 1
            elif item.queue_state in {"leased", "executing"}:
                lifecycle_counts["executing"] += 1
            elif self._workflow_episode_is_retry_wait(item, now=now):
                lifecycle_counts["retry_wait"] += 1
                retry_wait_episode_count += 1
                oldest_retry_wait_age_seconds = max(
                    oldest_retry_wait_age_seconds,
                    max(0.0, (now - item.updated_at).total_seconds()),
                )
            else:
                lifecycle_counts["queued"] += 1
            if self._workflow_episode_is_stale(item, now=now):
                stale_episode_count += 1
                if item.lease_expires_at is not None:
                    oldest_stale_episode_age_seconds = max(
                        oldest_stale_episode_age_seconds,
                        max(0.0, (now - item.lease_expires_at).total_seconds()),
                    )
            if self._workflow_episode_is_retryable(item, now=now) and int(item.attempt_count or 0) > 0:
                retryable_episode_count += 1
        return {
            "lifecycle_counts": lifecycle_counts,
            "retry_backlog_count": retry_wait_episode_count,
            "retryable_episode_count": retryable_episode_count,
            "retry_wait_episode_count": retry_wait_episode_count,
            "dead_lettered_episode_count": lifecycle_counts["dead_lettered"],
            "stale_episode_count": stale_episode_count,
            "oldest_stale_episode_age_seconds": oldest_stale_episode_age_seconds,
            "oldest_retry_wait_age_seconds": oldest_retry_wait_age_seconds,
            "task_backed_episode_count": task_backed_count,
            "internal_episode_count": internal_episode_count,
            "active_leased_count": sum(
                1
                for item in records
                if item.queue_state in {"leased", "executing"}
            ),
            "ok": stale_episode_count == 0,
        }
