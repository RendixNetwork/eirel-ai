from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from shared.common.database import Database
from shared.common.models import (
    ConsumerSessionState,
    DagExecutionRecord,
    DagNodeExecutionRecord,
    TaskRequestRecord,
)
from shared.contracts.models import AttributionRecord, ExecutionDAG, ExecutionResult, RoutingPlan, TaskObject


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class ExecutionStore:
    def __init__(self, db: Database):
        self.db = db

    def _task_workflow_execution_summary(
        self,
        *,
        record: TaskRequestRecord,
        include_internal: bool = False,
    ) -> dict[str, Any] | None:
        metadata = dict(record.metadata_json or {})
        routing_plan = dict(record.routing_plan_json or {})
        routing_metadata = dict(routing_plan.get("metadata", {}) or {})
        execution_dag = dict(record.execution_dag_json or {})
        dag_metadata = dict(execution_dag.get("metadata", {}) or {})
        execution_result = dict(record.execution_result_json or {})
        execution_metadata = dict(execution_result.get("metadata", {}) or {})

        workflow_episode_id = str(metadata.get("workflow_episode_id") or "").strip() or None
        workflow_spec_id = (
            str(metadata.get("workflow_spec_id") or "").strip()
            or str(routing_metadata.get("workflow_spec_id") or "").strip()
            or str(dag_metadata.get("workflow_spec_id") or "").strip()
            or str(execution_metadata.get("workflow_spec_id") or "").strip()
            or None
        )
        if workflow_episode_id is None and workflow_spec_id is None:
            return None

        workflow_class = (
            str(metadata.get("workflow_class") or "").strip()
            or str(execution_metadata.get("workflow_class") or "").strip()
            or (workflow_spec_id.replace("_v1", "") if workflow_spec_id else "")
        )

        selected_node_ids = list(metadata.get("workflow_selected_node_ids") or [])
        if not selected_node_ids:
            execution_nodes = list(execution_result.get("nodes") or [])
            selected_node_ids = [
                str(node.get("node_id"))
                for node in execution_nodes
                if isinstance(node, dict) and str(node.get("node_id") or "").strip()
            ]
        if not selected_node_ids:
            planned_nodes = list(routing_metadata.get("planned_nodes") or [])
            selected_node_ids = [
                str(node.get("node_id"))
                for node in planned_nodes
                if isinstance(node, dict)
                and node.get("execution_owner") == "family_worker"
                and str(node.get("node_id") or "").strip()
            ]

        selected_families = list(metadata.get("workflow_selected_families") or [])
        if not selected_families:
            execution_nodes = list(execution_result.get("nodes") or [])
            selected_families = [
                str(node.get("family_id"))
                for node in execution_nodes
                if isinstance(node, dict) and str(node.get("family_id") or "").strip()
            ]
        if not selected_families:
            selected_families = [
                str(item)
                for item in list(routing_metadata.get("family_sequence") or [])
                if str(item).strip()
            ]

        runtime_contract_modes = dict(metadata.get("runtime_contract_modes") or {})
        if not runtime_contract_modes:
            for node in list(execution_result.get("nodes") or []):
                if not isinstance(node, dict):
                    continue
                node_metadata = dict(node.get("metadata") or {})
                mode = str(node_metadata.get("runtime_contract_mode") or "").strip()
                if not mode:
                    continue
                runtime_contract_modes[mode] = runtime_contract_modes.get(mode, 0) + 1

        summary = {
            "workflow_episode_id": workflow_episode_id,
            "workflow_spec_id": workflow_spec_id,
            "workflow_class": workflow_class or None,
            "status": str(metadata.get("workflow_status") or record.status),
            "final_outcome_score": metadata.get("workflow_final_outcome_score"),
            "execution_path": (
                str(execution_metadata.get("execution_path") or metadata.get("execution_path") or "").strip()
                or None
            ),
            "selected_families": selected_families,
            "selected_node_ids": selected_node_ids,
            "runtime_contract_modes": runtime_contract_modes,
            "workflow_trace_url": (
                f"/v1/internal/workflow-episodes/{workflow_episode_id}/trace"
                if workflow_episode_id
                else None
            ),
            "workflow_composition_registry_url": (
                f"/v1/workflow-composition/registry?workflow_spec_id={workflow_spec_id}"
                if workflow_spec_id
                else "/v1/workflow-composition/registry"
            ),
            "workflow_composition_source": (
                str(metadata.get("workflow_composition_source") or "").strip() or None
            ),
            "workflow_composition_revision": (
                str(metadata.get("workflow_composition_revision") or "").strip() or None
            ),
            "workflow_composition_reason": (
                str(metadata.get("workflow_composition_reason") or "").strip() or None
            ),
        }
        if include_internal:
            summary.update(
                {
                    "replay_executed": metadata.get("replay_executed"),
                    "corpus_version": metadata.get("corpus_version"),
                    "corpus_manifest_digest": metadata.get("corpus_manifest_digest"),
                    "hidden_slice_count": metadata.get("hidden_slice_count"),
                    "selected_deployment_ids": list(metadata.get("selected_deployment_ids") or []),
                }
            )
        return {
            key: value
            for key, value in summary.items()
            if include_internal
            or value is not None
            or key in {"workflow_episode_id", "workflow_spec_id", "selected_families", "selected_node_ids", "runtime_contract_modes"}
        }

    def _session_task_summary_payload(self, record: TaskRequestRecord) -> dict[str, Any]:
        workflow_summary = self._task_workflow_execution_summary(record=record, include_internal=False)
        return {
            "task_id": record.task_id,
            "status": record.status,
            "workflow_episode_id": workflow_summary.get("workflow_episode_id") if workflow_summary else None,
            "workflow_spec_id": workflow_summary.get("workflow_spec_id") if workflow_summary else None,
            "workflow_class": workflow_summary.get("workflow_class") if workflow_summary else None,
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
            "task_status_url": f"/v1/tasks/{record.task_id}",
        }

    def ensure_session(self, *, session_id: str, user_id: str, initial_prompt: str) -> ConsumerSessionState:
        del initial_prompt
        with self.db.sessionmaker() as session:
            record = session.get(ConsumerSessionState, session_id)
            if record is None:
                record = ConsumerSessionState(
                    session_id=session_id,
                    user_id=user_id,
                    status="active",
                    messages_json=[],
                )
                session.add(record)
            else:
                record.user_id = user_id
                record.status = "active"
                record.updated_at = utcnow()
            session.commit()
            session.refresh(record)
            return record

    def create_task(self, *, task: TaskObject) -> TaskRequestRecord:
        with self.db.sessionmaker() as session:
            session_state = session.get(ConsumerSessionState, task.session_id)
            if session_state is None:
                session_state = ConsumerSessionState(
                    session_id=task.session_id,
                    user_id=task.user_id,
                    status="active",
                    latest_task_id=task.task_id,
                    messages_json=[],
                )
                session.add(session_state)
            session_state.user_id = task.user_id
            session_state.status = "active"
            session_state.latest_task_id = task.task_id
            session_state.messages_json = [
                *session_state.messages_json,
                {"role": "user", "content": task.raw_input, "task_id": task.task_id},
            ]
            session_state.updated_at = utcnow()
            record = TaskRequestRecord(
                task_id=task.task_id,
                session_id=task.session_id,
                user_id=task.user_id,
                raw_input=task.raw_input,
                mode=task.mode,
                status="queued",
                queue_state="queued",
                constraints_json=task.constraints.model_dump(mode="json"),
                metadata_json=task.metadata,
            )
            session.add(record)
            execution = DagExecutionRecord(task_id=task.task_id, status="queued", queue_state="queued")
            session.add(execution)
            session.commit()
            session.refresh(record)
            return record

    def task_payload(self, record: TaskRequestRecord, *, include_internal: bool = False) -> dict[str, Any]:
        workflow_summary = self._task_workflow_execution_summary(record=record, include_internal=include_internal)
        return {
            "task_id": record.task_id,
            "session_id": record.session_id,
            "user_id": record.user_id,
            "raw_input": record.raw_input,
            "mode": record.mode,
            "status": record.status,
            "queue_state": record.queue_state,
            "lease_owner": record.lease_owner,
            "lease_expires_at": (
                record.lease_expires_at.isoformat() if record.lease_expires_at else None
            ),
            "retry_count": record.retry_count,
            "routing_plan": record.routing_plan_json,
            "execution_dag": record.execution_dag_json,
            "execution_result": record.execution_result_json,
            "context_package": record.context_package_json,
            "attribution": record.attribution_json,
            "workflow_episode_id": workflow_summary.get("workflow_episode_id") if workflow_summary else None,
            "workflow_spec_id": workflow_summary.get("workflow_spec_id") if workflow_summary else None,
            "workflow_execution": workflow_summary,
            "error": record.error_text,
            "queued_at": record.queued_at.isoformat(),
            "started_at": record.started_at.isoformat() if record.started_at else None,
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
            "updated_at": record.updated_at.isoformat(),
        }

    def session_payload(self, record: ConsumerSessionState, *, include_internal: bool = False) -> dict[str, Any]:
        with self.db.sessionmaker() as session:
            tasks = list(
                session.execute(
                    select(TaskRequestRecord)
                    .where(TaskRequestRecord.session_id == record.session_id)
                    .order_by(TaskRequestRecord.updated_at.desc(), TaskRequestRecord.queued_at.desc())
                    .limit(10)
                ).scalars()
            )
        return {
            "session_id": record.session_id,
            "user_id": record.user_id,
            "status": record.status,
            "latest_task_id": record.latest_task_id,
            "messages": list(record.messages_json),
            "recent_tasks": [self._session_task_summary_payload(item) for item in tasks],
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }

    def get_task(self, *, task_id: str) -> TaskRequestRecord | None:
        with self.db.sessionmaker() as session:
            return session.get(TaskRequestRecord, task_id)

    def get_session(self, *, session_id: str) -> ConsumerSessionState | None:
        with self.db.sessionmaker() as session:
            return session.get(ConsumerSessionState, session_id)

    def is_task_cancelled(self, *, task_id: str) -> bool:
        with self.db.sessionmaker() as session:
            task = session.get(TaskRequestRecord, task_id)
            return bool(task is not None and task.status == "cancelled")

    def task_object(self, *, task_id: str) -> TaskObject | None:
        with self.db.sessionmaker() as session:
            record = session.get(TaskRequestRecord, task_id)
            if record is None:
                return None
            return TaskObject(
                task_id=record.task_id,
                raw_input=record.raw_input,
                mode=record.mode,
                user_id=record.user_id,
                session_id=record.session_id,
                constraints=record.constraints_json,
                metadata=record.metadata_json,
            )

    def mark_task_queued(self, *, task_id: str) -> None:
        with self.db.sessionmaker() as session:
            task = session.get(TaskRequestRecord, task_id)
            execution = session.execute(
                select(DagExecutionRecord)
                .where(DagExecutionRecord.task_id == task_id)
                .order_by(DagExecutionRecord.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if task is None:
                return
            task.status = "queued"
            task.queue_state = "queued"
            task.lease_owner = None
            task.lease_expires_at = None
            task.updated_at = utcnow()
            if execution is not None:
                execution.status = "queued"
                execution.queue_state = "queued"
                execution.worker_id = None
                execution.updated_at = utcnow()
            session.commit()

    def cancel_task(self, *, task_id: str, reason: str | None = None) -> dict[str, Any] | None:
        with self.db.sessionmaker() as session:
            task = session.get(TaskRequestRecord, task_id)
            execution = session.execute(
                select(DagExecutionRecord)
                .where(DagExecutionRecord.task_id == task_id)
                .order_by(DagExecutionRecord.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if task is None:
                return None
            metadata = dict(task.metadata_json or {})
            if reason:
                metadata["cancel_reason"] = reason
            task.metadata_json = metadata
            task.status = "cancelled"
            task.queue_state = "cancelled"
            task.error_text = reason or task.error_text
            task.lease_owner = None
            task.lease_expires_at = None
            task.completed_at = task.completed_at or utcnow()
            task.updated_at = utcnow()
            if execution is not None:
                execution.status = "cancelled"
                execution.queue_state = "cancelled"
                execution.error_text = reason or execution.error_text
                execution.worker_id = None
                execution.completed_at = execution.completed_at or utcnow()
                execution.updated_at = utcnow()
            session.commit()
            session.refresh(task)
            return self.task_payload(task, include_internal=True)

    def lease_task(
        self,
        *,
        task_id: str,
        worker_id: str,
        lease_seconds: int,
    ) -> bool:
        with self.db.sessionmaker() as session:
            task = session.get(TaskRequestRecord, task_id)
            execution = session.execute(
                select(DagExecutionRecord)
                .where(DagExecutionRecord.task_id == task_id)
                .order_by(DagExecutionRecord.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if task is None:
                return False
            now = utcnow()
            if task.status in {"completed", "failed"}:
                return False
            if task.lease_expires_at and task.lease_expires_at > now and task.lease_owner not in {
                None,
                worker_id,
            }:
                return False
            task.queue_state = "leased"
            task.lease_owner = worker_id
            task.lease_expires_at = datetime.fromtimestamp(
                now.timestamp() + max(1, lease_seconds),
                UTC,
            ).replace(tzinfo=None)
            task.updated_at = now
            if execution is not None:
                execution.queue_state = "leased"
                execution.worker_id = worker_id
                execution.updated_at = now
            session.commit()
            return True

    def extend_task_lease(
        self,
        *,
        task_id: str,
        worker_id: str,
        lease_seconds: int,
    ) -> bool:
        with self.db.sessionmaker() as session:
            task = session.get(TaskRequestRecord, task_id)
            execution = session.execute(
                select(DagExecutionRecord)
                .where(DagExecutionRecord.task_id == task_id)
                .order_by(DagExecutionRecord.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if task is None or task.status in {"completed", "failed"}:
                return False
            if task.lease_owner != worker_id:
                return False
            now = utcnow()
            task.lease_expires_at = datetime.fromtimestamp(
                now.timestamp() + max(1, lease_seconds),
                UTC,
            ).replace(tzinfo=None)
            task.updated_at = now
            if execution is not None:
                execution.queue_state = "executing"
                execution.worker_id = worker_id
                execution.updated_at = now
            session.commit()
            return True

    def mark_task_started(
        self,
        *,
        task_id: str,
        routing_plan: RoutingPlan,
        dag: ExecutionDAG,
        worker_id: str | None = None,
    ) -> None:
        with self.db.sessionmaker() as session:
            task = session.get(TaskRequestRecord, task_id)
            execution = session.execute(
                select(DagExecutionRecord)
                .where(DagExecutionRecord.task_id == task_id)
                .order_by(DagExecutionRecord.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if task is None or execution is None:
                return
            task.status = "executing"
            task.queue_state = "executing"
            task.routing_plan_json = routing_plan.model_dump(mode="json")
            task.execution_dag_json = dag.model_dump(mode="json")
            task.started_at = task.started_at or utcnow()
            task.updated_at = utcnow()
            execution.status = "running"
            execution.queue_state = "executing"
            execution.worker_id = worker_id
            execution.attempt_count += 1
            execution.started_at = execution.started_at or utcnow()
            execution.updated_at = utcnow()
            session.commit()

    def mark_task_retry(self, *, task_id: str, error_text: str) -> int:
        with self.db.sessionmaker() as session:
            task = session.get(TaskRequestRecord, task_id)
            execution = session.execute(
                select(DagExecutionRecord)
                .where(DagExecutionRecord.task_id == task_id)
                .order_by(DagExecutionRecord.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if task is None:
                return 0
            task.retry_count += 1
            task.status = "queued"
            task.queue_state = "queued"
            task.error_text = error_text
            task.lease_owner = None
            task.lease_expires_at = None
            task.updated_at = utcnow()
            if execution is not None:
                execution.status = "queued"
                execution.queue_state = "queued"
                execution.worker_id = None
                execution.errors_json = [*execution.errors_json, error_text]
                execution.updated_at = utcnow()
            session.commit()
            return task.retry_count

    def clear_task_lease(self, *, task_id: str) -> None:
        with self.db.sessionmaker() as session:
            task = session.get(TaskRequestRecord, task_id)
            if task is None:
                return
            task.lease_owner = None
            task.lease_expires_at = None
            task.updated_at = utcnow()
            session.commit()

    def attach_task_workflow_episode(
        self,
        *,
        task_id: str,
        workflow_episode_id: str,
        workflow_spec_id: str,
        summary: dict[str, Any] | None = None,
    ) -> None:
        with self.db.sessionmaker() as session:
            task = session.get(TaskRequestRecord, task_id)
            if task is None:
                return
            task.metadata_json = {
                **dict(task.metadata_json or {}),
                "workflow_episode_id": workflow_episode_id,
                "workflow_spec_id": workflow_spec_id,
                **dict(summary or {}),
            }
            task.updated_at = utcnow()
            session.commit()

    def recover_expired_leases(self) -> list[str]:
        recovered: list[str] = []
        with self.db.sessionmaker() as session:
            now = utcnow()
            rows = list(
                session.execute(
                    select(TaskRequestRecord).where(
                        TaskRequestRecord.queue_state.in_(("leased", "executing")),
                        TaskRequestRecord.status.not_in(("completed", "failed")),
                        TaskRequestRecord.lease_expires_at.is_not(None),
                        TaskRequestRecord.lease_expires_at < now,
                    )
                ).scalars()
            )
            for task in rows:
                task.status = "queued"
                task.queue_state = "queued"
                task.lease_owner = None
                task.lease_expires_at = None
                task.updated_at = now
                recovered.append(task.task_id)
                execution = session.execute(
                    select(DagExecutionRecord)
                    .where(DagExecutionRecord.task_id == task.task_id)
                    .order_by(DagExecutionRecord.created_at.desc())
                    .limit(1)
                ).scalar_one_or_none()
                if execution is not None:
                    execution.status = "queued"
                    execution.queue_state = "queued"
                    execution.worker_id = None
                    execution.updated_at = now
            session.commit()
        return recovered

    def record_node_attempt(
        self,
        *,
        task_id: str,
        execution_id: str,
        node_id: str,
        family_id: str | None,
        attempt_index: int,
        status: str,
        miner_hotkey: str | None,
        miner_endpoint: str | None,
        latency_ms: int,
        output: dict[str, Any],
        error_text: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.db.sessionmaker() as session:
            session.add(
                DagNodeExecutionRecord(
                    execution_id=execution_id,
                    task_id=task_id,
                    node_id=node_id,
                    family_id=family_id,
                    attempt_index=attempt_index,
                    miner_hotkey=miner_hotkey,
                    miner_endpoint=miner_endpoint,
                    status=status,
                    latency_ms=latency_ms,
                    output_json=output,
                    error_text=error_text,
                    metadata_json=metadata or {},
                    completed_at=utcnow(),
                )
            )
            session.commit()

    def latest_execution_id(self, *, task_id: str) -> str | None:
        with self.db.sessionmaker() as session:
            execution = session.execute(
                select(DagExecutionRecord)
                .where(DagExecutionRecord.task_id == task_id)
                .order_by(DagExecutionRecord.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            return execution.id if execution else None

    def complete_task(
        self,
        *,
        task_id: str,
        execution_result: ExecutionResult,
        context_package: dict[str, Any],
        attribution: AttributionRecord,
    ) -> None:
        with self.db.sessionmaker() as session:
            task = session.get(TaskRequestRecord, task_id)
            execution = session.execute(
                select(DagExecutionRecord)
                .where(DagExecutionRecord.task_id == task_id)
                .order_by(DagExecutionRecord.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if task is None or execution is None:
                return
            task.status = execution_result.status
            task.queue_state = "completed"
            task.execution_result_json = execution_result.model_dump(mode="json")
            task.context_package_json = context_package
            task.attribution_json = attribution.model_dump(mode="json")
            task.error_text = "\n".join(execution_result.errors) if execution_result.errors else None
            task.completed_at = utcnow()
            task.lease_owner = None
            task.lease_expires_at = None
            task.updated_at = utcnow()
            execution.status = execution_result.status
            execution.queue_state = "completed"
            execution.errors_json = list(execution_result.errors)
            execution.final_output_json = dict(execution_result.final_output)
            execution.completed_at = utcnow()
            execution.worker_id = None
            execution.updated_at = utcnow()
            session_state = session.get(ConsumerSessionState, task.session_id)
            if session_state is not None:
                session_state.latest_task_id = task_id
                session_state.status = "active"
                session_state.messages_json = [
                    *session_state.messages_json,
                    {
                        "role": "assistant",
                        "content": str(execution_result.final_output),
                        "task_id": task_id,
                    },
                ]
                session_state.updated_at = utcnow()
            session.commit()

    def fail_task(self, *, task_id: str, error_text: str) -> None:
        with self.db.sessionmaker() as session:
            task = session.get(TaskRequestRecord, task_id)
            execution = session.execute(
                select(DagExecutionRecord)
                .where(DagExecutionRecord.task_id == task_id)
                .order_by(DagExecutionRecord.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if task is None or execution is None:
                return
            task.status = "failed"
            task.queue_state = "failed"
            task.error_text = error_text
            task.completed_at = utcnow()
            task.lease_owner = None
            task.lease_expires_at = None
            task.updated_at = utcnow()
            execution.status = "failed"
            execution.queue_state = "failed"
            execution.errors_json = [error_text]
            execution.completed_at = utcnow()
            execution.worker_id = None
            execution.updated_at = utcnow()
            session_state = session.get(ConsumerSessionState, task.session_id)
            if session_state is not None:
                session_state.latest_task_id = task_id
                session_state.status = "active"
                session_state.messages_json = [
                    *session_state.messages_json,
                    {
                        "role": "assistant",
                        "content": error_text,
                        "task_id": task_id,
                        "status": "failed",
                    },
                ]
                session_state.updated_at = utcnow()
            session.commit()

    def accepted_payload(
        self,
        *,
        task_id: str,
        session_id: str,
        workflow_spec_id: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "status": "queued",
            "task_id": task_id,
            "session_id": session_id,
            "task_status_url": f"/v1/tasks/{task_id}",
            "session_url": f"/v1/sessions/{session_id}",
        }
        if workflow_spec_id:
            payload["workflow_spec_id"] = workflow_spec_id
        return payload
