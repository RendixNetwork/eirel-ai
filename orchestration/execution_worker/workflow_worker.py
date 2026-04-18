from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import os
import socket
from typing import Any

import httpx

from shared.common.config import Settings
from shared.common.exceptions import (
    WorkflowEpisodeAbortedError,
    WorkflowEpisodeCancelledError,
    WorkflowEpisodeLeaseFencedError,
)
from shared.common.registry import fetch_registry, workflow_composition_node_registry
from shared.workflow_specs import evaluate_workflow_episode
from shared.workflow_specs import build_workflow_episode, get_workflow_spec
from shared.contracts.models import ExecutionNodeResult, ExecutionResult, NodeTrace, TaskObject, WorkflowEpisode, WorkflowEpisodeResult
from shared.workflow_runtime import execute_episode_nodes


def default_workflow_worker_id() -> str:
    configured = os.getenv("WORKFLOW_EXECUTION_WORKER_NAME", "").strip()
    if configured:
        return configured
    return f"workflow-{socket.gethostname()}-{os.getpid()}"


@dataclass(slots=True)
class WorkflowWorkerMetrics:
    queued_total: int = 0
    processed_total: int = 0
    completed_total: int = 0
    failed_total: int = 0
    deferred_total: int = 0
    recovered_total: int = 0
    retried_total: int = 0
    dead_lettered_total: int = 0
    last_error: str | None = None
    last_activity_at: datetime | None = None


@dataclass(slots=True)
class WorkflowExecutionWorker:
    settings: Settings
    worker_id: str = field(default_factory=default_workflow_worker_id)
    metrics: WorkflowWorkerMetrics = field(default_factory=WorkflowWorkerMetrics)
    client_factory: Any | None = None

    def _mark_activity(self) -> None:
        self.metrics.last_activity_at = datetime.now(UTC).replace(tzinfo=None)

    def _headers(self) -> dict[str, str]:
        token = (os.getenv("EIREL_INTERNAL_SERVICE_TOKEN") or "").strip()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @asynccontextmanager
    async def _client(self):
        if self.client_factory is not None:
            async with self.client_factory() as client:
                yield client
            return
        async with httpx.AsyncClient(
            base_url=self.settings.owner_api_url.rstrip("/"),
            timeout=20.0,
        ) as client:
            yield client

    async def _recover_expired(self) -> int:
        async with self._client() as client:
            response = await client.post(
                "/v1/internal/workflow-episodes/recover-expired-leases",
                headers=self._headers(),
            )
            response.raise_for_status()
            payload = response.json()
            recovered = int(payload.get("recovered_count", 0) or 0)
            self.metrics.recovered_total += recovered
            self.metrics.retried_total += int(payload.get("retried_count", 0) or 0)
            self.metrics.dead_lettered_total += int(payload.get("dead_lettered_count", 0) or 0)
            if recovered or int(payload.get("retried_count", 0) or 0) or int(payload.get("dead_lettered_count", 0) or 0):
                self._mark_activity()
            return recovered

    async def _list_candidates(self, *, queue_state: str) -> list[dict[str, Any]]:
        async with self._client() as client:
            response = await client.get(
                "/v1/internal/workflow-episodes",
                params={"queue_state": queue_state},
                headers=self._headers(),
            )
            response.raise_for_status()
            payload = response.json()
        items = [item for item in payload if isinstance(item, dict)]
        items.sort(key=lambda item: str(item.get("created_at") or ""))
        return items

    async def _load_episode_trace(self, *, episode_id: str) -> dict[str, Any]:
        async with self._client() as client:
            response = await client.get(
                f"/v1/internal/workflow-episodes/{episode_id}/trace",
                headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _raise_for_episode_control_error(response: httpx.Response) -> None:
        if response.status_code != 409:
            response.raise_for_status()
            return
        detail: str | dict[str, Any]
        try:
            body = response.json()
            detail = body.get("detail") if isinstance(body, dict) else body
        except Exception:
            detail = response.text
        message = str(detail or "workflow episode conflict")
        if "cancelled" in message:
            raise WorkflowEpisodeCancelledError(message)
        raise WorkflowEpisodeLeaseFencedError(message)

    @staticmethod
    def _selection_metadata_from_payload(
        payload: dict[str, Any] | None,
        *,
        workflow_spec_id: str,
        selected_node_map: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = dict(payload or {})
        selected_node_map = dict(selected_node_map or {})
        selected_deployment_ids = sorted(
            {
                str(item.get("deployment_id"))
                for item in selected_node_map.values()
                if isinstance(item, dict) and str(item.get("deployment_id") or "").strip()
            }
        )
        selected = bool(selected_node_map)
        return {
            "workflow_spec_id": workflow_spec_id,
            "workflow_composition_source": str(payload.get("selection_reason") or "").strip() or None,
            "workflow_composition_revision": str(payload.get("source_serving_release_id") or "").strip() or None,
            "workflow_composition_registry_url": (
                f"/v1/workflow-composition/registry?workflow_spec_id={workflow_spec_id}"
                if workflow_spec_id
                else "/v1/workflow-composition/registry"
            ),
            "workflow_composition_reason": (
                str(payload.get("selection_reason") or "").strip() or None
            ),
            "selected_deployment_ids": selected_deployment_ids,
            "workflow_composition_selected": selected,
        }

    async def _lease_episode(
        self,
        *,
        episode_id: str,
        active_node_id: str | None = None,
        active_role_id: str | None = None,
    ) -> dict[str, Any] | None:
        async with self._client() as client:
            response = await client.post(
                f"/v1/internal/workflow-episodes/{episode_id}/lease",
                json={
                    "worker_id": self.worker_id,
                    "lease_seconds": self.settings.execution_worker_lease_seconds,
                    "active_node_id": active_node_id,
                    "active_role_id": active_role_id,
                },
                headers=self._headers(),
            )
            if response.status_code == 400:
                return None
            self._raise_for_episode_control_error(response)
            return response.json()

    async def _heartbeat_episode(
        self,
        *,
        episode_id: str,
        queue_state: str,
        active_node_id: str | None = None,
        active_role_id: str | None = None,
        checkpoint_state: dict[str, Any] | None = None,
        runtime_state_patch: dict[str, Any] | None = None,
        resume_tokens: dict[str, str] | None = None,
        deferred_node_ids: list[str] | None = None,
        metadata_patch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "worker_id": self.worker_id,
            "lease_seconds": self.settings.execution_worker_lease_seconds,
            "queue_state": queue_state,
        }
        if active_node_id is not None:
            payload["active_node_id"] = active_node_id
        if active_role_id is not None:
            payload["active_role_id"] = active_role_id
        if checkpoint_state is not None:
            payload["checkpoint_state"] = checkpoint_state
        if runtime_state_patch is not None:
            payload["runtime_state_patch"] = runtime_state_patch
        if resume_tokens is not None:
            payload["resume_tokens"] = resume_tokens
        if deferred_node_ids is not None:
            payload["deferred_node_ids"] = deferred_node_ids
        if metadata_patch is not None:
            payload["metadata_patch"] = metadata_patch
        async with self._client() as client:
            response = await client.post(
                f"/v1/internal/workflow-episodes/{episode_id}/heartbeat",
                json=payload,
                headers=self._headers(),
            )
            self._raise_for_episode_control_error(response)
            return response.json()

    async def _complete_episode(
        self,
        *,
        episode_id: str,
        status: str,
        error_text: str | None = None,
        final_outcome_score: float | None = None,
    ) -> dict[str, Any]:
        async with self._client() as client:
            response = await client.post(
                f"/v1/internal/workflow-episodes/{episode_id}/complete",
                json={
                    "worker_id": self.worker_id,
                    "status": status,
                    "error_text": error_text,
                    "final_outcome_score": final_outcome_score,
                },
                headers=self._headers(),
            )
            self._raise_for_episode_control_error(response)
            return response.json()

    async def _store_episode_result(
        self,
        *,
        episode: WorkflowEpisode,
        result: WorkflowEpisodeResult,
    ) -> dict[str, Any]:
        async with self._client() as client:
            response = await client.post(
                "/v1/internal/workflow-episodes",
                json={
                    "episode": episode.model_dump(mode="json"),
                    "result": result.model_dump(mode="json"),
                },
                headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()

    async def _register_episode(self, *, episode: WorkflowEpisode) -> dict[str, Any]:
        async with self._client() as client:
            response = await client.post(
                "/v1/internal/workflow-episodes/register",
                json={"episode": episode.model_dump(mode="json")},
                headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()

    async def _cancel_episode(
        self,
        *,
        episode_id: str,
        reason: str,
        cancellation_source: str,
    ) -> dict[str, Any]:
        async with self._client() as client:
            response = await client.post(
                f"/v1/internal/workflow-episodes/{episode_id}/cancel",
                json={
                    "reason": reason,
                    "requested_by": self.worker_id,
                    "cancellation_source": cancellation_source,
                },
                headers=self._headers(),
            )
            self._raise_for_episode_control_error(response)
            return response.json()

    async def _update_episode_selection(
        self,
        *,
        episode: WorkflowEpisode,
        metadata_patch: dict[str, Any],
    ) -> dict[str, Any]:
        async with self._client() as client:
            response = await client.post(
                f"/v1/internal/workflow-episodes/{episode.episode_id}/update-selection",
                json={
                    "episode": episode.model_dump(mode="json"),
                    "metadata_patch": metadata_patch,
                },
                headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()

    async def _requeue_episode(
        self,
        *,
        episode_id: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        async with self._client() as client:
            response = await client.post(
                f"/v1/internal/workflow-episodes/{episode_id}/requeue",
                json={"reason": reason},
                headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()

    def _heartbeat_interval_seconds(self) -> float:
        return max(
            0.25,
            min(
                float(self.settings.execution_worker_poll_interval_seconds),
                max(1, self.settings.execution_worker_lease_seconds) / 3,
            ),
        )

    async def _heartbeat_loop(
        self,
        *,
        episode_id: str,
        shared_state: dict[str, Any],
        stop_event: asyncio.Event,
    ) -> None:
        interval_seconds = self._heartbeat_interval_seconds()
        while not stop_event.is_set():
            await asyncio.sleep(interval_seconds)
            if stop_event.is_set():
                break
            try:
                await self._heartbeat_episode(
                    episode_id=episode_id,
                    queue_state=str(shared_state.get("queue_state") or "executing"),
                    active_node_id=shared_state.get("active_node_id"),
                    active_role_id=shared_state.get("active_role_id"),
                )
            except WorkflowEpisodeCancelledError as exc:
                shared_state["abort_reason"] = str(exc)
                self.metrics.last_error = str(exc)
                self._mark_activity()
                break
            except WorkflowEpisodeLeaseFencedError as exc:
                shared_state["abort_reason"] = str(exc)
                self.metrics.last_error = str(exc)
                self._mark_activity()
                break
            except Exception as exc:
                self.metrics.last_error = str(exc)
                shared_state["abort_reason"] = str(exc)
                self._mark_activity()
                break

    def _restore_episode_state(
        self,
        *,
        payload: dict[str, Any],
    ) -> tuple[WorkflowEpisode, dict[str, dict[str, Any]], list[NodeTrace]]:
        episode_payload = payload.get("episode") if isinstance(payload.get("episode"), dict) else {}
        if not episode_payload:
            raise ValueError("workflow episode trace is missing episode payload")
        episode = WorkflowEpisode.model_validate(episode_payload)
        checkpoint_state = payload.get("checkpoint_state") if isinstance(payload.get("checkpoint_state"), dict) else {}
        resume_tokens = payload.get("resume_tokens") if isinstance(payload.get("resume_tokens"), dict) else {}
        deferred_node_ids = [
            str(item)
            for item in list(payload.get("deferred_node_ids") or [])
            if str(item).strip()
        ]
        active_node_id = str(payload.get("active_node_id") or "").strip()
        if not active_node_id and deferred_node_ids:
            active_node_id = deferred_node_ids[0]
        if not active_node_id and resume_tokens:
            active_node_id = next(
                (str(node_id) for node_id, token in resume_tokens.items() if str(token).strip()),
                "",
            )
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        completed_outputs = metadata.get("completed_outputs") if isinstance(metadata.get("completed_outputs"), dict) else {}
        partial_traces_payload = metadata.get("partial_node_traces") if isinstance(metadata.get("partial_node_traces"), list) else []
        partial_traces = [NodeTrace.model_validate(item) for item in partial_traces_payload if isinstance(item, dict)]
        for node in episode.nodes:
            if node.node_id == active_node_id:
                node.checkpoint_state = dict(checkpoint_state or {})
                resume_token = resume_tokens.get(node.node_id)
                if isinstance(resume_token, str) and resume_token.strip():
                    node.resume_token = resume_token
        return episode, {str(key): dict(value) for key, value in completed_outputs.items()}, partial_traces

    async def _abort_reason(
        self,
        *,
        episode_id: str,
        shared_state: dict[str, Any],
    ) -> str | None:
        cached = str(shared_state.get("abort_reason") or "").strip()
        if cached:
            return cached
        trace_payload = await self._load_episode_trace(episode_id=episode_id)
        if str(trace_payload.get("status") or "") == "cancelled" or bool(trace_payload.get("cancel_requested")):
            shared_state["abort_reason"] = "workflow episode cancelled"
            return "workflow episode cancelled"
        lease_owner = str(trace_payload.get("lease_owner") or "").strip()
        if lease_owner and lease_owner != self.worker_id:
            shared_state["abort_reason"] = "workflow episode lease fenced"
            return "workflow episode lease fenced"
        return None

    async def _maybe_rebind_episode_selection(
        self,
        *,
        episode: WorkflowEpisode,
        trace_payload: dict[str, Any],
        completed_outputs: dict[str, dict[str, Any]],
        partial_traces: list[NodeTrace],
    ) -> tuple[WorkflowEpisode, dict[str, Any] | None]:
        if completed_outputs or partial_traces or str(trace_payload.get("active_node_id") or "").strip():
            return episode, None
        metadata = dict(episode.metadata or {})
        workflow_spec_id = str(episode.workflow_spec_id or "").strip()
        if not workflow_spec_id:
            return episode, None
        source = str(metadata.get("source") or "").strip()
        current_revision = str(metadata.get("workflow_composition_revision") or "").strip()
        if source != "task_execution" and not current_revision:
            return episode, None
        registry, workflow_composition_registry = await fetch_registry()
        workflow_composition_payload = (
            dict(workflow_composition_registry.get(workflow_spec_id) or {})
            if isinstance(workflow_composition_registry, dict)
            else {}
        )
        next_revision = str(workflow_composition_payload.get("source_serving_release_id") or "").strip()
        selection_metadata = self._selection_metadata_from_payload(
            workflow_composition_payload,
            workflow_spec_id=workflow_spec_id,
            selected_node_map=(
                dict(workflow_composition_payload.get("selected_node_map") or {})
                if isinstance(workflow_composition_payload.get("selected_node_map"), dict)
                else {}
            ),
        )
        if not next_revision or next_revision == current_revision:
            return episode, None
        selected_nodes = workflow_composition_node_registry(
            workflow_composition_registry,
            workflow_spec_id=workflow_spec_id,
        )
        selected_node_map = (
            dict(workflow_composition_payload.get("selected_node_map") or {})
            if isinstance(workflow_composition_payload.get("selected_node_map"), dict)
            else {}
        )
        updated = episode.model_copy(deep=True)
        for node in updated.nodes:
            selected = selected_nodes.get(node.node_id)
            if selected is None:
                continue
            selected_payload = selected_node_map.get(node.node_id) if isinstance(selected_node_map.get(node.node_id), dict) else {}
            node.miner_hotkey = selected.hotkey
            node.endpoint = selected.endpoint
            node.metadata = {
                **dict(node.metadata or {}),
                "deployment_id": selected.metadata.get("source_deployment_id"),
                "submission_id": selected.metadata.get("source_submission_id"),
                "workflow_composition_source": selection_metadata.get("workflow_composition_source"),
                "workflow_composition_revision": selection_metadata.get("workflow_composition_revision"),
                "workflow_composition_registry_url": selection_metadata.get("workflow_composition_registry_url"),
                "workflow_composition_reason": selection_metadata.get("workflow_composition_reason"),
                "role_id": selected_payload.get("role_id"),
            }
        updated.metadata = {
            **dict(updated.metadata or {}),
            **selection_metadata,
        }
        await self._update_episode_selection(
            episode=updated,
            metadata_patch=selection_metadata,
        )
        return updated, selection_metadata

    async def process_episode_id(self, *, episode_id: str) -> dict[str, Any]:
        leased = await self._lease_episode(episode_id=episode_id)
        if leased is None:
            return await self._load_episode_trace(episode_id=episode_id)
        await self._process_episode(episode_payload=leased)
        return await self._load_episode_trace(episode_id=episode_id)

    async def _build_task_episode(
        self,
        *,
        task: TaskObject,
        dag,
    ) -> WorkflowEpisode:
        workflow_spec_id = str(dag.metadata.get("workflow_spec_id") or "").strip()
        if not workflow_spec_id:
            raise ValueError("task DAG has no protocol workflow spec")
        workflow_spec = get_workflow_spec(workflow_spec_id)
        registry, workflow_composition_registry = await fetch_registry()
        workflow_composition_payload = (
            dict(workflow_composition_registry.get(workflow_spec_id) or {})
            if isinstance(workflow_composition_registry, dict)
            else {}
        )
        selected_node_map = (
            dict(workflow_composition_payload.get("selected_node_map") or {})
            if isinstance(workflow_composition_payload.get("selected_node_map"), dict)
            else {}
        )
        selected_nodes = workflow_composition_node_registry(
            workflow_composition_registry,
            workflow_spec_id=workflow_spec_id,
        )
        selection_metadata = self._selection_metadata_from_payload(
            workflow_composition_payload,
            workflow_spec_id=workflow_spec_id,
            selected_node_map=selected_node_map,
        )
        coalition: dict[str, dict[str, Any]] = {}
        for spec_node in workflow_spec.nodes:
            selected = selected_nodes.get(spec_node.node_id)
            if selected is not None:
                selected_payload = selected_node_map.get(spec_node.node_id) if isinstance(selected_node_map.get(spec_node.node_id), dict) else {}
                coalition[spec_node.node_id] = {
                    "miner_hotkey": selected.hotkey,
                    "endpoint": selected.endpoint,
                    "deployment_id": selected.metadata.get("source_deployment_id"),
                    "submission_id": selected.metadata.get("source_submission_id"),
                    "workflow_composition_source": selection_metadata.get("workflow_composition_source"),
                    "workflow_composition_revision": selection_metadata.get("workflow_composition_revision"),
                    "workflow_composition_registry_url": selection_metadata.get("workflow_composition_registry_url"),
                    "workflow_composition_reason": selection_metadata.get("workflow_composition_reason"),
                    "role_id": selected_payload.get("role_id"),
                }
                continue
            candidates = sorted(
                list(registry.get(spec_node.family_id, [])),
                key=lambda item: (item.quality_score, item.stake, item.latency_score, item.hotkey),
                reverse=True,
            )
            if not candidates:
                raise ValueError(f"no serving miner available for workflow node family {spec_node.family_id}")
            coalition[spec_node.node_id] = {
                "miner_hotkey": candidates[0].hotkey,
                "endpoint": candidates[0].endpoint,
                "workflow_composition_source": selection_metadata.get("workflow_composition_source"),
                "workflow_composition_revision": selection_metadata.get("workflow_composition_revision"),
                "workflow_composition_registry_url": selection_metadata.get("workflow_composition_registry_url"),
                "workflow_composition_reason": selection_metadata.get("workflow_composition_reason"),
            }
        episode = build_workflow_episode(
            workflow_spec=workflow_spec,
            task_prompt=task.raw_input,
            run_id=f"task:{task.task_id}",
            coalition=coalition,
            initial_context={
                "task_id": task.task_id,
                "session_id": task.session_id,
                "user_id": task.user_id,
                "structured_input": dict(task.structured_input or {}),
                "task_metadata": dict(task.metadata or {}),
            },
            metadata={
                "source": "task_execution",
                "task_id": task.task_id,
                "session_id": task.session_id,
                "workflow_template": dag.metadata.get("workflow_template"),
                "workflow_version": dag.metadata.get("workflow_version"),
                **selection_metadata,
            },
        )
        episode.episode_id = f"task-{task.task_id}-{workflow_spec_id}"
        return episode

    def _execution_result_from_trace(
        self,
        *,
        task: TaskObject,
        dag,
        trace_payload: dict[str, Any],
    ) -> ExecutionResult:
        result_payload = trace_payload.get("result") if isinstance(trace_payload.get("result"), dict) else {}
        episode_payload = trace_payload.get("episode") if isinstance(trace_payload.get("episode"), dict) else {}
        episode = WorkflowEpisode.model_validate(episode_payload) if episode_payload else None
        episode_metadata = dict((episode.metadata if episode is not None else {}) or {})
        if not result_payload:
            error_text = str((trace_payload.get("metadata") or {}).get("error_text") or "workflow episode result missing")
            return ExecutionResult(
                task_id=task.task_id,
                status="failed",
                nodes=[],
                final_output={},
                errors=[error_text],
                metadata={
                    "workflow_episode_id": trace_payload.get("episode_id"),
                    "workflow_spec_id": trace_payload.get("workflow_spec_id"),
                    "execution_path": "workflow_episode_bridge_v1",
                    "workflow_composition_source": episode_metadata.get("workflow_composition_source"),
                    "workflow_composition_revision": episode_metadata.get("workflow_composition_revision"),
                    "workflow_composition_registry_url": episode_metadata.get("workflow_composition_registry_url"),
                    "workflow_composition_reason": episode_metadata.get("workflow_composition_reason"),
                },
            )
        workflow_result = WorkflowEpisodeResult.model_validate(result_payload)
        episode_nodes = {node.node_id: node for node in (episode.nodes if episode is not None else [])}
        execution_nodes: list[ExecutionNodeResult] = []
        for trace in workflow_result.node_traces:
            episode_node = episode_nodes.get(trace.node_id)
            execution_nodes.append(
                ExecutionNodeResult(
                    node_id=trace.node_id,
                    family_id=trace.family_id,
                    status="completed" if trace.status == "completed" else "failed",
                    output=dict(trace.handoff_payload.get("output", {}) or {}),
                    latency_ms=int(trace.latency_ms),
                    miner_hotkey=trace.miner_hotkey,
                    error=(trace.metadata or {}).get("error"),
                    metadata={
                        "attempts": [
                            {
                                "attempt_index": 1,
                                "status": trace.status,
                                "latency_ms": trace.latency_ms,
                                "miner_hotkey": trace.miner_hotkey,
                                "miner_endpoint": episode_node.endpoint if episode_node is not None else None,
                            }
                        ],
                        "checkpoint_events": list(trace.checkpoint_events or []),
                        "runtime_state_patch": dict(trace.runtime_state_patch or {}),
                        "runtime_contract_mode": trace.runtime_contract_mode,
                        "role_id": trace.role_id,
                        "workflow_episode_node": True,
                    },
                )
            )
        final_family_id = next(
            (
                trace.family_id
                for trace in reversed(workflow_result.node_traces)
                if trace.status == "completed"
            ),
            None,
        )
        return ExecutionResult(
            task_id=task.task_id,
            status=workflow_result.status,
            nodes=execution_nodes,
            final_output={
                "response": dict(workflow_result.final_output or {}),
                "response_owner": "workflow_episode",
                "response_family": final_family_id,
                "workflow_episode_id": workflow_result.episode_id,
                "workflow_spec_id": workflow_result.workflow_spec_id,
                "workflow_class": workflow_result.workflow_class,
            },
            errors=list(workflow_result.metadata.get("failed_node_ids", []) or []) if workflow_result.status == "failed" else [],
            metadata={
                "workflow_episode_id": workflow_result.episode_id,
                "workflow_spec_id": workflow_result.workflow_spec_id,
                "workflow_class": workflow_result.workflow_class,
                "execution_path": "workflow_episode_bridge_v1",
                "workflow_final_outcome_score": workflow_result.final_outcome_score,
                "workflow_composition_source": episode_metadata.get("workflow_composition_source"),
                "workflow_composition_revision": episode_metadata.get("workflow_composition_revision"),
                "workflow_composition_registry_url": episode_metadata.get("workflow_composition_registry_url"),
                "workflow_composition_reason": episode_metadata.get("workflow_composition_reason"),
            },
            completed_at=workflow_result.completed_at,
        )

    async def run_task_workflow(
        self,
        *,
        task: TaskObject,
        dag,
        episode_id: str | None = None,
    ) -> tuple[str, ExecutionResult, dict[str, Any]]:
        trace_payload: dict[str, Any]
        resolved_episode_id = str(episode_id or "").strip()
        if resolved_episode_id:
            trace_payload = await self._load_episode_trace(episode_id=resolved_episode_id)
        else:
            episode = await self._build_task_episode(task=task, dag=dag)
            resolved_episode_id = episode.episode_id
            trace_payload = await self._register_episode(episode=episode)
        if str(trace_payload.get("status") or "") not in {"completed", "failed", "cancelled", "dead_lettered"}:
            trace_payload = await self.process_episode_id(episode_id=resolved_episode_id)
        execution_result = self._execution_result_from_trace(task=task, dag=dag, trace_payload=trace_payload)
        return resolved_episode_id, execution_result, trace_payload

    async def _process_episode(self, *, episode_payload: dict[str, Any]) -> None:
        episode_id = str(episode_payload.get("episode_id") or "").strip()
        trace_payload = await self._load_episode_trace(episode_id=episode_id)
        episode, completed_outputs, partial_traces = self._restore_episode_state(payload=trace_payload)
        episode, selection_metadata = await self._maybe_rebind_episode_selection(
            episode=episode,
            trace_payload=trace_payload,
            completed_outputs=completed_outputs,
            partial_traces=partial_traces,
        )
        stop_event = asyncio.Event()
        shared_state: dict[str, Any] = {
            "queue_state": "executing",
            "active_node_id": trace_payload.get("active_node_id"),
            "active_role_id": trace_payload.get("active_role_id"),
        }
        if selection_metadata:
            shared_state["selection_metadata"] = dict(selection_metadata)
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(
                episode_id=episode_id,
                shared_state=shared_state,
                stop_event=stop_event,
            )
        )

        def completed_output_payloads() -> dict[str, dict[str, Any]]:
            return {
                node_id: dict(value)
                for node_id, value in completed_outputs.items()
            }

        def partial_trace_payloads(traces: list[NodeTrace]) -> list[dict[str, Any]]:
            return [trace.model_dump(mode="json") for trace in traces]

        async def update_callback(event: dict[str, Any]) -> None:
            event_type = str(event.get("event_type") or "")
            shared_state["active_node_id"] = event.get("node_id")
            shared_state["active_role_id"] = event.get("role_id")
            if event_type == "deferred":
                self.metrics.deferred_total += 1
                shared_state["queue_state"] = "deferred"
                await self._heartbeat_episode(
                    episode_id=episode_id,
                    queue_state="deferred",
                    active_node_id=event.get("node_id"),
                    active_role_id=event.get("role_id"),
                    checkpoint_state=dict(event.get("checkpoint_state") or {}),
                    runtime_state_patch=dict(event.get("runtime_state_patch") or {}),
                    resume_tokens={str(event.get("node_id")): str(event.get("resume_token") or "")},
                    deferred_node_ids=[str(event.get("node_id"))],
                    metadata_patch={
                        "completed_outputs": completed_output_payloads(),
                        "partial_node_traces": partial_trace_payloads(partial_traces),
                        "runtime_contract_mode": event.get("runtime_contract_mode"),
                        **dict(shared_state.get("selection_metadata") or {}),
                    },
                )
                return
            trace_payload = event.get("trace") if isinstance(event.get("trace"), dict) else None
            if trace_payload is not None:
                trace = NodeTrace.model_validate(trace_payload)
                partial_traces[:] = [item for item in partial_traces if item.node_id != trace.node_id]
                partial_traces.append(trace)
            if event_type == "completed":
                completed_output = event.get("completed_output")
                if isinstance(completed_output, dict):
                    completed_outputs[str(event.get("node_id"))] = dict(completed_output)
                shared_state["queue_state"] = "executing"
                await self._heartbeat_episode(
                    episode_id=episode_id,
                    queue_state="executing",
                    active_node_id=event.get("node_id"),
                    active_role_id=event.get("role_id"),
                    checkpoint_state=dict(event.get("checkpoint_state") or {}),
                    runtime_state_patch={},
                    deferred_node_ids=[],
                    metadata_patch={
                        "completed_outputs": completed_output_payloads(),
                        "partial_node_traces": partial_trace_payloads(partial_traces),
                        "runtime_contract_mode": event.get("runtime_contract_mode"),
                        **dict(shared_state.get("selection_metadata") or {}),
                    },
                )
                return
            if event_type == "failed":
                shared_state["queue_state"] = "executing"
                await self._heartbeat_episode(
                    episode_id=episode_id,
                    queue_state="executing",
                    active_node_id=event.get("node_id"),
                    active_role_id=event.get("role_id"),
                    checkpoint_state=dict(event.get("checkpoint_state") or {}),
                    runtime_state_patch={},
                    deferred_node_ids=[],
                    metadata_patch={
                        "completed_outputs": completed_output_payloads(),
                        "partial_node_traces": partial_trace_payloads(partial_traces),
                        "runtime_contract_mode": event.get("runtime_contract_mode"),
                        "last_error": event.get("error"),
                        **dict(shared_state.get("selection_metadata") or {}),
                    },
                )

        async def abort_callback(episode_id: str, node_id: str, deferred_round: int) -> str | None:
            del node_id, deferred_round
            return await self._abort_reason(episode_id=episode_id, shared_state=shared_state)

        try:
            completed_outputs, node_traces, _ = await execute_episode_nodes(
                episode=episode,
                task_prompt=episode.task_prompt,
                completed_outputs=completed_outputs,
                existing_traces=partial_traces,
                update_callback=update_callback,
                abort_callback=abort_callback,
            )
            final_output: dict[str, Any] = {}
            final_trace = next((trace for trace in reversed(node_traces) if trace.status == "completed"), None)
            if final_trace is not None:
                final_output = dict(completed_outputs.get(final_trace.node_id, {}).get("output", {}) or {})
            result = evaluate_workflow_episode(
                episode=episode,
                node_traces=node_traces,
                final_output=final_output,
            )
            result.metadata = {
                **dict(result.metadata or {}),
                "execution_path": "workflow_runtime_v3",
                "runtime_orchestrated": True,
            }
            stored_payload = await self._store_episode_result(episode=episode, result=result)
            self.metrics.processed_total += 1
            if result.status == "completed":
                self.metrics.completed_total += 1
            else:
                finalized = await self._complete_episode(
                    episode_id=episode_id,
                    status="failed",
                    error_text=str((stored_payload.get("metadata") or {}).get("error_text") or "workflow episode failed"),
                    final_outcome_score=result.final_outcome_score,
                )
                if str(finalized.get("queue_state") or "") == "dead_lettered":
                    self.metrics.dead_lettered_total += 1
                else:
                    self.metrics.retried_total += 1
                self._mark_activity()
        except WorkflowEpisodeCancelledError as exc:
            self.metrics.last_error = str(exc)
            self.metrics.processed_total += 1
            await self._complete_episode(
                episode_id=episode_id,
                status="cancelled",
                error_text=str(exc),
            )
            self._mark_activity()
        except WorkflowEpisodeLeaseFencedError as exc:
            self.metrics.last_error = str(exc)
            self.metrics.processed_total += 1
            self._mark_activity()
        except WorkflowEpisodeAbortedError as exc:
            self.metrics.last_error = str(exc)
            self.metrics.processed_total += 1
            if "cancelled" in str(exc).lower():
                await self._complete_episode(
                    episode_id=episode_id,
                    status="cancelled",
                    error_text=str(exc),
                )
            else:
                finalized = await self._complete_episode(
                    episode_id=episode_id,
                    status="failed",
                    error_text=str(exc),
                )
                if str(finalized.get("queue_state") or "") == "dead_lettered":
                    self.metrics.dead_lettered_total += 1
                elif str(finalized.get("queue_state") or "") == "queued":
                    self.metrics.retried_total += 1
                else:
                    self.metrics.failed_total += 1
            self._mark_activity()
        except Exception as exc:
            self.metrics.last_error = str(exc)
            self.metrics.processed_total += 1
            finalized = await self._complete_episode(
                episode_id=episode_id,
                status="failed",
                error_text=str(exc),
            )
            if str(finalized.get("queue_state") or "") == "dead_lettered":
                self.metrics.dead_lettered_total += 1
            elif str(finalized.get("queue_state") or "") == "queued":
                self.metrics.retried_total += 1
            else:
                self.metrics.failed_total += 1
            self._mark_activity()
        finally:
            stop_event.set()
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    async def reclaim_expired(self) -> int:
        return await self._recover_expired()

    async def run_once(self) -> bool:
        await self.reclaim_expired()
        candidates = [
            *(await self._list_candidates(queue_state="deferred")),
            *(await self._list_candidates(queue_state="queued")),
        ]
        if not candidates:
            return False
        now = datetime.now(UTC).replace(tzinfo=None)
        for candidate in candidates:
            episode_id = str(candidate.get("episode_id") or "").strip()
            if not episode_id:
                continue
            next_eligible_at = str(candidate.get("next_eligible_at") or "").strip()
            if next_eligible_at:
                try:
                    if datetime.fromisoformat(next_eligible_at) > now:
                        continue
                except ValueError:
                    pass
            leased = await self._lease_episode(
                episode_id=episode_id,
                active_node_id=candidate.get("active_node_id"),
                active_role_id=candidate.get("active_role_id"),
            )
            if leased is None:
                continue
            self.metrics.queued_total += 1
            self._mark_activity()
            await self._process_episode(episode_payload=leased)
            return True
        return False

    async def run_forever(self, *, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                processed = await self.run_once()
                if not processed:
                    await asyncio.sleep(self.settings.execution_worker_poll_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.metrics.last_error = str(exc)
                self._mark_activity()
                await asyncio.sleep(self.settings.execution_worker_poll_interval_seconds)

    def status_payload(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "up": True,
            "processed_total": self.metrics.processed_total,
            "completed_total": self.metrics.completed_total,
            "failed_total": self.metrics.failed_total,
            "deferred_total": self.metrics.deferred_total,
            "recovered_total": self.metrics.recovered_total,
            "retried_total": self.metrics.retried_total,
            "dead_lettered_total": self.metrics.dead_lettered_total,
            "last_error": self.metrics.last_error,
            "poll_interval_seconds": float(self.settings.execution_worker_poll_interval_seconds),
            "lease_seconds": int(self.settings.execution_worker_lease_seconds),
            "last_activity_at": (
                self.metrics.last_activity_at.isoformat()
                if self.metrics.last_activity_at is not None
                else None
            ),
            "active_runtime_mode": "workflow_episode_v3",
        }

    def metrics_payload(self) -> str:
        last_error = (self.metrics.last_error or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        return (
            "# TYPE eirel_workflow_execution_worker_up gauge\n"
            "eirel_workflow_execution_worker_up 1\n"
            "# TYPE eirel_workflow_execution_worker_processed_total counter\n"
            f"eirel_workflow_execution_worker_processed_total {self.metrics.processed_total}\n"
            "# TYPE eirel_workflow_execution_worker_completed_total counter\n"
            f"eirel_workflow_execution_worker_completed_total {self.metrics.completed_total}\n"
            "# TYPE eirel_workflow_execution_worker_failed_total counter\n"
            f"eirel_workflow_execution_worker_failed_total {self.metrics.failed_total}\n"
            "# TYPE eirel_workflow_execution_worker_deferred_total counter\n"
            f"eirel_workflow_execution_worker_deferred_total {self.metrics.deferred_total}\n"
            "# TYPE eirel_workflow_execution_worker_recovered_total counter\n"
            f"eirel_workflow_execution_worker_recovered_total {self.metrics.recovered_total}\n"
            "# TYPE eirel_workflow_execution_worker_retried_total counter\n"
            f"eirel_workflow_execution_worker_retried_total {self.metrics.retried_total}\n"
            "# TYPE eirel_workflow_execution_worker_dead_lettered_total counter\n"
            f"eirel_workflow_execution_worker_dead_lettered_total {self.metrics.dead_lettered_total}\n"
            f'eirel_workflow_execution_worker_last_error{{message="{last_error}"}} 1\n'
        )
