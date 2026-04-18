from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import os
import socket
from dataclasses import dataclass, field
from typing import Any

import httpx

from shared.common.config import Settings, get_settings
from shared.common.execution_store import ExecutionStore
from shared.common.redis_queue import QueueMessage


def default_worker_id() -> str:
    configured = os.getenv("EXECUTION_WORKER_CONSUMER_NAME", "").strip()
    if configured:
        return configured
    return f"{socket.gethostname()}-{os.getpid()}"


@dataclass(slots=True)
class WorkerMetrics:
    queued_total: int = 0
    processed_total: int = 0
    completed_total: int = 0
    failed_total: int = 0
    retried_total: int = 0
    recovered_total: int = 0
    last_error: str | None = None
    last_activity_at: datetime | None = None


@dataclass(slots=True)
class ExecutionWorker:
    queue: Any
    store: ExecutionStore
    settings: Settings
    workflow_worker: Any | None = None
    worker_id: str = field(default_factory=default_worker_id)
    metrics: WorkerMetrics = field(default_factory=WorkerMetrics)

    def _mark_activity(self) -> None:
        self.metrics.last_activity_at = datetime.now(UTC).replace(tzinfo=None)

    async def _heartbeat_leased_task(self, *, task_id: str, stop_event: asyncio.Event) -> None:
        interval_seconds = max(
            0.25,
            min(
                float(self.settings.execution_worker_poll_interval_seconds),
                max(1, self.settings.execution_worker_lease_seconds) / 3,
            ),
        )
        while not stop_event.is_set():
            await asyncio.sleep(interval_seconds)
            if stop_event.is_set():
                break
            extended = self.store.extend_task_lease(
                task_id=task_id,
                worker_id=self.worker_id,
                lease_seconds=self.settings.execution_worker_lease_seconds,
            )
            if not extended:
                break

    async def process_message(self, message: QueueMessage) -> None:
        leased = self.store.lease_task(
            task_id=message.task_id,
            worker_id=self.worker_id,
            lease_seconds=self.settings.execution_worker_lease_seconds,
        )
        if not leased:
            await self.queue.ack(message_id=message.message_id)
            return
        heartbeat_stop = asyncio.Event()
        heartbeat_task = asyncio.create_task(
            self._heartbeat_leased_task(task_id=message.task_id, stop_event=heartbeat_stop)
        )
        try:
            await self._execute_task(task_id=message.task_id)
            self.metrics.processed_total += 1
            self.metrics.completed_total += 1
            self._mark_activity()
        except Exception as exc:
            self.metrics.last_error = str(exc)
            retry_count = self.store.mark_task_retry(
                task_id=message.task_id,
                error_text=str(exc),
            )
            if retry_count < max(1, self.settings.execution_worker_max_retries):
                await self.queue.enqueue_task(task_id=message.task_id)
                self.metrics.retried_total += 1
            else:
                self.store.fail_task(task_id=message.task_id, error_text=str(exc))
                self.metrics.failed_total += 1
            self._mark_activity()
        finally:
            heartbeat_stop.set()
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            await self.queue.ack(message_id=message.message_id)

    async def _execute_task(self, *, task_id: str) -> None:
        if self.store.is_task_cancelled(task_id=task_id):
            return
        task = self.store.task_object(task_id=task_id)
        if task is None:
            raise ValueError("task not found")

        orchestrator_url = get_settings().orchestrator_url.rstrip("/")

        # Delegate to the orchestrator service
        payload = {
            "prompt": task.raw_input,
            "user_id": task.user_id,
            "session_id": task.session_id,
            "context_history": task.context_history,
            "constraints": task.constraints.model_dump(mode="json") if task.constraints else {},
            "metadata": task.metadata or {},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{orchestrator_url}/v1/orchestrate",
                json=payload,
            )
            resp.raise_for_status()
            orchestrator_result = resp.json()

        if self.store.is_task_cancelled(task_id=task_id):
            return

        # Store orchestrator result as the execution output
        from shared.contracts.models import (
            AttributionContribution,
            AttributionRecord,
            ExecutionNodeResult,
            ExecutionResult,
        )

        node_results = []
        for step in orchestrator_result.get("steps", []):
            node_results.append(ExecutionNodeResult(
                node_id=step.get("step_id", "unknown"),
                family_id=step.get("family_id"),
                status="completed" if step.get("status") == "completed" else "failed",
                output=step.get("output", {}),
                latency_ms=int(step.get("latency_ms", 0)),
                miner_hotkey=step.get("miner_hotkey"),
                error=step.get("error"),
                metadata={"step_type": step.get("step_type")},
            ))

        execution_result = ExecutionResult(
            task_id=task_id,
            status="completed" if orchestrator_result.get("status") == "completed" else "failed",
            nodes=node_results,
            final_output=orchestrator_result.get("response", {}),
            metadata={
                "orchestrator_request_id": orchestrator_result.get("request_id"),
                "route_type": orchestrator_result.get("route_type"),
                "workflow_template": orchestrator_result.get("workflow_template"),
            },
        )

        # Build attribution inline
        completed_nodes = [n for n in node_results if n.miner_hotkey]
        depth = max(1, len(node_results))
        weight_denom = sum(range(1, len(completed_nodes) + 1)) or 1
        contributions = [
            AttributionContribution(
                family_id=n.family_id,
                miner_hotkey=n.miner_hotkey or "unknown",
                depth=i,
                contribution_weight=i / weight_denom,
                latency_ms=n.latency_ms,
            )
            for i, n in enumerate(completed_nodes, start=1)
        ]
        attribution = AttributionRecord(
            task_id=task_id,
            pipeline_depth=depth,
            contributions=contributions,
            query_volume_families=[n.family_id for n in node_results if n.family_id],
        )

        self.store.complete_task(
            task_id=task_id,
            execution_result=execution_result,
            context_package={"orchestrator_response": orchestrator_result},
            attribution=attribution,
        )

    async def reclaim_expired(self) -> list[QueueMessage]:
        recovered_task_ids = self.store.recover_expired_leases()
        for task_id in recovered_task_ids:
            await self.queue.enqueue_task(task_id=task_id)
        recovered_messages = await self.queue.recover_idle(
            consumer_name=self.worker_id,
            min_idle_ms=self.settings.execution_worker_idle_reclaim_ms,
        )
        self.metrics.recovered_total += len(recovered_task_ids) + len(recovered_messages)
        if recovered_task_ids or recovered_messages:
            self._mark_activity()
        return recovered_messages

    async def run_once(self, *, block_ms: int | None = None) -> bool:
        recovered_messages = await self.reclaim_expired()
        if recovered_messages:
            await self.process_message(recovered_messages[0])
            return True
        message = await self.queue.claim_next(
            consumer_name=self.worker_id,
            block_ms=(
                self.settings.execution_worker_block_ms
                if block_ms is None
                else max(0, int(block_ms))
            ),
        )
        if message is None:
            return False
        await self.process_message(message)
        return True

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
            "retried_total": self.metrics.retried_total,
            "recovered_total": self.metrics.recovered_total,
            "last_error": self.metrics.last_error,
            "poll_interval_seconds": float(self.settings.execution_worker_poll_interval_seconds),
            "lease_seconds": int(self.settings.execution_worker_lease_seconds),
            "last_activity_at": (
                self.metrics.last_activity_at.isoformat()
                if self.metrics.last_activity_at is not None
                else None
            ),
            "active_runtime_mode": "task_dag_v1",
        }

    def metrics_payload(self) -> str:
        last_error = (self.metrics.last_error or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        return (
            "# TYPE eirel_execution_worker_up gauge\n"
            "eirel_execution_worker_up 1\n"
            "# TYPE eirel_execution_worker_processed_total counter\n"
            f"eirel_execution_worker_processed_total {self.metrics.processed_total}\n"
            "# TYPE eirel_execution_worker_completed_total counter\n"
            f"eirel_execution_worker_completed_total {self.metrics.completed_total}\n"
            "# TYPE eirel_execution_worker_failed_total counter\n"
            f"eirel_execution_worker_failed_total {self.metrics.failed_total}\n"
            "# TYPE eirel_execution_worker_retried_total counter\n"
            f"eirel_execution_worker_retried_total {self.metrics.retried_total}\n"
            "# TYPE eirel_execution_worker_recovered_total counter\n"
            f"eirel_execution_worker_recovered_total {self.metrics.recovered_total}\n"
            f'eirel_execution_worker_last_error{{message="{last_error}"}} 1\n'
        )
