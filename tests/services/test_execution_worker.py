from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch
from sqlalchemy import select

from shared.common.config import get_settings
from shared.common.database import Database
from shared.common.execution_store import ExecutionStore
from shared.common.redis_queue import InMemoryExecutionQueue
from orchestration.execution_worker.worker import ExecutionWorker
from shared.contracts.models import ExecutionNodeResult, ExecutionResult, TaskConstraints, TaskObject


def _fake_orchestrator_response(*, task_id: str, status: str = "completed") -> dict:
    """Build a fake orchestrator response payload."""
    return {
        "request_id": "test-req-1",
        "session_id": "session-1",
        "status": status,
        "route_type": "specialist",
        "workflow_template": "direct_analysis",
        "plan_id": "plan-1",
        "response": {"summary": "done"},
        "steps": [
            {
                "step_id": "general_chat-1",
                "step_type": "specialist",
                "status": "completed",
                "output": {"summary": "done"},
                "latency_ms": 5,
                "family_id": "general_chat",
                "miner_hotkey": "miner-1",
                "error": None,
                "tool_name": None,
            }
        ],
        "total_latency_ms": 10.0,
        "metadata": {"step_count": 1, "completed_steps": 1, "failed_steps": 0},
    }


class FakeHTTPResponse:
    """Mimics httpx.Response for mocking."""
    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self._data = data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self) -> dict:
        return self._data


class FakeAsyncClient:
    """Mimics httpx.AsyncClient for mocking."""
    def __init__(self, response: FakeHTTPResponse):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url, json=None, **kwargs):
        return self._response


async def test_execution_worker_processes_queued_task(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'worker.db'}")
    monkeypatch.setenv("ORCHESTRATOR_URL", "http://fake-orchestrator:8050")
    db = Database(get_settings().database_url)
    db.create_all()
    store = ExecutionStore(db)
    task = TaskObject(
        task_id="task-1",
        session_id="session-1",
        user_id="user-1",
        raw_input="answer this",
        constraints=TaskConstraints(),
    )
    store.ensure_session(session_id=task.session_id, user_id=task.user_id, initial_prompt=task.raw_input)
    store.create_task(task=task)

    fake_response = FakeHTTPResponse(200, _fake_orchestrator_response(task_id="task-1"))
    with patch("httpx.AsyncClient", return_value=FakeAsyncClient(fake_response)):
        queue = InMemoryExecutionQueue()
        await queue.ensure_consumer_group()
        await queue.enqueue_task(task_id=task.task_id)
        worker = ExecutionWorker(queue=queue, store=store, settings=get_settings(), worker_id="worker-a")

        processed = await worker.run_once()

    assert processed is True
    record = store.get_task(task_id=task.task_id)
    assert record is not None
    assert record.status == "completed"
    assert record.queue_state == "completed"
    assert record.retry_count == 0


async def test_execution_worker_retries_then_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'worker-retry.db'}")
    monkeypatch.setenv("EXECUTION_WORKER_MAX_RETRIES", "2")
    monkeypatch.setenv("ORCHESTRATOR_URL", "http://fake-orchestrator:8050")
    db = Database(get_settings().database_url)
    db.create_all()
    store = ExecutionStore(db)
    task = TaskObject(
        task_id="task-2",
        session_id="session-2",
        user_id="user-2",
        raw_input="this should fail",
        constraints=TaskConstraints(),
    )
    store.ensure_session(session_id=task.session_id, user_id=task.user_id, initial_prompt=task.raw_input)
    store.create_task(task=task)

    fake_response = FakeHTTPResponse(500, {"detail": "internal error"})
    with patch("httpx.AsyncClient", return_value=FakeAsyncClient(fake_response)):
        queue = InMemoryExecutionQueue()
        await queue.ensure_consumer_group()
        await queue.enqueue_task(task_id=task.task_id)
        worker = ExecutionWorker(queue=queue, store=store, settings=get_settings(), worker_id="worker-b")

        assert await worker.run_once() is True
        retried = store.get_task(task_id=task.task_id)
        assert retried is not None
        assert retried.status == "queued"
        assert retried.retry_count == 1

        assert await worker.run_once() is True
        failed = store.get_task(task_id=task.task_id)
        assert failed is not None
        assert failed.status == "failed"
        assert failed.queue_state == "failed"
        assert failed.retry_count == 2


async def test_execution_worker_heartbeats_long_running_lease(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'worker-heartbeat.db'}")
    monkeypatch.setenv("EXECUTION_WORKER_LEASE_SECONDS", "1")
    monkeypatch.setenv("EXECUTION_WORKER_POLL_INTERVAL_SECONDS", "0.1")
    monkeypatch.setenv("ORCHESTRATOR_URL", "http://fake-orchestrator:8050")
    db = Database(get_settings().database_url)
    db.create_all()
    store = ExecutionStore(db)
    task = TaskObject(
        task_id="task-3",
        session_id="session-3",
        user_id="user-3",
        raw_input="long running task",
        constraints=TaskConstraints(),
    )
    store.ensure_session(session_id=task.session_id, user_id=task.user_id, initial_prompt=task.raw_input)
    store.create_task(task=task)

    call_count = 0

    class SlowFakeAsyncClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, url, json=None, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(1.3)
            return FakeHTTPResponse(200, _fake_orchestrator_response(task_id="task-3"))

    with patch("httpx.AsyncClient", return_value=SlowFakeAsyncClient()):
        queue = InMemoryExecutionQueue()
        await queue.ensure_consumer_group()
        await queue.enqueue_task(task_id=task.task_id)
        worker = ExecutionWorker(queue=queue, store=store, settings=get_settings(), worker_id="worker-c")

        run_task = asyncio.create_task(worker.run_once())
        await asyncio.sleep(1.05)

        recovered = store.recover_expired_leases()
        assert recovered == []

        assert await run_task is True
        record = store.get_task(task_id=task.task_id)
        assert record is not None
        assert record.status == "completed"
        assert record.queue_state == "completed"
        assert record.retry_count == 0
        assert call_count == 1
