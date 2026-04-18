from __future__ import annotations

from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient

from orchestration.execution_worker.main import app


async def test_execution_worker_runtime_status_endpoints_expose_task_and_workflow_workers():
    app.state.worker = SimpleNamespace(
        worker_id="task-worker-1",
        status_payload=lambda: {
            "worker_id": "task-worker-1",
            "up": True,
            "processed_total": 3,
            "completed_total": 2,
            "failed_total": 1,
            "retried_total": 1,
            "recovered_total": 0,
            "last_error": "task error",
            "poll_interval_seconds": 0.5,
            "lease_seconds": 60,
            "last_activity_at": "2026-04-02T00:00:00",
            "active_runtime_mode": "task_dag_v1",
        },
    )
    app.state.workflow_worker = SimpleNamespace(
        worker_id="workflow-worker-1",
        status_payload=lambda: {
            "worker_id": "workflow-worker-1",
            "up": True,
            "processed_total": 5,
            "completed_total": 4,
            "failed_total": 1,
            "deferred_total": 2,
            "recovered_total": 1,
            "retried_total": 1,
            "dead_lettered_total": 1,
            "last_error": "workflow error",
            "poll_interval_seconds": 0.5,
            "lease_seconds": 60,
            "last_activity_at": "2026-04-02T00:00:01",
            "active_runtime_mode": "workflow_episode_v3",
        },
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        runtime_status = await client.get("/v1/operators/runtime-status")
        assert runtime_status.status_code == 200
        payload = runtime_status.json()
        assert payload["service"] == "execution-worker"
        assert payload["task_worker"]["worker_id"] == "task-worker-1"
        assert payload["workflow_worker"]["dead_lettered_total"] == 1

        task_status = await client.get("/v1/operators/runtime-status/tasks")
        assert task_status.status_code == 200
        assert task_status.json()["active_runtime_mode"] == "task_dag_v1"

        workflow_status = await client.get("/v1/operators/runtime-status/workflow")
        assert workflow_status.status_code == 200
        assert workflow_status.json()["active_runtime_mode"] == "workflow_episode_v3"


async def test_execution_worker_runtime_action_endpoints_expose_recover_and_run_once():
    state = {
        "task_runs": [],
        "workflow_runs": [],
    }

    async def task_reclaim_expired():
        return [SimpleNamespace(task_id="task-1")]

    async def workflow_reclaim_expired():
        return 2

    async def task_run_once(*, block_ms=None):
        state["task_runs"].append(block_ms)
        return True

    async def workflow_run_once():
        state["workflow_runs"].append("workflow")
        return False

    app.state.worker = SimpleNamespace(
        reclaim_expired=task_reclaim_expired,
        run_once=task_run_once,
        status_payload=lambda: {
            "worker_id": "task-worker-1",
            "up": True,
            "processed_total": 4,
            "completed_total": 3,
            "failed_total": 1,
            "retried_total": 1,
            "recovered_total": 1,
            "last_error": None,
            "poll_interval_seconds": 0.5,
            "lease_seconds": 60,
            "last_activity_at": "2026-04-02T00:00:02",
            "active_runtime_mode": "task_dag_v1",
        },
    )
    app.state.workflow_worker = SimpleNamespace(
        reclaim_expired=workflow_reclaim_expired,
        run_once=workflow_run_once,
        status_payload=lambda: {
            "worker_id": "workflow-worker-1",
            "up": True,
            "processed_total": 8,
            "completed_total": 6,
            "failed_total": 1,
            "deferred_total": 1,
            "recovered_total": 2,
            "retried_total": 1,
            "dead_lettered_total": 0,
            "last_error": None,
            "poll_interval_seconds": 0.5,
            "lease_seconds": 60,
            "last_activity_at": "2026-04-02T00:00:03",
            "active_runtime_mode": "workflow_episode_v3",
        },
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        recover = await client.post("/v1/operators/runtime-actions/recover")
        assert recover.status_code == 200
        recover_payload = recover.json()
        assert recover_payload["action"] == "recover"
        assert recover_payload["task_worker"]["recovered"] == 1
        assert recover_payload["workflow_worker"]["recovered"] == 2

        run_once = await client.post(
            "/v1/operators/runtime-actions/run-once",
            json={"non_blocking": True},
        )
        assert run_once.status_code == 200
        run_once_payload = run_once.json()
        assert run_once_payload["action"] == "run_once"
        assert run_once_payload["task_worker"]["processed"] is True
        assert run_once_payload["workflow_worker"]["processed"] is False

    assert state["task_runs"] == [0]
    assert state["workflow_runs"] == ["workflow"]
