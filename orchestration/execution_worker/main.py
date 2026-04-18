from __future__ import annotations

import asyncio
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.responses import Response
from pydantic import BaseModel

from shared.common.config import get_settings
from shared.common.database import Database
from shared.common.execution_store import ExecutionStore
from shared.common.redis_queue import create_execution_queue
from orchestration.execution_worker.worker import ExecutionWorker
from orchestration.execution_worker.workflow_worker import WorkflowExecutionWorker


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    db = Database(settings.database_url)
    db.create_all()
    queue = create_execution_queue(
        redis_url=settings.redis_url,
        stream_name=settings.execution_stream_name,
        group_name=settings.execution_stream_group,
    )
    await queue.ensure_consumer_group()
    store = ExecutionStore(db)
    workflow_worker = WorkflowExecutionWorker(settings=settings)
    worker = ExecutionWorker(
        queue=queue,
        store=store,
        settings=settings,
        workflow_worker=workflow_worker,
    )
    stop_event = asyncio.Event()
    task = asyncio.create_task(worker.run_forever(stop_event=stop_event))
    workflow_task = asyncio.create_task(workflow_worker.run_forever(stop_event=stop_event))
    app.state.settings = settings
    app.state.db = db
    app.state.queue = queue
    app.state.worker = worker
    app.state.workflow_worker = workflow_worker
    app.state.stop_event = stop_event
    app.state.worker_task = task
    app.state.workflow_worker_task = workflow_task
    try:
        yield
    finally:
        stop_event.set()
        task.cancel()
        workflow_task.cancel()
        for t in (task, workflow_task):
            try:
                await asyncio.wait_for(t, timeout=10.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        await queue.close()


app = FastAPI(title="execution-worker", lifespan=lifespan)


class RuntimeActionRequest(BaseModel):
    include_task_worker: bool = True
    include_workflow_worker: bool = True


class RuntimeRunOnceRequest(RuntimeActionRequest):
    non_blocking: bool = True


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, str]:
    worker: ExecutionWorker = request.app.state.worker
    workflow_worker: WorkflowExecutionWorker = request.app.state.workflow_worker
    return {
        "status": "ok",
        "worker_id": worker.worker_id,
        "workflow_worker_id": workflow_worker.worker_id,
    }


@app.get("/metrics")
async def metrics(request: Request) -> Response:
    worker: ExecutionWorker = request.app.state.worker
    workflow_worker: WorkflowExecutionWorker = request.app.state.workflow_worker
    return Response(
        content=worker.metrics_payload() + workflow_worker.metrics_payload(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/v1/operators/runtime-status/tasks")
async def task_runtime_status(request: Request) -> dict[str, object]:
    worker: ExecutionWorker = request.app.state.worker
    return worker.status_payload()


@app.get("/v1/operators/runtime-status/workflow")
async def workflow_runtime_status(request: Request) -> dict[str, object]:
    workflow_worker: WorkflowExecutionWorker = request.app.state.workflow_worker
    return workflow_worker.status_payload()


@app.get("/v1/operators/runtime-status")
async def runtime_status(request: Request) -> dict[str, object]:
    worker: ExecutionWorker = request.app.state.worker
    workflow_worker: WorkflowExecutionWorker = request.app.state.workflow_worker
    return {
        "service": "execution-worker",
        "task_worker": worker.status_payload(),
        "workflow_worker": workflow_worker.status_payload(),
    }


@app.post("/v1/leases/recover")
async def recover_leases(request: Request) -> dict[str, int]:
    worker: ExecutionWorker = request.app.state.worker
    workflow_worker: WorkflowExecutionWorker = request.app.state.workflow_worker
    recovered_messages = await worker.reclaim_expired()
    recovered_workflow_episodes = await workflow_worker.reclaim_expired()
    return {
        "recovered": len(recovered_messages),
        "workflow_recovered": recovered_workflow_episodes,
    }


@app.post("/v1/operators/runtime-actions/recover")
async def runtime_action_recover(
    request: Request,
    payload: RuntimeActionRequest | None = None,
) -> dict[str, object]:
    worker: ExecutionWorker = request.app.state.worker
    workflow_worker: WorkflowExecutionWorker = request.app.state.workflow_worker
    include_task_worker = True if payload is None else payload.include_task_worker
    include_workflow_worker = True if payload is None else payload.include_workflow_worker
    recovered_messages = await worker.reclaim_expired() if include_task_worker else []
    recovered_workflow_episodes = (
        await workflow_worker.reclaim_expired()
        if include_workflow_worker
        else 0
    )
    return {
        "service": "execution-worker",
        "action": "recover",
        "task_worker": {
            "included": include_task_worker,
            "recovered": len(recovered_messages),
            "status": worker.status_payload(),
        },
        "workflow_worker": {
            "included": include_workflow_worker,
            "recovered": recovered_workflow_episodes,
            "status": workflow_worker.status_payload(),
        },
    }


@app.post("/v1/operators/runtime-actions/run-once")
async def runtime_action_run_once(
    request: Request,
    payload: RuntimeRunOnceRequest | None = None,
) -> dict[str, object]:
    worker: ExecutionWorker = request.app.state.worker
    workflow_worker: WorkflowExecutionWorker = request.app.state.workflow_worker
    include_task_worker = True if payload is None else payload.include_task_worker
    include_workflow_worker = True if payload is None else payload.include_workflow_worker
    non_blocking = True if payload is None else payload.non_blocking
    task_processed = (
        await worker.run_once(block_ms=0 if non_blocking else None)
        if include_task_worker
        else False
    )
    workflow_processed = (
        await workflow_worker.run_once()
        if include_workflow_worker
        else False
    )
    return {
        "service": "execution-worker",
        "action": "run_once",
        "non_blocking": non_blocking,
        "task_worker": {
            "included": include_task_worker,
            "processed": task_processed,
            "status": worker.status_payload(),
        },
        "workflow_worker": {
            "included": include_workflow_worker,
            "processed": workflow_processed,
            "status": workflow_worker.status_payload(),
        },
    }


def main() -> None:
    uvicorn.run("orchestration.execution_worker.main:app", host="0.0.0.0", port=8006, reload=False)
