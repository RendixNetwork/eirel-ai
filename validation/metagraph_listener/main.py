from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import Response
from pydantic import BaseModel, Field

from shared.common.config import get_settings
from shared.common.database import Database
from shared.common.metagraph import MetagraphNeuron
from validation.metagraph_listener.listener import MetagraphSyncService


logger = logging.getLogger(__name__)


class SyncRunRequest(BaseModel):
    neurons: list[MetagraphNeuron] = Field(default_factory=list)


async def _periodic_sync_loop(
    service: MetagraphSyncService, *, interval_seconds: int,
) -> None:
    """Background task: re-sync the metagraph every ``interval_seconds``.

    Runs a sync immediately so the DB is populated without waiting a full
    interval after startup. Catches and logs per-iteration failures so a
    transient subtensor RPC error doesn't kill the loop. Re-raises
    ``asyncio.CancelledError`` for clean shutdown.
    """
    interval = max(5, int(interval_seconds or 60))
    while True:
        try:
            result = await service.run_sync(neurons=None)
            logger.info(
                "metagraph sync ok: neurons=%s validators=%s",
                result.get("neuron_count"), result.get("validator_count"),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("metagraph sync failed: %s", exc)
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    db = Database(settings.database_url)
    db.create_all()
    service = MetagraphSyncService(
        db=db,
        network=settings.bittensor_network,
        netuid=settings.bittensor_netuid,
        snapshot_path=settings.metagraph_snapshot_path,
    )
    app.state.sync_service = service

    sync_task = asyncio.create_task(
        _periodic_sync_loop(
            service,
            interval_seconds=settings.metagraph_sync_interval_seconds,
        )
    )
    try:
        yield
    finally:
        sync_task.cancel()
        try:
            await sync_task
        except asyncio.CancelledError:
            pass
        db.engine.dispose()


app = FastAPI(title="metagraph-listener", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics(request: Request) -> Response:
    payload = request.app.state.sync_service.status_payload()
    is_success = 1 if payload.get("status") == "success" else 0
    neuron_count = int(payload.get("neuron_count", 0))
    last_sync_ts = _iso_to_epoch_seconds(payload.get("created_at"))
    validator_count = int(payload.get("validator_count", 0))
    return Response(
        content=(
            "# TYPE eirel_metagraph_listener_up gauge\n"
            "eirel_metagraph_listener_up 1\n"
            "# TYPE eirel_metagraph_listener_last_sync_success gauge\n"
            f"eirel_metagraph_listener_last_sync_success {is_success}\n"
            "# TYPE eirel_metagraph_listener_last_sync_timestamp_seconds gauge\n"
            f"eirel_metagraph_listener_last_sync_timestamp_seconds {last_sync_ts}\n"
            "# TYPE eirel_metagraph_listener_registered_neurons gauge\n"
            f"eirel_metagraph_listener_registered_neurons {neuron_count}\n"
            "# TYPE eirel_metagraph_listener_active_validators gauge\n"
            f"eirel_metagraph_listener_active_validators {validator_count}\n"
        ),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


def _iso_to_epoch_seconds(value: Any) -> float:
    if not value:
        return 0.0
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


@app.get("/v1/sync/status")
async def sync_status() -> dict[str, Any]:
    return app.state.sync_service.status_payload()


@app.post("/v1/sync/run")
async def sync_run(payload: SyncRunRequest | None = None) -> dict[str, Any]:
    try:
        neurons = payload.neurons if payload and payload.neurons else None
        return await app.state.sync_service.run_sync(neurons=neurons)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def main() -> None:
    uvicorn.run("validation.metagraph_listener.main:app", host="0.0.0.0", port=8011, reload=False)
