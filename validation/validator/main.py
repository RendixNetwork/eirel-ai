from __future__ import annotations

import asyncio
import logging
import os

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response
from pydantic import BaseModel, Field

from validation.validator.engine import run_distributed_benchmarks, run_validator_loop, run_weight_setting_loop
from validation.validator import metrics as _metrics

logger = logging.getLogger(__name__)


class DistributedBenchmarkRequest(BaseModel):
    run_id: str | None = None
    family_id: str
    batch_size: int = Field(default=1, ge=1, le=40)
    max_parallel: int = Field(default=2, ge=1, le=20)
    rubric_version: str = "family_rubric_v2"
    judge_model: str = Field(default_factory=lambda: os.getenv("EIREL_EVAL_JUDGE_MODEL", "local-rubric-judge"))


_AUTO_LOOP_ENABLED = os.getenv("EIREL_VALIDATOR_AUTO_LOOP", "true").lower() not in ("0", "false", "no")


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# Undo bittensor's ``LoggingMachine.before_enable_default`` side effect,
# which blanket-sets every non-bittensor logger to CRITICAL.  Without
# this the run_weight_setting_loop INFO/WARNING lines from engine.py
# are swallowed silently, leaving operators with no visibility into
# whether ``set_weights`` succeeded or got rate-limited.
#
# bittensor sets *child* loggers directly (not the parent), so setting
# ``validation`` to INFO wouldn't propagate — we have to walk every
# existing logger under our namespaces and reset it to NOTSET so it
# inherits from the root logger again.
_EIREL_NAMESPACES = ("validation.", "shared.", "eirel.", "control_plane.", "infra.")
for _name in list(logging.root.manager.loggerDict.keys()):
    if any(_name == ns[:-1] or _name.startswith(ns) for ns in _EIREL_NAMESPACES):
        logging.getLogger(_name).setLevel(logging.NOTSET)
# Force the root to INFO in case basicConfig was a no-op (it's a no-op
# when uvicorn preloads handlers before app import).
logging.getLogger().setLevel(logging.INFO)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Start evaluation and weight-setting background loops on startup."""
    tasks: list[asyncio.Task] = []
    if _AUTO_LOOP_ENABLED:
        logger.info("validator auto-loop enabled — starting evaluation + weight-setting loops")
        tasks.append(asyncio.create_task(run_validator_loop()))
        tasks.append(asyncio.create_task(run_weight_setting_loop()))
    else:
        logger.info("validator auto-loop disabled (set EIREL_VALIDATOR_AUTO_LOOP=true to enable)")
    yield
    for task in tasks:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=10.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


app = FastAPI(title="validator-engine", lifespan=_lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    stub = b"# TYPE eirel_validator_engine_up gauge\neirel_validator_engine_up 1\n"
    return Response(
        content=stub + generate_latest(_metrics.registry),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/v1/distributed-benchmark-runs")
async def distributed_benchmark_runs(payload: DistributedBenchmarkRequest) -> dict:
    return await run_distributed_benchmarks(
        run_id=payload.run_id,
        family_id=payload.family_id,
        batch_size=payload.batch_size,
        max_parallel=payload.max_parallel,
        rubric_version=payload.rubric_version,
        judge_model=payload.judge_model,
    )


def main() -> None:
    uvicorn.run("validation.validator.main:app", host="0.0.0.0", port=8010, reload=False)
