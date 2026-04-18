"""Orchestrator service — FastAPI entry point.

The orchestrator is the central brain of the EIREL subnet. It receives
requests from the consumer API (conversation gateway) and coordinates
platform tools and specialist families to produce responses.

Replaces: api_gateway, classifier_service, dag_executor
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response
from pydantic import BaseModel, Field

from shared.common.request_context import RequestIdMiddleware
from shared.common.tracing import init_tracing, get_tracer
from orchestration.orchestrator.orchestrator import Orchestrator

init_tracing("orchestrator")
_tracer = get_tracer(__name__)


class OrchestratorRequest(BaseModel):
    prompt: str
    user_id: str = "anonymous"
    session_id: str | None = None
    context_history: list[dict[str, Any]] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = logging.getLogger(__name__)
    logger.info("orchestrator starting up")
    app.state.orchestrator = Orchestrator()
    yield
    logger.info("orchestrator shutting down")


app = FastAPI(title="eirel-orchestrator", lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    checks: dict[str, str] = {}
    try:
        orchestrator = request.app.state.orchestrator
        tool_count = len(orchestrator.available_tools())
        checks["tools_registry"] = f"ok ({tool_count} tools)"
    except Exception as exc:
        checks["tools_registry"] = f"degraded: {exc}"
    overall = "ok" if all("ok" in v for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}


@app.get("/metrics")
async def metrics(request: Request) -> Response:
    return Response(
        content="# TYPE eirel_orchestrator_up gauge\neirel_orchestrator_up 1\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/v1/tools")
async def list_tools(request: Request) -> dict[str, Any]:
    """List available platform tools."""
    orchestrator: Orchestrator = request.app.state.orchestrator
    return {
        "tools": orchestrator.tool_schemas(),
        "count": len(orchestrator.available_tools()),
    }


@app.post("/v1/orchestrate")
async def orchestrate(payload: OrchestratorRequest, request: Request):
    """Main orchestration endpoint.

    Receives a chat request, routes it through family selection →
    composition planning → execution, and returns the assembled response.
    """
    orchestrator: Orchestrator = request.app.state.orchestrator

    with _tracer.start_as_current_span("orchestrate") as span:
        span.set_attribute("orchestrator.user_id", payload.user_id)
        span.set_attribute("orchestrator.prompt_length", len(payload.prompt))

        result = await orchestrator.handle_request(
            prompt=payload.prompt,
            user_id=payload.user_id,
            session_id=payload.session_id,
            context_history=payload.context_history,
            constraints=payload.constraints,
        )

        status = result.get("status", "completed")
        span.set_attribute("orchestrator.status", status)
        span.set_attribute("orchestrator.route_type", result.get("route_type", ""))

        http_status = 200 if status in ("completed", "partial") else 500
        return JSONResponse(status_code=http_status, content=result)


def main() -> None:
    uvicorn.run("orchestration.orchestrator.main:app", host="0.0.0.0", port=8050, reload=False)
