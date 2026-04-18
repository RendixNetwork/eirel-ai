from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import PlainTextResponse

from shared.common.redis_job_ledger import (
    JobLedger,
    JobUsageRecord,
    create_job_ledger,
)
from tool_platforms._charge_tool import charge_tool_cost
from tool_platforms.sandbox_tool_service.backends import (
    SandboxBackend,
    SubprocessBackend,
)
from tool_platforms.sandbox_tool_service.models import (
    SandboxExecuteRequest,
    SandboxExecuteResponse,
)


def generate_job_token(master_token: str, job_id: str) -> str:
    return hmac.new(
        master_token.encode(), job_id.encode(), "sha256"
    ).hexdigest()


def verify_job_token(master_token: str, job_id: str, token: str) -> bool:
    expected = generate_job_token(master_token, job_id)
    return hmac.compare_digest(expected, token)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def create_app(
    *,
    backend: SandboxBackend | None = None,
    ledger: JobLedger | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.auth_token = os.getenv("EIREL_SANDBOX_TOOL_API_TOKEN", "")
        app.state.default_max_requests = int(
            os.getenv("EIREL_SANDBOX_TOOL_DEFAULT_MAX_REQUESTS", "12")
        )
        app.state.default_timeout_seconds = float(
            os.getenv("EIREL_SANDBOX_TOOL_DEFAULT_TIMEOUT_SECONDS", "5.0")
        )
        app.state.default_memory_mb = int(
            os.getenv("EIREL_SANDBOX_TOOL_DEFAULT_MEMORY_MB", "128")
        )
        app.state.max_code_bytes = int(
            os.getenv("EIREL_SANDBOX_TOOL_MAX_CODE_BYTES", str(64 * 1024))
        )
        app.state.backend = backend or SubprocessBackend(
            default_timeout=app.state.default_timeout_seconds,
            default_memory_mb=app.state.default_memory_mb,
        )
        if ledger is not None:
            app.state.job_ledger = ledger
        else:
            app.state.job_ledger = create_job_ledger(os.getenv("REDIS_URL", ""))
        app.state.metrics = {
            "requests_total": 0,
            "quota_rejections_total": 0,
            "execute_requests_total": 0,
            "execute_failures_total": 0,
            "execute_timeouts_total": 0,
        }
        try:
            yield
        finally:
            await app.state.job_ledger.close()

    app = FastAPI(
        title="sandbox-tool-service",
        version="0.1.0",
        lifespan=lifespan,
    )

    async def require_auth(
        authorization: str | None = Header(default=None),
        x_eirel_job_id: str | None = Header(default=None),
    ) -> None:
        auth_token: str = app.state.auth_token
        if not auth_token:
            return
        if authorization == f"Bearer {auth_token}":
            return
        if x_eirel_job_id and authorization:
            bearer = authorization.removeprefix("Bearer ").strip()
            if verify_job_token(auth_token, x_eirel_job_id.strip(), bearer):
                return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid sandbox tool auth token",
        )

    async def require_job(
        x_eirel_job_id: str | None = Header(default=None),
        x_eirel_max_requests: str | None = Header(default=None),
    ) -> tuple[str, int]:
        job_id = (x_eirel_job_id or "").strip()
        if not job_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="missing X-Eirel-Job-Id",
            )
        try:
            max_requests = int(x_eirel_max_requests or app.state.default_max_requests)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="invalid X-Eirel-Max-Requests",
            ) from exc
        return job_id, max(1, max_requests)

    async def check_budget(
        *, job_id: str, max_requests: int, tool_name: str
    ) -> JobUsageRecord:
        ledger: JobLedger = app.state.job_ledger
        usage = await ledger.get_or_create(job_id)
        if usage.request_count >= max_requests:
            app.state.metrics["quota_rejections_total"] += 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="sandbox tool request budget exceeded",
            )
        usage.request_count += 1
        usage.tool_counts[tool_name] = usage.tool_counts.get(tool_name, 0) + 1
        app.state.metrics["requests_total"] += 1
        await ledger.save(job_id, usage)
        return usage

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> PlainTextResponse:
        m = app.state.metrics
        lines = [
            "# HELP eirel_sandbox_tool_requests_total Total successful sandbox tool requests.",
            "# TYPE eirel_sandbox_tool_requests_total counter",
            f"eirel_sandbox_tool_requests_total {m['requests_total']}",
            "# HELP eirel_sandbox_tool_quota_rejections_total Total sandbox tool quota rejections.",
            "# TYPE eirel_sandbox_tool_quota_rejections_total counter",
            f"eirel_sandbox_tool_quota_rejections_total {m['quota_rejections_total']}",
            "# HELP eirel_sandbox_execute_timeouts_total Total sandbox executions that hit the wall clock limit.",
            "# TYPE eirel_sandbox_execute_timeouts_total counter",
            f"eirel_sandbox_execute_timeouts_total {m['execute_timeouts_total']}",
            "# HELP eirel_sandbox_execute_failures_total Total sandbox executions that exited non-zero.",
            "# TYPE eirel_sandbox_execute_failures_total counter",
            f"eirel_sandbox_execute_failures_total {m['execute_failures_total']}",
        ]
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.get("/v1/jobs/{job_id}/usage")
    async def job_usage(
        job_id: str, _: None = Depends(require_auth)
    ) -> dict[str, Any]:
        ledger: JobLedger = app.state.job_ledger
        usage = await ledger.get_usage(job_id)
        if usage is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="job usage not found",
            )
        return {
            "job_id": job_id,
            "retrieval_ledger_id": usage.ledger_id,
            "request_count": usage.request_count,
            "tool_counts": dict(usage.tool_counts),
        }

    @app.post("/v1/execute", response_model=SandboxExecuteResponse)
    async def execute(
        payload: SandboxExecuteRequest,
        _: None = Depends(require_auth),
        job: tuple[str, int] = Depends(require_job),
    ) -> SandboxExecuteResponse:
        if len(payload.code.encode("utf-8")) > app.state.max_code_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"code exceeds {app.state.max_code_bytes} bytes",
            )
        job_id, max_requests = job
        usage = await check_budget(
            job_id=job_id,
            max_requests=max_requests,
            tool_name="sandbox_execute",
        )
        app.state.metrics["execute_requests_total"] += 1
        retrieved_at = _utcnow()

        result = await app.state.backend.execute(
            code=payload.code,
            timeout_seconds=payload.timeout_seconds,
            memory_mb=payload.memory_mb,
        )

        if result.timed_out:
            app.state.metrics["execute_timeouts_total"] += 1
        elif result.exit_code != 0:
            app.state.metrics["execute_failures_total"] += 1

        ledger: JobLedger = app.state.job_ledger
        await ledger.save(job_id, usage)

        per_exec_cost = float(os.getenv("EIREL_SANDBOX_PER_EXECUTION_COST_USD", "0.0"))
        await charge_tool_cost(
            job_id=job_id, tool_name="sandbox", amount_usd=per_exec_cost,
        )

        return SandboxExecuteResponse(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
            truncated=result.truncated,
            retrieved_at=retrieved_at,
            retrieval_ledger_id=usage.ledger_id,
            metadata={
                "backend": type(app.state.backend).__name__,
                "timed_out": result.timed_out,
            },
        )

    return app


app = create_app()


def main() -> None:
    port = int(os.getenv("EIREL_SANDBOX_TOOL_PORT", "8091"))
    uvicorn.run(
        "tool_platforms.sandbox_tool_service.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
