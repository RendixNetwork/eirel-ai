from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from tool_platforms.sandbox_tool_service.app import create_app, generate_job_token
from tool_platforms.sandbox_tool_service.backends import (
    ExecutionResult,
    SandboxBackend,
    SubprocessBackend,
)


# -- SubprocessBackend direct tests ----------------------------------------


async def test_subprocess_backend_happy_path_math():
    backend = SubprocessBackend()
    result = await backend.execute(
        code="print(2 + 2)",
        timeout_seconds=5.0,
        memory_mb=128,
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == "4"
    assert result.stderr == ""
    assert result.duration_ms > 0
    assert result.timed_out is False


async def test_subprocess_backend_multiline_script():
    code = """
import math
import statistics
values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(f"mean={statistics.mean(values)}")
print(f"stdev={round(statistics.stdev(values), 3)}")
print(f"sqrt2={round(math.sqrt(2), 6)}")
"""
    backend = SubprocessBackend()
    result = await backend.execute(code=code, timeout_seconds=5.0, memory_mb=128)
    assert result.exit_code == 0
    assert "mean=5.5" in result.stdout
    assert "stdev=3.028" in result.stdout
    assert "sqrt2=1.414214" in result.stdout


async def test_subprocess_backend_stderr_capture():
    code = 'import sys; print("warn", file=sys.stderr); print("out")'
    backend = SubprocessBackend()
    result = await backend.execute(code=code, timeout_seconds=5.0, memory_mb=128)
    assert result.exit_code == 0
    assert result.stdout.strip() == "out"
    assert result.stderr.strip() == "warn"


async def test_subprocess_backend_non_zero_exit():
    backend = SubprocessBackend()
    result = await backend.execute(
        code="import sys; sys.exit(7)",
        timeout_seconds=5.0,
        memory_mb=128,
    )
    assert result.exit_code == 7
    assert result.timed_out is False


async def test_subprocess_backend_runtime_error_captured():
    backend = SubprocessBackend()
    result = await backend.execute(
        code="raise ValueError('boom')",
        timeout_seconds=5.0,
        memory_mb=128,
    )
    assert result.exit_code != 0
    assert "ValueError" in result.stderr
    assert "boom" in result.stderr


async def test_subprocess_backend_timeout():
    backend = SubprocessBackend()
    # Spin forever; should be killed by wall-clock timeout.
    result = await backend.execute(
        code="while True: pass",
        timeout_seconds=1.0,
        memory_mb=128,
    )
    assert result.timed_out is True
    assert result.exit_code == -9
    # Duration should be close to the timeout (within reason for CI jitter).
    assert 500 < result.duration_ms < 3500


# -- Import blocking (defense in depth) ------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "socket",
        "urllib",
        "urllib.request",
        "http.client",
        "subprocess",
        "ctypes",
        "multiprocessing",
        "ssl",
    ],
)
async def test_subprocess_backend_blocks_dangerous_imports(module: str):
    backend = SubprocessBackend()
    code = f"import {module}; print('imported')"
    result = await backend.execute(code=code, timeout_seconds=5.0, memory_mb=128)
    assert result.exit_code != 0
    assert "is blocked" in result.stderr
    assert "imported" not in result.stdout


async def test_subprocess_backend_strips_os_system():
    backend = SubprocessBackend()
    code = "import os; os.system('echo pwned')"
    result = await backend.execute(code=code, timeout_seconds=5.0, memory_mb=128)
    # os.system was deleted in the preamble, so accessing it raises AttributeError.
    assert result.exit_code != 0
    assert "AttributeError" in result.stderr or "os" in result.stderr
    assert "pwned" not in result.stdout


async def test_subprocess_backend_allows_safe_stdlib():
    code = """
import json, re, datetime, collections, itertools, functools
data = {"x": 1, "y": [2, 3, 4]}
print(json.dumps(data))
print(re.sub(r"\\d", "X", "abc123"))
print(datetime.date(2026, 1, 1).isoformat())
print(list(itertools.islice(itertools.count(), 5)))
"""
    backend = SubprocessBackend()
    result = await backend.execute(code=code, timeout_seconds=5.0, memory_mb=128)
    assert result.exit_code == 0
    assert '"x": 1' in result.stdout
    assert "abcXXX" in result.stdout
    assert "2026-01-01" in result.stdout
    assert "[0, 1, 2, 3, 4]" in result.stdout


# -- Resource limits --------------------------------------------------------


async def test_subprocess_backend_memory_limit_kills_runaway():
    # Try to allocate 2 GB — should be killed by the 128 MB limit.
    code = "x = bytearray(2 * 1024 * 1024 * 1024)"
    backend = SubprocessBackend()
    result = await backend.execute(code=code, timeout_seconds=5.0, memory_mb=128)
    assert result.exit_code != 0
    # MemoryError or killed by RLIMIT_AS
    assert "MemoryError" in result.stderr or result.exit_code < 0


# -- HTTP service tests -----------------------------------------------------


class _FakeBackend:
    """In-process fake that returns canned results and records calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._next: ExecutionResult | None = None

    def set_result(self, result: ExecutionResult) -> None:
        self._next = result

    async def execute(
        self,
        *,
        code: str,
        timeout_seconds: float | None = None,
        memory_mb: int | None = None,
        session_id: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> ExecutionResult:
        self.calls.append({
            "code": code,
            "timeout_seconds": timeout_seconds,
            "memory_mb": memory_mb,
            "session_id": session_id,
            "attachments": attachments,
        })
        if self._next is not None:
            return self._next
        return ExecutionResult(
            stdout="4\n",
            stderr="",
            exit_code=0,
            duration_ms=12,
            truncated=False,
            timed_out=False,
        )


async def test_service_execute_happy_path(monkeypatch):
    monkeypatch.setenv("EIREL_SANDBOX_TOOL_API_TOKEN", "sbx-token")
    backend = _FakeBackend()
    app = create_app(backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/v1/execute",
                json={"code": "print(2 + 2)"},
                headers={
                    "Authorization": "Bearer sbx-token",
                    "X-Eirel-Job-Id": "job-1",
                    "X-Eirel-Max-Requests": "5",
                },
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["stdout"] == "4\n"
    assert body["exit_code"] == 0
    assert body["duration_ms"] == 12
    assert body["metadata"]["backend"] == "_FakeBackend"
    assert body["retrieval_ledger_id"] == "ledger:job-1"
    assert len(backend.calls) == 1
    assert backend.calls[0]["code"] == "print(2 + 2)"


async def test_service_enforces_auth(monkeypatch):
    monkeypatch.setenv("EIREL_SANDBOX_TOOL_API_TOKEN", "sbx-token")
    app = create_app(backend=_FakeBackend())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/v1/execute",
                json={"code": "print(1)"},
                headers={"X-Eirel-Job-Id": "job-1"},
            )
    assert resp.status_code == 401


async def test_service_per_job_token_auth(monkeypatch):
    monkeypatch.setenv("EIREL_SANDBOX_TOOL_API_TOKEN", "master-token")
    app = create_app(backend=_FakeBackend())
    job_token = generate_job_token("master-token", "job-scoped")
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/v1/execute",
                json={"code": "print(1)"},
                headers={
                    "Authorization": f"Bearer {job_token}",
                    "X-Eirel-Job-Id": "job-scoped",
                },
            )
    assert resp.status_code == 200


async def test_service_budget_enforcement():
    app = create_app(backend=_FakeBackend())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            headers = {
                "X-Eirel-Job-Id": "job-budget",
                "X-Eirel-Max-Requests": "2",
            }
            a = await client.post("/v1/execute", json={"code": "print(1)"}, headers=headers)
            b = await client.post("/v1/execute", json={"code": "print(2)"}, headers=headers)
            c = await client.post("/v1/execute", json={"code": "print(3)"}, headers=headers)
    assert a.status_code == 200
    assert b.status_code == 200
    assert c.status_code == 429
    assert "budget" in c.json()["detail"]


async def test_service_missing_job_id():
    app = create_app(backend=_FakeBackend())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post("/v1/execute", json={"code": "print(1)"})
    assert resp.status_code == 400


async def test_service_rejects_oversized_code(monkeypatch):
    monkeypatch.setenv("EIREL_SANDBOX_TOOL_MAX_CODE_BYTES", "100")
    app = create_app(backend=_FakeBackend())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/v1/execute",
                json={"code": "x = '" + "a" * 200 + "'"},
                headers={"X-Eirel-Job-Id": "job-big"},
            )
    assert resp.status_code == 413
    assert "exceeds" in resp.json()["detail"]


async def test_service_healthz_and_metrics():
    app = create_app(backend=_FakeBackend())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            health = await client.get("/healthz")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}

            await client.post(
                "/v1/execute",
                json={"code": "print(1)"},
                headers={"X-Eirel-Job-Id": "job-m"},
            )
            metrics = await client.get("/metrics")
    assert metrics.status_code == 200
    assert "eirel_sandbox_tool_requests_total 1" in metrics.text


async def test_service_timeout_metric_incremented():
    backend = _FakeBackend()
    backend.set_result(
        ExecutionResult(
            stdout="",
            stderr="",
            exit_code=-9,
            duration_ms=1000,
            truncated=False,
            timed_out=True,
        )
    )
    app = create_app(backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            await client.post(
                "/v1/execute",
                json={"code": "while True: pass"},
                headers={"X-Eirel-Job-Id": "job-t"},
            )
            metrics = await client.get("/metrics")
    assert "eirel_sandbox_execute_timeouts_total 1" in metrics.text


async def test_service_failure_metric_incremented():
    backend = _FakeBackend()
    backend.set_result(
        ExecutionResult(
            stdout="",
            stderr="ValueError",
            exit_code=1,
            duration_ms=10,
            truncated=False,
            timed_out=False,
        )
    )
    app = create_app(backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            await client.post(
                "/v1/execute",
                json={"code": "raise ValueError()"},
                headers={"X-Eirel-Job-Id": "job-f"},
            )
            metrics = await client.get("/metrics")
    assert "eirel_sandbox_execute_failures_total 1" in metrics.text
