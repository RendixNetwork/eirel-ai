from __future__ import annotations

"""Tests for validator_engine FastAPI app and distributed benchmark runner."""

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeTransport(httpx.AsyncBaseTransport):
    """Route fake HTTP responses by URL path."""

    def __init__(self, routes: dict[str, tuple[int, dict[str, Any]]]) -> None:
        self._routes = routes

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        for pattern, (status_code, body) in self._routes.items():
            if pattern in path:
                return httpx.Response(
                    status_code=status_code,
                    json=body,
                    request=request,
                )
        return httpx.Response(status_code=404, json={"detail": "not found"}, request=request)


# ===================================================================
# FastAPI app tests
# ===================================================================

async def test_healthz(monkeypatch):
    monkeypatch.setenv("EIREL_VALIDATOR_AUTO_LOOP", "false")
    # Force re-import to pick up env
    import importlib
    import validation.validator.main as ve_main
    importlib.reload(ve_main)
    try:
        app = ve_main.app
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/healthz")
                assert resp.status_code == 200
                assert resp.json()["status"] == "ok"
    finally:
        importlib.reload(ve_main)


async def test_metrics(monkeypatch):
    monkeypatch.setenv("EIREL_VALIDATOR_AUTO_LOOP", "false")
    import importlib
    import validation.validator.main as ve_main
    importlib.reload(ve_main)
    try:
        app = ve_main.app
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/metrics")
                assert resp.status_code == 200
                assert "eirel_validator_engine_up" in resp.text
    finally:
        importlib.reload(ve_main)


# ===================================================================
# run_distributed_benchmarks tests
# ===================================================================

async def test_benchmarks_no_tasks_available(monkeypatch):
    """When owner-api returns empty task list, total_claimed should be 0."""
    monkeypatch.setenv("EIREL_VALIDATOR_MNEMONIC", "abandon " * 11 + "about")
    monkeypatch.setenv("OWNER_API_URL", "http://fake-owner:8000")

    claim_response = {"tasks": []}

    async def _fake_post(self, *args, **kwargs):
        url = str(args[0]) if args else str(kwargs.get("url", ""))
        return httpx.Response(200, json=claim_response, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

    from validation.validator.engine import run_distributed_benchmarks
    result = await run_distributed_benchmarks(family_id="general_chat")
    assert result["total_claimed"] == 0
    assert result["total_submitted"] == 0


async def test_benchmarks_claim_failure(monkeypatch):
    """When claim returns 500, exception should propagate."""
    monkeypatch.setenv("EIREL_VALIDATOR_MNEMONIC", "abandon " * 11 + "about")
    monkeypatch.setenv("OWNER_API_URL", "http://fake-owner:8000")

    async def _fake_post(self, *args, **kwargs):
        url = str(args[0]) if args else str(kwargs.get("url", ""))
        return httpx.Response(500, json={"error": "server error"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

    from validation.validator.engine import run_distributed_benchmarks
    with pytest.raises(httpx.HTTPStatusError):
        await run_distributed_benchmarks(family_id="general_chat")
