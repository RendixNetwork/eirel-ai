from __future__ import annotations

"""Tests for consumer_api — non-legacy endpoints (healthz, tasks/sessions, metrics)."""

import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def consumer_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'consumer.db'}")
    monkeypatch.setenv("CONSUMER_API_KEYS", "test-key-1,test-key-2")
    monkeypatch.setenv("CONSUMER_RATE_LIMIT_REQUESTS", "100")
    monkeypatch.setenv("CONSUMER_RATE_LIMIT_WINDOW_SECONDS", "60")
    monkeypatch.setenv("ORCHESTRATOR_URL", "http://fake-orchestrator:8050")
    monkeypatch.setenv("USE_REDIS_POOL", "0")
    monkeypatch.setenv("METAGRAPH_SYNC_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("RESULT_AGGREGATION_INTERVAL_SECONDS", "3600")
    from shared.common.config import reset_settings
    reset_settings()
    yield
    reset_settings()


async def _make_client():
    from orchestration.consumer_api.main import app
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def test_healthz(consumer_env):
    async for client in _make_client():
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


async def test_get_task_not_found(consumer_env):
    async for client in _make_client():
        resp = await client.get(
            "/v1/tasks/nonexistent",
            headers={"X-API-Key": "test-key-1"},
        )
        assert resp.status_code == 404


async def test_get_session_not_found(consumer_env):
    async for client in _make_client():
        resp = await client.get(
            "/v1/sessions/nonexistent",
            headers={"X-API-Key": "test-key-1"},
        )
        assert resp.status_code == 404


async def test_metrics_endpoint(consumer_env):
    async for client in _make_client():
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        assert "eirel_consumer_api_up" in resp.text
