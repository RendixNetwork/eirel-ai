from __future__ import annotations

"""Tests for consumer_api — FastAPI endpoints and route_chat_request."""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


# ===================================================================
# Endpoint tests
# ===================================================================

async def test_healthz(consumer_env):
    async for client in _make_client():
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


async def test_chat_without_api_key(consumer_env):
    async for client in _make_client():
        resp = await client.post("/v1/chat", json={"prompt": "hello"})
        assert resp.status_code == 401


async def test_chat_with_wrong_api_key(consumer_env):
    async for client in _make_client():
        resp = await client.post(
            "/v1/chat",
            json={"prompt": "hello"},
            headers={"X-API-Key": "bad-key"},
        )
        assert resp.status_code == 401


async def test_chat_happy_path(consumer_env, monkeypatch):
    async def _fake_route(**kwargs):
        return 200, {"status": "completed", "response": {"text": "hi"}}

    monkeypatch.setattr("orchestration.consumer_api.main.route_chat_request", _fake_route)

    async for client in _make_client():
        resp = await client.post(
            "/v1/chat",
            json={"prompt": "hello", "user_id": "u1"},
            headers={"X-API-Key": "test-key-1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"


async def test_chat_returns_error_on_orchestrator_failure(consumer_env, monkeypatch):
    async def _fake_route(**kwargs):
        return 500, {"status": "failed", "error": "boom"}

    monkeypatch.setattr("orchestration.consumer_api.main.route_chat_request", _fake_route)

    async for client in _make_client():
        resp = await client.post(
            "/v1/chat",
            json={"prompt": "hello"},
            headers={"X-API-Key": "test-key-1"},
        )
        assert resp.status_code == 500


async def test_rate_limiting(consumer_env, monkeypatch):
    monkeypatch.setenv("CONSUMER_RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("CONSUMER_RATE_LIMIT_WINDOW_SECONDS", "60")
    from shared.common.config import reset_settings
    reset_settings()

    async def _fake_route(**kwargs):
        return 200, {"status": "completed"}

    monkeypatch.setattr("orchestration.consumer_api.main.route_chat_request", _fake_route)

    async for client in _make_client():
        for _ in range(2):
            resp = await client.post(
                "/v1/chat",
                json={"prompt": "hi"},
                headers={"X-API-Key": "test-key-1"},
            )
            assert resp.status_code == 200

        resp = await client.post(
            "/v1/chat",
            json={"prompt": "hi"},
            headers={"X-API-Key": "test-key-1"},
        )
        assert resp.status_code == 429


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


# ===================================================================
# Streaming chat tests (POST /v1/chat/stream → SSE)
# ===================================================================

def _parse_sse(body: bytes) -> list[tuple[str, dict]]:
    """Split SSE body into (event_name, data_dict) pairs."""
    import json as _json
    events: list[tuple[str, dict]] = []
    for frame in body.decode("utf-8").split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        event_name = ""
        data_str = ""
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:"):].strip()
        if event_name and data_str:
            events.append((event_name, _json.loads(data_str)))
    return events


async def test_chat_stream_proxies_ndjson_as_sse(consumer_env, monkeypatch):
    """Happy path: serving miner returns NDJSON stream → consumer-api re-emits as SSE."""
    import json as _json

    async def _fake_resolve(family_id):
        return {"endpoint": "http://miner.local", "hotkey": "hk"}

    monkeypatch.setattr(
        "orchestration.consumer_api.chat._resolve_serving_miner", _fake_resolve,
    )

    ndjson_body = (
        _json.dumps({"event": "delta", "text": "Hello "}) + "\n"
        + _json.dumps({"event": "delta", "text": "world"}) + "\n"
        + _json.dumps({
            "event": "done",
            "output": {"answer": "Hello world"},
            "citations": [],
            "tool_calls": [],
            "status": "completed",
        }) + "\n"
    ).encode("utf-8")

    class _Transport(httpx.AsyncBaseTransport):
        def __init__(self, body: bytes) -> None:
            self.body = body
            self.calls: list[str] = []

        async def handle_async_request(self, request):
            self.calls.append(request.url.path)
            return httpx.Response(
                200, content=self.body,
                headers={"content-type": "application/x-ndjson"},
            )

    transport = _Transport(ndjson_body)

    import orchestration.consumer_api.chat as chat_mod
    original = chat_mod.httpx.AsyncClient

    def _patched(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(chat_mod.httpx, "AsyncClient", _patched)

    async for client in _make_client():
        resp = await client.post(
            "/v1/chat/stream",
            json={"prompt": "hi", "user_id": "u1"},
            headers={"X-API-Key": "test-key-1"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(resp.content)
        names = [e[0] for e in events]
        assert names[0] == "started"
        assert "delta" in names
        assert names[-1] == "done"

        deltas = [e[1].get("text", "") for e in events if e[0] == "delta"]
        assert "".join(deltas) == "Hello world"

        # Stream URL was hit (not the unary fallback).
        assert any(p.endswith("/v1/agent/infer/stream") for p in transport.calls)


async def test_chat_stream_falls_back_to_unary_on_404(consumer_env, monkeypatch):
    """Older miners on eirel SDK <0.2.3 lack the stream route — consumer-api
    falls back to the unary endpoint and emits the answer as one delta+done
    so the client UX is unchanged."""

    async def _fake_resolve(family_id):
        return {"endpoint": "http://miner.local", "hotkey": "hk"}

    monkeypatch.setattr(
        "orchestration.consumer_api.chat._resolve_serving_miner", _fake_resolve,
    )

    class _Transport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def handle_async_request(self, request):
            self.calls.append(request.url.path)
            if request.url.path.endswith("/stream"):
                return httpx.Response(404, text="not found")
            return httpx.Response(
                200, json={
                    "status": "completed",
                    "output": {"answer": "legacy reply"},
                    "citations": [],
                    "tool_calls": [],
                },
            )

    transport = _Transport()

    import orchestration.consumer_api.chat as chat_mod
    original = chat_mod.httpx.AsyncClient

    def _patched(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(chat_mod.httpx, "AsyncClient", _patched)

    async for client in _make_client():
        resp = await client.post(
            "/v1/chat/stream",
            json={"prompt": "hi"},
            headers={"X-API-Key": "test-key-1"},
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.content)
        names = [e[0] for e in events]
        assert names == ["started", "delta", "done"]
        delta_text = events[1][1].get("text")
        assert delta_text == "legacy reply"
        # Hit stream first (404), then unary.
        assert transport.calls[0].endswith("/v1/agent/infer/stream")
        assert transport.calls[1].endswith("/v1/agent/infer")


async def test_chat_stream_emits_error_when_no_serving_miner(
    consumer_env, monkeypatch,
):
    async def _fake_resolve(family_id):
        return None

    monkeypatch.setattr(
        "orchestration.consumer_api.chat._resolve_serving_miner", _fake_resolve,
    )

    async for client in _make_client():
        resp = await client.post(
            "/v1/chat/stream",
            json={"prompt": "hi"},
            headers={"X-API-Key": "test-key-1"},
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.content)
        names = [e[0] for e in events]
        assert "error" in names
        err = next(e[1] for e in events if e[0] == "error")
        assert "no serving miner" in err["message"]


async def test_chat_stream_requires_api_key(consumer_env):
    async for client in _make_client():
        resp = await client.post("/v1/chat/stream", json={"prompt": "hi"})
        assert resp.status_code == 401


# ===================================================================
# route_chat_request unit tests
# ===================================================================

async def test_route_chat_request_success(consumer_env, monkeypatch):
    orchestrator_response = {"status": "completed", "response": {"text": "answer"}}

    async def _fake_post(self, url, **kwargs):
        return httpx.Response(
            200,
            json=orchestrator_response,
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

    from orchestration.consumer_api.chat import route_chat_request
    status, body = await route_chat_request(prompt="test question")
    assert status == 200
    assert body["status"] == "completed"


async def test_traffic_logging_records_entry(consumer_env, monkeypatch):
    monkeypatch.setenv("EIREL_TRAFFIC_LOGGING_ENABLED", "1")

    async def _fake_post(self, url, **kwargs):
        return httpx.Response(
            200,
            json={"status": "completed"},
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

    import orchestration.consumer_api.chat as chat_mod
    chat_mod._TRAFFIC_LOGGING_ENABLED = True
    chat_mod._traffic_log.clear()
    try:
        await chat_mod.route_chat_request(prompt="test")
        assert len(chat_mod._traffic_log) >= 1
        assert chat_mod._traffic_log[-1]["status"] == "completed"
    finally:
        chat_mod._TRAFFIC_LOGGING_ENABLED = False
        chat_mod._traffic_log.clear()


async def test_traffic_log_eviction(consumer_env, monkeypatch):
    import orchestration.consumer_api.chat as chat_mod
    chat_mod._TRAFFIC_LOGGING_ENABLED = True
    chat_mod._TRAFFIC_LOG_MAX_SIZE = 10
    chat_mod._traffic_log.clear()
    try:
        for i in range(15):
            await chat_mod._record_traffic(
                prompt=f"p{i}", user_id="u", session_id=None,
                status_code=200, latency_ms=1.0,
            )
        assert len(chat_mod._traffic_log) <= 10
    finally:
        chat_mod._TRAFFIC_LOGGING_ENABLED = False
        chat_mod._traffic_log.clear()
        chat_mod._TRAFFIC_LOG_MAX_SIZE = 10000


async def test_traffic_logging_disabled_no_records(consumer_env, monkeypatch):
    import orchestration.consumer_api.chat as chat_mod
    chat_mod._TRAFFIC_LOGGING_ENABLED = False
    chat_mod._traffic_log.clear()
    await chat_mod._record_traffic(
        prompt="test", user_id="u", session_id=None,
        status_code=200, latency_ms=1.0,
    )
    assert len(chat_mod._traffic_log) == 0
