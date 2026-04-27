from __future__ import annotations

import json

import httpx
import pytest

from shared.benchmark._invocation import _invoke_task
from shared.core.evaluation_models import MinerBenchmarkTarget


class _Task:
    def __init__(self) -> None:
        self.task_id = "t-1"
        self.family_id = "general_chat"
        self.prompt = "hello"
        self.expected_output = {}
        self.inputs = {"mode": "instant"}
        self.metadata = {}


def _miner() -> MinerBenchmarkTarget:
    return MinerBenchmarkTarget(
        hotkey="5X" * 32, endpoint="http://miner.local", stake=0, metadata={},
    )


def _ndjson_body(*chunks: dict) -> bytes:
    return ("\n".join(json.dumps(c) for c in chunks) + "\n").encode("utf-8")


class _RoutedTransport(httpx.AsyncBaseTransport):
    """Dispatches by URL path, lets each test script its own responses."""

    def __init__(self, handlers: dict[str, list[httpx.Response]]) -> None:
        self._handlers = {p: list(rs) for p, rs in handlers.items()}
        self.call_count: dict[str, int] = {p: 0 for p in handlers}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path not in self._handlers:
            raise RuntimeError(f"no handler scripted for {path}")
        self.call_count[path] += 1
        bucket = self._handlers[path]
        if not bucket:
            raise RuntimeError(f"no more responses for {path}")
        return bucket.pop(0)


@pytest.fixture
def _patch_client(monkeypatch):
    import shared.benchmark._invocation as mod
    original = mod.httpx.AsyncClient
    transport_holder: dict[str, httpx.AsyncBaseTransport] = {}

    def _install(transport: httpx.AsyncBaseTransport) -> None:
        transport_holder["t"] = transport

    def _patched(*args, **kwargs):
        kwargs["transport"] = transport_holder["t"]
        return original(*args, **kwargs)

    monkeypatch.setattr(mod.httpx, "AsyncClient", _patched)
    monkeypatch.setattr(mod, "_RETRY_BACKOFF_SECONDS", 0.0)
    return _install


async def test_streaming_path_assembles_answer(_patch_client):
    body = _ndjson_body(
        {"event": "delta", "text": "Hello "},
        {"event": "delta", "text": "world"},
        {
            "event": "done",
            "output": {"answer": "Hello world"},
            "citations": ["https://example.com/x"],
            "tool_calls": [],
            "status": "completed",
            "metadata": {},
        },
    )
    transport = _RoutedTransport({
        "/v1/agent/infer/stream": [
            httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"}),
        ],
    })
    _patch_client(transport)

    run = await _invoke_task(miner=_miner(), task=_Task(), timeout_seconds=5.0)

    assert run.status == "completed"
    assert run.metadata.get("streamed") is True
    # Total completion latency recorded; TTFB intentionally NOT measured.
    assert run.metadata.get("latency_seconds") is not None
    assert "first_token_seconds" not in run.metadata
    # Answer assembled from deltas + done.output.
    assert run.response.get("output", {}).get("answer") == "Hello world"
    assert run.response.get("citations") == ["https://example.com/x"]
    assert transport.call_count["/v1/agent/infer/stream"] == 1


async def test_falls_back_to_unary_on_404(_patch_client):
    """Older miners on eirel SDK <0.2.3 don't have the stream endpoint."""
    transport = _RoutedTransport({
        "/v1/agent/infer/stream": [httpx.Response(404, text="not found")],
        "/v1/agent/infer": [
            httpx.Response(
                200, json={
                    "status": "completed",
                    "output": {"answer": "legacy reply"},
                    "citations": [], "tool_calls": [],
                },
            ),
        ],
    })
    _patch_client(transport)

    run = await _invoke_task(miner=_miner(), task=_Task(), timeout_seconds=5.0)

    assert run.status == "completed"
    assert run.metadata.get("streamed") is False
    assert run.response.get("output", {}).get("answer") == "legacy reply"
    assert transport.call_count["/v1/agent/infer/stream"] == 1
    assert transport.call_count["/v1/agent/infer"] == 1


async def test_streaming_502_retries_streaming_path(_patch_client):
    """A transient 502 on the stream URL retries on the same URL — we
    don't fall back to unary unless we see a 404 (route-missing)."""
    good_body = _ndjson_body(
        {"event": "delta", "text": "ok"},
        {"event": "done", "output": {"answer": "ok"}, "status": "completed"},
    )
    transport = _RoutedTransport({
        "/v1/agent/infer/stream": [
            httpx.Response(502, text="bad gateway"),
            httpx.Response(200, content=good_body, headers={"content-type": "application/x-ndjson"}),
        ],
    })
    _patch_client(transport)

    run = await _invoke_task(miner=_miner(), task=_Task(), timeout_seconds=5.0)

    assert run.status == "completed"
    assert run.metadata.get("streamed") is True
    assert transport.call_count["/v1/agent/infer/stream"] == 2


async def test_done_chunk_with_failed_status_propagates_error(_patch_client):
    body = _ndjson_body(
        {"event": "done", "status": "failed", "error": "downstream LLM 500"},
    )
    transport = _RoutedTransport({
        "/v1/agent/infer/stream": [
            httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"}),
        ],
    })
    _patch_client(transport)

    run = await _invoke_task(miner=_miner(), task=_Task(), timeout_seconds=5.0)

    # Failed status from the miner's done chunk must propagate to
    # BenchmarkTaskRun so the validator's _judge_miner gates this as
    # verdict=error rather than scoring an empty completed response.
    assert run.status == "failed"
    assert run.error == "downstream LLM 500"
    assert run.response.get("status") == "failed"
