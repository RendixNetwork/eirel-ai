from __future__ import annotations

import httpx
import pytest

from shared.benchmark._invocation import _invoke_task
from shared.core.evaluation_models import MinerBenchmarkTarget


class _Task:
    def __init__(self, task_id: str = "t-1", family_id: str = "general_chat") -> None:
        self.task_id = task_id
        self.family_id = family_id
        self.prompt = "hello"
        self.expected_output = {}
        self.inputs = {"mode": "instant"}
        self.metadata = {}


class _ScriptedTransport(httpx.AsyncBaseTransport):
    """Returns a scripted sequence of responses, one per request."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        if not self._responses:
            raise RuntimeError("no more scripted responses")
        return self._responses.pop(0)


def _miner() -> MinerBenchmarkTarget:
    return MinerBenchmarkTarget(
        hotkey="5X" * 32,
        endpoint="http://miner.local",
        stake=0,
        metadata={},
    )


@pytest.fixture
def _patch_client(monkeypatch):
    """Force httpx.AsyncClient to use whatever transport we register."""
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


async def test_retry_on_502_then_success(_patch_client):
    transport = _ScriptedTransport([
        httpx.Response(502, text="bad gateway"),
        httpx.Response(200, json={"output": {"content": "hi"}, "status": "completed"}),
    ])
    _patch_client(transport)

    run = await _invoke_task(miner=_miner(), task=_Task(), timeout_seconds=5.0)

    assert transport.call_count == 2
    assert run.status == "completed"
    assert run.error is None


async def test_fails_after_second_502(_patch_client):
    transport = _ScriptedTransport([
        httpx.Response(502, text="bad gateway"),
        httpx.Response(502, text="still bad"),
    ])
    _patch_client(transport)

    run = await _invoke_task(miner=_miner(), task=_Task(), timeout_seconds=5.0)

    assert transport.call_count == 2
    assert run.status == "failed"
    assert "502" in (run.error or "")


async def test_does_not_retry_on_400(_patch_client):
    """Bad request from the miner means the task payload was malformed —
    retrying would just burn time with the same result."""
    transport = _ScriptedTransport([
        httpx.Response(400, text="bad request"),
    ])
    _patch_client(transport)

    run = await _invoke_task(miner=_miner(), task=_Task(), timeout_seconds=5.0)

    assert transport.call_count == 1
    assert run.status == "failed"


async def test_retries_on_503(_patch_client):
    transport = _ScriptedTransport([
        httpx.Response(503, text="unavailable"),
        httpx.Response(200, json={"output": {}, "status": "completed"}),
    ])
    _patch_client(transport)

    run = await _invoke_task(miner=_miner(), task=_Task(), timeout_seconds=5.0)

    assert transport.call_count == 2
    assert run.status == "completed"


async def test_retries_on_504(_patch_client):
    transport = _ScriptedTransport([
        httpx.Response(504, text="gateway timeout"),
        httpx.Response(200, json={"output": {}, "status": "completed"}),
    ])
    _patch_client(transport)

    run = await _invoke_task(miner=_miner(), task=_Task(), timeout_seconds=5.0)

    assert transport.call_count == 2
    assert run.status == "completed"
