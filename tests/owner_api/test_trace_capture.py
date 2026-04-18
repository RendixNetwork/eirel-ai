from __future__ import annotations

import httpx
import pytest

from control_plane.owner_api.middleware.trace_capture import (
    InMemoryTraceStore,
    ToolProxy,
    TraceStore,
    default_trace_store,
    reset_default_trace_store,
)
from shared.core.evaluation_models import TraceEntry


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_default_trace_store()
    yield
    reset_default_trace_store()


async def test_in_memory_store_appends_and_reads_by_conversation():
    store = InMemoryTraceStore()
    await store.append("c1", TraceEntry(tool_name="web_search", args={"q": "x"}))
    await store.append("c1", TraceEntry(tool_name="semantic_scholar", args={"q": "y"}))
    await store.append("c2", TraceEntry(tool_name="x_api", args={"q": "z"}))

    c1 = await store.get_trace("c1")
    c2 = await store.get_trace("c2")

    assert [e.tool_name for e in c1.entries] == ["web_search", "semantic_scholar"]
    assert [e.tool_name for e in c2.entries] == ["x_api"]
    assert c1.conversation_id == "c1"


async def test_in_memory_store_clear_removes_entries():
    store = InMemoryTraceStore()
    await store.append("c1", TraceEntry(tool_name="web_search", args={}))
    await store.clear("c1")
    trace = await store.get_trace("c1")
    assert trace.entries == []


async def test_tool_proxy_records_trace_entry_on_success():
    store = InMemoryTraceStore()

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"title": "hit"}]})

    transport = httpx.MockTransport(_handler)
    client = httpx.AsyncClient(transport=transport)
    proxy = ToolProxy(store=store, http_client=client)
    try:
        result = await proxy.proxy_call(
            conversation_id="c1",
            tool_name="web_search",
            target_url="http://web-search-tool-service:8085/v1/search",
            args={"query": "bananas"},
        )
    finally:
        await proxy.close()
        await client.aclose()

    assert result["status_code"] == 200
    assert result["payload"]["results"][0]["title"] == "hit"
    assert result["cost_usd"] == pytest.approx(0.001)

    trace = await store.get_trace("c1")
    assert len(trace.entries) == 1
    entry = trace.entries[0]
    assert entry.tool_name == "web_search"
    assert entry.args == {"query": "bananas"}
    assert entry.latency_ms >= 0
    assert entry.cost_usd == pytest.approx(0.001)
    assert entry.result_digest != ""
    # Body excerpt is captured lowercased for the grounding-overlap check.
    assert "hit" in entry.result_body_excerpt
    assert entry.result_body_excerpt == entry.result_body_excerpt.lower()


async def test_tool_proxy_body_excerpt_truncated_to_cap():
    store = InMemoryTraceStore()
    big_text = "A" * 10_000  # 10 KB, must be truncated to <= 2048 chars.

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"huge": big_text})

    transport = httpx.MockTransport(_handler)
    client = httpx.AsyncClient(transport=transport)
    proxy = ToolProxy(store=store, http_client=client)
    try:
        await proxy.proxy_call(
            conversation_id="c-huge",
            tool_name="web_search",
            target_url="http://web-search-tool-service:8085/v1/search",
            args={"query": "huge"},
        )
    finally:
        await proxy.close()
        await client.aclose()

    trace = await store.get_trace("c-huge")
    entry = trace.entries[0]
    assert 0 < len(entry.result_body_excerpt) <= 2048


async def test_tool_proxy_records_trace_entry_on_failure():
    store = InMemoryTraceStore()

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "upstream_down"})

    transport = httpx.MockTransport(_handler)
    client = httpx.AsyncClient(transport=transport)
    proxy = ToolProxy(store=store, http_client=client)
    try:
        result = await proxy.proxy_call(
            conversation_id="c-err",
            tool_name="x_api",
            target_url="http://x-tool-service:8086/v1/search",
            args={"query": "NVDA"},
        )
    finally:
        await proxy.close()
        await client.aclose()

    assert result["status_code"] == 503
    assert result["error"] is not None

    trace = await store.get_trace("c-err")
    assert len(trace.entries) == 1
    assert trace.entries[0].tool_name == "x_api"
    assert trace.entries[0].cost_usd == pytest.approx(0.050)


def test_default_trace_store_is_singleton():
    a = default_trace_store()
    b = default_trace_store()
    assert a is b


async def test_tool_proxy_injects_honeytoken_into_search_results():
    from shared.core.honeytokens import HONEYTOKEN_MARKER, generate_honeytoken_set

    store = InMemoryTraceStore()
    active = generate_honeytoken_set("run-proxy-inject", count=4)

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"url": "https://real.example/a"}]})

    transport = httpx.MockTransport(_handler)
    client = httpx.AsyncClient(transport=transport)
    proxy = ToolProxy(
        store=store,
        http_client=client,
        active_honeytokens=active,
        honeytoken_injection_rate=1.0,  # force hit every call
    )
    try:
        result = await proxy.proxy_call(
            conversation_id="c-proxy-ht",
            tool_name="web_search",
            target_url="http://web-search-tool-service:8085/v1/search",
            args={"query": "x"},
        )
    finally:
        await proxy.close()
        await client.aclose()

    results = result["payload"]["results"]
    assert len(results) == 2  # one real + one honeytoken
    injected_url = results[-1]["url"]
    assert HONEYTOKEN_MARKER in injected_url
    assert injected_url in active


async def test_tool_proxy_skips_injection_when_active_set_empty():
    store = InMemoryTraceStore()

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"url": "https://real.example/a"}]})

    transport = httpx.MockTransport(_handler)
    client = httpx.AsyncClient(transport=transport)
    proxy = ToolProxy(
        store=store,
        http_client=client,
        active_honeytokens=None,  # no active set
        honeytoken_injection_rate=1.0,
    )
    try:
        result = await proxy.proxy_call(
            conversation_id="c-no-ht",
            tool_name="web_search",
            target_url="http://web-search-tool-service:8085/v1/search",
            args={"query": "x"},
        )
    finally:
        await proxy.close()
        await client.aclose()

    assert len(result["payload"]["results"]) == 1


# -- TraceStore contract tests -----------


def _make_store(name: str) -> TraceStore:
    if name == "memory":
        return InMemoryTraceStore()
    if name == "redis":
        import fakeredis.aioredis

        from control_plane.owner_api.middleware.redis_trace_store import (
            RedisTraceStore,
        )

        client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        return RedisTraceStore(client)
    raise ValueError(f"unknown backend: {name}")


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_trace_store_contract_append_and_get(backend: str):
    store = _make_store(backend)
    await store.append("c1", TraceEntry(tool_name="web_search", args={"q": "a"}))
    await store.append("c1", TraceEntry(tool_name="semantic_scholar", args={"q": "b"}))
    trace = await store.get_trace("c1")
    assert trace.conversation_id == "c1"
    assert [e.tool_name for e in trace.entries] == ["web_search", "semantic_scholar"]


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_trace_store_contract_get_missing_returns_empty(backend: str):
    store = _make_store(backend)
    trace = await store.get_trace("nonexistent")
    assert trace.conversation_id == "nonexistent"
    assert trace.entries == []


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_trace_store_contract_clear(backend: str):
    store = _make_store(backend)
    await store.append("c1", TraceEntry(tool_name="web_search", args={}))
    await store.clear("c1")
    trace = await store.get_trace("c1")
    assert trace.entries == []


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_trace_store_contract_clear_many(backend: str):
    store = _make_store(backend)
    await store.append("c1", TraceEntry(tool_name="web_search", args={}))
    await store.append("c2", TraceEntry(tool_name="semantic_scholar", args={}))
    await store.append("c3", TraceEntry(tool_name="x_api", args={}))
    await store.clear_many(["c1", "c2"])
    assert (await store.get_trace("c1")).entries == []
    assert (await store.get_trace("c2")).entries == []
    assert len((await store.get_trace("c3")).entries) == 1


# -- Backend selection tests -----------


def test_default_trace_store_returns_in_memory_by_default(monkeypatch):
    monkeypatch.delenv("EIREL_TRACE_STORE_BACKEND", raising=False)
    from shared.common.config import reset_settings

    reset_settings()
    store = default_trace_store()
    assert isinstance(store, InMemoryTraceStore)


def test_default_trace_store_returns_redis_when_configured(monkeypatch):
    import fakeredis.aioredis

    monkeypatch.setenv("EIREL_TRACE_STORE_BACKEND", "redis")
    monkeypatch.setenv("EIREL_TRACE_STORE_REDIS_URL", "redis://fake:6379")

    from redis import asyncio as redis_asyncio

    from control_plane.owner_api.middleware.redis_trace_store import RedisTraceStore
    from shared.common.config import reset_settings

    monkeypatch.setattr(
        redis_asyncio,
        "from_url",
        lambda *a, **kw: fakeredis.aioredis.FakeRedis(decode_responses=True),
    )
    reset_settings()
    store = default_trace_store()
    assert isinstance(store, RedisTraceStore)
