from __future__ import annotations

import httpx
import pytest

from control_plane.owner_api.middleware.trace_capture import (
    InMemoryTraceStore,
    ToolBudgetExhaustedError,
    ToolProxy,
)


async def test_tool_proxy_429s_when_charge_tool_returns_429():
    store = InMemoryTraceStore()

    call_log: list[tuple[str, str]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        call_log.append((request.method, url))
        if "/charge_tool" in url:
            return httpx.Response(429, json={"detail": "run budget exhausted"})
        return httpx.Response(200, json={"results": []})

    transport = httpx.MockTransport(_handler)
    client = httpx.AsyncClient(transport=transport)
    proxy = ToolProxy(
        store=store,
        http_client=client,
        provider_proxy_url="http://proxy:8082",
        provider_proxy_token="tok",
        job_id="job-1",
    )
    try:
        with pytest.raises(ToolBudgetExhaustedError):
            await proxy.proxy_call(
                conversation_id="c1",
                tool_name="web_search",
                target_url="http://web-search:8085/v1/search",
                args={"query": "test"},
            )
    finally:
        await proxy.close()
        await client.aclose()

    trace = await store.get_trace("c1")
    assert len(trace.entries) == 1
    entry = trace.entries[0]
    assert entry.tool_name == "web_search"
    assert entry.cost_usd == 0.0
    assert entry.metadata["error"] == "run_budget_exhausted"
    assert entry.metadata["status_code"] == 429

    charge_calls = [url for method, url in call_log if "/charge_tool" in url]
    assert len(charge_calls) == 1
    tool_calls = [url for method, url in call_log if "/v1/search" in url]
    assert len(tool_calls) == 0


async def test_successful_tool_call_records_trace_without_double_charge():
    store = InMemoryTraceStore()

    call_log: list[tuple[str, str]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        call_log.append((request.method, url))
        if "/charge_tool" in url:
            return httpx.Response(200, json={"cost_usd_used": 0.001})
        return httpx.Response(200, json={"results": [{"title": "hit"}]})

    transport = httpx.MockTransport(_handler)
    client = httpx.AsyncClient(transport=transport)
    proxy = ToolProxy(
        store=store,
        http_client=client,
        provider_proxy_url="http://proxy:8082",
        provider_proxy_token="tok",
        job_id="job-2",
    )
    try:
        result = await proxy.proxy_call(
            conversation_id="c2",
            tool_name="web_search",
            target_url="http://web-search:8085/v1/search",
            args={"query": "test"},
        )
    finally:
        await proxy.close()
        await client.aclose()

    assert result["status_code"] == 200
    assert result["cost_usd"] == pytest.approx(0.001)

    trace = await store.get_trace("c2")
    assert len(trace.entries) == 1
    entry = trace.entries[0]
    assert entry.tool_name == "web_search"
    assert entry.cost_usd == pytest.approx(0.001)
    assert entry.metadata["error"] == ""

    charge_calls = [url for _, url in call_log if "/charge_tool" in url]
    assert len(charge_calls) == 1
    tool_calls = [url for _, url in call_log if "/v1/search" in url]
    assert len(tool_calls) == 1


async def test_tool_proxy_without_budget_config_works_as_before():
    store = InMemoryTraceStore()

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    transport = httpx.MockTransport(_handler)
    client = httpx.AsyncClient(transport=transport)
    proxy = ToolProxy(store=store, http_client=client)
    try:
        result = await proxy.proxy_call(
            conversation_id="c3",
            tool_name="semantic_scholar",
            target_url="http://semantic-scholar:8088/v1/search",
            args={"query": "test"},
        )
    finally:
        await proxy.close()
        await client.aclose()

    assert result["status_code"] == 200
    trace = await store.get_trace("c3")
    assert len(trace.entries) == 1
