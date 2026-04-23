from __future__ import annotations

from unittest.mock import AsyncMock, patch

import fakeredis.aioredis

from control_plane.owner_api.middleware.redis_trace_store import (
    TRACE_TTL_SECONDS,
    RedisTraceStore,
    _key,
)
from shared.core.evaluation_models import TraceEntry


def _make_store() -> tuple[fakeredis.aioredis.FakeRedis, RedisTraceStore]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return client, RedisTraceStore(client)


async def test_append_and_get_round_trip():
    _, store = _make_store()
    await store.append("c1", TraceEntry(tool_name="web_search", args={"q": "a"}))
    await store.append("c1", TraceEntry(tool_name="sandbox", args={"q": "b"}))
    trace = await store.get_trace("c1")
    assert trace.conversation_id == "c1"
    assert [e.tool_name for e in trace.entries] == ["web_search", "sandbox"]


async def test_get_missing_key_returns_empty():
    _, store = _make_store()
    trace = await store.get_trace("nonexistent")
    assert trace.conversation_id == "nonexistent"
    assert trace.entries == []


async def test_clear_removes_entries():
    _, store = _make_store()
    await store.append("c1", TraceEntry(tool_name="web_search", args={}))
    await store.clear("c1")
    trace = await store.get_trace("c1")
    assert trace.entries == []


async def test_clear_many_batches_deletes():
    _, store = _make_store()
    await store.append("c1", TraceEntry(tool_name="web_search", args={}))
    await store.append("c2", TraceEntry(tool_name="sandbox", args={}))
    await store.append("c3", TraceEntry(tool_name="x_api", args={}))
    await store.clear_many(["c1", "c2"])
    assert (await store.get_trace("c1")).entries == []
    assert (await store.get_trace("c2")).entries == []
    assert len((await store.get_trace("c3")).entries) == 1


async def test_ttl_set_after_append():
    client, store = _make_store()
    await store.append("c1", TraceEntry(tool_name="web_search", args={}))
    ttl = await client.ttl(_key("c1"))
    assert 0 < ttl <= TRACE_TTL_SECONDS


async def test_append_failure_logs_warning(caplog):
    client, store = _make_store()
    with patch.object(client, "pipeline", side_effect=ConnectionError("boom")):
        await store.append("c1", TraceEntry(tool_name="web_search", args={}))
    assert "trace append failed" in caplog.text
    trace = await store.get_trace("c1")
    assert trace.entries == []


async def test_get_failure_returns_empty(caplog):
    client, store = _make_store()
    with patch.object(client, "lrange", side_effect=ConnectionError("boom")):
        trace = await store.get_trace("c1")
    assert trace.conversation_id == "c1"
    assert trace.entries == []
    assert "trace get failed" in caplog.text


async def test_clear_failure_logs_warning(caplog):
    client, store = _make_store()
    await store.append("c1", TraceEntry(tool_name="web_search", args={}))
    with patch.object(client, "delete", side_effect=ConnectionError("boom")):
        await store.clear("c1")
    assert "trace clear failed" in caplog.text


async def test_clear_many_failure_logs_warning(caplog):
    client, store = _make_store()
    await store.append("c1", TraceEntry(tool_name="web_search", args={}))
    with patch.object(client, "pipeline", side_effect=ConnectionError("boom")):
        await store.clear_many(["c1"])
    assert "trace clear_many failed" in caplog.text


async def test_none_client_all_methods_noop():
    store = RedisTraceStore(None)
    await store.append("c1", TraceEntry(tool_name="web_search", args={}))
    trace = await store.get_trace("c1")
    assert trace.entries == []
    await store.clear("c1")
    await store.clear_many(["c1", "c2"])
