from __future__ import annotations

from typing import Any

import fakeredis.aioredis
import httpx
from httpx import ASGITransport, AsyncClient

from shared.common.redis_job_ledger import RedisJobLedger


def _make_redis_ledger() -> RedisJobLedger:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisJobLedger(client, ttl_seconds=86400)


async def test_redis_ledger_persists_across_instances():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)

    ledger_a = RedisJobLedger(client, ttl_seconds=86400)
    await ledger_a.record_usage("job-1", 1, {"search": 1})
    await ledger_a.record_usage("job-1", 1, {"search": 1})

    ledger_b = RedisJobLedger(client, ttl_seconds=86400)
    record = await ledger_b.get_usage("job-1")
    assert record is not None
    assert record.request_count == 2
    assert record.tool_counts["search"] == 2
    assert record.ledger_id == "ledger:job-1"


async def test_redis_ledger_reset():
    ledger = _make_redis_ledger()
    await ledger.record_usage("job-2", 3, {"fetch": 3})
    await ledger.reset("job-2")
    assert await ledger.get_usage("job-2") is None


async def test_redis_ledger_searches_persist():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)

    ledger_a = RedisJobLedger(client, ttl_seconds=86400)
    record = await ledger_a.get_or_create("job-3")
    record.searches.append({"query": "test", "result_count": 5})
    await ledger_a.save("job-3", record)

    ledger_b = RedisJobLedger(client, ttl_seconds=86400)
    restored = await ledger_b.get_usage("job-3")
    assert restored is not None
    assert len(restored.searches) == 1
    assert restored.searches[0]["query"] == "test"


def _fake_x_transport(tweets: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": tweets,
                "includes": {
                    "users": [
                        {"id": t.get("author_id", "u1"), "username": "testuser", "verified": False}
                        for t in tweets
                    ]
                },
                "meta": {"result_count": len(tweets)},
            },
        )
    return httpx.MockTransport(handler)


async def test_x_tool_usage_persists_across_restart(monkeypatch):
    monkeypatch.setenv("EIREL_X_BEARER_TOKEN", "bearer-xxx")
    from tool_platforms.x_tool_service.app import create_app

    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    tweets = [{"id": "t1", "author_id": "u1", "text": "hello", "created_at": "2026-01-01T00:00:00Z", "public_metrics": {}}]
    transport = _fake_x_transport(tweets)

    ledger_a = RedisJobLedger(client, ttl_seconds=86400)
    app_a = create_app(x_transport=transport, ledger=ledger_a)
    async with app_a.router.lifespan_context(app_a):
        async with AsyncClient(transport=ASGITransport(app=app_a), base_url="http://test") as http:
            headers = {"X-Eirel-Job-Id": "job-persist", "X-Eirel-Max-Requests": "5"}
            resp = await http.post("/v1/search", json={"query": "q"}, headers=headers)
            assert resp.status_code == 200

    ledger_b = RedisJobLedger(client, ttl_seconds=86400)
    app_b = create_app(x_transport=transport, ledger=ledger_b)
    async with app_b.router.lifespan_context(app_b):
        async with AsyncClient(transport=ASGITransport(app=app_b), base_url="http://test") as http:
            usage = await http.get("/v1/jobs/job-persist/usage")
            assert usage.status_code == 200
            assert usage.json()["request_count"] == 1


_SEMANTIC_SCHOLAR_RESPONSE = {
    "total": 1,
    "offset": 0,
    "data": [
        {
            "paperId": "abc123",
            "title": "Test Paper",
            "abstract": "A summary.",
            "authors": [{"authorId": "1", "name": "Author"}],
            "year": 2026,
            "venue": "Test",
            "citationCount": 0,
            "influentialCitationCount": 0,
            "externalIds": {"ArXiv": "2601.00001"},
            "openAccessPdf": None,
            "url": "https://www.semanticscholar.org/paper/abc123",
        }
    ],
}


def _fake_semantic_scholar_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_SEMANTIC_SCHOLAR_RESPONSE)
    return httpx.MockTransport(handler)


async def test_semantic_scholar_usage_persists_across_restart(monkeypatch):
    from tool_platforms.semantic_scholar_tool_service.app import create_app

    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    transport = _fake_semantic_scholar_transport()

    ledger_a = RedisJobLedger(client, ttl_seconds=86400)
    app_a = create_app(semantic_scholar_transport=transport, ledger=ledger_a)
    async with app_a.router.lifespan_context(app_a):
        async with AsyncClient(transport=ASGITransport(app=app_a), base_url="http://test") as http:
            headers = {"X-Eirel-Job-Id": "job-s2", "X-Eirel-Max-Requests": "5"}
            resp = await http.post("/v1/search", json={"query": "AI"}, headers=headers)
            assert resp.status_code == 200

    ledger_b = RedisJobLedger(client, ttl_seconds=86400)
    app_b = create_app(semantic_scholar_transport=transport, ledger=ledger_b)
    async with app_b.router.lifespan_context(app_b):
        async with AsyncClient(transport=ASGITransport(app=app_b), base_url="http://test") as http:
            usage = await http.get("/v1/jobs/job-s2/usage")
            assert usage.status_code == 200
            assert usage.json()["request_count"] == 1
