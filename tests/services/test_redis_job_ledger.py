from __future__ import annotations

import fakeredis.aioredis

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


