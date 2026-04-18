from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from redis import asyncio as redis_asyncio

_logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 86400
KEY_PREFIX = "tool_job_usage:"


@dataclass(slots=True)
class JobUsageRecord:
    request_count: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    ledger_id: str = ""
    searches: list[dict[str, Any]] = field(default_factory=list)


def _deserialize(raw: str | bytes) -> JobUsageRecord:
    data = json.loads(raw)
    return JobUsageRecord(
        request_count=data.get("request_count", 0),
        tool_counts=data.get("tool_counts", {}),
        ledger_id=data.get("ledger_id", ""),
        searches=data.get("searches", []),
    )


class RedisJobLedger:
    def __init__(
        self,
        client: redis_asyncio.Redis,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds

    async def record_usage(
        self,
        job_id: str,
        tokens: int,
        tool_calls: dict[str, int],
    ) -> None:
        record = await self.get_or_create(job_id)
        record.request_count += tokens
        for tool, count in tool_calls.items():
            record.tool_counts[tool] = record.tool_counts.get(tool, 0) + count
        await self.save(job_id, record)

    async def get_usage(self, job_id: str) -> JobUsageRecord | None:
        key = f"{KEY_PREFIX}{job_id}"
        raw = await self._client.get(key)
        if raw is None:
            return None
        return _deserialize(raw)

    async def reset(self, job_id: str) -> None:
        key = f"{KEY_PREFIX}{job_id}"
        await self._client.delete(key)

    async def get_or_create(self, job_id: str) -> JobUsageRecord:
        record = await self.get_usage(job_id)
        if record is not None:
            return record
        record = JobUsageRecord(ledger_id=f"ledger:{job_id}")
        await self.save(job_id, record)
        return record

    async def save(self, job_id: str, record: JobUsageRecord) -> None:
        key = f"{KEY_PREFIX}{job_id}"
        data = json.dumps(asdict(record))
        await self._client.set(key, data, ex=self._ttl_seconds)

    async def close(self) -> None:
        await self._client.aclose()


class InMemoryJobLedger:
    def __init__(self) -> None:
        self._store: dict[str, JobUsageRecord] = {}

    async def record_usage(
        self,
        job_id: str,
        tokens: int,
        tool_calls: dict[str, int],
    ) -> None:
        record = await self.get_or_create(job_id)
        record.request_count += tokens
        for tool, count in tool_calls.items():
            record.tool_counts[tool] = record.tool_counts.get(tool, 0) + count

    async def get_usage(self, job_id: str) -> JobUsageRecord | None:
        return self._store.get(job_id)

    async def reset(self, job_id: str) -> None:
        self._store.pop(job_id, None)

    async def get_or_create(self, job_id: str) -> JobUsageRecord:
        record = self._store.get(job_id)
        if record is not None:
            return record
        record = JobUsageRecord(ledger_id=f"ledger:{job_id}")
        self._store[job_id] = record
        return record

    async def save(self, job_id: str, record: JobUsageRecord) -> None:
        self._store[job_id] = record

    async def close(self) -> None:
        pass


JobLedger = RedisJobLedger | InMemoryJobLedger


def create_job_ledger(
    redis_url: str,
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> JobLedger:
    if redis_url.strip():
        client = redis_asyncio.from_url(
            redis_url,
            decode_responses=True,
            max_connections=10,
            health_check_interval=30,
        )
        return RedisJobLedger(client, ttl_seconds=ttl_seconds)
    _logger.warning("REDIS_URL not set; job usage ledger will use in-memory storage")
    return InMemoryJobLedger()
