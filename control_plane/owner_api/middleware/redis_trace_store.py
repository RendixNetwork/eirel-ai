from __future__ import annotations

import logging

from redis import asyncio as redis_asyncio
from redis.exceptions import RedisError

from shared.core.evaluation_models import ConversationTrace, TraceEntry

_logger = logging.getLogger(__name__)

TRACE_TTL_SECONDS = 24 * 3600


def _key(conversation_id: str) -> str:
    return f"eirel:trace:v1:{conversation_id}"


class RedisTraceStore:

    def __init__(self, redis_client: redis_asyncio.Redis | None) -> None:
        self._r = redis_client

    async def append(self, conversation_id: str, entry: TraceEntry) -> None:
        if self._r is None:
            return
        key = _key(conversation_id)
        try:
            async with self._r.pipeline(transaction=False) as pipe:
                pipe.rpush(key, entry.model_dump_json())
                pipe.expire(key, TRACE_TTL_SECONDS)
                await pipe.execute()
        except (RedisError, ConnectionError):
            _logger.warning("trace append failed for %s", conversation_id)

    async def get_trace(self, conversation_id: str) -> ConversationTrace:
        if self._r is None:
            return ConversationTrace(conversation_id=conversation_id)
        key = _key(conversation_id)
        try:
            raw_items = await self._r.lrange(key, 0, -1)
            entries = [TraceEntry.model_validate_json(item) for item in raw_items]
            return ConversationTrace(
                conversation_id=conversation_id, entries=entries
            )
        except (RedisError, ConnectionError):
            _logger.warning("trace get failed for %s", conversation_id)
            return ConversationTrace(conversation_id=conversation_id)

    async def clear(self, conversation_id: str) -> None:
        if self._r is None:
            return
        try:
            await self._r.delete(_key(conversation_id))
        except (RedisError, ConnectionError):
            _logger.warning("trace clear failed for %s", conversation_id)

    async def clear_many(self, conversation_ids: list[str]) -> None:
        if self._r is None:
            return
        if not conversation_ids:
            return
        try:
            async with self._r.pipeline(transaction=False) as pipe:
                for cid in conversation_ids:
                    pipe.delete(_key(cid))
                await pipe.execute()
        except (RedisError, ConnectionError):
            _logger.warning("trace clear_many failed")
