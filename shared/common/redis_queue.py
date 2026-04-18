from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

from redis import asyncio as redis_asyncio
from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError

_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QueueMessage:
    message_id: str
    task_id: str


class InMemoryExecutionQueue:
    def __init__(self) -> None:
        self._messages: deque[QueueMessage] = deque()
        self._pending: dict[str, QueueMessage] = {}
        self._next_id = 1
        self._condition = asyncio.Condition()

    async def ensure_consumer_group(self) -> None:
        return None

    async def enqueue_task(self, *, task_id: str) -> str:
        async with self._condition:
            message = QueueMessage(message_id=str(self._next_id), task_id=task_id)
            self._next_id += 1
            self._messages.append(message)
            self._condition.notify_all()
            return message.message_id

    async def claim_next(
        self,
        *,
        consumer_name: str,
        block_ms: int,
    ) -> QueueMessage | None:
        del consumer_name
        timeout = max(0.0, block_ms / 1000.0)
        async with self._condition:
            if not self._messages:
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=timeout)
                except TimeoutError:
                    return None
            if not self._messages:
                return None
            message = self._messages.popleft()
            self._pending[message.message_id] = message
            return message

    async def recover_idle(
        self,
        *,
        consumer_name: str,
        min_idle_ms: int,
        count: int = 32,
    ) -> list[QueueMessage]:
        del consumer_name, min_idle_ms, count
        return []

    async def ack(self, *, message_id: str) -> None:
        self._pending.pop(message_id, None)

    async def close(self) -> None:
        return None


class RedisExecutionQueue:
    def __init__(
        self,
        *,
        redis_url: str,
        stream_name: str,
        group_name: str,
    ) -> None:
        self.stream_name = stream_name
        self.group_name = group_name
        self.client = redis_asyncio.from_url(
            redis_url,
            decode_responses=True,
            max_connections=20,
            health_check_interval=30,
        )

    async def ensure_consumer_group(self) -> None:
        try:
            await self.client.xgroup_create(
                name=self.stream_name,
                groupname=self.group_name,
                id="0",
                mkstream=True,
            )
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def enqueue_task(self, *, task_id: str) -> str:
        for attempt in range(2):
            try:
                return str(
                    await self.client.xadd(self.stream_name, {"task_id": task_id})
                )
            except (RedisConnectionError, RedisTimeoutError):
                if attempt == 0:
                    _logger.warning("redis enqueue transient failure, retrying")
                    await asyncio.sleep(0.5)
                    continue
                raise
        raise RuntimeError("unreachable")

    async def claim_next(
        self,
        *,
        consumer_name: str,
        block_ms: int,
    ) -> QueueMessage | None:
        for attempt in range(2):
            try:
                response = await self.client.xreadgroup(
                    groupname=self.group_name,
                    consumername=consumer_name,
                    streams={self.stream_name: ">"},
                    count=1,
                    block=block_ms,
                )
                for _, entries in response:
                    for message_id, fields in entries:
                        task_id = fields.get("task_id")
                        if task_id:
                            return QueueMessage(message_id=str(message_id), task_id=str(task_id))
                return None
            except (RedisConnectionError, RedisTimeoutError):
                if attempt == 0:
                    _logger.warning("redis claim_next transient failure, retrying")
                    await asyncio.sleep(0.5)
                    continue
                raise
        return None

    async def recover_idle(
        self,
        *,
        consumer_name: str,
        min_idle_ms: int,
        count: int = 32,
    ) -> list[QueueMessage]:
        recovered: list[QueueMessage] = []
        next_id = "0-0"
        while len(recovered) < count:
            response = await self.client.xautoclaim(
                name=self.stream_name,
                groupname=self.group_name,
                consumername=consumer_name,
                min_idle_time=min_idle_ms,
                start_id=next_id,
                count=count - len(recovered),
            )
            if isinstance(response, tuple) and len(response) >= 2:
                next_id = str(response[0])
                messages = response[1]
            else:
                next_id = "0-0"
                messages = []
            if not messages:
                break
            for message_id, fields in messages:
                task_id = fields.get("task_id")
                if task_id:
                    recovered.append(
                        QueueMessage(message_id=str(message_id), task_id=str(task_id))
                    )
            if next_id == "0-0":
                break
        return recovered

    async def ack(self, *, message_id: str) -> None:
        await self.client.xack(self.stream_name, self.group_name, message_id)

    async def close(self) -> None:
        await self.client.aclose()


_IN_MEMORY_QUEUES: dict[str, InMemoryExecutionQueue] = {}


def create_execution_queue(
    *,
    redis_url: str,
    stream_name: str,
    group_name: str,
) -> RedisExecutionQueue | InMemoryExecutionQueue:
    if redis_url.strip():
        return RedisExecutionQueue(
            redis_url=redis_url,
            stream_name=stream_name,
            group_name=group_name,
        )
    key = f"{stream_name}:{group_name}"
    queue = _IN_MEMORY_QUEUES.get(key)
    if queue is None:
        queue = InMemoryExecutionQueue()
        _IN_MEMORY_QUEUES[key] = queue
    return queue
