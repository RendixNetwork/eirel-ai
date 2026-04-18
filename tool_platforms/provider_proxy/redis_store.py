from __future__ import annotations

"""Redis-backed ledger for provider-proxy cost accounting.

Replaces the in-memory ``usage: dict[str, JobUsage]`` with a durable
store so proxy restarts (config rolls, image rolls, crashes) don't
wipe per-deployment spend mid-run.  Runs are 3 days long; TTL is set
to 7 days to give comfortable headroom across a full run plus
aggregation.

Data layout, per job_id:
    provider_proxy:usage:<job_id>            (hash)
        started_at           (float, monotonic clock of first call)
        request_count        (int)
        estimated_total_tokens (int)
        actual_total_tokens  (int)
        cost_usd_used        (float)
        max_usd_budget       (float)
        cost_rejections      (int)
    provider_proxy:provider_counts:<job_id>  (hash: provider → int)
    provider_proxy:model_counts:<job_id>     (hash: provider:model → int)
    provider_proxy:cost_by_provider:<job_id> (hash: key → float)
        keys are either a bare provider name (LLM), ``tool:<name>``,
        or ``penalty:<reason>`` — the split ScoringManager consumes.

Concurrent request safety:
    ``reserve_estimate`` and ``charge_tool`` execute a Lua script so
    the budget check and the HINCRBYFLOAT reservation are atomic.
    Without this, two concurrent calls could both pass a budget
    check that together exceeds it.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

from redis import asyncio as redis_asyncio
from redis.exceptions import ResponseError

_logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # 7 days

_USAGE_PREFIX = "provider_proxy:usage:"
_PROVIDER_COUNTS_PREFIX = "provider_proxy:provider_counts:"
_MODEL_COUNTS_PREFIX = "provider_proxy:model_counts:"
_COST_BY_PROVIDER_PREFIX = "provider_proxy:cost_by_provider:"


@dataclass(slots=True)
class JobUsage:
    """Read-only view of a single job's accumulated usage."""

    started_at: float
    request_count: int = 0
    estimated_total_tokens: int = 0
    actual_total_tokens: int = 0
    provider_request_counts: dict[str, int] = field(default_factory=dict)
    model_request_counts: dict[str, int] = field(default_factory=dict)
    cost_usd_used: float = 0.0
    max_usd_budget: float = 0.0
    cost_by_provider: dict[str, float] = field(default_factory=dict)
    cost_rejections: int = 0


# -- Lua scripts -------------------------------------------------------------
#
# Both scripts are intentionally short — Redis is single-threaded inside a
# script so holding them too long stalls other requests.

_RESERVE_SCRIPT = """
-- KEYS: usage_key, provider_counts_key, model_counts_key
-- ARGV: estimated_cost, max_budget, estimated_tokens, provider, model_key, now, ttl
local usage_key = KEYS[1]
local provider_counts_key = KEYS[2]
local model_counts_key = KEYS[3]
local estimated_cost = tonumber(ARGV[1])
local max_budget = tonumber(ARGV[2])
local estimated_tokens = tonumber(ARGV[3])
local provider = ARGV[4]
local model_key = ARGV[5]
local now = ARGV[6]
local ttl = tonumber(ARGV[7])

local cost_str = redis.call('HGET', usage_key, 'cost_usd_used')
local cost = tonumber(cost_str) or 0.0
if cost + estimated_cost > max_budget then
    redis.call('HINCRBY', usage_key, 'cost_rejections', 1)
    redis.call('HSETNX', usage_key, 'started_at', now)
    redis.call('HSETNX', usage_key, 'max_usd_budget', max_budget)
    redis.call('EXPIRE', usage_key, ttl)
    return {1, tostring(cost)}
end
redis.call('HSETNX', usage_key, 'started_at', now)
redis.call('HSETNX', usage_key, 'max_usd_budget', max_budget)
redis.call('HINCRBYFLOAT', usage_key, 'cost_usd_used', estimated_cost)
redis.call('HINCRBY', usage_key, 'request_count', 1)
redis.call('HINCRBY', usage_key, 'estimated_total_tokens', estimated_tokens)
redis.call('HINCRBY', provider_counts_key, provider, 1)
redis.call('HINCRBY', model_counts_key, model_key, 1)
redis.call('EXPIRE', usage_key, ttl)
redis.call('EXPIRE', provider_counts_key, ttl)
redis.call('EXPIRE', model_counts_key, ttl)
return {0, tostring(cost + estimated_cost)}
"""


_CHARGE_TOOL_SCRIPT = """
-- KEYS: usage_key, cost_by_provider_key
-- ARGV: amount, bucket_key, ttl
local usage_key = KEYS[1]
local cost_key = KEYS[2]
local amount = tonumber(ARGV[1])
local bucket_key = ARGV[2]
local ttl = tonumber(ARGV[3])

local cost_str = redis.call('HGET', usage_key, 'cost_usd_used')
local cost = tonumber(cost_str) or 0.0
local max_str = redis.call('HGET', usage_key, 'max_usd_budget')
local max_budget = tonumber(max_str) or 0.0
if cost + amount > max_budget then
    redis.call('HINCRBY', usage_key, 'cost_rejections', 1)
    redis.call('EXPIRE', usage_key, ttl)
    return {1, tostring(cost)}
end
redis.call('HINCRBYFLOAT', usage_key, 'cost_usd_used', amount)
redis.call('HINCRBYFLOAT', cost_key, bucket_key, amount)
redis.call('EXPIRE', usage_key, ttl)
redis.call('EXPIRE', cost_key, ttl)
return {0, tostring(cost + amount)}
"""


class ProviderJobStore:
    """Async Redis-backed ledger.  Single source of truth — no local cache.

    On real Redis, the two budget-sensitive operations (reserve_estimate
    and charge_tool) run as atomic Lua scripts so two concurrent requests
    can't both pass the same budget check.  On a Redis backend that
    doesn't support EVAL/EVALSHA (e.g. fakeredis without lupa installed),
    the store transparently falls back to a Python path serialised by a
    per-job asyncio lock — still race-free within a single proxy
    replica, which is what the current deployment ships.
    """

    def __init__(
        self,
        client: redis_asyncio.Redis,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._client = client
        self._ttl = ttl_seconds
        self._reserve_script = client.register_script(_RESERVE_SCRIPT)
        self._charge_tool_script = client.register_script(_CHARGE_TOOL_SCRIPT)
        # State for the Python fallback path.  ``_lua_available`` starts
        # optimistic and flips once on the first NOSCRIPT / "unknown
        # command" response — we don't want to probe on every call.
        self._lua_available: bool = True
        self._per_job_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    async def reserve_estimate(
        self,
        *,
        job_id: str,
        estimated_cost: float,
        max_usd_budget: float,
        estimated_tokens: int,
        provider: str,
        model: str,
    ) -> tuple[bool, float]:
        """Atomically reserve an estimate against the run budget.

        Returns ``(accepted, cost_usd_used)``.  ``accepted=False`` means
        the run budget would have been exceeded; the caller must 429
        the client.  ``cost_rejections`` is bumped on rejection.
        """
        if self._lua_available:
            try:
                result = await self._reserve_script(
                    keys=[
                        _USAGE_PREFIX + job_id,
                        _PROVIDER_COUNTS_PREFIX + job_id,
                        _MODEL_COUNTS_PREFIX + job_id,
                    ],
                    args=[
                        str(estimated_cost),
                        str(max_usd_budget),
                        str(int(estimated_tokens)),
                        provider,
                        f"{provider}:{model}",
                        str(time.monotonic()),
                        str(self._ttl),
                    ],
                )
                rejected = int(result[0]) == 1
                cost_used = float(result[1])
                return (not rejected, cost_used)
            except ResponseError as exc:
                if "unknown command" not in str(exc).lower():
                    raise
                self._lua_available = False
                _logger.info("Lua scripting unavailable; using Python fallback path")
        return await self._reserve_estimate_fallback(
            job_id=job_id,
            estimated_cost=estimated_cost,
            max_usd_budget=max_usd_budget,
            estimated_tokens=estimated_tokens,
            provider=provider,
            model=model,
        )

    async def _reserve_estimate_fallback(
        self,
        *,
        job_id: str,
        estimated_cost: float,
        max_usd_budget: float,
        estimated_tokens: int,
        provider: str,
        model: str,
    ) -> tuple[bool, float]:
        usage_key = _USAGE_PREFIX + job_id
        provider_counts_key = _PROVIDER_COUNTS_PREFIX + job_id
        model_counts_key = _MODEL_COUNTS_PREFIX + job_id
        async with self._per_job_locks[job_id]:
            cost_str = await self._client.hget(usage_key, "cost_usd_used")
            cost = float(cost_str) if cost_str is not None else 0.0
            if cost + estimated_cost > max_usd_budget:
                await self._client.hincrby(usage_key, "cost_rejections", 1)
                await self._client.hsetnx(usage_key, "started_at", str(time.monotonic()))
                await self._client.hsetnx(usage_key, "max_usd_budget", str(max_usd_budget))
                await self._client.expire(usage_key, self._ttl)
                return (False, cost)
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.hsetnx(usage_key, "started_at", str(time.monotonic()))
                pipe.hsetnx(usage_key, "max_usd_budget", str(max_usd_budget))
                pipe.hincrbyfloat(usage_key, "cost_usd_used", estimated_cost)
                pipe.hincrby(usage_key, "request_count", 1)
                pipe.hincrby(usage_key, "estimated_total_tokens", int(estimated_tokens))
                pipe.hincrby(provider_counts_key, provider, 1)
                pipe.hincrby(model_counts_key, f"{provider}:{model}", 1)
                pipe.expire(usage_key, self._ttl)
                pipe.expire(provider_counts_key, self._ttl)
                pipe.expire(model_counts_key, self._ttl)
                await pipe.execute()
            return (True, cost + estimated_cost)

    async def refund_estimate(
        self,
        *,
        job_id: str,
        estimated_cost: float,
        estimated_tokens: int,
    ) -> None:
        """Undo a reservation after an upstream failure.

        Callers that previously reserved via ``reserve_estimate`` must
        call this on any exception before reconciliation — otherwise
        the estimate leaks into ``cost_usd_used`` as ghost spend.
        """
        usage_key = _USAGE_PREFIX + job_id
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.hincrbyfloat(usage_key, "cost_usd_used", -float(estimated_cost))
            pipe.hincrby(usage_key, "estimated_total_tokens", -int(estimated_tokens))
            pipe.expire(usage_key, self._ttl)
            await pipe.execute()

    async def reconcile_actual_cost(
        self,
        *,
        job_id: str,
        provider: str,
        delta_cost: float,
        actual_cost: float,
        actual_total_tokens: int,
    ) -> None:
        """Apply the post-call reconciliation after the upstream returned."""
        usage_key = _USAGE_PREFIX + job_id
        cost_key = _COST_BY_PROVIDER_PREFIX + job_id
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.hincrbyfloat(usage_key, "cost_usd_used", float(delta_cost))
            pipe.hincrby(usage_key, "actual_total_tokens", int(actual_total_tokens))
            pipe.hincrbyfloat(cost_key, provider, float(actual_cost))
            pipe.expire(usage_key, self._ttl)
            pipe.expire(cost_key, self._ttl)
            await pipe.execute()

    async def charge_tool(
        self,
        *,
        job_id: str,
        tool_name: str,
        amount_usd: float,
    ) -> tuple[bool, float]:
        """Atomically check budget and charge a tool call.

        Returns ``(accepted, cost_usd_used)``.  ``accepted=False`` → 429.
        """
        if self._lua_available:
            try:
                result = await self._charge_tool_script(
                    keys=[
                        _USAGE_PREFIX + job_id,
                        _COST_BY_PROVIDER_PREFIX + job_id,
                    ],
                    args=[
                        str(float(amount_usd)),
                        f"tool:{tool_name}",
                        str(self._ttl),
                    ],
                )
                rejected = int(result[0]) == 1
                cost_used = float(result[1])
                return (not rejected, cost_used)
            except ResponseError as exc:
                if "unknown command" not in str(exc).lower():
                    raise
                self._lua_available = False
                _logger.info("Lua scripting unavailable; using Python fallback path")
        return await self._charge_tool_fallback(
            job_id=job_id, tool_name=tool_name, amount_usd=amount_usd,
        )

    async def _charge_tool_fallback(
        self,
        *,
        job_id: str,
        tool_name: str,
        amount_usd: float,
    ) -> tuple[bool, float]:
        usage_key = _USAGE_PREFIX + job_id
        cost_key = _COST_BY_PROVIDER_PREFIX + job_id
        async with self._per_job_locks[job_id]:
            cost_str = await self._client.hget(usage_key, "cost_usd_used")
            max_str = await self._client.hget(usage_key, "max_usd_budget")
            cost = float(cost_str) if cost_str is not None else 0.0
            max_budget = float(max_str) if max_str is not None else 0.0
            if cost + amount_usd > max_budget:
                await self._client.hincrby(usage_key, "cost_rejections", 1)
                await self._client.expire(usage_key, self._ttl)
                return (False, cost)
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.hincrbyfloat(usage_key, "cost_usd_used", float(amount_usd))
                pipe.hincrbyfloat(cost_key, f"tool:{tool_name}", float(amount_usd))
                pipe.expire(usage_key, self._ttl)
                pipe.expire(cost_key, self._ttl)
                await pipe.execute()
            return (True, cost + amount_usd)

    async def charge_penalty(
        self,
        *,
        job_id: str,
        reason: str,
        amount_usd: float,
    ) -> float:
        """Charge an unconditional penalty (always lands, no 429)."""
        usage_key = _USAGE_PREFIX + job_id
        cost_key = _COST_BY_PROVIDER_PREFIX + job_id
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.hincrbyfloat(usage_key, "cost_usd_used", float(amount_usd))
            pipe.hincrbyfloat(cost_key, f"penalty:{reason}", float(amount_usd))
            pipe.expire(usage_key, self._ttl)
            pipe.expire(cost_key, self._ttl)
            pipe.hget(usage_key, "cost_usd_used")
            values = await pipe.execute()
        return float(values[-1] or 0.0)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def exists(self, job_id: str) -> bool:
        return bool(await self._client.exists(_USAGE_PREFIX + job_id))

    async def get(self, job_id: str) -> JobUsage | None:
        usage_key = _USAGE_PREFIX + job_id
        provider_counts_key = _PROVIDER_COUNTS_PREFIX + job_id
        model_counts_key = _MODEL_COUNTS_PREFIX + job_id
        cost_key = _COST_BY_PROVIDER_PREFIX + job_id
        async with self._client.pipeline(transaction=False) as pipe:
            pipe.hgetall(usage_key)
            pipe.hgetall(provider_counts_key)
            pipe.hgetall(model_counts_key)
            pipe.hgetall(cost_key)
            usage_raw, provider_raw, model_raw, cost_raw = await pipe.execute()
        if not usage_raw:
            return None
        return JobUsage(
            started_at=float(usage_raw.get("started_at", 0.0) or 0.0),
            request_count=int(usage_raw.get("request_count", 0) or 0),
            estimated_total_tokens=int(usage_raw.get("estimated_total_tokens", 0) or 0),
            actual_total_tokens=int(usage_raw.get("actual_total_tokens", 0) or 0),
            cost_usd_used=float(usage_raw.get("cost_usd_used", 0.0) or 0.0),
            max_usd_budget=float(usage_raw.get("max_usd_budget", 0.0) or 0.0),
            cost_rejections=int(usage_raw.get("cost_rejections", 0) or 0),
            provider_request_counts={k: int(v) for k, v in (provider_raw or {}).items()},
            model_request_counts={k: int(v) for k, v in (model_raw or {}).items()},
            cost_by_provider={k: float(v) for k, v in (cost_raw or {}).items()},
        )

    async def list_all(self) -> dict[str, JobUsage]:
        """Used by /v1/operators/summary — scans all jobs in the DB."""
        job_ids: list[str] = []
        async for key in self._client.scan_iter(match=_USAGE_PREFIX + "*"):
            job_id = key[len(_USAGE_PREFIX):] if isinstance(key, str) else key.decode()[len(_USAGE_PREFIX):]
            job_ids.append(job_id)
        jobs: dict[str, JobUsage] = {}
        for job_id in job_ids:
            usage = await self.get(job_id)
            if usage is not None:
                jobs[job_id] = usage
        return jobs


__all__ = ["JobUsage", "ProviderJobStore", "DEFAULT_TTL_SECONDS"]
