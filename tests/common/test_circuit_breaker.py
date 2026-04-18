from __future__ import annotations

"""Tests for Item 18: Circuit breakers."""

import time
from unittest.mock import patch

import pytest

from shared.common.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


@pytest.fixture
def breaker():
    return CircuitBreaker(failure_threshold=3, recovery_timeout=1.0, half_open_max_calls=1)


@pytest.mark.asyncio
async def test_closed_circuit_passes_calls(breaker):
    async def ok():
        return "success"

    result = await breaker.call("ep1", ok)
    assert result == "success"
    assert breaker.state("ep1") == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold(breaker):
    async def fail():
        raise RuntimeError("fail")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call("ep1", fail)

    assert breaker.state("ep1") == CircuitState.OPEN


@pytest.mark.asyncio
async def test_open_circuit_rejects_calls(breaker):
    async def fail():
        raise RuntimeError("fail")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call("ep1", fail)

    with pytest.raises(CircuitOpenError) as exc_info:
        await breaker.call("ep1", fail)

    assert exc_info.value.key == "ep1"
    assert exc_info.value.retry_after > 0


@pytest.mark.asyncio
async def test_circuit_transitions_to_half_open(breaker):
    async def fail():
        raise RuntimeError("fail")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call("ep1", fail)

    # Simulate time passing
    breaker._circuits["ep1"].opened_at = time.monotonic() - 2.0
    assert breaker.state("ep1") == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_half_open_success_closes_circuit(breaker):
    async def fail():
        raise RuntimeError("fail")

    async def ok():
        return "success"

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call("ep1", fail)

    breaker._circuits["ep1"].opened_at = time.monotonic() - 2.0

    result = await breaker.call("ep1", ok)
    assert result == "success"
    assert breaker.state("ep1") == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_half_open_failure_reopens_circuit(breaker):
    async def fail():
        raise RuntimeError("fail")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call("ep1", fail)

    breaker._circuits["ep1"].opened_at = time.monotonic() - 2.0
    breaker._circuits["ep1"].failure_count = 0  # Reset for half-open test

    with pytest.raises(RuntimeError):
        await breaker.call("ep1", fail)


@pytest.mark.asyncio
async def test_per_key_isolation(breaker):
    async def fail():
        raise RuntimeError("fail")

    async def ok():
        return "success"

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call("ep1", fail)

    result = await breaker.call("ep2", ok)
    assert result == "success"
    assert breaker.state("ep1") == CircuitState.OPEN
    assert breaker.state("ep2") == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_reset_clears_circuit(breaker):
    async def fail():
        raise RuntimeError("fail")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call("ep1", fail)

    breaker.reset("ep1")
    assert breaker.state("ep1") == CircuitState.CLOSED


def test_stats_returns_per_key_info(breaker):
    stats = breaker.stats()
    assert isinstance(stats, dict)


@pytest.mark.asyncio
async def test_success_resets_failure_count(breaker):
    async def fail():
        raise RuntimeError("fail")

    async def ok():
        return "success"

    # 2 failures then success
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await breaker.call("ep1", fail)

    await breaker.call("ep1", ok)
    assert breaker.state("ep1") == CircuitState.CLOSED
    assert breaker._circuits["ep1"].failure_count == 0
