from __future__ import annotations

"""Circuit breaker for async calls.

Implements the standard CLOSED → OPEN → HALF_OPEN pattern to prevent
cascading failures when downstream services are unhealthy.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, TypeVar

_logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, key: str, retry_after: float):
        self.key = key
        self.retry_after = retry_after
        super().__init__(f"Circuit open for {key!r}, retry after {retry_after:.1f}s")


class CircuitBreaker:
    """Per-endpoint circuit breaker.

    Parameters:
        failure_threshold: Number of consecutive failures before opening.
        recovery_timeout: Seconds to wait before transitioning to half-open.
        half_open_max_calls: Max concurrent calls in half-open state.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._circuits: dict[str, _CircuitEntry] = {}

    def state(self, key: str) -> CircuitState:
        entry = self._circuits.get(key)
        if entry is None:
            return CircuitState.CLOSED
        return entry.effective_state(self.recovery_timeout)

    async def call(self, key: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute ``func`` through the circuit breaker for ``key``."""
        entry = self._circuits.setdefault(key, _CircuitEntry())
        state = entry.effective_state(self.recovery_timeout)

        if state == CircuitState.OPEN:
            retry_after = entry.opened_at + self.recovery_timeout - time.monotonic()
            raise CircuitOpenError(key, max(0.0, retry_after))

        if state == CircuitState.HALF_OPEN and entry.half_open_calls >= self.half_open_max_calls:
            raise CircuitOpenError(key, self.recovery_timeout)

        if state == CircuitState.HALF_OPEN:
            entry.half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            entry.record_success()
            return result
        except Exception:
            entry.record_failure(self.failure_threshold)
            raise

    def reset(self, key: str) -> None:
        """Manually reset a circuit to closed."""
        self._circuits.pop(key, None)

    def reset_all(self) -> None:
        """Reset all circuits."""
        self._circuits.clear()

    def record_success(self, key: str) -> None:
        """Record a successful call for the given key."""
        entry = self._circuits.get(key)
        if entry is not None:
            entry.record_success()

    def record_failure(self, key: str) -> None:
        """Record a failed call for the given key."""
        entry = self._circuits.setdefault(key, _CircuitEntry())
        entry.record_failure(self.failure_threshold)

    def stats(self) -> dict[str, dict[str, Any]]:
        """Return per-key circuit state and counters."""
        return {
            key: {
                "state": entry.effective_state(self.recovery_timeout).value,
                "failure_count": entry.failure_count,
                "success_count": entry.success_count,
                "last_failure_at": entry.last_failure_at,
            }
            for key, entry in self._circuits.items()
        }


class _CircuitEntry:
    __slots__ = (
        "state",
        "failure_count",
        "success_count",
        "opened_at",
        "last_failure_at",
        "half_open_calls",
    )

    def __init__(self) -> None:
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.opened_at = 0.0
        self.last_failure_at = 0.0
        self.half_open_calls = 0

    def effective_state(self, recovery_timeout: float) -> CircuitState:
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.opened_at >= recovery_timeout:
                return CircuitState.HALF_OPEN
        return self.state

    def record_success(self) -> None:
        prev_state = self.state
        self.failure_count = 0
        self.success_count += 1
        self.state = CircuitState.CLOSED
        self.half_open_calls = 0
        if prev_state == CircuitState.HALF_OPEN:
            _logger.info("circuit breaker closed after half-open success")

    def record_failure(self, threshold: int) -> None:
        self.failure_count += 1
        self.last_failure_at = time.monotonic()
        if self.failure_count >= threshold:
            was_open = self.state == CircuitState.OPEN
            self.state = CircuitState.OPEN
            self.opened_at = time.monotonic()
            self.half_open_calls = 0
            if not was_open:
                _logger.warning(
                    "circuit breaker opened (failures=%d, threshold=%d)",
                    self.failure_count, threshold,
                )
