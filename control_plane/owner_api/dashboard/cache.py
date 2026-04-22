from __future__ import annotations

import time
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, default_ttl_seconds: float = 30.0) -> None:
        self._default_ttl = default_ttl_seconds
        self._store: dict[tuple[Any, ...], tuple[float, T]] = {}

    def get(self, key: tuple[Any, ...]) -> T | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at < time.monotonic():
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: tuple[Any, ...], value: T, ttl: float | None = None) -> None:
        expires_at = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
        self._store[key] = (expires_at, value)

    def clear(self) -> None:
        self._store.clear()
