from __future__ import annotations

import asyncio

import pytest

from tool_platforms.web_search_tool_service.backends import (
    AllBackendsFailedError,
    FallbackSearchBackend,
    HardBackendError,
    RetryableBackendError,
)
from tool_platforms.web_search_tool_service.models import SearchDocument


def _doc(doc_id: str = "d1") -> SearchDocument:
    return SearchDocument(
        document_id=doc_id, title="T", url="http://x", snippet="S", score=1.0,
    )


class _SuccessBackend:
    def __init__(self, name: str, documents: list[SearchDocument] | None = None) -> None:
        self.name = name
        self.call_count = 0
        self._documents = documents or []

    async def search(
        self, *, query: str, count: int, freshness: str | None = None, site: str | None = None,
    ) -> list[SearchDocument]:
        self.call_count += 1
        return self._documents


class _RetryableFailBackend:
    def __init__(self, name: str, message: str = "fail") -> None:
        self.name = name
        self._message = message

    async def search(
        self, *, query: str, count: int, freshness: str | None = None, site: str | None = None,
    ) -> list[SearchDocument]:
        raise RetryableBackendError(self._message)


class _HardFailBackend:
    def __init__(self, name: str, message: str = "hard fail") -> None:
        self.name = name
        self._message = message

    async def search(
        self, *, query: str, count: int, freshness: str | None = None, site: str | None = None,
    ) -> list[SearchDocument]:
        raise HardBackendError(self._message)


class _SlowBackend:
    def __init__(self, name: str, delay: float) -> None:
        self.name = name

        self._delay = delay

    async def search(
        self, *, query: str, count: int, freshness: str | None = None, site: str | None = None,
    ) -> list[SearchDocument]:
        await asyncio.sleep(self._delay)
        return [_doc()]


async def test_fallback_primary_success():
    docs = [_doc()]
    primary = _SuccessBackend("primary", docs)
    secondary = _SuccessBackend("secondary")
    fb = FallbackSearchBackend([primary, secondary])
    result = await fb.search(query="test", count=5)
    assert result.documents == docs
    assert result.backend_name == "primary"
    assert result.attempted == ["primary"]
    assert result.latency_ms >= 0
    assert secondary.call_count == 0


async def test_fallback_primary_retryable_falls_to_secondary():
    docs = [_doc()]
    primary = _RetryableFailBackend("primary")
    secondary = _SuccessBackend("secondary", docs)
    fb = FallbackSearchBackend([primary, secondary])
    result = await fb.search(query="test", count=5)
    assert result.documents == docs
    assert result.backend_name == "secondary"
    assert result.attempted == ["primary", "secondary"]


async def test_fallback_primary_timeout_falls_to_secondary():
    docs = [_doc()]
    primary = _SlowBackend("primary", delay=5.0)
    secondary = _SuccessBackend("secondary", docs)
    fb = FallbackSearchBackend([primary, secondary], per_backend_timeout=0.05)
    result = await fb.search(query="test", count=5)
    assert result.documents == docs
    assert result.backend_name == "secondary"
    assert result.attempted == ["primary", "secondary"]


async def test_fallback_hard_error_no_fallback():
    primary = _HardFailBackend("primary")
    secondary = _SuccessBackend("secondary")
    fb = FallbackSearchBackend([primary, secondary])
    with pytest.raises(HardBackendError, match="hard fail"):
        await fb.search(query="test", count=5)
    assert secondary.call_count == 0


async def test_fallback_all_backends_fail():
    b1 = _RetryableFailBackend("b1", "err1")
    b2 = _RetryableFailBackend("b2", "err2")
    b3 = _RetryableFailBackend("b3", "err3")
    fb = FallbackSearchBackend([b1, b2, b3])
    with pytest.raises(AllBackendsFailedError) as exc_info:
        await fb.search(query="test", count=5)
    assert exc_info.value.failures == [("b1", "err1"), ("b2", "err2"), ("b3", "err3")]


async def test_fallback_deterministic_ordering():
    call_order: list[str] = []

    class _OrderRecorder:
        def __init__(self, name: str) -> None:
            self.name = name

        async def search(
            self, *, query: str, count: int, freshness: str | None = None, site: str | None = None,
        ) -> list[SearchDocument]:
            call_order.append(self.name)
            raise RetryableBackendError("fail")

    fb = FallbackSearchBackend([_OrderRecorder("a"), _OrderRecorder("b"), _OrderRecorder("c")])
    with pytest.raises(AllBackendsFailedError):
        await fb.search(query="test", count=5)
    assert call_order == ["a", "b", "c"]
