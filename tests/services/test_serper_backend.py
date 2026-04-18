from __future__ import annotations

import json
import logging

import httpx
import pytest

from tool_platforms.web_search_tool_service.backends import (
    HardBackendError,
    RetryableBackendError,
    SerperBackend,
)


def _transport(handler):
    return httpx.MockTransport(handler)


def _json_response(body, status_code=200):
    return httpx.Response(
        status_code,
        json=body,
        headers={"Content-Type": "application/json"},
    )


CANNED_ORGANIC = [
    {
        "title": "Example Result",
        "link": "https://example.com/page",
        "snippet": "A short description.",
        "position": 1,
        "date": "2026-01-10",
    },
    {
        "title": "Second Result",
        "link": "https://example.com/page2",
        "snippet": "Another description.",
        "position": 2,
    },
]


# -- Happy path --------

async def test_happy_path_returns_documents():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"organic": CANNED_ORGANIC})

    backend = SerperBackend(api_key="sk-test", transport=_transport(handler))
    docs = await backend.search(query="test query", count=5)

    assert len(docs) == 2
    assert docs[0].document_id == "serper-1"
    assert docs[0].title == "Example Result"
    assert docs[0].url == "https://example.com/page"
    assert docs[0].snippet == "A short description."
    assert docs[0].metadata["source"] == "serper"
    assert docs[0].metadata["position"] == 1
    assert docs[0].metadata["date"] == "2026-01-10"
    assert docs[1].document_id == "serper-2"
    assert docs[1].metadata["date"] == ""


# -- Empty results --------

async def test_empty_organic_returns_empty_list():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"organic": []})

    backend = SerperBackend(api_key="sk-test", transport=_transport(handler))
    docs = await backend.search(query="nothing", count=5)
    assert docs == []


# -- Error responses --------

async def test_5xx_raises_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="Bad Gateway")

    backend = SerperBackend(api_key="sk-test", transport=_transport(handler))
    with pytest.raises(RetryableBackendError, match="502"):
        await backend.search(query="test", count=5)


async def test_429_raises_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="Too Many Requests")

    backend = SerperBackend(api_key="sk-test", transport=_transport(handler))
    with pytest.raises(RetryableBackendError, match="429"):
        await backend.search(query="test", count=5)


async def test_401_raises_hard():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Unauthorized")

    backend = SerperBackend(api_key="sk-bad", transport=_transport(handler))
    with pytest.raises(HardBackendError, match="401"):
        await backend.search(query="test", count=5)


async def test_400_raises_hard():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Bad Request")

    backend = SerperBackend(api_key="sk-test", transport=_transport(handler))
    with pytest.raises(HardBackendError, match="400"):
        await backend.search(query="test", count=5)


# -- Timeout --------

async def test_timeout_raises_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out")

    backend = SerperBackend(api_key="sk-test", transport=_transport(handler))
    with pytest.raises(RetryableBackendError):
        await backend.search(query="test", count=5)


# -- Missing 'organic' key --------

async def test_missing_organic_key_raises_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"results": []})

    backend = SerperBackend(api_key="sk-test", transport=_transport(handler))
    with pytest.raises(RetryableBackendError, match="organic"):
        await backend.search(query="test", count=5)


# -- Site filter --------

async def test_site_filter_appended_to_query():
    captured_body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content))
        return _json_response({"organic": CANNED_ORGANIC[:1]})

    backend = SerperBackend(api_key="sk-test", transport=_transport(handler))
    await backend.search(query="test", count=5, site="example.com")

    assert " site:example.com" in captured_body["q"]


# -- Freshness filter --------

async def test_freshness_mapped_to_tbs():
    captured_body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content))
        return _json_response({"organic": []})

    backend = SerperBackend(api_key="sk-test", transport=_transport(handler))
    await backend.search(query="test", count=5, freshness="week")

    assert captured_body["tbs"] == "qdr:w"


# -- Missing API key at construction --------

def test_missing_api_key_raises_at_construction():
    with pytest.raises((HardBackendError, ValueError)):
        SerperBackend(api_key="")


# -- API key not leaked in logs --------

async def test_api_key_not_in_logged_error(caplog):
    secret = "sk-super-secret-key-12345"

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"no_organic": True})

    backend = SerperBackend(api_key=secret, transport=_transport(handler))
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RetryableBackendError):
            await backend.search(query="test", count=5)

    full_log = caplog.text
    assert secret not in full_log
