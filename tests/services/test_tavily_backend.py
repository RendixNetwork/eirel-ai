from __future__ import annotations

import json
import logging

import httpx
import pytest

from tool_platforms.web_search_tool_service.backends import (
    HardBackendError,
    RetryableBackendError,
    TavilyBackend,
)


def _transport(handler):
    return httpx.MockTransport(handler)


def _json_response(body, status_code=200):
    return httpx.Response(
        status_code,
        json=body,
        headers={"Content-Type": "application/json"},
    )


CANNED_RESULTS = [
    {
        "title": "Tavily Result One",
        "url": "https://example.com/tavily1",
        "content": "This is the extracted page content from Tavily for the first result.",
        "score": 0.95,
    },
    {
        "title": "Tavily Result Two",
        "url": "https://example.com/tavily2",
        "content": "Second result content extracted by Tavily.",
        "score": 0.82,
    },
]

CANNED_ANSWER = "Tavily synthesized answer across all results."

CANNED_RESPONSE = {
    "answer": CANNED_ANSWER,
    "results": CANNED_RESULTS,
}


# -- Happy path --------

async def test_happy_path_returns_documents():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(CANNED_RESPONSE)

    backend = TavilyBackend(api_key="tvly-test", transport=_transport(handler))
    docs = await backend.search(query="test query", count=5)

    assert len(docs) == 2
    assert docs[0].document_id == "tavily-0"
    assert docs[0].title == "Tavily Result One"
    assert docs[0].url == "https://example.com/tavily1"
    assert docs[0].snippet == CANNED_RESULTS[0]["content"]
    assert docs[0].metadata["source"] == "tavily"
    assert docs[0].metadata["score"] == 0.95
    assert docs[1].document_id == "tavily-1"
    assert docs[1].metadata["source"] == "tavily"
    assert docs[1].metadata["score"] == 0.82


# -- Answer in first document metadata --------

async def test_answer_in_first_document_metadata_only():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(CANNED_RESPONSE)

    backend = TavilyBackend(api_key="tvly-test", transport=_transport(handler))
    docs = await backend.search(query="test", count=5)

    assert docs[0].metadata["answer"] == CANNED_ANSWER
    assert "answer" not in docs[1].metadata


async def test_no_answer_key_when_answer_is_none():
    response_no_answer = {"results": CANNED_RESULTS}

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(response_no_answer)

    backend = TavilyBackend(api_key="tvly-test", transport=_transport(handler))
    docs = await backend.search(query="test", count=5)

    assert "answer" not in docs[0].metadata


# -- Site filter --------

async def test_site_filter_mapped_to_include_domains():
    captured_body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content))
        return _json_response(CANNED_RESPONSE)

    backend = TavilyBackend(api_key="tvly-test", transport=_transport(handler))
    await backend.search(query="test", count=5, site="example.com")

    assert captured_body["include_domains"] == ["example.com"]


async def test_no_include_domains_when_site_is_none():
    captured_body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content))
        return _json_response(CANNED_RESPONSE)

    backend = TavilyBackend(api_key="tvly-test", transport=_transport(handler))
    await backend.search(query="test", count=5)

    assert "include_domains" not in captured_body


# -- Freshness filter --------

async def test_freshness_mapped_to_time_range():
    captured_body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content))
        return _json_response({"answer": None, "results": []})

    backend = TavilyBackend(api_key="tvly-test", transport=_transport(handler))
    await backend.search(query="test", count=5, freshness="week")

    assert captured_body["time_range"] == "week"


# -- Error responses --------

async def test_5xx_raises_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="Bad Gateway")

    backend = TavilyBackend(api_key="tvly-test", transport=_transport(handler))
    with pytest.raises(RetryableBackendError, match="502"):
        await backend.search(query="test", count=5)


async def test_429_raises_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="Too Many Requests")

    backend = TavilyBackend(api_key="tvly-test", transport=_transport(handler))
    with pytest.raises(RetryableBackendError, match="429"):
        await backend.search(query="test", count=5)


async def test_401_raises_hard():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Unauthorized")

    backend = TavilyBackend(api_key="tvly-bad", transport=_transport(handler))
    with pytest.raises(HardBackendError, match="401"):
        await backend.search(query="test", count=5)


async def test_400_raises_hard():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Bad Request")

    backend = TavilyBackend(api_key="tvly-test", transport=_transport(handler))
    with pytest.raises(HardBackendError, match="400"):
        await backend.search(query="test", count=5)


# -- Timeout --------

async def test_timeout_raises_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out")

    backend = TavilyBackend(api_key="tvly-test", transport=_transport(handler))
    with pytest.raises(RetryableBackendError):
        await backend.search(query="test", count=5)


# -- Missing API key at construction --------

def test_missing_api_key_raises_at_construction():
    with pytest.raises((HardBackendError, ValueError)):
        TavilyBackend(api_key="")


# -- API key not leaked in errors or logs --------

async def test_api_key_not_in_logged_error(caplog):
    secret = "tvly-super-secret-key-12345"

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"no_results": True})

    backend = TavilyBackend(api_key=secret, transport=_transport(handler))
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RetryableBackendError):
            await backend.search(query="test", count=5)

    full_log = caplog.text
    assert secret not in full_log


async def test_api_key_not_in_error_message_on_http_failure():
    secret = "tvly-super-secret-key-99999"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    backend = TavilyBackend(api_key=secret, transport=_transport(handler))
    with pytest.raises(RetryableBackendError) as exc_info:
        await backend.search(query="test", count=5)

    assert secret not in str(exc_info.value)


# -- search_depth passthrough --------

async def test_search_depth_advanced_passthrough():
    captured_body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content))
        return _json_response(CANNED_RESPONSE)

    backend = TavilyBackend(
        api_key="tvly-test",
        transport=_transport(handler),
        search_depth="advanced",
    )
    await backend.search(query="deep research", count=5)

    assert captured_body["search_depth"] == "advanced"


# -- Snippet truncation --------

async def test_snippet_truncated_to_400_chars():
    long_content = "x" * 600
    results = [{"title": "Long", "url": "https://example.com", "content": long_content, "score": 0.9}]

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"results": results})

    backend = TavilyBackend(api_key="tvly-test", transport=_transport(handler))
    docs = await backend.search(query="test", count=5)

    assert len(docs[0].snippet) == 400


# -- Request body structure --------

async def test_api_key_in_request_body_not_header():
    captured_headers: dict = {}
    captured_body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        captured_body.update(json.loads(request.content))
        return _json_response(CANNED_RESPONSE)

    backend = TavilyBackend(api_key="tvly-test", transport=_transport(handler))
    await backend.search(query="test", count=5)

    assert captured_body["api_key"] == "tvly-test"
    assert "x-api-key" not in captured_headers
    assert "authorization" not in captured_headers
