"""Tests for the OpenAI Responses API baseline client."""

from __future__ import annotations

import pytest
import httpx

from validation.validator.openai_baseline import (
    OpenAIBaselineClient,
    OpenAIBaselineError,
)


def _mock_transport(
    response_body: dict | None = None,
    status_code: int = 200,
) -> httpx.MockTransport:
    if response_body is None:
        response_body = _default_response_body()

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=response_body)

    return httpx.MockTransport(_handler)


def _default_response_body() -> dict:
    return {
        "id": "resp_abc123",
        "output": [
            {"type": "web_search_call", "status": "completed"},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "The speed of light is 299,792,458 m/s.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://example.com/physics",
                                "title": "Physics Constants",
                                "start_index": 23,
                                "end_index": 43,
                            }
                        ],
                    }
                ],
            },
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 50},
    }


async def test_generate_returns_normalized_response():
    client = OpenAIBaselineClient(
        api_key="sk-test",
        model="gpt-5",
        transport=_mock_transport(),
    )
    result = await client.generate(prompt="What is the speed of light?")
    assert "299,792,458" in result.response_text
    assert len(result.citations) == 1
    assert result.citations[0]["url"] == "https://example.com/physics"
    assert result.model == "gpt-5"
    assert result.latency_seconds >= 0.0
    await client.aclose()


async def test_generate_raises_without_api_key():
    import os

    # Explicitly pass empty api_key and also clear env to defeat fallback
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        client = OpenAIBaselineClient(api_key="", transport=_mock_transport())
        with pytest.raises(OpenAIBaselineError, match="OPENAI_API_KEY is not set"):
            await client.generate(prompt="x")
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved


async def test_generate_raises_on_http_error():
    client = OpenAIBaselineClient(
        api_key="sk-test",
        transport=_mock_transport(response_body={"error": "down"}, status_code=500),
    )
    with pytest.raises(OpenAIBaselineError, match="status=500"):
        await client.generate(prompt="x")
    await client.aclose()


async def test_budget_guard_stops_when_exhausted():
    # Budget cap lower than one call's estimated cost
    client = OpenAIBaselineClient(
        api_key="sk-test",
        max_cost_usd_per_run=0.001,
        transport=_mock_transport(),
    )
    with pytest.raises(OpenAIBaselineError, match="budget exhausted"):
        await client.generate(prompt="x")


async def test_extracts_multiple_citations():
    body = _default_response_body()
    body["output"][1]["content"][0]["annotations"] = [
        {"type": "url_citation", "url": "https://a.com", "title": "A", "start_index": 0, "end_index": 1},
        {"type": "url_citation", "url": "https://b.com", "title": "B", "start_index": 2, "end_index": 3},
    ]
    client = OpenAIBaselineClient(
        api_key="sk-test",
        transport=_mock_transport(response_body=body),
    )
    result = await client.generate(prompt="x")
    urls = [c["url"] for c in result.citations]
    assert urls == ["https://a.com", "https://b.com"]
    await client.aclose()


async def test_handles_empty_output_array():
    body = {"id": "resp_1", "output": [], "usage": {}}
    client = OpenAIBaselineClient(
        api_key="sk-test",
        transport=_mock_transport(response_body=body),
    )
    result = await client.generate(prompt="x")
    assert result.response_text == ""
    assert result.citations == []
    await client.aclose()


async def test_spent_usd_tracks_across_calls():
    client = OpenAIBaselineClient(
        api_key="sk-test",
        max_cost_usd_per_run=10.0,
        transport=_mock_transport(),
    )
    await client.generate(prompt="one")
    spent_after_first = client.spent_usd
    await client.generate(prompt="two")
    assert client.spent_usd > spent_after_first
    await client.aclose()
