"""HTTP-level tests for the validator's OpenAI-compatible client.

Exercises:
  * Successful structured-output call → ProviderResponse with text +
    latency populated.
  * Retry on 429/502/503/504 (bounded by max_retries).
  * Non-retryable 4xx surfaces immediately as ProviderError.
  * Timeout paths surface ProviderTimeout.
  * ``content`` returned as a list of parts joins correctly (some
    OpenAI-compatible providers emit this shape).
"""

from __future__ import annotations

import json

import httpx
import pytest

from validation.validator.eval_config import ProviderConfig
from validation.validator.providers.openai_compatible import (
    OpenAICompatibleClient,
)
from validation.validator.providers.types import (
    ProviderError,
    ProviderResponse,
    ProviderTimeout,
)


pytestmark = pytest.mark.asyncio


def _cfg(**overrides) -> ProviderConfig:
    base = dict(
        base_url="http://provider.test",
        api_key="tok",
        model="test-model",
        timeout_seconds=5.0,
        max_tokens=512,
    )
    base.update(overrides)
    return ProviderConfig(**base)


def _ok_response(content: str | list) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_cost_usd": 0.0123},
        },
    )


async def test_complete_structured_returns_text_and_latency():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _ok_response(json.dumps({"answer": "ok"}))

    transport = httpx.MockTransport(handler)
    client = OpenAICompatibleClient(_cfg(), transport=transport)

    resp = await client.complete_structured(
        system="you are a judge",
        user=json.dumps({"prompt": "foo"}),
        response_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )
    await client.aclose()

    assert isinstance(resp, ProviderResponse)
    assert json.loads(resp.text) == {"answer": "ok"}
    assert resp.latency_ms >= 0
    assert resp.usage_usd == pytest.approx(0.0123)
    assert resp.finish_reason == "stop"

    body = captured["body"]
    assert body["model"] == "test-model"
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 512
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"]["type"] == "object"
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"


async def test_retry_succeeds_after_503():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": "transient"})
        return _ok_response(json.dumps({"ok": True}))

    transport = httpx.MockTransport(handler)
    client = OpenAICompatibleClient(
        _cfg(), transport=transport,
        max_retries=2, backoff_base_seconds=0.001,
    )
    resp = await client.complete_structured(
        system="s", user="u",
        response_schema={"type": "object"},
    )
    await client.aclose()
    assert calls["n"] == 2
    assert json.loads(resp.text) == {"ok": True}


async def test_retry_exhausted_raises_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "always 503"})

    transport = httpx.MockTransport(handler)
    client = OpenAICompatibleClient(
        _cfg(), transport=transport,
        max_retries=2, backoff_base_seconds=0.001,
    )
    with pytest.raises(ProviderError) as exc:
        await client.complete_structured(
            system="s", user="u",
            response_schema={"type": "object"},
        )
    await client.aclose()
    assert "503" in str(exc.value)


async def test_non_retryable_4xx_raises_immediately():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    transport = httpx.MockTransport(handler)
    client = OpenAICompatibleClient(
        _cfg(), transport=transport,
        max_retries=3, backoff_base_seconds=0.001,
    )
    with pytest.raises(ProviderError) as exc:
        await client.complete_structured(
            system="s", user="u",
            response_schema={"type": "object"},
        )
    await client.aclose()
    assert calls["n"] == 1  # NOT retried
    assert "400" in str(exc.value)


async def test_timeout_raises_provider_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated")

    transport = httpx.MockTransport(handler)
    client = OpenAICompatibleClient(
        _cfg(), transport=transport,
        max_retries=1, backoff_base_seconds=0.001,
    )
    with pytest.raises(ProviderTimeout):
        await client.complete_structured(
            system="s", user="u",
            response_schema={"type": "object"},
        )
    await client.aclose()


async def test_content_as_parts_list_joins():
    """Some OpenAI-compatible providers return content as
    [{type: 'text', text: '...'}, ...]. Client must concatenate text."""
    parts = [
        {"type": "text", "text": '{"a":'},
        {"type": "text", "text": ' "b"}'},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response(parts)

    transport = httpx.MockTransport(handler)
    client = OpenAICompatibleClient(_cfg(), transport=transport)
    resp = await client.complete_structured(
        system="s", user="u",
        response_schema={"type": "object"},
    )
    await client.aclose()
    assert json.loads(resp.text) == {"a": "b"}


async def test_unconfigured_raises_at_init():
    with pytest.raises(ProviderError):
        OpenAICompatibleClient(_cfg(api_key=""))
