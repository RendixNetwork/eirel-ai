"""HTTP-level tests for the validator's Gemini client.

Exercises:
  * Successful generateContent call → ProviderResponse with text and
    finish_reason populated.
  * Retry on 429/502/503/504; non-retryable 4xx surfaces immediately.
  * Timeout → ProviderTimeout.
  * Safety-block (no candidates + promptFeedback.blockReason) surfaces
    a ProviderError with the block reason.
  * Request shape: ``contents`` + ``systemInstruction`` + camelCase
    ``responseSchema`` + ``responseMimeType: application/json``.
"""

from __future__ import annotations

import json

import httpx
import pytest

from validation.validator.eval_config import ProviderConfig
from validation.validator.providers.gemini import (
    DEFAULT_GEMINI_BASE_URL,
    GeminiClient,
)
from validation.validator.providers.types import (
    ProviderError,
    ProviderResponse,
    ProviderTimeout,
)


pytestmark = pytest.mark.asyncio


def _cfg(**overrides) -> ProviderConfig:
    base = dict(
        base_url=DEFAULT_GEMINI_BASE_URL,
        api_key="tok",
        model="gemini-3.1-pro-preview",
        timeout_seconds=5.0,
        max_tokens=512,
    )
    base.update(overrides)
    return ProviderConfig(**base)


def _ok_response(text: str, finish_reason: str = "STOP") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {"parts": [{"text": text}]},
                    "finishReason": finish_reason,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100, "candidatesTokenCount": 50,
            },
        },
    )


async def test_complete_structured_request_shape():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _ok_response(json.dumps({"answer": "42"}))

    transport = httpx.MockTransport(handler)
    client = GeminiClient(_cfg(), transport=transport)

    resp = await client.complete_structured(
        system="you are a judge",
        user='{"prompt": "foo"}',
        response_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    )
    await client.aclose()

    assert isinstance(resp, ProviderResponse)
    assert json.loads(resp.text) == {"answer": "42"}
    assert resp.finish_reason == "STOP"
    # gemini-3.1-pro-preview: 100 prompt × $2/Mtok + 50 output × $12/Mtok
    # = 0.0002 + 0.0006 = 0.0008 (computed client-side from rate card).
    assert resp.usage_usd == pytest.approx(0.0008)

    # URL: ?key=... + :generateContent suffix
    assert "/models/gemini-3.1-pro-preview:generateContent" in captured["url"]
    assert "key=tok" in captured["url"]
    body = captured["body"]
    # System prompt → top-level systemInstruction (NOT a role)
    assert body["systemInstruction"]["parts"][0]["text"] == "you are a judge"
    # User content nested in contents[].parts[]
    assert body["contents"][0]["role"] == "user"
    assert body["contents"][0]["parts"][0]["text"] == '{"prompt": "foo"}'
    # JSON-mode lives in generationConfig with camelCase keys
    cfg = body["generationConfig"]
    assert cfg["responseMimeType"] == "application/json"
    assert cfg["responseSchema"]["type"] == "object"
    assert cfg["temperature"] == 0.0
    assert cfg["maxOutputTokens"] == 512


async def test_retry_succeeds_after_503():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": {"message": "transient"}})
        return _ok_response('{"ok":true}')

    transport = httpx.MockTransport(handler)
    client = GeminiClient(
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
        return httpx.Response(503, json={"error": {"message": "always 503"}})

    transport = httpx.MockTransport(handler)
    client = GeminiClient(
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
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    transport = httpx.MockTransport(handler)
    client = GeminiClient(
        _cfg(), transport=transport,
        max_retries=3, backoff_base_seconds=0.001,
    )
    with pytest.raises(ProviderError) as exc:
        await client.complete_structured(
            system="s", user="u",
            response_schema={"type": "object"},
        )
    await client.aclose()
    assert calls["n"] == 1
    assert "400" in str(exc.value)


async def test_timeout_raises_provider_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated")

    transport = httpx.MockTransport(handler)
    client = GeminiClient(
        _cfg(), transport=transport,
        max_retries=1, backoff_base_seconds=0.001,
    )
    with pytest.raises(ProviderTimeout):
        await client.complete_structured(
            system="s", user="u",
            response_schema={"type": "object"},
        )
    await client.aclose()


async def test_safety_block_surfaces_block_reason():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [],
                "promptFeedback": {"blockReason": "SAFETY"},
            },
        )

    transport = httpx.MockTransport(handler)
    client = GeminiClient(_cfg(), transport=transport)
    with pytest.raises(ProviderError) as exc:
        await client.complete_structured(
            system="s", user="u",
            response_schema={"type": "object"},
        )
    await client.aclose()
    assert "SAFETY" in str(exc.value)


async def test_unconfigured_raises_at_init():
    with pytest.raises(ProviderError):
        GeminiClient(_cfg(api_key=""))
