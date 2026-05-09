"""Per-vendor oracle client tests.

Exercises the OpenAI / Gemini / Grok oracle wrappers against an
httpx-mocked provider client. Verifies:
  * Successful structured response → OracleGrounding(status="ok").
  * Provider errors / timeouts → status="error" with diagnostic msg.
  * Gemini safety-block surfaces as status="blocked".
  * Malformed JSON in the answer field → status="error".
"""

from __future__ import annotations

import json

import httpx
import pytest

from validation.validator.eval_config import ProviderConfig
from validation.validator.oracles.base import OracleContext
from validation.validator.oracles.gemini_oracle import GeminiOracle
from validation.validator.oracles.grok_oracle import GrokOracle
from validation.validator.oracles.openai_oracle import OpenAIOracle
from validation.validator.providers.gemini import (
    DEFAULT_GEMINI_BASE_URL,
    GeminiClient,
)
from validation.validator.providers.openai_compatible import (
    OpenAICompatibleClient,
)


pytestmark = pytest.mark.asyncio


def _openai_cfg() -> ProviderConfig:
    # Use the production host so vendor detection in
    # ``cost_calc.vendor_from_base_url`` picks the OpenAI extractor.
    # The httpx MockTransport in each test intercepts the request
    # regardless of the real host.
    return ProviderConfig(
        base_url="https://api.openai.com/v1",
        api_key="tok",
        model="gpt-5.4",
        timeout_seconds=5.0,
        max_tokens=512,
    )


def _grok_cfg() -> ProviderConfig:
    return ProviderConfig(
        base_url="https://api.x.ai/v1",
        api_key="tok",
        model="grok-4.3",
        timeout_seconds=5.0,
        max_tokens=512,
    )


def _gemini_cfg() -> ProviderConfig:
    return ProviderConfig(
        base_url=DEFAULT_GEMINI_BASE_URL,
        api_key="tok",
        model="gemini-3.1-pro-preview",
        timeout_seconds=5.0,
        max_tokens=512,
    )


def _ok_openai(answer: str) -> httpx.Response:
    """OpenAI/Grok ``/v1/responses`` payload shape (Responses API).

    Both vendors share the same wire format here — a top-level
    ``output`` array of ``{type: message, status, content: [...]}``
    items, where each content part of type ``output_text`` carries the
    text. See ``openai_compatible.py:_parse_responses_api_response``.

    The ``usage`` block uses Responses-API token-count keys
    (``input_tokens`` / ``output_tokens``) — cost extraction
    multiplies by the gpt-5.4 rate card (5/Mtok input, 30/Mtok
    output) for a deterministic value the tests can assert on.
    """
    return httpx.Response(
        200,
        json={
            "output": [
                {
                    "type": "message",
                    "status": "stop",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"answer": answer}),
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        },
    )


def _ok_gemini(answer: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [{
                "content": {"parts": [{"text": json.dumps({"answer": answer})}]},
                "finishReason": "STOP",
            }],
        },
    )


def _ctx() -> OracleContext:
    return OracleContext(
        task_id="t1",
        prompt="What is the capital of France?",
    )


# -- OpenAI oracle ---------------------------------------------------------


async def test_openai_oracle_ok():
    transport = httpx.MockTransport(lambda req: _ok_openai("Paris"))
    client = OpenAICompatibleClient(_openai_cfg(), transport=transport)
    oracle = OpenAIOracle(client=client)

    g = await oracle.produce_grounding(_ctx())
    await oracle.aclose()

    assert g.vendor == "openai"
    assert g.status == "ok"
    assert g.raw_text == "Paris"
    # gpt-5.4 short tier: 100 input × $2.5/Mtok + 50 output × $15/Mtok
    # = 0.00025 + 0.00075 = 0.001
    assert g.cost_usd == pytest.approx(0.001)
    assert g.finish_reason == "stop"


async def test_openai_oracle_timeout_surfaces_error():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated")

    transport = httpx.MockTransport(handler)
    client = OpenAICompatibleClient(
        _openai_cfg(), transport=transport,
        max_retries=0, backoff_base_seconds=0.001,
    )
    oracle = OpenAIOracle(client=client)
    g = await oracle.produce_grounding(_ctx())
    await oracle.aclose()

    assert g.vendor == "openai"
    assert g.status == "error"
    assert "timeout" in (g.error_msg or "")


async def test_openai_oracle_malformed_answer_surfaces_error():
    """Server returned a valid Responses-API envelope but the
    ``output_text`` is non-JSON garbage — oracle parses ``raw_text``
    successfully but fails the JSON-shape extraction step."""
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "status": "stop",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "not even close to JSON",
                                "annotations": [],
                            }
                        ],
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    client = OpenAICompatibleClient(_openai_cfg(), transport=transport)
    oracle = OpenAIOracle(client=client)
    g = await oracle.produce_grounding(_ctx())
    await oracle.aclose()

    assert g.status == "error"
    assert "malformed_response" in (g.error_msg or "")


async def test_openai_oracle_missing_answer_field_surfaces_error():
    """JSON parses fine but the required ``answer`` field is absent."""
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "status": "stop",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps({"foo": "bar"}),
                                "annotations": [],
                            }
                        ],
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    client = OpenAICompatibleClient(_openai_cfg(), transport=transport)
    oracle = OpenAIOracle(client=client)
    g = await oracle.produce_grounding(_ctx())
    await oracle.aclose()

    assert g.status == "error"


# -- Grok oracle (same shape as OpenAI) ------------------------------------


async def test_grok_oracle_ok():
    transport = httpx.MockTransport(lambda req: _ok_openai("Paris"))
    client = OpenAICompatibleClient(_grok_cfg(), transport=transport)
    oracle = GrokOracle(client=client)
    g = await oracle.produce_grounding(_ctx())
    await oracle.aclose()

    assert g.vendor == "grok"
    assert g.status == "ok"
    assert g.raw_text == "Paris"


# -- Gemini oracle ---------------------------------------------------------


async def test_gemini_oracle_ok():
    transport = httpx.MockTransport(lambda req: _ok_gemini("Paris"))
    client = GeminiClient(_gemini_cfg(), transport=transport)
    oracle = GeminiOracle(client=client)

    g = await oracle.produce_grounding(_ctx())
    await oracle.aclose()

    assert g.vendor == "gemini"
    assert g.status == "ok"
    assert g.raw_text == "Paris"
    assert g.finish_reason == "STOP"


async def test_gemini_oracle_safety_block_surfaces_blocked_status():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [],
                "promptFeedback": {"blockReason": "SAFETY"},
            },
        )

    transport = httpx.MockTransport(handler)
    client = GeminiClient(_gemini_cfg(), transport=transport)
    oracle = GeminiOracle(client=client)
    g = await oracle.produce_grounding(_ctx())
    await oracle.aclose()

    assert g.vendor == "gemini"
    assert g.status == "blocked"
    assert "SAFETY" in (g.error_msg or "")
