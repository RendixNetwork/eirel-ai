"""Vendor-aware exact-cost extractor tests.

Each vendor's response payload has been hand-built to mirror the
shape returned by the real provider in May 2026 — these are the
shapes ``OpenAICompatibleClient._parse_responses_api_response`` and
``GeminiClient._parse_response`` actually parse, so test fidelity to
the wire format is what matters here.
"""

from __future__ import annotations

import pytest

from validation.validator.providers.cost_calc import (
    extract_chutes_chat_cost,
    extract_gemini_generate_cost,
    extract_grok_responses_cost,
    extract_openai_compatible_chat_cost,
    extract_openai_responses_cost,
    vendor_from_base_url,
)


# -- vendor_from_base_url --------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://api.openai.com/v1", "openai"),
        ("https://api.x.ai/v1", "xai"),
        ("https://llm.chutes.ai/v1", "chutes"),
        ("https://generativelanguage.googleapis.com/v1beta", "gemini"),
        ("https://my-self-hosted.example/v1", "unknown"),
        ("", "unknown"),
    ],
)
def test_vendor_from_base_url(url: str, expected: str) -> None:
    assert vendor_from_base_url(url) == expected


# -- OpenAI Responses API --------------------------------------------------


def test_openai_responses_cost_token_only() -> None:
    """gpt-5.4 short tier: input $2.50/Mtok, output $15/Mtok.
    1000 * 2.5 / 1M + 500 * 15 / 1M = 0.0025 + 0.0075 = 0.01"""
    payload = {
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
        },
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "ok"}]},
        ],
    }
    cost = extract_openai_responses_cost(payload, model="gpt-5.4")
    assert cost == pytest.approx(0.01)


def test_openai_responses_cost_with_cached_input() -> None:
    """gpt-5.4 short tier: cached $0.25/Mtok, uncached $2.50/Mtok,
    output $15/Mtok. 500 cached → $0.000125; 500 uncached → $0.00125;
    200 output → $0.003. Total = $0.004375"""
    payload = {
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 200,
            "input_tokens_details": {"cached_tokens": 500},
        },
        "output": [],
    }
    cost = extract_openai_responses_cost(payload, model="gpt-5.4")
    assert cost == pytest.approx(0.004375)


def test_openai_responses_cost_with_web_search() -> None:
    """gpt-5.4 short tier + 2 web_search_call items. Web search adds
    $10/1k regardless of token count."""
    payload = {
        "usage": {"input_tokens": 1000, "output_tokens": 0},
        "output": [
            {"type": "web_search_call", "action": {"type": "search"}},
            {"type": "web_search_call", "action": {"type": "search"}},
            {"type": "message", "content": []},
        ],
    }
    # token cost = 1000 * 2.5 / 1M = $0.0025; web = 2 * 0.01 = $0.02
    # total = 0.0225
    cost = extract_openai_responses_cost(payload, model="gpt-5.4")
    assert cost == pytest.approx(0.0225)


def test_openai_responses_cost_missing_usage() -> None:
    assert extract_openai_responses_cost({}, model="gpt-5.4") is None
    assert extract_openai_responses_cost({"usage": "not a dict"}, model="gpt-5.4") is None


# -- xAI Grok Responses API ------------------------------------------------


def test_grok_responses_cost_uses_ticks() -> None:
    """37756000 ticks / 10^10 = $0.0037756 — official xAI example value."""
    payload = {
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cost_in_usd_ticks": 37_756_000,
        },
    }
    cost = extract_grok_responses_cost(payload, model="grok-4.3")
    assert cost == pytest.approx(0.0037756)


def test_grok_responses_cost_falls_back_to_token_rate() -> None:
    """Without ``cost_in_usd_ticks``, fall back to the static rate
    table. Short-context (≤200K input): xai:grok-4.3 = $1.25/$2.50 per Mtok."""
    payload = {"usage": {"prompt_tokens": 100_000, "completion_tokens": 100_000}}
    cost = extract_grok_responses_cost(payload, model="grok-4.3")
    # 100K * 1.25 / 1M + 100K * 2.5 / 1M = 0.125 + 0.25 = $0.375
    assert cost == pytest.approx(0.375)


def test_grok_responses_cost_long_context_doubles() -> None:
    """xAI Grok 4.3: above 200K input tokens, both input + output
    rates double (1.25→2.50 input, 2.50→5.00 output)."""
    payload = {"usage": {
        "prompt_tokens": 250_000, "completion_tokens": 50_000,
    }}
    cost = extract_grok_responses_cost(payload, model="grok-4.3")
    # 250K * 2.5 / 1M + 50K * 5.0 / 1M = 0.625 + 0.25 = $0.875
    assert cost == pytest.approx(0.875)


def test_openai_responses_cost_long_context_at_272k() -> None:
    """gpt-5.4 long-context kicks in at >272K input. At exactly 272K
    the call still bills at the short rate (strict-greater)."""
    short_payload = {
        "usage": {"input_tokens": 272_000, "output_tokens": 1000},
    }
    short_cost = extract_openai_responses_cost(
        short_payload, model="gpt-5.4",
    )
    # 272K * 2.5 / 1M + 1K * 15 / 1M = 0.68 + 0.015 = $0.695
    assert short_cost == pytest.approx(0.695)

    long_payload = {
        "usage": {"input_tokens": 272_001, "output_tokens": 1000},
    }
    long_cost = extract_openai_responses_cost(
        long_payload, model="gpt-5.4",
    )
    # 272001 * 5 / 1M + 1K * 22.5 / 1M = 1.360005 + 0.0225 = $1.382505
    assert long_cost == pytest.approx(1.382505)


def test_openai_responses_cost_long_context_with_cached_input() -> None:
    """Cached-input rate also jumps tiers at 272K: $0.25 → $0.50 /Mtok."""
    payload = {
        "usage": {
            "input_tokens": 300_000,
            "output_tokens": 0,
            "input_tokens_details": {"cached_tokens": 100_000},
        },
        "output": [],
    }
    # 200K uncached × $5/Mtok + 100K cached × $0.50/Mtok = 1.00 + 0.05 = $1.05
    cost = extract_openai_responses_cost(payload, model="gpt-5.4")
    assert cost == pytest.approx(1.05)


def test_grok_responses_cost_zero_ticks_falls_through() -> None:
    """Tick value of 0 / negative shouldn't be treated as authoritative —
    fall back to token rate (so a misconfigured upstream doesn't show
    free calls)."""
    payload = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 100,
            "cost_in_usd_ticks": 0,
        },
    }
    cost = extract_grok_responses_cost(payload, model="grok-4.3")
    # Token fallback: 100*1.25/1M + 100*2.5/1M = 0.000125 + 0.00025 = 0.000375
    assert cost == pytest.approx(0.000375)


# -- Gemini generateContent ------------------------------------------------


def test_gemini_cost_token_only() -> None:
    """gemini-3.1-pro-preview: 2/12/Mtok. 1000 prompt + 500 candidates
    = 1000*2/1M + 500*12/1M = 0.002 + 0.006 = 0.008"""
    payload = {
        "usageMetadata": {
            "promptTokenCount": 1000,
            "candidatesTokenCount": 500,
        },
    }
    cost = extract_gemini_generate_cost(
        payload, model="gemini-3.1-pro-preview",
    )
    assert cost == pytest.approx(0.008)


def test_gemini_cost_with_search_queries() -> None:
    """Each query in webSearchQueries adds $0.014. Three queries =
    $0.042 on top of token cost."""
    payload = {
        "usageMetadata": {
            "promptTokenCount": 1000,
            "candidatesTokenCount": 0,
        },
        "candidates": [{
            "content": {"parts": []},
            "groundingMetadata": {
                "webSearchQueries": ["q1", "q2", "q3"],
            },
        }],
    }
    cost = extract_gemini_generate_cost(
        payload, model="gemini-3.1-pro-preview",
    )
    # 1000 * 2 / 1M = 0.002 + 3 * 0.014 = 0.044
    assert cost == pytest.approx(0.044)


def test_gemini_cost_long_context_doubles() -> None:
    """Above 200K tokens, both rates jump to the long-context tier
    ($2→$4 input, $12→$18 output). Tier selection lives in
    ``llm_cost_for`` so the extractor stays generic."""
    payload = {
        "usageMetadata": {
            "promptTokenCount": 200_001,
            "candidatesTokenCount": 1000,
        },
    }
    cost = extract_gemini_generate_cost(
        payload, model="gemini-3.1-pro-preview",
    )
    # 200_001 * 4 / 1M + 1000 * 18 / 1M = 0.800004 + 0.018 = 0.818004
    assert cost == pytest.approx(0.818004)


def test_gemini_cost_at_threshold_stays_short_tier() -> None:
    """Strict-greater threshold: exactly 200K tokens → short rates."""
    payload = {
        "usageMetadata": {
            "promptTokenCount": 200_000,
            "candidatesTokenCount": 1000,
        },
    }
    cost = extract_gemini_generate_cost(
        payload, model="gemini-3.1-pro-preview",
    )
    # 200K * 2 / 1M + 1K * 12 / 1M = 0.4 + 0.012 = 0.412
    assert cost == pytest.approx(0.412)


def test_gemini_cost_with_cached_content() -> None:
    """Cached tokens at the cached rate ($0.20/Mtok = 0.1× input)."""
    payload = {
        "usageMetadata": {
            "promptTokenCount": 1000,
            "candidatesTokenCount": 0,
            "cachedContentTokenCount": 1000,
        },
    }
    # All 1000 prompt tokens are cached at $0.20/Mtok = $0.0002
    cost = extract_gemini_generate_cost(
        payload, model="gemini-3.1-pro-preview",
    )
    assert cost == pytest.approx(0.0002)


def test_gemini_cost_missing_metadata() -> None:
    assert extract_gemini_generate_cost({}, model="gemini-3-pro") is None


# -- Chutes chat-completions -----------------------------------------------


def test_chutes_chat_cost_known_model() -> None:
    """zai-org/GLM-5.1-TEE static rate is 0.5/2.0 /Mtok.
    1M prompt + 1M completion = 0.5 + 2.0 = $2.5"""
    payload = {
        "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
    }
    cost = extract_chutes_chat_cost(payload, model="zai-org/GLM-5.1-TEE")
    assert cost == pytest.approx(2.5)


def test_chutes_chat_cost_unknown_model_falls_back_to_glob() -> None:
    """``chutes:*`` glob entry is 0.5/2.0 — same as GLM. So unknown
    model with the same tokens = same cost."""
    payload = {"usage": {"prompt_tokens": 1000, "completion_tokens": 500}}
    cost = extract_chutes_chat_cost(payload, model="some-unreleased-model")
    # 1000 * 0.5 / 1M + 500 * 2 / 1M = 0.0005 + 0.001 = 0.0015
    assert cost == pytest.approx(0.0015)


# -- Dispatcher ------------------------------------------------------------


def test_dispatcher_routes_to_chutes() -> None:
    payload = {"usage": {"prompt_tokens": 1000, "completion_tokens": 500}}
    cost = extract_openai_compatible_chat_cost(
        payload,
        base_url="https://llm.chutes.ai/v1",
        model="zai-org/GLM-5.1-TEE",
    )
    # 1000 * 0.5 / 1M + 500 * 2 / 1M = 0.0015
    assert cost == pytest.approx(0.0015)


def test_dispatcher_routes_to_xai() -> None:
    payload = {"usage": {"prompt_tokens": 1000, "completion_tokens": 500,
                          "cost_in_usd_ticks": 158_500}}
    cost = extract_openai_compatible_chat_cost(
        payload, base_url="https://api.x.ai/v1", model="grok-4.3",
    )
    # 158500 / 10^10 = 0.00001585 — official xAI example.
    assert cost == pytest.approx(1.585e-5)


def test_dispatcher_routes_to_openai_chat_completions() -> None:
    """OpenAI chat-completions has different ``usage`` keys
    (``prompt_tokens`` / ``completion_tokens``) than Responses API."""
    payload = {
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "prompt_tokens_details": {"cached_tokens": 200},
        },
    }
    cost = extract_openai_compatible_chat_cost(
        payload, base_url="https://api.openai.com/v1", model="gpt-5.4",
    )
    # 800 uncached * $2.5/Mtok + 200 cached * $0.25/Mtok + 500 * $15/Mtok
    # = 0.002 + 0.00005 + 0.0075 = 0.00955
    assert cost == pytest.approx(0.00955)


def test_dispatcher_unknown_host_reads_total_cost_usd() -> None:
    """Self-hosted / unknown vendor: trust ``usage.total_cost_usd``
    when provided, else None — preserves the legacy behavior."""
    payload = {"usage": {"total_cost_usd": 0.42}}
    cost = extract_openai_compatible_chat_cost(
        payload, base_url="https://my.example/v1", model="anything",
    )
    assert cost == pytest.approx(0.42)
