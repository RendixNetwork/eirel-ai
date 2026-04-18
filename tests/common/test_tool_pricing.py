from __future__ import annotations

import pytest

from shared.common.tool_pricing import LLM_PRICING, LLMPrice, llm_cost_for


def test_llm_cost_for_anthropic_opus():
    cost = llm_cost_for(
        provider="anthropic",
        model="claude-opus-4-6",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert cost == pytest.approx(15.0 + 75.0)


def test_llm_cost_for_anthropic_sonnet():
    cost = llm_cost_for(
        provider="anthropic",
        model="claude-sonnet-4-6",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert cost == pytest.approx(3.0 + 15.0)


def test_llm_cost_for_anthropic_haiku():
    cost = llm_cost_for(
        provider="anthropic",
        model="claude-haiku-4-5",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert cost == pytest.approx(1.0 + 5.0)


def test_llm_cost_for_openai_gpt4o():
    cost = llm_cost_for(
        provider="openai",
        model="gpt-4o",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert cost == pytest.approx(2.5 + 10.0)


def test_llm_cost_for_openai_gpt4o_mini():
    cost = llm_cost_for(
        provider="openai",
        model="gpt-4o-mini",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert cost == pytest.approx(0.15 + 0.60)


def test_llm_cost_for_unknown_model_falls_through_to_wildcard():
    cost = llm_cost_for(
        provider="openrouter",
        model="some-unknown-model",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    fallback = LLM_PRICING["openrouter:*"]
    expected = fallback.input_per_mtok_usd + fallback.output_per_mtok_usd
    assert cost == pytest.approx(expected)


def test_llm_cost_for_chutes_wildcard_fallback():
    cost = llm_cost_for(
        provider="chutes",
        model="anything",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    fallback = LLM_PRICING["chutes:*"]
    expected = fallback.input_per_mtok_usd + fallback.output_per_mtok_usd
    assert cost == pytest.approx(expected)


def test_llm_cost_for_completely_unknown_provider_returns_zero():
    cost = llm_cost_for(
        provider="unknown_provider",
        model="unknown_model",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert cost == 0.0


def test_reasoning_tokens_priced_at_reasoning_rate():
    cost = llm_cost_for(
        provider="anthropic",
        model="claude-opus-4-6",
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=1_000_000,
    )
    price = LLM_PRICING["anthropic:claude-opus-4-6"]
    assert cost == pytest.approx(price.reasoning_per_mtok_usd)


def test_reasoning_tokens_fall_back_to_output_rate_when_reasoning_rate_is_zero():
    price = LLM_PRICING["openai:gpt-4o"]
    assert price.reasoning_per_mtok_usd == 0.0
    cost = llm_cost_for(
        provider="openai",
        model="gpt-4o",
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=1_000_000,
    )
    assert cost == pytest.approx(price.output_per_mtok_usd)


def test_llm_cost_for_scales_with_token_count():
    cost_1k = llm_cost_for(
        provider="anthropic",
        model="claude-sonnet-4-6",
        prompt_tokens=1_000,
        completion_tokens=500,
    )
    cost_10k = llm_cost_for(
        provider="anthropic",
        model="claude-sonnet-4-6",
        prompt_tokens=10_000,
        completion_tokens=5_000,
    )
    assert cost_10k == pytest.approx(cost_1k * 10)


def test_llm_cost_for_zero_tokens_returns_zero():
    cost = llm_cost_for(
        provider="anthropic",
        model="claude-opus-4-6",
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
    )
    assert cost == 0.0


def test_llm_cost_for_chutes_kimi_uses_per_model_rate_not_fallback():
    # Canonical Chutes rate for moonshotai/Kimi-K2.5-TEE from
    # https://llm.chutes.ai/v1/models — $0.3827/M input, $1.72/M output.
    # Verifies the per-model entry in LLM_PRICING beats the ``chutes:*``
    # $0.50/$2.00 fallback, so cost tracking reflects what Chutes bills
    # rather than a generic guess.
    cost = llm_cost_for(
        provider="chutes",
        model="moonshotai/Kimi-K2.5-TEE",
        prompt_tokens=1_000_000,
        completion_tokens=0,
    )
    assert cost == pytest.approx(0.3827)
    cost = llm_cost_for(
        provider="chutes",
        model="moonshotai/Kimi-K2.5-TEE",
        prompt_tokens=0,
        completion_tokens=1_000_000,
    )
    assert cost == pytest.approx(1.72)


def test_llm_cost_for_chutes_unknown_model_uses_fallback():
    cost = llm_cost_for(
        provider="chutes",
        model="some/unseen-model",
        prompt_tokens=1_000_000,
        completion_tokens=0,
    )
    # ``chutes:*`` fallback is $0.50/M input.  Not Kimi's rate.
    assert cost == pytest.approx(0.50)
