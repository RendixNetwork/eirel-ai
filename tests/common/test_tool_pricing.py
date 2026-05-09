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


# -- Long-context tier behaviour ---------------------------------------------


def test_short_tier_at_threshold_uses_short_rates_openai():
    """gpt-5.4 threshold is 272K, strict-greater. Exactly 272K → short
    rate ($2.50/Mtok input)."""
    cost = llm_cost_for(
        provider="openai", model="gpt-5.4",
        prompt_tokens=272_000, completion_tokens=0,
    )
    # 272K * $2.50/Mtok = 0.68
    assert cost == pytest.approx(0.68)


def test_long_tier_above_threshold_uses_long_rates_openai():
    """One token above 272K → long rates: input $5/Mtok, output $22.50/Mtok."""
    cost = llm_cost_for(
        provider="openai", model="gpt-5.4",
        prompt_tokens=272_001, completion_tokens=10_000,
    )
    # 272001 * 5 / 1M + 10K * 22.5 / 1M = 1.360005 + 0.225 = 1.585005
    assert cost == pytest.approx(1.585005)


def test_long_tier_grok_doubles_rates():
    """grok-4.3 threshold is 200K. Above: $1.25→$2.50 input,
    $2.50→$5 output."""
    cost = llm_cost_for(
        provider="xai", model="grok-4.3",
        prompt_tokens=200_001, completion_tokens=10_000,
    )
    # 200001 * 2.5 / 1M + 10K * 5 / 1M = 0.500003 + 0.05 = 0.550003
    assert cost == pytest.approx(0.550003)


def test_long_tier_gemini_doubles_rates():
    """gemini-3.1-pro-preview threshold is 200K. Above:
    input $2→$4, output $12→$18."""
    cost = llm_cost_for(
        provider="gemini", model="gemini-3.1-pro-preview",
        prompt_tokens=300_000, completion_tokens=5_000,
    )
    # 300K * 4 / 1M + 5K * 18 / 1M = 1.2 + 0.09 = 1.29
    assert cost == pytest.approx(1.29)


def test_long_tier_applies_long_cached_rate():
    """Cached-input rate also tiers: gpt-5.4 cached
    short $0.25 → long $0.50 /Mtok."""
    cost = llm_cost_for(
        provider="openai", model="gpt-5.4",
        prompt_tokens=300_000, completion_tokens=0,
        cached_prompt_tokens=100_000,
    )
    # uncached 200K × $5/Mtok + cached 100K × $0.50/Mtok
    # = 1.00 + 0.05 = 1.05
    assert cost == pytest.approx(1.05)


def test_long_tier_falls_back_to_scaled_cached_rate_when_long_unset():
    """Operator overrides via EIREL_LLM_PRICING_JSON may set
    ``long_input`` / ``long_output`` but skip ``long_cached_input``.
    The cost calc scales the short cached rate by the input ratio to
    preserve the discount proportion rather than charging the full
    long-input rate for cached tokens."""
    from shared.common.tool_pricing import LLM_PRICING
    LLM_PRICING["test_vendor:test_model"] = LLMPrice(
        input_per_mtok_usd=2.0,
        output_per_mtok_usd=10.0,
        cached_input_per_mtok_usd=0.20,  # 0.1× input
        long_context_threshold_tokens=100_000,
        long_input_per_mtok_usd=4.0,
        long_output_per_mtok_usd=20.0,
        # long_cached_input_per_mtok_usd intentionally unset (0).
    )
    try:
        cost = llm_cost_for(
            provider="test_vendor", model="test_model",
            prompt_tokens=200_000, completion_tokens=0,
            cached_prompt_tokens=200_000,
        )
        # Long tier triggers (200K > 100K threshold).
        # Scaled cached_rate = 0.20 × (4.0 / 2.0) = 0.40
        # 200K × $0.40/Mtok = 0.08
        assert cost == pytest.approx(0.08)
    finally:
        del LLM_PRICING["test_vendor:test_model"]


def test_chutes_no_long_tier_threshold_zero():
    """Chutes models have ``long_context_threshold_tokens=0`` — they
    always bill at the single rate regardless of prompt size."""
    cost = llm_cost_for(
        provider="chutes", model="zai-org/GLM-5.1-TEE",
        prompt_tokens=10_000_000, completion_tokens=0,
    )
    # 10M × $0.50/Mtok = 5.0 (no long-tier kick-in)
    assert cost == pytest.approx(5.0)
