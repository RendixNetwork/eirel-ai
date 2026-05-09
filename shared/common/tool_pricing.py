from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ToolPrice:
    per_call_usd: float


# Initial per-call pricing estimates for each tool service. These are used by
# the trace capture middleware to record cost per tool call and by the
# general_chat 4D scorer to compute the cost dimension.
TOOL_PRICING: dict[str, ToolPrice] = {
    "web_search": ToolPrice(per_call_usd=0.001),
    "sandbox": ToolPrice(per_call_usd=0.002),
    "url_fetch": ToolPrice(per_call_usd=0.0005),
    # text-embedding-3-small averages ~$0.0002 per query (one
    # ~600-token query embed + amortized index cost). Indexing is
    # operator-paid, retrieval is the per-call cost the miner sees.
    "rag.retrieve": ToolPrice(per_call_usd=0.0002),
}


def cost_for_call(tool_name: str) -> float:
    """Return the per-call USD cost for a tool, or 0.0 if unknown."""
    price = TOOL_PRICING.get(tool_name)
    return price.per_call_usd if price is not None else 0.0


# -- LLM pricing -------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class LLMPrice:
    """Per-model token rate card.

    ``input_per_mtok_usd`` / ``output_per_mtok_usd`` are the
    short-context rates. When ``long_context_threshold_tokens > 0``
    AND the call's prompt exceeds that threshold, the
    ``long_context_*`` rates apply to the ENTIRE call (input + output)
    instead — that's how all three vendors (Gemini, Grok, OpenAI)
    bill long-context calls per their published price pages:

      * Gemini 3 family:  threshold 200,000  → both rates ~2× short
      * xAI Grok 4.3:     threshold 200,000  → both rates ~2× short
      * OpenAI gpt-5.4:   threshold 272,000  → both rates ~2× short

    Leave ``long_context_threshold_tokens=0`` (the default) for
    models that publish a single tier — Anthropic / Chutes / OpenRouter
    don't tier on context length.
    """

    input_per_mtok_usd: float
    output_per_mtok_usd: float
    reasoning_per_mtok_usd: float = 0.0
    # Cached-input rate used by OpenAI / Gemini context caching. When 0,
    # cached tokens are charged at the full input rate. OpenAI typically
    # quotes 0.1× input; Gemini quotes 0.1× input on Gemini 3.
    cached_input_per_mtok_usd: float = 0.0
    # Long-context tier — see class docstring. ``threshold_tokens`` is
    # the *strict-greater-than* gate (``prompt_tokens > threshold``
    # selects the long tier). When threshold is 0 the long-tier fields
    # are ignored regardless of value.
    long_context_threshold_tokens: int = 0
    long_input_per_mtok_usd: float = 0.0
    long_output_per_mtok_usd: float = 0.0
    long_cached_input_per_mtok_usd: float = 0.0
    long_reasoning_per_mtok_usd: float = 0.0


_DEFAULT_LLM_PRICING: dict[str, LLMPrice] = {
    "anthropic:claude-opus-4-6": LLMPrice(15.0, 75.0, 75.0),
    "anthropic:claude-sonnet-4-6": LLMPrice(3.0, 15.0, 15.0),
    "anthropic:claude-haiku-4-5": LLMPrice(1.0, 5.0, 5.0),
    "openai:gpt-4o": LLMPrice(2.5, 10.0, 0.0),
    "openai:gpt-4o-mini": LLMPrice(0.15, 0.60, 0.0),
    # GPT-5 family rates per https://developers.openai.com/api/docs/pricing
    # (May 2026). gpt-5.4 long-context tier kicks in above 272K input
    # tokens — both input + output + cached-input rates roughly double.
    # gpt-5.4-mini has a 400K context window but a single rate tier
    # (the published price page does not split short/long for mini).
    "openai:gpt-5.4": LLMPrice(
        input_per_mtok_usd=2.5,
        output_per_mtok_usd=15.0,
        cached_input_per_mtok_usd=0.25,
        long_context_threshold_tokens=272_000,
        long_input_per_mtok_usd=5.0,
        long_output_per_mtok_usd=22.50,
        long_cached_input_per_mtok_usd=0.50,
    ),
    "openai:gpt-5.4-mini": LLMPrice(
        input_per_mtok_usd=0.75,
        output_per_mtok_usd=4.5,
        cached_input_per_mtok_usd=0.075,
    ),
    "openrouter:*": LLMPrice(5.0, 15.0, 0.0),
    # xAI Grok 4.3 rates per https://docs.x.ai/developers/models/grok-4.3
    # (May 2026). Long-context tier kicks in above 200K input tokens —
    # input/output/cached-input all double. Note that Grok responses
    # include ``usage.cost_in_usd_ticks`` which is the authoritative
    # per-call cost; this table is the fallback for mocked / older
    # payloads.
    "xai:grok-4.3": LLMPrice(
        input_per_mtok_usd=1.25,
        output_per_mtok_usd=2.50,
        cached_input_per_mtok_usd=0.20,
        long_context_threshold_tokens=200_000,
        long_input_per_mtok_usd=2.50,
        long_output_per_mtok_usd=5.00,
        long_cached_input_per_mtok_usd=0.40,
    ),
    "xai:*": LLMPrice(
        input_per_mtok_usd=1.25,
        output_per_mtok_usd=2.50,
        cached_input_per_mtok_usd=0.20,
        long_context_threshold_tokens=200_000,
        long_input_per_mtok_usd=2.50,
        long_output_per_mtok_usd=5.00,
        long_cached_input_per_mtok_usd=0.40,
    ),
    # Gemini 3 family rates per https://ai.google.dev/gemini-api/docs/pricing
    # (May 2026). gemini-3.1-pro-preview tiers above 200K input tokens
    # (input doubles 2→4, output 12→18, cached 0.20→0.40).
    # gemini-3.1-flash-lite-preview is single-tier — its 1M context
    # window doesn't have a per-token rate jump.
    "gemini:gemini-3.1-pro-preview": LLMPrice(
        input_per_mtok_usd=2.0,
        output_per_mtok_usd=12.0,
        cached_input_per_mtok_usd=0.20,
        long_context_threshold_tokens=200_000,
        long_input_per_mtok_usd=4.0,
        long_output_per_mtok_usd=18.0,
        long_cached_input_per_mtok_usd=0.40,
    ),
    "gemini:gemini-3-pro": LLMPrice(
        input_per_mtok_usd=2.0,
        output_per_mtok_usd=12.0,
        cached_input_per_mtok_usd=0.20,
        long_context_threshold_tokens=200_000,
        long_input_per_mtok_usd=4.0,
        long_output_per_mtok_usd=18.0,
        long_cached_input_per_mtok_usd=0.40,
    ),
    "gemini:gemini-3.1-flash-lite-preview": LLMPrice(
        input_per_mtok_usd=0.25,
        output_per_mtok_usd=1.50,
        cached_input_per_mtok_usd=0.025,
    ),
    # Legacy aliases retained so configs that still reference the
    # ``gemini-3-flash`` / ``gemini-3-flash-preview`` names resolve to
    # the same single-tier rate card as the new flash-lite preview.
    "gemini:gemini-3-flash": LLMPrice(
        input_per_mtok_usd=0.25,
        output_per_mtok_usd=1.50,
        cached_input_per_mtok_usd=0.025,
    ),
    "gemini:gemini-3-flash-preview": LLMPrice(
        input_per_mtok_usd=0.25,
        output_per_mtok_usd=1.50,
        cached_input_per_mtok_usd=0.025,
    ),
    "gemini:*": LLMPrice(
        input_per_mtok_usd=2.0,
        output_per_mtok_usd=12.0,
        cached_input_per_mtok_usd=0.20,
        long_context_threshold_tokens=200_000,
        long_input_per_mtok_usd=4.0,
        long_output_per_mtok_usd=18.0,
        long_cached_input_per_mtok_usd=0.40,
    ),
    # Chutes per-model rates, sourced from the canonical endpoint
    # ``https://llm.chutes.ai/v1/models`` (price.input.usd / price.output.usd
    # are $/1M tokens). Chutes does not tier on context length.
    "chutes:moonshotai/Kimi-K2.5-TEE": LLMPrice(0.3827, 1.72, 0.0),
    "chutes:Qwen/Qwen3-32B-TEE": LLMPrice(0.08, 0.24, 0.0),
    "chutes:MiniMaxAI/MiniMax-M2.5-TEE": LLMPrice(0.118, 0.99, 0.0),
    "chutes:zai-org/GLM-5.1-TEE": LLMPrice(0.50, 2.0, 0.0),
    "chutes:*": LLMPrice(0.50, 2.0, 0.0),
}


def _load_llm_pricing() -> dict[str, LLMPrice]:
    pricing = dict(_DEFAULT_LLM_PRICING)
    raw = os.getenv("EIREL_LLM_PRICING_JSON")
    if raw:
        try:
            overrides = json.loads(raw)
            for key, entry in overrides.items():
                pricing[key] = LLMPrice(
                    input_per_mtok_usd=entry["input_per_mtok_usd"],
                    output_per_mtok_usd=entry["output_per_mtok_usd"],
                    reasoning_per_mtok_usd=entry.get("reasoning_per_mtok_usd", 0.0),
                    cached_input_per_mtok_usd=entry.get(
                        "cached_input_per_mtok_usd", 0.0,
                    ),
                    long_context_threshold_tokens=int(
                        entry.get("long_context_threshold_tokens", 0)
                    ),
                    long_input_per_mtok_usd=entry.get(
                        "long_input_per_mtok_usd", 0.0,
                    ),
                    long_output_per_mtok_usd=entry.get(
                        "long_output_per_mtok_usd", 0.0,
                    ),
                    long_cached_input_per_mtok_usd=entry.get(
                        "long_cached_input_per_mtok_usd", 0.0,
                    ),
                    long_reasoning_per_mtok_usd=entry.get(
                        "long_reasoning_per_mtok_usd", 0.0,
                    ),
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            _logger.warning("invalid EIREL_LLM_PRICING_JSON, using defaults")
    return pricing


def llm_price_for(provider: str, model: str) -> LLMPrice | None:
    """Resolve the rate card for ``provider:model``.

    Resolution order: dynamic overlay → static table[exact] → static
    table[``provider:*`` glob] → ``None``. Returned dataclass exposes
    every per-token rate so callers can apply per-vendor discounts
    (cached-input, reasoning, long-context multipliers) themselves.
    """
    key = f"{provider}:{model}"
    price = _DYNAMIC_LLM_PRICING.get(key) or LLM_PRICING.get(key)
    if price is None:
        price = LLM_PRICING.get(f"{provider}:*")
    return price


LLM_PRICING: dict[str, LLMPrice] = _load_llm_pricing()


# Dynamic overrides populated at runtime from authoritative provider
# endpoints (e.g. https://llm.chutes.ai/v1/models).  When present, they
# take precedence over the static table so we never charge a stale price
# after a provider publishes a rate change.  Keys have the same shape as
# ``LLM_PRICING`` (``provider:model``).
_DYNAMIC_LLM_PRICING: dict[str, LLMPrice] = {}


def update_dynamic_pricing(entries: dict[str, LLMPrice]) -> None:
    """Replace the dynamic pricing overlay.

    Pass an empty dict to clear the overlay and fall back to the static
    table.  Callers are responsible for validating rates before calling;
    an empty or partial dict is treated as authoritative for the keys
    provided.
    """
    global _DYNAMIC_LLM_PRICING
    _DYNAMIC_LLM_PRICING = dict(entries)
    _logger.info("dynamic LLM pricing overlay updated: %d entries", len(entries))


def get_dynamic_pricing() -> dict[str, LLMPrice]:
    """Current overlay — used for tests / ops observability."""
    return dict(_DYNAMIC_LLM_PRICING)


def _is_long_context(price: LLMPrice, prompt_tokens: int) -> bool:
    """Whether the call's prompt size triggers the long-context tier.

    Strict-greater: at exactly the threshold the call still bills at
    the short-tier rate. Threshold of 0 disables the tier entirely so
    single-rate models always bill at the short rate.
    """
    return (
        price.long_context_threshold_tokens > 0
        and prompt_tokens > price.long_context_threshold_tokens
    )


def llm_cost_for(
    *,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int = 0,
    cached_prompt_tokens: int = 0,
) -> float:
    """Compute USD token cost for a single LLM call.

    Tier selection: when ``prompt_tokens > price.long_context_threshold_tokens``
    AND the threshold is configured (>0), the entire call (input,
    cached input, output, reasoning) bills at the long-context rates.
    This matches all three frontier vendors' policy as of May 2026:

      * Gemini 3 (200K) / Grok 4.3 (200K) / OpenAI gpt-5.4 (272K)

    ``cached_prompt_tokens`` is split out from ``prompt_tokens`` —
    these tokens are charged at the vendor's cached-input rate
    (typically 0.1× the uncached input rate at whichever tier
    applies). Pass 0 when the vendor doesn't expose a cache-hit count.
    """
    price = llm_price_for(provider, model)
    if price is None:
        _logger.warning("unknown LLM pricing key %s:%s, returning 0.0", provider, model)
        return 0.0

    if _is_long_context(price, prompt_tokens):
        input_rate = price.long_input_per_mtok_usd
        output_rate = price.long_output_per_mtok_usd
        # Long-tier cached/reasoning fall back to the short-tier rate
        # × the same multiplier the input rate jumped by — that's how
        # OpenAI/Gemini publish their long-context cached-input price.
        # When the operator hasn't quoted a long-tier cached rate, we
        # scale the short-tier cached rate by the input ratio.
        if price.long_cached_input_per_mtok_usd > 0:
            cached_rate = price.long_cached_input_per_mtok_usd
        elif price.cached_input_per_mtok_usd > 0 and price.input_per_mtok_usd > 0:
            ratio = price.long_input_per_mtok_usd / price.input_per_mtok_usd
            cached_rate = price.cached_input_per_mtok_usd * ratio
        else:
            cached_rate = price.long_input_per_mtok_usd
        if price.long_reasoning_per_mtok_usd > 0:
            reasoning_rate = price.long_reasoning_per_mtok_usd
        elif price.reasoning_per_mtok_usd > 0 and price.output_per_mtok_usd > 0:
            ratio = price.long_output_per_mtok_usd / price.output_per_mtok_usd
            reasoning_rate = price.reasoning_per_mtok_usd * ratio
        else:
            reasoning_rate = price.long_output_per_mtok_usd
    else:
        input_rate = price.input_per_mtok_usd
        output_rate = price.output_per_mtok_usd
        cached_rate = (
            price.cached_input_per_mtok_usd or price.input_per_mtok_usd
        )
        reasoning_rate = price.reasoning_per_mtok_usd or price.output_per_mtok_usd

    uncached_prompt = max(0, prompt_tokens - max(0, cached_prompt_tokens))
    return (
        uncached_prompt * input_rate
        + max(0, cached_prompt_tokens) * cached_rate
        + completion_tokens * output_rate
        + reasoning_tokens * reasoning_rate
    ) / 1_000_000
