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
    "x_api": ToolPrice(per_call_usd=0.050),
    "semantic_scholar": ToolPrice(per_call_usd=0.000),
    "sandbox": ToolPrice(per_call_usd=0.002),
}


def cost_for_call(tool_name: str) -> float:
    """Return the per-call USD cost for a tool, or 0.0 if unknown."""
    price = TOOL_PRICING.get(tool_name)
    return price.per_call_usd if price is not None else 0.0


# -- LLM pricing -------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class LLMPrice:
    input_per_mtok_usd: float
    output_per_mtok_usd: float
    reasoning_per_mtok_usd: float = 0.0


_DEFAULT_LLM_PRICING: dict[str, LLMPrice] = {
    "anthropic:claude-opus-4-6": LLMPrice(15.0, 75.0, 75.0),
    "anthropic:claude-sonnet-4-6": LLMPrice(3.0, 15.0, 15.0),
    "anthropic:claude-haiku-4-5": LLMPrice(1.0, 5.0, 5.0),
    "openai:gpt-4o": LLMPrice(2.5, 10.0, 0.0),
    "openai:gpt-4o-mini": LLMPrice(0.15, 0.60, 0.0),
    "openrouter:*": LLMPrice(5.0, 15.0, 0.0),
    # Chutes per-model rates, sourced from the canonical endpoint
    # ``https://llm.chutes.ai/v1/models`` (price.input.usd / price.output.usd
    # are $/1M tokens).  Keep rates in sync with that endpoint when adding
    # new models; the ``chutes:*`` fallback is intentionally generous to
    # avoid silently undercharging unknown models.
    "chutes:moonshotai/Kimi-K2.5-TEE": LLMPrice(0.3827, 1.72, 0.0),
    "chutes:Qwen/Qwen3-32B-TEE": LLMPrice(0.08, 0.24, 0.0),
    "chutes:MiniMaxAI/MiniMax-M2.5-TEE": LLMPrice(0.118, 0.99, 0.0),
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
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            _logger.warning("invalid EIREL_LLM_PRICING_JSON, using defaults")
    return pricing


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


def llm_cost_for(
    *,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int = 0,
) -> float:
    key = f"{provider}:{model}"
    # Dynamic overlay wins — it reflects the provider's current published
    # rates.  Static table is the fallback if the refresh never completed
    # or the model isn't in the overlay yet.
    price = _DYNAMIC_LLM_PRICING.get(key) or LLM_PRICING.get(key)
    if price is None:
        fallback_key = f"{provider}:*"
        price = LLM_PRICING.get(fallback_key)
    if price is None:
        _logger.warning("unknown LLM pricing key %s, returning 0.0", key)
        return 0.0
    reasoning_rate = price.reasoning_per_mtok_usd or price.output_per_mtok_usd
    return (
        prompt_tokens * price.input_per_mtok_usd
        + completion_tokens * price.output_per_mtok_usd
        + reasoning_tokens * reasoning_rate
    ) / 1_000_000
