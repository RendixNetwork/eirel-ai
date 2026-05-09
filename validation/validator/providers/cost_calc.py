"""Exact USD cost extraction for validator-side LLM provider calls.

Each vendor exposes call cost differently — some return a precise
post-hoc figure, others require client-side computation from token
counts plus tool-invocation counts. This module gives one extractor
per (vendor, endpoint shape) and returns the same scalar everywhere
so the rest of the validator never has to know vendor-specific shapes.

  * **OpenAI Responses API (gpt-5.4)** — no native cost field. Compute
    from ``usage.{input_tokens, output_tokens}``,
    ``usage.input_tokens_details.cached_tokens`` (charged at the cached
    rate), plus a per-call charge for every ``web_search_call`` item in
    ``output[]`` ($10 per 1k calls per the OpenAI price page).

  * **xAI Grok Responses API (grok-4.3)** — returns
    ``usage.cost_in_usd_ticks`` (int64, 1 USD = 10^10 ticks) which is
    already the all-inclusive cost (tokens + tool calls + reasoning).
    Falls back to token-based estimation when ticks are absent (older
    deployments / mock responses).

  * **Gemini generateContent (gemini-3.x)** — returns ``usageMetadata``
    with token counts and an optional ``cachedContentTokenCount``.
    Search billing is **per query**, not per call: count the entries
    in ``candidates[0].groundingMetadata.webSearchQueries`` and charge
    $14 per 1k. Prompts above 200k tokens double both input AND output
    rates per Google's published policy.

  * **Chutes chat-completions (zai-org/GLM-5.1-TEE et al.)** — token
    cost only; no web search, no special tools. Pricing comes from the
    dynamic overlay (refreshed hourly from ``llm.chutes.ai/v1/models``)
    when available, otherwise the static ``chutes:*`` rate.

All functions return ``None`` when the upstream payload doesn't carry
enough info to compute a cost — caller treats that as "unknown" rather
than zero. The conservative default avoids silently undercharging
when a vendor changes its response shape.
"""

from __future__ import annotations

import logging
from typing import Any

from shared.common.tool_pricing import llm_cost_for, llm_price_for

_logger = logging.getLogger(__name__)


# Web-search per-call rates (USD).
#   OpenAI: $10 / 1k calls for the ``web_search`` tool on Responses API.
#   xAI Grok: included in cost_in_usd_ticks; no separate adder.
#   Gemini: $14 / 1k queries (counted from grounding webSearchQueries).
_OPENAI_WEB_SEARCH_PER_CALL_USD = 10.0 / 1000
_GEMINI_WEB_SEARCH_PER_QUERY_USD = 14.0 / 1000

# NOTE: long-context token-rate tiering (Gemini @200K, Grok @200K,
# OpenAI gpt-5.4 @272K) is owned by ``shared.common.tool_pricing``.
# This module passes ``prompt_tokens`` straight to ``llm_cost_for`` and
# the rate card resolves the right tier — keeping vendor-specific
# thresholds in one place avoids drift between cost calculators.

# Vendor tags resolved from base_url substrings — used by clients that
# host multiple providers behind one HTTP shape (e.g. the validator's
# ``OpenAICompatibleClient`` covers OpenAI, xAI, and Chutes via the
# same client class, dispatched by host).
VENDOR_OPENAI = "openai"
VENDOR_XAI = "xai"
VENDOR_CHUTES = "chutes"
VENDOR_GEMINI = "gemini"


def vendor_from_base_url(base_url: str) -> str:
    """Map a provider base_url to a stable vendor tag.

    Returns ``"openai"`` / ``"xai"`` / ``"chutes"`` / ``"gemini"`` /
    ``"unknown"``. Used to pick the right cost extractor and pricing
    table key for a generic OpenAI-compatible HTTP client.
    """
    host = (base_url or "").lower()
    if "api.openai.com" in host:
        return VENDOR_OPENAI
    if "api.x.ai" in host:
        return VENDOR_XAI
    if "chutes.ai" in host:
        return VENDOR_CHUTES
    if "googleapis.com" in host or "generativelanguage" in host:
        return VENDOR_GEMINI
    return "unknown"


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):  # bool is int — must reject explicitly
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


# -- OpenAI Responses API ---------------------------------------------------


def extract_openai_responses_cost(
    payload: dict[str, Any], model: str,
) -> float | None:
    """Compute exact USD cost for an OpenAI ``/v1/responses`` call.

    Returns ``None`` when ``usage`` is absent (request errored before
    metering). The web_search tool is billed even when the model
    chose not to search ($0 in that case because there's no
    ``web_search_call`` entry in ``output[]``).
    """
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = _safe_int(usage.get("input_tokens"))
    output_tokens = _safe_int(usage.get("output_tokens"))
    cached_tokens = 0
    details = usage.get("input_tokens_details")
    if isinstance(details, dict):
        cached_tokens = _safe_int(details.get("cached_tokens"))
    if input_tokens == 0 and output_tokens == 0:
        return None

    token_cost = llm_cost_for(
        provider=VENDOR_OPENAI,
        model=model,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        cached_prompt_tokens=cached_tokens,
    )

    web_search_calls = _count_openai_web_search_calls(payload)
    web_cost = web_search_calls * _OPENAI_WEB_SEARCH_PER_CALL_USD
    return round(token_cost + web_cost, 8)


def _count_openai_web_search_calls(payload: dict[str, Any]) -> int:
    output = payload.get("output")
    if not isinstance(output, list):
        return 0
    count = 0
    for item in output:
        if not isinstance(item, dict):
            continue
        # OpenAI emits ``{"type": "web_search_call", "action": {"type": "search", ...}}``
        # for each search invocation. ``tool_choice: "auto"`` may
        # produce zero of these — that's a free call.
        if item.get("type") == "web_search_call":
            count += 1
    return count


# -- xAI Grok Responses API -------------------------------------------------


def extract_grok_responses_cost(
    payload: dict[str, Any], model: str,
) -> float | None:
    """Compute exact USD cost for an xAI Grok ``/v1/responses`` call.

    Prefers ``usage.cost_in_usd_ticks`` — xAI's authoritative integer
    cost field (1 USD = 10^10 ticks) which folds in tokens, search,
    reasoning, and any other server-side tools. Falls back to a
    token-based estimate via the static rate table when ticks are
    absent (older payload shapes / mocked responses).
    """
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None

    ticks = usage.get("cost_in_usd_ticks")
    if isinstance(ticks, (int, float)) and not isinstance(ticks, bool) and ticks > 0:
        return round(float(ticks) / 10_000_000_000, 8)

    input_tokens = _safe_int(
        usage.get("input_tokens") or usage.get("prompt_tokens")
    )
    output_tokens = _safe_int(
        usage.get("output_tokens") or usage.get("completion_tokens")
    )
    if input_tokens == 0 and output_tokens == 0:
        return None
    return round(llm_cost_for(
        provider=VENDOR_XAI,
        model=model,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
    ), 8)


# -- Gemini generateContent -------------------------------------------------


def extract_gemini_generate_cost(
    payload: dict[str, Any], model: str,
) -> float | None:
    """Compute exact USD cost for a Gemini ``:generateContent`` call.

    Charges:
      * ``promptTokenCount - cachedContentTokenCount`` at uncached input rate
      * ``cachedContentTokenCount`` at cached-input rate (``input × 0.1``)
      * ``candidatesTokenCount`` at output rate
      * ``len(grounding.webSearchQueries) × $0.014`` for grounded calls

    Long-context tier (>200K input tokens) is applied automatically by
    ``llm_cost_for`` from the rate card — both input and output rates
    double per Google's published policy.
    """
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usageMetadata")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = _safe_int(usage.get("promptTokenCount"))
    candidate_tokens = _safe_int(usage.get("candidatesTokenCount"))
    cached_tokens = _safe_int(usage.get("cachedContentTokenCount"))
    if prompt_tokens == 0 and candidate_tokens == 0:
        return None

    token_cost = llm_cost_for(
        provider=VENDOR_GEMINI,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=candidate_tokens,
        cached_prompt_tokens=cached_tokens,
    )

    search_query_count = _count_gemini_search_queries(payload)
    web_cost = search_query_count * _GEMINI_WEB_SEARCH_PER_QUERY_USD
    return round(token_cost + web_cost, 8)


def _count_gemini_search_queries(payload: dict[str, Any]) -> int:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return 0
    first = candidates[0]
    if not isinstance(first, dict):
        return 0
    grounding = first.get("groundingMetadata")
    if not isinstance(grounding, dict):
        return 0
    queries = grounding.get("webSearchQueries")
    if not isinstance(queries, list):
        return 0
    return len(queries)


# -- Chutes chat-completions ------------------------------------------------


def extract_chutes_chat_cost(
    payload: dict[str, Any], model: str,
) -> float | None:
    """Compute exact USD cost for a Chutes ``/chat/completions`` call.

    Token cost only — no web search, no per-call overhead. Resolves
    rates via :func:`llm_cost_for` so the dynamic overlay (auto-
    refreshed from ``llm.chutes.ai/v1/models``) takes precedence over
    the static table.
    """
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = _safe_int(usage.get("prompt_tokens"))
    completion_tokens = _safe_int(usage.get("completion_tokens"))
    if prompt_tokens == 0 and completion_tokens == 0:
        return None
    if llm_price_for(VENDOR_CHUTES, model) is None:
        # Chutes-style ``provider/model`` slug not in either table —
        # skip rather than charging the generous fallback.
        return None
    return round(llm_cost_for(
        provider=VENDOR_CHUTES,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    ), 8)


# -- OpenAI-compatible chat completions (dispatched) ------------------------


def extract_openai_compatible_chat_cost(
    payload: dict[str, Any], *, base_url: str, model: str,
) -> float | None:
    """Dispatch on ``base_url`` to the right chat-completions extractor.

    The validator's ``OpenAICompatibleClient`` is reused for OpenAI,
    xAI, and Chutes — the chat-completions response shape is shared
    across them but cost-extraction differs (Grok uses ticks, Chutes
    uses token rates). When the host doesn't match a known vendor we
    fall back to ``usage.total_cost_usd`` if the upstream surfaced one,
    else ``None``.
    """
    vendor = vendor_from_base_url(base_url)
    if vendor == VENDOR_CHUTES:
        return extract_chutes_chat_cost(payload, model)
    if vendor == VENDOR_XAI:
        # Grok also exposes the chat-completions endpoint; the ticks
        # field is documented as appearing on every xAI usage block.
        return extract_grok_responses_cost(payload, model)
    if vendor == VENDOR_OPENAI:
        # OpenAI chat-completions: same usage shape as Responses API
        # without the ``output`` array — so no web_search adder.
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if not isinstance(usage, dict):
            return None
        prompt_tokens = _safe_int(usage.get("prompt_tokens"))
        completion_tokens = _safe_int(usage.get("completion_tokens"))
        cached_tokens = 0
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached_tokens = _safe_int(details.get("cached_tokens"))
        if prompt_tokens == 0 and completion_tokens == 0:
            return None
        return round(llm_cost_for(
            provider=VENDOR_OPENAI,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_tokens,
        ), 8)

    # Unknown host — preserve any explicit total surfaced by the upstream.
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if isinstance(usage, dict):
        total = usage.get("total_cost_usd")
        if isinstance(total, (int, float)) and not isinstance(total, bool):
            return float(total)
    return None


__all__ = [
    "VENDOR_CHUTES",
    "VENDOR_GEMINI",
    "VENDOR_OPENAI",
    "VENDOR_XAI",
    "extract_chutes_chat_cost",
    "extract_gemini_generate_cost",
    "extract_grok_responses_cost",
    "extract_openai_compatible_chat_cost",
    "extract_openai_responses_cost",
    "vendor_from_base_url",
]
