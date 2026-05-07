"""Shared response/error types for validator-side LLM provider clients."""

from __future__ import annotations

from dataclasses import dataclass


class ProviderError(RuntimeError):
    """Generic provider failure that callers should treat as
    "this oracle/reconciler vendor produced no usable result."

    Includes 4xx that aren't retryable, malformed JSON responses, and
    response-schema validation failures.
    """


class ProviderTimeout(ProviderError):
    """Subclass for timeouts and exhausted-retry network errors so
    callers can surface 'vendor was unavailable' separately from
    'vendor returned nonsense'."""


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized completion response across providers.

    ``text`` is the raw structured-output payload (caller decodes JSON
    when ``response_schema`` was supplied). ``finish_reason`` is the
    upstream-specific stop reason — useful for telemetry but not load-
    bearing in scoring. ``usage_usd`` is best-effort: providers don't
    all expose cost in the response, so the caller may treat ``None``
    as "unknown" without a scoring impact.
    """

    text: str
    latency_ms: int
    usage_usd: float | None = None
    finish_reason: str | None = None
    # URLs the vendor cited during a web-search-enabled call.
    # OpenAI / Grok surface them in ``output[].content[].annotations``
    # plus a top-level ``citations`` array; Gemini surfaces them in
    # ``candidates[].groundingMetadata.groundingChunks[].web.uri``.
    # Empty when web search wasn't used / nothing cited.
    citations: tuple[str, ...] = ()
