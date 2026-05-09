"""Gemini Generative Language API client.

Google's API uses a different request shape from OpenAI's:
  * Endpoint: ``POST {base_url}/models/{model}:generateContent``
  * System prompt is a top-level ``systemInstruction`` field (NOT a role).
  * User content is a list of ``contents[].parts[].text``.
  * JSON-mode is configured via ``generationConfig.responseMimeType``
    (``"application/json"``) + ``generationConfig.responseSchema`` (the
    JSON Schema, with camelCase keys per Google's spec).
  * Auth via ``?key={api_key}`` query param (not bearer header).

Usage cost is computed client-side: Google returns ``usageMetadata``
with token counts (and an optional ``cachedContentTokenCount`` for
context caching) but no USD figure, so the client multiplies by the
configured rate card and adds search-grounding billing per
``candidates[0].groundingMetadata.webSearchQueries``. See
``cost_calc.extract_gemini_generate_cost`` for the policy.
``finish_reason`` maps from Google's ``finishReason`` field directly.

We do NOT wrap this in an OpenAI-compatible adapter — Google's
schema/tool/finish-reason shapes drift enough that a leaky adapter is
worse than two clean clients.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from validation.validator.eval_config import ProviderConfig
from validation.validator.providers.cost_calc import (
    extract_gemini_generate_cost,
)
from validation.validator.providers.types import (
    ProviderError,
    ProviderResponse,
    ProviderTimeout,
)

_logger = logging.getLogger(__name__)


_RETRY_STATUSES: frozenset[int] = frozenset({429, 502, 503, 504})
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE_SECONDS = 0.5
# Google's published default base URL. Operators can override via
# EIREL_VALIDATOR_ORACLE_GEMINI_BASE_URL when using a regional
# endpoint or a vertex-style host.
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Google's responseSchema accepts a strict subset of OpenAPI 3 schema
# — keys not in that subset trigger a 400 INVALID_ARGUMENT. The most
# common culprit from JSON-Schema-flavored callers is
# ``additionalProperties`` (we use it for ``strict: true`` on OpenAI).
# Strip the unsupported keys recursively before sending.
_GEMINI_UNSUPPORTED_SCHEMA_KEYS: frozenset[str] = frozenset({
    "additionalProperties",
    "$schema",
    "$ref",
    "$defs",
    "definitions",
})


def _gemini_compatible_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {
            k: _gemini_compatible_schema(v)
            for k, v in schema.items()
            if k not in _GEMINI_UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [_gemini_compatible_schema(item) for item in schema]
    return schema


class GeminiClient:
    """Thin wrapper around ``POST {base_url}/models/{model}:generateContent``."""

    def __init__(
        self,
        cfg: ProviderConfig,
        *,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_base_seconds: float = _DEFAULT_BACKOFF_BASE_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not cfg.configured:
            raise ProviderError(
                "GeminiClient requires base_url + api_key + model"
            )
        self._cfg = cfg
        self._max_retries = max(0, int(max_retries))
        self._backoff_base = max(0.0, float(backoff_base_seconds))
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return self._cfg.model

    @property
    def base_url(self) -> str:
        return self._cfg.base_url

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(transport=self._transport)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        response_schema: dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        schema_name: str = "response",  # accepted for API parity; unused
        enable_web_search: bool = False,
    ) -> ProviderResponse:
        """Single ``generateContent`` call with strict JSON output.

        Returns the raw text content of the model response — caller
        decodes JSON. Schema is passed in Google's camelCase shape.

        ``enable_web_search`` adds Google's server-side
        ``googleSearch`` tool so the model grounds against fresh
        results before answering — needed for live-lookup / current-
        events items where the training cutoff would otherwise
        produce stale answers.
        """
        del schema_name  # API parity with OpenAI-compatible client
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": "user", "parts": [{"text": user}]}
            ],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens or self._cfg.max_tokens),
                "responseMimeType": "application/json",
                "responseSchema": _gemini_compatible_schema(response_schema),
            },
        }
        if enable_web_search:
            body["tools"] = [{"googleSearch": {}}]
        params = {"key": self._cfg.api_key}
        url = (
            f"{self._cfg.base_url}/models/{self._cfg.model}:generateContent"
        )
        headers = {"Content-Type": "application/json"}
        client = await self._get_client()
        return await self._post_with_retry(
            client=client,
            url=url,
            params=params,
            body=body,
            headers=headers,
        )

    async def _post_with_retry(
        self,
        *,
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, str],
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> ProviderResponse:
        last_exc: Exception | None = None
        attempt = 0
        while True:
            attempt += 1
            t0 = time.perf_counter()
            try:
                response = await client.post(
                    url,
                    params=params,
                    json=body,
                    headers=headers,
                    timeout=self._cfg.timeout_seconds,
                )
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt > self._max_retries:
                    raise ProviderTimeout(
                        f"timeout after {attempt} attempt(s): {exc}"
                    ) from exc
                await self._sleep_backoff(attempt)
                continue
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt > self._max_retries:
                    raise ProviderError(
                        f"network error after {attempt} attempt(s): {exc}"
                    ) from exc
                await self._sleep_backoff(attempt)
                continue
            latency_ms = int((time.perf_counter() - t0) * 1000)
            if response.status_code in _RETRY_STATUSES and attempt <= self._max_retries:
                await self._sleep_backoff(attempt)
                continue
            if response.status_code != 200:
                raise ProviderError(
                    f"HTTP {response.status_code}: "
                    f"{(response.text or '')[:512]}"
                )
            parsed = self._parse_response(response, latency_ms)
            cost = parsed.usage_usd
            cost_str = "?" if cost is None else f"${cost:.6f}"
            _logger.info(
                "validator_provider_call: vendor=gemini model=%s latency_ms=%d cost_usd=%s",
                self._cfg.model, latency_ms, cost_str,
            )
            return parsed
        raise ProviderError(  # pragma: no cover
            f"unreachable after {attempt} attempts; last_exc={last_exc!r}"
        )

    async def _sleep_backoff(self, attempt: int) -> None:
        delay = self._backoff_base * (2 ** (attempt - 1))
        await asyncio.sleep(delay)

    def _parse_response(
        self, response: httpx.Response, latency_ms: int,
    ) -> ProviderResponse:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(f"non-JSON response: {exc}") from exc
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            # Gemini returns no candidates when the request was blocked
            # by safety filters — surface the prompt-feedback message
            # so the caller can distinguish "vendor refused" from
            # "vendor errored."
            feedback = payload.get("promptFeedback") or {}
            block_reason = feedback.get("blockReason")
            if block_reason:
                raise ProviderError(
                    f"gemini blocked the prompt: {block_reason}"
                )
            raise ProviderError(
                f"missing candidates in response: {str(payload)[:512]}"
            )
        candidate = candidates[0] or {}
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        # Concatenate text parts; ignore non-text (function calls,
        # images). For structured-output mode, the model returns one
        # text part containing the JSON string.
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in parts
        )
        finish_reason = candidate.get("finishReason")
        # ``groundingMetadata`` is populated when ``tools=[{googleSearch:{}}]``
        # was active. Each grounding chunk carries a ``web.uri``
        # (sometimes a redirect through Google's vertex search service —
        # operators reading the URL still get a working pointer).
        citations: list[str] = []
        seen: set[str] = set()
        gm = candidate.get("groundingMetadata") or {}
        for chunk in gm.get("groundingChunks") or []:
            if not isinstance(chunk, dict):
                continue
            web = chunk.get("web") or {}
            uri = web.get("uri") if isinstance(web, dict) else None
            if isinstance(uri, str) and uri not in seen:
                seen.add(uri)
                citations.append(uri)
        return ProviderResponse(
            text=text,
            latency_ms=latency_ms,
            usage_usd=extract_gemini_generate_cost(payload, self._cfg.model),
            finish_reason=finish_reason,
            citations=tuple(citations),
        )
