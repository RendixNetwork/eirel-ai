"""OpenAI-compatible chat-completions client.

Used by three roles in the validator:
  * ``OpenAI`` oracle (``gpt-5.4``)
  * ``Grok`` oracle via xAI's OpenAI-compatible endpoint (``grok-4.3``)
  * ``Chutes`` reconciler (``zai-org/GLM-5.1-TEE``)

The Gemini oracle does NOT use this client — Google's Generative
Language API has a different request shape. See ``gemini.py``.

Hardening:
  * Async ``httpx`` with explicit timeout per call (caller-controlled).
  * Bounded retry on 429 / 502 / 503 / 504 (transient classes only).
    Other 4xx surface immediately as :class:`ProviderError` so caller
    can distinguish "vendor said no" from "network blip."
  * ``response_format={"type": "json_schema", ...}`` enforces structured
    output. Caller passes a JSON Schema; client returns the raw text
    (caller decodes / validates).
  * Cost extraction is best-effort: provider responses include
    ``usage`` in different shapes; we read what we can and surface
    ``None`` otherwise. Scoring does not depend on this field.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

import httpx

from validation.validator.eval_config import ProviderConfig
from validation.validator.providers.types import (
    ProviderError,
    ProviderResponse,
    ProviderTimeout,
)

_logger = logging.getLogger(__name__)


_RETRY_STATUSES: frozenset[int] = frozenset({429, 502, 503, 504})
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE_SECONDS = 0.5


class OpenAICompatibleClient:
    """Thin wrapper around ``POST {base_url}/chat/completions``.

    One client per provider/role. Reuses a single ``httpx.AsyncClient``
    across calls; close via :meth:`aclose` when the validator process
    finishes the batch.
    """

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
                "OpenAICompatibleClient requires base_url + api_key + model"
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
        schema_name: str = "response",
    ) -> ProviderResponse:
        """Single chat-completions call with strict structured output.

        ``response_schema`` is passed through as ``response_format =
        {"type": "json_schema", "json_schema": {"name", "schema", "strict": true}}``.
        Returns the raw text content of the assistant message — caller
        decodes JSON. Doesn't validate against the schema itself
        (provider's responsibility on supported models).
        """
        # OpenAI's official API rejects ``max_tokens`` on the GPT-5
        # family (and most reasoning models since o1) — the parameter
        # was renamed to ``max_completion_tokens``. Detect by host so
        # Grok / Chutes / vLLM-compatible deployments keep the legacy
        # name they recognize.
        token_param_name = (
            "max_completion_tokens"
            if "api.openai.com" in self._cfg.base_url
            else "max_tokens"
        )
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": response_schema,
                    "strict": True,
                },
            },
            "temperature": float(temperature),
            token_param_name: int(max_tokens or self._cfg.max_tokens),
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._cfg.api_key}",
        }
        url = f"{self._cfg.base_url}/chat/completions"
        client = await self._get_client()
        return await self._post_with_retry(
            client=client, url=url, payload=payload, headers=headers,
        )

    async def complete_responses_with_web_search(
        self,
        *,
        system: str,
        user: str,
        response_schema: dict[str, Any],
        max_tokens: int | None = None,
        schema_name: str = "response",
    ) -> ProviderResponse:
        """Single ``POST /responses`` call with web_search + json_schema.

        Both OpenAI (``api.openai.com``) and xAI (``api.x.ai``) expose
        the ``/v1/responses`` Responses API with the same shape:
        ``input`` (not ``messages``), ``tools=[{type:"web_search"}]``,
        ``text.format.json_schema``, ``max_output_tokens``. Chat
        Completions rejects ``tools[].type=web_search`` on both
        vendors — the Responses surface is the only path to native
        server-side web search.

        Chutes / vLLM-compatible deployments don't ship a
        ``/v1/responses`` endpoint, so this method is reserved for the
        OpenAI + Grok oracles.
        """
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": [{"type": "web_search"}],
            "max_output_tokens": int(max_tokens or self._cfg.max_tokens),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": response_schema,
                    "strict": True,
                },
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._cfg.api_key}",
        }
        url = f"{self._cfg.base_url}/responses"
        client = await self._get_client()
        return await self._post_with_retry(
            client=client,
            url=url,
            payload=payload,
            headers=headers,
            response_parser=self._parse_responses_api_response,
        )

    async def _post_with_retry(
        self,
        *,
        client: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        response_parser: "Callable[[httpx.Response, int], ProviderResponse] | None" = None,
    ) -> ProviderResponse:
        parser = response_parser or self._parse_response
        last_exc: Exception | None = None
        attempt = 0
        while True:
            attempt += 1
            t0 = time.perf_counter()
            try:
                response = await client.post(
                    url,
                    json=payload,
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
            return parser(response, latency_ms)
        # Unreachable: the loop either returns or raises.
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
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError(
                f"missing choices in response: {str(payload)[:512]}"
            )
        choice = choices[0] or {}
        message = choice.get("message") or {}
        raw_content = message.get("content")
        if isinstance(raw_content, list):
            # OpenAI 'content' can be a list of {type, text} parts; join
            # the text-typed pieces. Non-text parts (e.g. image refs)
            # are dropped — the caller is doing structured-output, so
            # only text matters.
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            )
        else:
            text = str(raw_content or "")
        finish_reason = choice.get("finish_reason")
        usage_usd = self._extract_usage_usd(payload)
        return ProviderResponse(
            text=text,
            latency_ms=latency_ms,
            usage_usd=usage_usd,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _extract_usage_usd(payload: dict[str, Any]) -> float | None:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        total_usd = usage.get("total_cost_usd")
        if isinstance(total_usd, (int, float)):
            return float(total_usd)
        return None

    def _parse_responses_api_response(
        self, response: httpx.Response, latency_ms: int,
    ) -> ProviderResponse:
        """Parse OpenAI Responses-API output.

        The Responses API returns ``output`` as a list of items mixing
        ``web_search_call`` (tool invocations) and one terminal
        ``message`` carrying the structured-output text. Find the
        message and return its ``content[0].text``.
        """
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(f"non-JSON response: {exc}") from exc
        output = payload.get("output")
        if not isinstance(output, list):
            raise ProviderError(
                f"missing output array in /responses payload: {str(payload)[:512]}"
            )
        text_pieces: list[str] = []
        finish_reason: str | None = None
        citation_urls: list[str] = []
        seen_urls: set[str] = set()

        def _add_url(raw: Any) -> None:
            if not isinstance(raw, str):
                return
            url = raw.strip()
            if not url or url in seen_urls:
                return
            seen_urls.add(url)
            citation_urls.append(url)

        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text":
                    text_pieces.append(str(part.get("text", "")))
                # Inline citations live as ``annotations`` per content
                # part on both OpenAI's and xAI's Responses APIs:
                # ``[{type:"url_citation", url, title, start_index, end_index}]``.
                for ann in part.get("annotations") or []:
                    if isinstance(ann, dict):
                        _add_url(ann.get("url"))
            finish_reason = item.get("status") or finish_reason
        # xAI also surfaces a top-level ``citations`` array (flat URL
        # list). OpenAI's Responses API doesn't, but reading the field
        # defensively is fine.
        for url in payload.get("citations") or []:
            _add_url(url)
        if not text_pieces:
            raise ProviderError(
                f"no output_text in /responses payload: {str(payload)[:512]}"
            )
        return ProviderResponse(
            text="".join(text_pieces),
            latency_ms=latency_ms,
            usage_usd=self._extract_usage_usd(payload),
            finish_reason=finish_reason,
            citations=tuple(citation_urls),
        )
