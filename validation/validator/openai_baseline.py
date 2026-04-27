"""OpenAI Responses API client for baseline generation.

Each task produces exactly one baseline response per run: the validator
calls this client with the task prompt, gets back a normalized
:class:`~shared.core.evaluation_models.BaselineResponse`, and uses it as side
B in every per-miner pairwise judgment for that task.

The client targets the Responses API (``POST /v1/responses``) with the
built-in ``web_search`` tool so the baseline has the same class of
capability as miner agents (search-grounded answers with citations).

Failures are not retried inside the 3-minute fan-out window. If the call
fails, the validator reports ``baseline_failed`` to owner-api and the task
returns to the pending queue for another validator.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
from pydantic import ValidationError

from shared.core.evaluation_models import BaselineResponse

logger = logging.getLogger(__name__)


_DEFAULT_BASE_URL = "https://api.openai.com"
_DEFAULT_MODEL = "gpt-5"
_DEFAULT_TIMEOUT_SECONDS = 120.0
# gpt-5 pricing (input + output) is rough; actual USD is pulled from
# the API response usage when available. This constant is only a fallback
# used by the budget guard when usage is missing.
_APPROX_COST_PER_CALL_USD = 0.05


class OpenAIBaselineError(RuntimeError):
    """Raised when the OpenAI baseline call fails and should be surfaced
    as ``baseline_failed`` to owner-api."""


class OpenAIBaselineClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        max_cost_usd_per_run: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_BASELINE_MODEL", _DEFAULT_MODEL)
        self.base_url = (base_url or os.getenv("OPENAI_BASELINE_BASE_URL", _DEFAULT_BASE_URL)).rstrip("/")
        self.timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv("OPENAI_BASELINE_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS)
        )
        self.max_cost_usd_per_run = float(
            max_cost_usd_per_run
            if max_cost_usd_per_run is not None
            else os.getenv("OPENAI_BASELINE_MAX_COST_USD_PER_RUN", "10.0")
        )
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._spent_usd = 0.0

    @property
    def spent_usd(self) -> float:
        return self._spent_usd

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(transport=self._transport)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def generate(
        self,
        *,
        prompt: str,
        use_web_search: bool = False,
    ) -> BaselineResponse:
        """Call the Responses API, optionally with the built-in web_search tool.

        ``use_web_search`` is set from the task's own ``web_search`` flag by
        the caller — this mirrors the end-user toggle, so the baseline has
        the same information access as the miner would in real chat usage.

        Returns a normalized BaselineResponse; raises ``OpenAIBaselineError``
        on any failure (network, HTTP error, malformed output, budget
        exceeded).
        """
        if not self.api_key:
            raise OpenAIBaselineError("OPENAI_API_KEY is not set")
        if self._spent_usd + _APPROX_COST_PER_CALL_USD > self.max_cost_usd_per_run:
            raise OpenAIBaselineError(
                f"per-run OpenAI budget exhausted "
                f"(spent={self._spent_usd:.4f} cap={self.max_cost_usd_per_run:.4f})"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
        }
        if use_web_search:
            payload["tools"] = [{"type": "web_search"}]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        started = time.perf_counter()
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/v1/responses",
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise OpenAIBaselineError(f"network error calling baseline: {exc}") from exc

        latency = max(0.0, time.perf_counter() - started)

        if response.status_code >= 400:
            raise OpenAIBaselineError(
                f"baseline call failed: status={response.status_code} "
                f"body={response.text[:400]}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise OpenAIBaselineError(f"baseline returned non-JSON: {exc}") from exc

        response_text, citations, raw_output = _extract_output(body)
        cost_usd = _extract_cost_usd(body)
        if cost_usd <= 0.0:
            cost_usd = _APPROX_COST_PER_CALL_USD
        self._spent_usd += cost_usd

        try:
            return BaselineResponse(
                response_text=response_text,
                citations=citations,
                raw_output=raw_output,
                latency_seconds=latency,
                cost_usd=cost_usd,
                model=self.model,
                metadata={
                    "usage": body.get("usage") or {},
                    "response_id": body.get("id"),
                    "web_search_enabled": use_web_search,
                },
            )
        except ValidationError as exc:
            raise OpenAIBaselineError(f"baseline payload failed validation: {exc}") from exc


def _extract_output(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Pull response_text + url_citation annotations out of a Responses API body.

    The Responses API returns an ``output`` array containing one or more items.
    ``web_search_call`` items carry the search action; ``message`` items carry
    the assistant's output, where each ``output_text`` content block may
    include a list of ``url_citation`` annotations.
    """
    output = body.get("output") or []
    if not isinstance(output, list):
        return "", [], []
    text_parts: list[str] = []
    citations: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for content_block in item.get("content") or []:
            if not isinstance(content_block, dict):
                continue
            if content_block.get("type") != "output_text":
                continue
            text_parts.append(str(content_block.get("text") or ""))
            for annotation in content_block.get("annotations") or []:
                if not isinstance(annotation, dict):
                    continue
                if annotation.get("type") != "url_citation":
                    continue
                citations.append({
                    "url": annotation.get("url") or "",
                    "title": annotation.get("title") or "",
                    "start_index": annotation.get("start_index"),
                    "end_index": annotation.get("end_index"),
                })
    return "\n\n".join(p for p in text_parts if p).strip(), citations, output


def _extract_cost_usd(body: dict[str, Any]) -> float:
    """Pull approximate cost from ``usage.total_cost_usd`` if the provider
    surfaces it; otherwise return 0.0 and let the caller substitute the
    fallback estimate.
    """
    usage = body.get("usage") or {}
    if not isinstance(usage, dict):
        return 0.0
    for key in ("total_cost_usd", "cost_usd", "total_cost"):
        if key in usage:
            try:
                return float(usage[key])
            except (TypeError, ValueError):
                return 0.0
    return 0.0
