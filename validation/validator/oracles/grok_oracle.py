"""Grok oracle client.

xAI's REST ``/v1/chat/completions`` rejects ``tools[].type=web_search``
(only ``function`` and the deprecated ``live_search``). Native
server-side web search is exposed via xAI's OpenAI-compatible
``/v1/responses`` endpoint — same Responses-API shape as OpenAI's
GPT-5 family. We reuse ``OpenAICompatibleClient`` for the call;
the only thing that changes from OpenAI is the base_url + key + model.

Provider errors and refusals become ``status="error"`` rather than
propagating, so the reconciler sees a stable shape across vendors.
"""

from __future__ import annotations

import logging

from validation.validator.eval_config import grok_oracle_config
from validation.validator.oracles._helpers import (
    build_oracle_messages,
    extract_answer,
    response_schema,
)
from validation.validator.oracles.base import (
    OracleClient,
    OracleContext,
    OracleGrounding,
)
from validation.validator.providers.openai_compatible import (
    OpenAICompatibleClient,
)
from validation.validator.providers.types import (
    ProviderError,
    ProviderTimeout,
)

_logger = logging.getLogger(__name__)


class GrokOracle(OracleClient):
    """xAI ``grok-4.3`` oracle via ``/v1/responses`` + ``web_search``."""

    def __init__(
        self,
        client: OpenAICompatibleClient | None = None,
    ) -> None:
        self._client = client or OpenAICompatibleClient(grok_oracle_config())

    @property
    def vendor(self) -> str:
        return "grok"

    async def produce_grounding(
        self, context: OracleContext,
    ) -> OracleGrounding:
        system, user = build_oracle_messages(context)
        try:
            if context.web_search:
                resp = await self._client.complete_responses_with_web_search(
                    system=system,
                    user=user,
                    response_schema=response_schema(),
                    schema_name="oracle_answer",
                )
            else:
                # Self-contained task — chat-completions, no web_search
                # adder. Mirrors the miner's tooling for that task.
                resp = await self._client.complete_structured(
                    system=system,
                    user=user,
                    response_schema=response_schema(),
                    schema_name="oracle_answer",
                )
        except ProviderTimeout as exc:
            return OracleGrounding(
                vendor=self.vendor, status="error",
                error_msg=f"timeout: {exc}",
            )
        except ProviderError as exc:
            return OracleGrounding(
                vendor=self.vendor, status="error",
                error_msg=f"provider_error: {exc}",
            )
        try:
            answer = extract_answer(resp.text)
        except ValueError as exc:
            return OracleGrounding(
                vendor=self.vendor, status="error",
                error_msg=f"malformed_response: {exc}",
                latency_ms=resp.latency_ms,
                cost_usd=resp.usage_usd,
                finish_reason=resp.finish_reason,
            )
        return OracleGrounding(
            vendor=self.vendor,
            status="ok",
            raw_text=answer,
            latency_ms=resp.latency_ms,
            cost_usd=resp.usage_usd,
            finish_reason=resp.finish_reason,
            citations=resp.citations,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
