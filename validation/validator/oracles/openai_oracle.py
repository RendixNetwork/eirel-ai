"""OpenAI oracle client.

Wraps the OpenAI-compatible provider client pointed at OpenAI's native
API (``gpt-5.4`` by default). Returns a single ``OracleGrounding``;
provider errors become ``status="error"`` rather than propagating.
"""

from __future__ import annotations

import logging

from validation.validator.eval_config import openai_oracle_config
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


class OpenAIOracle(OracleClient):
    """OpenAI ``gpt-5.4`` oracle."""

    def __init__(
        self,
        client: OpenAICompatibleClient | None = None,
    ) -> None:
        self._client = client or OpenAICompatibleClient(openai_oracle_config())

    @property
    def vendor(self) -> str:
        return "openai"

    async def produce_grounding(
        self, context: OracleContext,
    ) -> OracleGrounding:
        system, user = build_oracle_messages(context)
        try:
            # OpenAI exposes the server-side ``web_search`` tool only on
            # the Responses API. Live-lookup / current-events items
            # require it; static items still benefit (model double-
            # checks recall against fresh web results).
            resp = await self._client.complete_responses_with_web_search(
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
