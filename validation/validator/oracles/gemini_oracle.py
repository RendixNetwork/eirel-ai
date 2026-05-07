"""Gemini oracle client.

Wraps the Gemini provider client (``gemini-3.1-pro-preview`` by
default). Surfaces safety blocks and provider errors as
``OracleGrounding(status="error" or "blocked")`` for graceful
degradation in the reconciler.
"""

from __future__ import annotations

import logging

from validation.validator.eval_config import gemini_oracle_config
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
from validation.validator.providers.gemini import GeminiClient
from validation.validator.providers.types import (
    ProviderError,
    ProviderTimeout,
)

_logger = logging.getLogger(__name__)


class GeminiOracle(OracleClient):
    """Gemini ``gemini-3.1-pro-preview`` oracle."""

    def __init__(
        self,
        client: GeminiClient | None = None,
    ) -> None:
        self._client = client or GeminiClient(gemini_oracle_config())

    @property
    def vendor(self) -> str:
        return "gemini"

    async def produce_grounding(
        self, context: OracleContext,
    ) -> OracleGrounding:
        system, user = build_oracle_messages(context)
        try:
            resp = await self._client.complete_structured(
                system=system,
                user=user,
                response_schema=response_schema(),
                schema_name="oracle_answer",
                enable_web_search=True,
            )
        except ProviderTimeout as exc:
            return OracleGrounding(
                vendor=self.vendor, status="error",
                error_msg=f"timeout: {exc}",
            )
        except ProviderError as exc:
            # Gemini blocks via ``promptFeedback.blockReason`` — the
            # provider client raises ProviderError("gemini blocked the
            # prompt: ..."). Map that to status="blocked" so operators
            # can distinguish vendor-refusal from network errors.
            msg = str(exc)
            status: str = "blocked" if "blocked" in msg.lower() else "error"
            return OracleGrounding(
                vendor=self.vendor,
                status=status,  # type: ignore[arg-type]
                error_msg=f"provider_error: {msg}",
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
