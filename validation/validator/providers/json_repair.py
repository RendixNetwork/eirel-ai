"""JSON-repair retry wrapper for structured-output provider calls.

Some judge / reconciler models (especially TEE-hosted ones with
imperfect JSON-mode adherence) emit malformed JSON intermittently.
A model with a 92-95% parse rate is unusable for production scoring
out of the box but recoverable with a 2-retry repair loop:

  1. First call → response not parseable as JSON.
  2. Re-prompt with the parse error appended and "previous response was
     malformed; emit ONLY valid JSON matching the schema."
  3. If retry 2 also fails, surface the original ProviderError.

The wrapper sits between the per-role caller (judge module / reconciler)
and the underlying ``OpenAICompatibleClient`` / ``GeminiClient``. It's
pure plumbing — no schema awareness; the caller passes the same
schema dict to the wrapper as it would to the underlying client.

Below 90% raw parse rate the model isn't ready and should be swapped;
the wrapper is for the 90-98% recovery zone.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from validation.validator.providers.types import (
    ProviderError,
    ProviderResponse,
    ProviderTimeout,
)

_logger = logging.getLogger(__name__)


class _StructuredCompleter(Protocol):
    """Subset of provider client surface ``with_json_repair`` wraps."""

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        response_schema: dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        schema_name: str = "response",
    ) -> ProviderResponse: ...


_REPAIR_INSTRUCTION = (
    "Your previous response was malformed JSON: {error}\n"
    "Re-emit your answer as STRICT JSON matching the response schema. "
    "Do not include any prose, code fences, or explanation outside "
    "the JSON object."
)


class JsonRepairClient:
    """Wraps a provider client with bounded retry on JSON parse errors.

    Caller-facing API matches ``_StructuredCompleter`` so the wrapper
    is drop-in. Returns the same ``ProviderResponse`` whose ``text``
    field is the (now-parseable) JSON string. The caller still calls
    ``json.loads`` on the result; the wrapper just gives them more
    chances at a valid response.
    """

    def __init__(
        self,
        wrapped: _StructuredCompleter,
        *,
        max_retries: int = 2,
    ) -> None:
        self._wrapped = wrapped
        self._max_retries = max(0, int(max_retries))

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
        last_error: Exception | None = None
        # Attempt 0 = original call; attempts 1..max_retries are repairs.
        # Each repair re-prompts with the parse error included.
        current_user = user
        attempt = 0
        while True:
            response = await self._wrapped.complete_structured(
                system=system,
                user=current_user,
                response_schema=response_schema,
                temperature=temperature,
                max_tokens=max_tokens,
                schema_name=schema_name,
            )
            try:
                json.loads(response.text)
            except (TypeError, ValueError) as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    raise ProviderError(
                        f"JSON parse failed after "
                        f"{attempt + 1} attempt(s): {exc}"
                    ) from exc
                _logger.warning(
                    "JSON parse failed on attempt %d, retrying: %s",
                    attempt + 1, exc,
                )
                # Append the repair instruction to the user prompt so
                # the model sees both the original ask AND the parse
                # error. Keeps the schema/system intact.
                current_user = (
                    user
                    + "\n\n"
                    + _REPAIR_INSTRUCTION.format(error=str(exc)[:200])
                )
                attempt += 1
                continue
            return response


def with_json_repair(
    client: _StructuredCompleter, *, max_retries: int = 2,
) -> JsonRepairClient:
    """Construction helper. Returns a wrapper that delegates to
    ``client`` with up to ``max_retries`` re-prompts on parse failure."""
    return JsonRepairClient(client, max_retries=max_retries)


__all__ = [
    "JsonRepairClient",
    "with_json_repair",
]
