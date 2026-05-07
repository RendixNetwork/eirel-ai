"""Validator-side LLM provider clients.

Two transport classes cover all four roles the validator drives:

  * ``OpenAICompatibleClient`` — OpenAI native API, xAI/Grok native API,
    Chutes (used for the GLM-5.1-TEE reconciler). All speak
    OpenAI-style ``POST /chat/completions`` with ``response_format`` for
    structured-output mode.
  * ``GeminiClient`` — Google's Generative Language API uses a
    different request shape (``contents`` + ``systemInstruction``,
    ``responseSchema`` + ``responseMimeType``). Implemented as a
    separate client rather than wrapping it in an OpenAI-compatible
    adapter, since the response/tool shapes drift.

Both expose the same method signature so callers can treat them
interchangeably:

    async def complete_structured(
        *,
        system: str,
        user: str,
        response_schema: dict,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse: ...
"""

from __future__ import annotations

from validation.validator.providers.gemini import GeminiClient
from validation.validator.providers.openai_compatible import (
    OpenAICompatibleClient,
)
from validation.validator.providers.types import (
    ProviderError,
    ProviderResponse,
    ProviderTimeout,
)

__all__ = [
    "GeminiClient",
    "OpenAICompatibleClient",
    "ProviderError",
    "ProviderResponse",
    "ProviderTimeout",
]
