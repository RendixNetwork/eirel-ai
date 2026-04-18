from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


SUPPORTED_PROVIDER_IDS = ("chutes",)
PROVIDER_ENV_MAP = {
    "openai": (
        "EIREL_PROVIDER_OPENAI_BASE_URL",
        "EIREL_PROVIDER_OPENAI_API_KEY",
        "https://api.openai.com/v1/chat/completions",
    ),
    "anthropic": (
        "EIREL_PROVIDER_ANTHROPIC_BASE_URL",
        "EIREL_PROVIDER_ANTHROPIC_API_KEY",
        "https://api.anthropic.com/v1/messages",
    ),
    "openrouter": (
        "EIREL_PROVIDER_OPENROUTER_BASE_URL",
        "EIREL_PROVIDER_OPENROUTER_API_KEY",
        "https://openrouter.ai/api/v1/chat/completions",
    ),
    "chutes": (
        "EIREL_PROVIDER_CHUTES_BASE_URL",
        "EIREL_PROVIDER_CHUTES_API_KEY",
        "https://api.chutes.ai/v1/chat/completions",
    ),
}


class ProviderProxyRequest(BaseModel):
    provider: str
    model: str
    payload: dict[str, Any]
    max_requests: int = Field(default=24, ge=1)
    max_total_tokens: int = Field(default=60000, ge=1)
    max_wall_clock_seconds: int = Field(default=300, ge=1)
    per_request_timeout_seconds: int = Field(default=60, ge=1)


class ProviderProxyUsage(BaseModel):
    provider: str
    model: str
    request_count: int
    estimated_total_tokens: int
    max_requests: int
    max_total_tokens: int
    actual_prompt_tokens: int = 0
    actual_completion_tokens: int = 0
    actual_total_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    cost_usd_used: float = 0.0
    cost_remaining_usd: float = 0.0


class ProviderProxyResponse(BaseModel):
    upstream_response: dict[str, Any]
    usage: ProviderProxyUsage
