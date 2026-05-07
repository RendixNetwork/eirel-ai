from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RelayCallRequest(BaseModel):
    """Body for ``POST /v1/relay/connections/{connection_id}/call``."""

    tool_name: str = Field(min_length=1, max_length=128)
    args: dict[str, Any] = Field(default_factory=dict)
    capabilities_hash: str | None = Field(default=None, max_length=64)


class RelayCallResponse(BaseModel):
    """Outcome returned to the orchestrator's :class:`MCPToolDispatcher`."""

    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: int = 0
    cost_usd: float = 0.0


class RelayListToolsResponse(BaseModel):
    """Operator/admin tool — used to refresh ``capabilities_hash``."""

    tools: list[dict[str, Any]] = Field(default_factory=list)
    capabilities_hash: str = ""
