"""Base class for platform tools.

Platform tools are Tier 1 capabilities owned by the subnet infrastructure.
They run deterministic operations (code execution, web search, file I/O)
that don't need specialist miner competition.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any


class ToolResult:
    """Standardized result from a platform tool invocation."""

    def __init__(
        self,
        *,
        tool_name: str,
        success: bool,
        output: Any = None,
        error: str | None = None,
        latency_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ):
        self.tool_name = tool_name
        self.success = success
        self.output = output
        self.error = error
        self.latency_ms = latency_ms
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
            "metadata": self.metadata,
        }


class PlatformTool(ABC):
    """Abstract base for all platform tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier (e.g. 'code_exec', 'web_search')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the orchestrator's tool selection."""

    @abstractmethod
    async def execute(self, *, params: dict[str, Any]) -> ToolResult:
        """Execute the tool with the given parameters."""

    async def invoke(self, *, params: dict[str, Any]) -> ToolResult:
        """Wrapper that adds timing and error handling."""
        start = time.monotonic()
        try:
            result = await self.execute(params=params)
            result.latency_ms = (time.monotonic() - start) * 1000
            return result
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
                latency_ms=elapsed,
            )

    def schema(self) -> dict[str, Any]:
        """Return a JSON-serializable tool schema for the orchestrator."""
        return {
            "name": self.name,
            "description": self.description,
        }
