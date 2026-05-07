"""Platform tool: web_search

Performs web searches via the existing research-tool-service. Returns
structured search results that the orchestrator can use to augment
specialist responses or answer directly.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from orchestration.orchestrator.platform_tools.base import PlatformTool, ToolResult

_logger = logging.getLogger(__name__)

WEB_SEARCH_TOOL_URL = os.getenv(
    "EIREL_WEB_SEARCH_TOOL_URL", "http://web-search-tool-service:8085"
)
SEARCH_TIMEOUT = float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "20"))
MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "10"))


class WebSearchTool(PlatformTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for current information. Returns structured results "
            "with titles, snippets, and URLs."
        )

    async def execute(self, *, params: dict[str, Any]) -> ToolResult:
        query = params.get("query", "")
        if not query.strip():
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="no search query provided",
            )

        payload = {
            "query": query,
            "max_results": min(params.get("max_results", MAX_RESULTS), MAX_RESULTS),
        }
        try:
            async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
                resp = await client.post(
                    f"{WEB_SEARCH_TOOL_URL}/v1/search",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output={
                        "query": query,
                        "results": results[:MAX_RESULTS],
                        "total": len(results),
                    },
                )
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"research-tool-service returned {exc.response.status_code}",
            )
        except httpx.ConnectError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="research-tool-service unavailable",
            )
