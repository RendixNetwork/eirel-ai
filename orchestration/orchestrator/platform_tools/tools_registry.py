"""Platform tools registry.

Central registry for all Tier 1 platform tools. The orchestrator uses this
to discover available tools and invoke them by name.
"""

from __future__ import annotations

import logging
from typing import Any

from orchestration.orchestrator.platform_tools.base import PlatformTool, ToolResult
from orchestration.orchestrator.platform_tools.code_executor import CodeExecutorTool
from orchestration.orchestrator.platform_tools.web_search import WebSearchTool
from orchestration.orchestrator.platform_tools.file_manager import FileManagerTool
from orchestration.orchestrator.platform_tools.image_gen import ImageGenTool
from orchestration.orchestrator.platform_tools.memory_tool import MemoryRecallTool

_logger = logging.getLogger(__name__)


class ToolsRegistry:
    """Registry of all available platform tools."""

    def __init__(self) -> None:
        self._tools: dict[str, PlatformTool] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        for tool in [
            CodeExecutorTool(),
            WebSearchTool(),
            FileManagerTool(),
            ImageGenTool(),
            MemoryRecallTool(),
        ]:
            self.register(tool)

    def register(self, tool: PlatformTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> PlatformTool | None:
        return self._tools.get(name)

    def available_tools(self) -> list[str]:
        return list(self._tools.keys())

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    async def invoke(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"unknown tool: {tool_name}",
            )
        return await tool.invoke(params=params)
