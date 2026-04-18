"""Platform tool: memory_recall

Retrieves relevant context from conversation history and the retrieval
service. Enables multi-turn conversations by recalling prior exchanges
and relevant knowledge.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from orchestration.orchestrator.platform_tools.base import PlatformTool, ToolResult

_logger = logging.getLogger(__name__)

RETRIEVAL_SERVICE_URL = os.getenv(
    "EIREL_RETRIEVAL_SERVICE_URL", "http://retrieval-service:8091"
)
RETRIEVAL_TIMEOUT = float(os.getenv("MEMORY_RECALL_TIMEOUT_SECONDS", "10"))


class MemoryRecallTool(PlatformTool):
    @property
    def name(self) -> str:
        return "memory_recall"

    @property
    def description(self) -> str:
        return (
            "Recall relevant information from conversation history and "
            "the knowledge retrieval system."
        )

    async def execute(self, *, params: dict[str, Any]) -> ToolResult:
        query = params.get("query", "")
        session_id = params.get("session_id")
        context_history = params.get("context_history", [])

        results: dict[str, Any] = {
            "conversation_context": [],
            "retrieved_knowledge": [],
        }

        # Extract relevant conversation context
        if context_history:
            relevant = self._search_context(query, context_history)
            results["conversation_context"] = relevant

        # Query retrieval service for external knowledge
        if query.strip():
            knowledge = await self._retrieve_knowledge(query)
            results["retrieved_knowledge"] = knowledge

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=results,
        )

    def _search_context(
        self, query: str, history: list[dict[str, Any]], max_results: int = 5
    ) -> list[dict[str, Any]]:
        """Simple keyword-based context search over conversation history."""
        query_terms = set(query.lower().split())
        scored: list[tuple[float, dict[str, Any]]] = []
        for msg in history:
            content = str(msg.get("content", "")).lower()
            overlap = len(query_terms & set(content.split()))
            if overlap > 0:
                scored.append((overlap, msg))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:max_results]]

    async def _retrieve_knowledge(self, query: str) -> list[dict[str, Any]]:
        """Query the retrieval service for relevant knowledge."""
        try:
            async with httpx.AsyncClient(timeout=RETRIEVAL_TIMEOUT) as client:
                resp = await client.post(
                    f"{RETRIEVAL_SERVICE_URL}/v1/search",
                    json={"query": query, "top_k": 5},
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("results", [])
        except Exception as exc:
            _logger.warning("retrieval service error: %s", exc)
            return []
