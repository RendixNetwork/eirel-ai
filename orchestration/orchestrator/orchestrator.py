"""Core orchestrator — the brain of the EIREL subnet.

This is the central module that ties together family selection,
composition planning, and execution coordination. It replaces the
old classifier → DAG → executor pipeline with a conversational,
streaming-ready architecture.

The orchestrator is subnet-owned infrastructure (not a competing family).
It controls the conversation experience and routes to platform tools
(Tier 1) and specialist families (Tier 2) as needed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any
from uuid import uuid4

_REQUEST_TIMEOUT = float(os.getenv("ORCHESTRATOR_REQUEST_TIMEOUT_SECONDS", "120"))

from orchestration.orchestrator.family_selector import RoutingDecision, select_route
from orchestration.orchestrator.composition_planner import CompositionPlan, build_plan
from orchestration.orchestrator.execution_coordinator import ExecutionCoordinator
from orchestration.orchestrator.platform_tools.tools_registry import ToolsRegistry

_logger = logging.getLogger(__name__)


class Orchestrator:
    """Main orchestrator that processes chat requests end-to-end."""

    def __init__(self) -> None:
        self.tools_registry = ToolsRegistry()
        self.coordinator = ExecutionCoordinator(tools_registry=self.tools_registry)

    async def handle_request(
        self,
        *,
        prompt: str,
        user_id: str = "anonymous",
        session_id: str | None = None,
        context_history: list[dict[str, Any]] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process a chat request through the full orchestration pipeline.

        1. Route selection (family_selector)
        2. Plan composition (composition_planner)
        3. Plan execution (execution_coordinator)
        4. Response assembly
        """
        try:
            return await asyncio.wait_for(
                self._handle_request_inner(
                    prompt=prompt,
                    user_id=user_id,
                    session_id=session_id,
                    context_history=context_history,
                    constraints=constraints,
                ),
                timeout=_REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            _logger.error("orchestrator request timed out after %ss", _REQUEST_TIMEOUT)
            return {
                "request_id": "timeout",
                "status": "failed",
                "error": f"orchestrator request timed out after {_REQUEST_TIMEOUT}s",
            }

    async def _handle_request_inner(
        self,
        *,
        prompt: str,
        user_id: str = "anonymous",
        session_id: str | None = None,
        context_history: list[dict[str, Any]] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_id = str(uuid4())[:12]
        session_id = session_id or str(uuid4())
        start = time.monotonic()

        _logger.info(
            "orchestrator request %s: user=%s prompt_len=%d",
            request_id, user_id, len(prompt),
        )

        # Step 1: Route selection
        excluded_families = []
        modalities = ["text"]
        if constraints:
            excluded_families = constraints.get("families_excluded", [])
            modalities = constraints.get("modalities_allowed", ["text"])

        decision: RoutingDecision = select_route(
            prompt=prompt,
            context_history=context_history,
            modalities_allowed=modalities,
            families_excluded=excluded_families,
        )

        _logger.info(
            "orchestrator %s routed as %s (template=%s, families=%s, tools=%s)",
            request_id,
            decision.route_type,
            decision.workflow_template,
            decision.families,
            decision.platform_tools,
        )

        # Step 2: Composition planning
        plan: CompositionPlan = build_plan(
            decision=decision,
            prompt=prompt,
            context_history=context_history,
            session_id=session_id,
        )

        # Step 3: Execute the plan
        result = await self.coordinator.execute_plan(
            plan=plan,
            prompt=prompt,
            user_id=user_id,
            session_id=session_id,
            context_history=context_history,
        )

        total_ms = (time.monotonic() - start) * 1000

        # Step 4: Assemble final response
        response = {
            "request_id": request_id,
            "session_id": session_id,
            "user_id": user_id,
            **result,
            "routing": decision.to_dict(),
            "plan": plan.to_dict(),
            "orchestrator_latency_ms": round(total_ms, 1),
        }

        _logger.info(
            "orchestrator %s completed in %.1fms (status=%s, steps=%d)",
            request_id, total_ms,
            result.get("status", "unknown"),
            len(result.get("steps", [])),
        )

        return response

    def available_tools(self) -> list[str]:
        """List available platform tools."""
        return self.tools_registry.available_tools()

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return schemas for all available platform tools."""
        return self.tools_registry.tool_schemas()
