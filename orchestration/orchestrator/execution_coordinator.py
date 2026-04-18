"""Execution coordinator — executes a CompositionPlan step by step.

The coordinator walks through the plan, invoking platform tools and
specialist families as needed, collecting results, and producing the
final response. It handles retries, circuit breaking, and step
dependency resolution.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

from shared.common.circuit_breaker import CircuitBreaker
from orchestration.orchestrator.composition_planner import CompositionPlan, ExecutionStep
from orchestration.orchestrator.platform_tools.tools_registry import ToolsRegistry

_logger = logging.getLogger(__name__)
_miner_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)

OWNER_API_URL = os.getenv("OWNER_API_URL", "http://owner-api:8000")
INTERNAL_SERVICE_TOKEN = os.getenv("EIREL_INTERNAL_SERVICE_TOKEN", "")


class StepResult:
    """Result from executing a single plan step."""

    def __init__(
        self,
        *,
        step_id: str,
        step_type: str,
        status: str = "completed",
        output: dict[str, Any] | None = None,
        error: str | None = None,
        latency_ms: float = 0.0,
        family_id: str | None = None,
        tool_name: str | None = None,
        miner_hotkey: str | None = None,
    ):
        self.step_id = step_id
        self.step_type = step_type
        self.status = status
        self.output = output or {}
        self.error = error
        self.latency_ms = latency_ms
        self.family_id = family_id
        self.tool_name = tool_name
        self.miner_hotkey = miner_hotkey

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
            "family_id": self.family_id,
            "tool_name": self.tool_name,
            "miner_hotkey": self.miner_hotkey,
        }


class ExecutionCoordinator:
    """Executes a CompositionPlan and collects results."""

    def __init__(self, *, tools_registry: ToolsRegistry):
        self.tools_registry = tools_registry
        self._step_results: dict[str, StepResult] = {}

    async def execute_plan(
        self,
        *,
        plan: CompositionPlan,
        prompt: str,
        user_id: str = "anonymous",
        session_id: str | None = None,
        context_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Execute all steps in the plan and return the final response."""
        self._step_results.clear()
        overall_start = time.monotonic()

        for step in plan.steps:
            # Wait for dependencies
            if step.depends_on:
                for dep_id in step.depends_on:
                    if dep_id not in self._step_results:
                        _logger.warning("missing dependency %s for step %s", dep_id, step.step_id)

            result = await self._execute_step(
                step=step,
                prompt=prompt,
                user_id=user_id,
                session_id=session_id,
                context_history=context_history,
                prior_results=dict(self._step_results),
            )
            self._step_results[step.step_id] = result

            # If a critical step fails, abort
            if result.status == "failed" and step.step_type == "specialist":
                _logger.error("specialist step %s failed: %s", step.step_id, result.error)
                break

        total_ms = (time.monotonic() - overall_start) * 1000
        return self._build_response(
            plan=plan,
            prompt=prompt,
            total_ms=total_ms,
        )

    async def _execute_step(
        self,
        *,
        step: ExecutionStep,
        prompt: str,
        user_id: str,
        session_id: str | None,
        context_history: list[dict[str, Any]] | None,
        prior_results: dict[str, StepResult],
    ) -> StepResult:
        """Execute a single step."""
        start = time.monotonic()

        if step.step_type == "direct":
            return StepResult(
                step_id=step.step_id,
                step_type="direct",
                status="completed",
                output={"response": prompt},
                latency_ms=(time.monotonic() - start) * 1000,
            )

        if step.step_type == "platform_tool":
            return await self._execute_tool_step(step=step, prompt=prompt, session_id=session_id)

        if step.step_type == "specialist":
            return await self._execute_specialist_step(
                step=step,
                prompt=prompt,
                user_id=user_id,
                session_id=session_id,
                context_history=context_history,
                prior_results=prior_results,
            )

        if step.step_type == "synthesis":
            return self._execute_synthesis_step(
                step=step,
                prior_results=prior_results,
            )

        return StepResult(
            step_id=step.step_id,
            step_type=step.step_type,
            status="failed",
            error=f"unknown step type: {step.step_type}",
            latency_ms=(time.monotonic() - start) * 1000,
        )

    async def _execute_tool_step(
        self,
        *,
        step: ExecutionStep,
        prompt: str,
        session_id: str | None,
    ) -> StepResult:
        """Invoke a platform tool."""
        tool_name = step.tool_name or ""
        params = dict(step.params)
        params.setdefault("query", prompt)
        params.setdefault("session_id", session_id)

        tool_result = await self.tools_registry.invoke(tool_name, params)
        return StepResult(
            step_id=step.step_id,
            step_type="platform_tool",
            status="completed" if tool_result.success else "failed",
            output=tool_result.to_dict(),
            error=tool_result.error,
            latency_ms=tool_result.latency_ms,
            tool_name=tool_name,
        )

    async def _execute_specialist_step(
        self,
        *,
        step: ExecutionStep,
        prompt: str,
        user_id: str,
        session_id: str | None,
        context_history: list[dict[str, Any]] | None,
        prior_results: dict[str, StepResult],
    ) -> StepResult:
        """Route to a specialist family via the owner-api miner registry."""
        family_id = step.family_id or ""
        start = time.monotonic()

        # Build context from prior step results
        prior_context = []
        for dep_id in step.depends_on:
            dep = prior_results.get(dep_id)
            if dep and dep.status == "completed":
                prior_context.append({
                    "step_id": dep.step_id,
                    "type": dep.step_type,
                    "output": dep.output,
                })

        # Get winner miner for this family from owner-api
        miner = await self._get_serving_miner(family_id)
        if miner is None:
            return StepResult(
                step_id=step.step_id,
                step_type="specialist",
                status="failed",
                error=f"no serving miner available for family {family_id}",
                latency_ms=(time.monotonic() - start) * 1000,
                family_id=family_id,
            )

        # Invoke the miner
        invocation_payload = {
            "task_id": session_id or "ephemeral",
            "family_id": family_id,
            "raw_input": prompt,
            "subtask": step.subtask,
            "context": prior_context,
            "context_history": context_history or [],
            "constraints": {},
            "metadata": {"user_id": user_id},
        }

        try:
            async def _invoke() -> httpx.Response:
                async with httpx.AsyncClient(timeout=step.timeout_seconds) as client:
                    resp = await client.post(
                        f"{miner['endpoint'].rstrip('/')}/v1/invoke",
                        json=invocation_payload,
                    )
                    resp.raise_for_status()
                    return resp

            response = await _miner_circuit_breaker.call(
                f"miner-{family_id}-{miner['hotkey'][:8]}",
                _invoke,
            )
            result_data = response.json()
            elapsed = (time.monotonic() - start) * 1000
            return StepResult(
                step_id=step.step_id,
                step_type="specialist",
                status="completed",
                output=result_data,
                latency_ms=elapsed,
                family_id=family_id,
                miner_hotkey=miner.get("hotkey"),
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            _logger.error("specialist invocation failed for %s: %s", family_id, exc)
            return StepResult(
                step_id=step.step_id,
                step_type="specialist",
                status="failed",
                error=str(exc),
                latency_ms=elapsed,
                family_id=family_id,
            )

    async def _get_serving_miner(self, family_id: str) -> dict[str, Any] | None:
        """Fetch the winner (serving) miner for a family from owner-api."""
        headers: dict[str, str] = {}
        if INTERNAL_SERVICE_TOKEN:
            headers["Authorization"] = f"Bearer {INTERNAL_SERVICE_TOKEN}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{OWNER_API_URL}/v1/internal/serving/{family_id}",
                    headers=headers,
                )
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                data = resp.json()
                if not data.get("endpoint"):
                    return None
                return data
        except Exception as exc:
            _logger.error("failed to fetch serving miner for %s: %s", family_id, exc)
            return None

    def _execute_synthesis_step(
        self,
        *,
        step: ExecutionStep,
        prior_results: dict[str, StepResult],
    ) -> StepResult:
        """Combine results from prior steps."""
        combined: list[dict[str, Any]] = []
        for dep_id in step.depends_on:
            dep = prior_results.get(dep_id)
            if dep and dep.status == "completed":
                combined.append(dep.output)
        return StepResult(
            step_id=step.step_id,
            step_type="synthesis",
            status="completed",
            output={"combined_results": combined},
        )

    def _build_response(
        self,
        *,
        plan: CompositionPlan,
        prompt: str,
        total_ms: float,
    ) -> dict[str, Any]:
        """Build the final response from all step results."""
        step_results = [r.to_dict() for r in self._step_results.values()]
        all_completed = all(r.status == "completed" for r in self._step_results.values())

        # Extract the main response content
        final_output = self._extract_final_output()

        return {
            "status": "completed" if all_completed else "partial",
            "route_type": plan.route_type,
            "workflow_template": plan.workflow_template,
            "plan_id": plan.plan_id,
            "response": final_output,
            "steps": step_results,
            "total_latency_ms": round(total_ms, 1),
            "metadata": {
                "step_count": len(step_results),
                "completed_steps": sum(1 for r in self._step_results.values() if r.status == "completed"),
                "failed_steps": sum(1 for r in self._step_results.values() if r.status == "failed"),
            },
        }

    def _extract_final_output(self) -> dict[str, Any]:
        """Extract the primary response from step results."""
        results_list = list(self._step_results.values())
        if not results_list:
            return {}

        # For direct responses, return the direct step output
        for r in results_list:
            if r.step_type == "direct" and r.status == "completed":
                return r.output

        # For synthesis steps, return the combined output
        for r in reversed(results_list):
            if r.step_type == "synthesis" and r.status == "completed":
                return r.output

        # Otherwise return the last successful specialist or tool result
        for r in reversed(results_list):
            if r.status == "completed":
                return r.output

        # All failed
        errors = [r.error for r in results_list if r.error]
        return {"error": "; ".join(errors) if errors else "execution failed"}
