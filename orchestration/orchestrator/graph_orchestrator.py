"""Graph-runtime orchestrator entrypoint.

Stitches family selection + plan + miner picker + executor into one
call. This is the surface the consumer-api hands a user prompt to.

The orchestrator is single-family today (general_chat); multi-family
routing is reserved for later milestones. The function signatures
deliberately accept all the primitives that future families will
need (``available_families``, multi-step plans) so adding a family is
purely additive — the consumer-api never needs to change.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from eirel.groups import FamilyId

from orchestration.orchestrator.family_selection import (
    FamilySelection,
    select_family_for_prompt,
)
from orchestration.orchestrator.graph_executor import (
    ConsumerStreamEvent,
    GraphExecutor,
)
from orchestration.orchestrator.graph_plan import GraphPlan, build_graph_plan
from orchestration.orchestrator.miner_picker import (
    MinerPicker,
    NoEligibleMinerError,
)

_logger = logging.getLogger(__name__)

__all__ = ["GraphOrchestrator", "OrchestratorError"]


class OrchestratorError(RuntimeError):
    """Raised when the orchestrator can't serve the request.

    Wraps the underlying cause (no eligible miner, plan empty, etc.)
    so the consumer-api can surface a coherent 5xx without leaking
    internal stack traces.
    """


class GraphOrchestrator:
    """User-prompt-in, NDJSON-out orchestrator for the graph runtime."""

    def __init__(
        self,
        *,
        miner_picker: MinerPicker,
        executor: GraphExecutor,
        active_families: tuple[FamilyId, ...] = ("general_chat",),
    ):
        if not active_families:
            raise ValueError("at least one active family is required")
        self._picker = miner_picker
        self._executor = executor
        self._active_families = tuple(active_families)

    @property
    def active_families(self) -> tuple[FamilyId, ...]:
        return self._active_families

    def plan(
        self,
        *,
        prompt: str,
        context: dict[str, Any] | None = None,
        timeout_seconds: float = 60.0,
    ) -> GraphPlan:
        """Build the plan for a prompt.

        Surface separated from :meth:`invoke` / :meth:`astream` so the
        consumer-api can log/audit the plan before commit.
        """
        selection = select_family_for_prompt(
            prompt=prompt,
            available_families=self._active_families,
            context=context,
        )
        return build_graph_plan(
            selection=selection,
            timeout_seconds=timeout_seconds,
        )

    async def invoke(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]] | None = None,
        thread_id: str | None = None,
        resume_token: str | None = None,
        mode: str = "instant",
        web_search: bool = False,
        run_budget_usd: float | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        plan = self.plan(prompt=prompt, context=metadata)
        try:
            return await self._executor.invoke(
                plan=plan,
                prompt=prompt,
                history=history,
                thread_id=thread_id,
                resume_token=resume_token,
                mode=mode,
                web_search=web_search,
                run_budget_usd=run_budget_usd,
                run_id=run_id,
                metadata=metadata,
            )
        except NoEligibleMinerError as exc:
            raise OrchestratorError(str(exc)) from exc

    async def astream(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]] | None = None,
        thread_id: str | None = None,
        resume_token: str | None = None,
        mode: str = "instant",
        web_search: bool = False,
        run_budget_usd: float | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[ConsumerStreamEvent]:
        plan = self.plan(prompt=prompt, context=metadata)
        try:
            async for event in self._executor.astream(
                plan=plan,
                prompt=prompt,
                history=history,
                thread_id=thread_id,
                resume_token=resume_token,
                mode=mode,
                web_search=web_search,
                run_budget_usd=run_budget_usd,
                run_id=run_id,
                metadata=metadata,
            ):
                yield event
        except NoEligibleMinerError as exc:
            # Surface as a final failed-done event so the NDJSON
            # contract holds even for empty-fleet outages.
            yield ConsumerStreamEvent({
                "event": "done",
                "status": "failed",
                "error": f"orchestrator: {exc}",
                "metadata": {"plan": plan.to_dict()},
            })
