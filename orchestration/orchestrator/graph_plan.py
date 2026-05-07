"""Graph-runtime composition planner.

Today every plan is single-step: route the prompt to one miner
running the selected family. The data shape is kept generic so future
multi-family plans (e.g. ``deep_research → planner → coding``) can be
added without changing the executor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eirel.groups import FamilyId

from orchestration.orchestrator.family_selection import FamilySelection

__all__ = ["GraphPlanStep", "GraphPlan", "build_graph_plan"]


@dataclass(frozen=True, slots=True)
class GraphPlanStep:
    """One specialist invocation."""

    step_id: str
    family_id: FamilyId
    timeout_seconds: float = 60.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "family_id": self.family_id,
            "timeout_seconds": self.timeout_seconds,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class GraphPlan:
    """An ordered execution plan for the graph-runtime orchestrator."""

    plan_id: str
    steps: tuple[GraphPlanStep, ...]
    selection: FamilySelection
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "steps": [s.to_dict() for s in self.steps],
            "selection": self.selection.to_dict(),
            "metadata": dict(self.metadata),
        }


def build_graph_plan(
    *,
    selection: FamilySelection,
    plan_id: str | None = None,
    timeout_seconds: float = 60.0,
) -> GraphPlan:
    """Build a single-step plan for the selected family.

    Multi-step plans are reserved for cross-family workflows that
    aren't active today.
    """
    from uuid import uuid4

    pid = plan_id or uuid4().hex[:12]
    step = GraphPlanStep(
        step_id="step-1",
        family_id=selection.family_id,
        timeout_seconds=timeout_seconds,
    )
    return GraphPlan(
        plan_id=pid,
        steps=(step,),
        selection=selection,
        metadata={
            "single_family_fast_path": len(selection.available_families) == 1,
        },
    )
