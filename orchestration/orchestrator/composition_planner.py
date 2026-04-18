"""Composition planner — builds an execution plan from a routing decision.

Given a RoutingDecision from the family_selector, the composition planner
creates a step-by-step execution plan that the execution_coordinator will
follow. Plans can include platform tool invocations, specialist family
calls, and synthesis steps.
"""

from __future__ import annotations

import logging
from typing import Any, Literal
from uuid import uuid4

from orchestration.orchestrator.family_selector import RoutingDecision

_logger = logging.getLogger(__name__)


class ExecutionStep:
    """A single step in an orchestrator execution plan."""

    def __init__(
        self,
        *,
        step_id: str | None = None,
        step_type: Literal["platform_tool", "specialist", "synthesis", "direct"],
        tool_name: str | None = None,
        family_id: str | None = None,
        params: dict[str, Any] | None = None,
        subtask: str = "",
        depends_on: list[str] | None = None,
        timeout_seconds: float = 45.0,
    ):
        self.step_id = step_id or str(uuid4())[:8]
        self.step_type = step_type
        self.tool_name = tool_name
        self.family_id = family_id
        self.params = params or {}
        self.subtask = subtask
        self.depends_on = depends_on or []
        self.timeout_seconds = timeout_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "tool_name": self.tool_name,
            "family_id": self.family_id,
            "subtask": self.subtask,
            "depends_on": self.depends_on,
            "timeout_seconds": self.timeout_seconds,
        }


class CompositionPlan:
    """An ordered execution plan produced by the composition planner."""

    def __init__(
        self,
        *,
        plan_id: str | None = None,
        steps: list[ExecutionStep] | None = None,
        route_type: str = "direct",
        workflow_template: str = "direct_response",
        metadata: dict[str, Any] | None = None,
    ):
        self.plan_id = plan_id or str(uuid4())[:12]
        self.steps = steps or []
        self.route_type = route_type
        self.workflow_template = workflow_template
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "steps": [s.to_dict() for s in self.steps],
            "route_type": self.route_type,
            "workflow_template": self.workflow_template,
            "metadata": self.metadata,
        }


# Family timeout defaults (carried from dag_executor)
FAMILY_TIMEOUT_SECONDS: dict[str, float] = {
    "analyst": 45.0,
    "browser": 90.0,
    "builder": 300.0,
    "data": 60.0,
    "media": 120.0,
    "memory": 30.0,
    "planner": 15.0,
    "verifier": 45.0,
}


def build_plan(
    *,
    decision: RoutingDecision,
    prompt: str,
    context_history: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
) -> CompositionPlan:
    """Build an execution plan from a routing decision."""
    if decision.route_type == "direct":
        return _build_direct_plan(prompt=prompt)

    if decision.route_type == "platform_tool":
        return _build_tool_plan(
            decision=decision,
            prompt=prompt,
            session_id=session_id,
        )

    if decision.route_type == "specialist":
        return _build_specialist_plan(
            decision=decision,
            prompt=prompt,
        )

    # Composite: both tools and specialists
    return _build_composite_plan(
        decision=decision,
        prompt=prompt,
        session_id=session_id,
    )


def _build_direct_plan(*, prompt: str) -> CompositionPlan:
    """Plan for direct orchestrator response (no tools or specialists)."""
    return CompositionPlan(
        steps=[
            ExecutionStep(
                step_id="direct",
                step_type="direct",
                subtask=prompt,
                timeout_seconds=5.0,
            ),
        ],
        route_type="direct",
        workflow_template="direct_response",
    )


def _build_tool_plan(
    *,
    decision: RoutingDecision,
    prompt: str,
    session_id: str | None,
) -> CompositionPlan:
    """Plan that uses platform tools only."""
    steps: list[ExecutionStep] = []
    for tool_name in decision.platform_tools:
        steps.append(
            ExecutionStep(
                step_type="platform_tool",
                tool_name=tool_name,
                params={"query": prompt, "session_id": session_id},
                subtask=f"Execute {tool_name} for: {prompt[:120]}",
                timeout_seconds=30.0,
            )
        )
    # Add synthesis step to combine tool results
    if len(steps) > 1:
        steps.append(
            ExecutionStep(
                step_id="synthesis",
                step_type="synthesis",
                subtask="Combine platform tool results into a coherent response.",
                depends_on=[s.step_id for s in steps],
                timeout_seconds=5.0,
            )
        )
    return CompositionPlan(
        steps=steps,
        route_type="platform_tool",
        workflow_template="tool_execution",
    )


def _build_specialist_plan(
    *,
    decision: RoutingDecision,
    prompt: str,
) -> CompositionPlan:
    """Plan that routes to specialist families."""
    steps: list[ExecutionStep] = []
    previous_ids: list[str] = []

    for family_id in decision.families:
        if family_id == "verifier":
            continue  # Added at end
        step = ExecutionStep(
            step_type="specialist",
            family_id=family_id,
            subtask=_subtask_for_family(family_id, prompt),
            depends_on=list(previous_ids),
            timeout_seconds=FAMILY_TIMEOUT_SECONDS.get(family_id, 45.0),
        )
        steps.append(step)
        previous_ids.append(step.step_id)

    # Verifier at the end if present
    if "verifier" in decision.families:
        steps.append(
            ExecutionStep(
                step_type="specialist",
                family_id="verifier",
                subtask="Verify the output for accuracy, evidence alignment, and policy compliance.",
                depends_on=list(previous_ids),
                timeout_seconds=FAMILY_TIMEOUT_SECONDS["verifier"],
            )
        )

    return CompositionPlan(
        steps=steps,
        route_type="specialist",
        workflow_template=decision.workflow_template,
    )


def _build_composite_plan(
    *,
    decision: RoutingDecision,
    prompt: str,
    session_id: str | None,
) -> CompositionPlan:
    """Plan combining platform tools and specialist families."""
    steps: list[ExecutionStep] = []

    # Platform tools first (they provide context for specialists)
    tool_step_ids: list[str] = []
    for tool_name in decision.platform_tools:
        step = ExecutionStep(
            step_type="platform_tool",
            tool_name=tool_name,
            params={"query": prompt, "session_id": session_id},
            subtask=f"Execute {tool_name} for context gathering.",
            timeout_seconds=30.0,
        )
        steps.append(step)
        tool_step_ids.append(step.step_id)

    # Specialist families (depend on tool results)
    specialist_ids: list[str] = []
    for family_id in decision.families:
        if family_id == "verifier":
            continue
        step = ExecutionStep(
            step_type="specialist",
            family_id=family_id,
            subtask=_subtask_for_family(family_id, prompt),
            depends_on=list(tool_step_ids),
            timeout_seconds=FAMILY_TIMEOUT_SECONDS.get(family_id, 45.0),
        )
        steps.append(step)
        specialist_ids.append(step.step_id)

    # Verifier
    if "verifier" in decision.families:
        steps.append(
            ExecutionStep(
                step_type="specialist",
                family_id="verifier",
                subtask="Verify output accuracy and policy compliance.",
                depends_on=list(specialist_ids),
                timeout_seconds=FAMILY_TIMEOUT_SECONDS["verifier"],
            )
        )

    return CompositionPlan(
        steps=steps,
        route_type="composite",
        workflow_template=decision.workflow_template,
    )


def _subtask_for_family(family_id: str, prompt: str) -> str:
    """Generate a subtask description for a family."""
    prefix_map = {
        "analyst": "Analyze the request and prepare an evidence-backed response",
        "builder": "Implement the requested deliverable",
        "browser": "Navigate and extract content from web pages",
        "data": "Execute the data transformation or query",
        "media": "Generate the requested media assets",
        "memory": "Recall relevant context and prior knowledge",
        "planner": "Create a structured plan for the task",
    }
    prefix = prefix_map.get(family_id, "Process the request")
    return f"{prefix}: {prompt[:120]}"
