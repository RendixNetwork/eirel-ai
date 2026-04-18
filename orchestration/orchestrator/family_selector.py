"""Family selector — determines which specialist families a request needs.

Post-refactor there is only one launch family (``general_chat``), so the
selector is trivial. The function signatures are kept so the orchestrator
still has a single place to plug future multi-family dispatch for
``deep_research`` and ``coding``.
"""

from __future__ import annotations

from typing import Any, Literal

from eirel.groups import FamilyId


class RoutingDecision:
    """Result of the family selector analysis."""

    def __init__(
        self,
        *,
        route_type: Literal["direct", "platform_tool", "specialist", "composite"] = "specialist",
        families: list[FamilyId] | None = None,
        platform_tools: list[str] | None = None,
        workflow_template: str = "direct_analysis",
        task_type: Literal["conversational", "creative", "analytical", "agentic"] = "analytical",
        confidence: float = 1.0,
        reasoning: list[str] | None = None,
    ):
        self.route_type = route_type
        self.families = families or []
        self.platform_tools = platform_tools or []
        self.workflow_template = workflow_template
        self.task_type = task_type
        self.confidence = confidence
        self.reasoning = reasoning or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_type": self.route_type,
            "families": list(self.families),
            "platform_tools": list(self.platform_tools),
            "workflow_template": self.workflow_template,
            "task_type": self.task_type,
            "confidence": self.confidence,
            "reasoning": list(self.reasoning),
        }


def select_route(
    *,
    prompt: str,
    context_history: list[dict[str, Any]] | None = None,
    modalities_allowed: list[str] | None = None,
    families_excluded: list[str] | None = None,
) -> RoutingDecision:
    """Route every request to the ``general_chat`` family.

    Kept as a single-point-of-extension: future families will land here.
    """
    del prompt, context_history, modalities_allowed
    excluded = set(families_excluded or [])
    if "general_chat" in excluded:
        return RoutingDecision(
            route_type="direct",
            families=[],
            workflow_template="direct_response",
            task_type="conversational",
            confidence=1.0,
            reasoning=["general_chat_excluded_by_caller"],
        )
    return RoutingDecision(
        route_type="specialist",
        families=["general_chat"],
        workflow_template="direct_analysis",
        task_type="analytical",
        confidence=1.0,
        reasoning=["general_chat_default_route"],
    )


def select_family() -> FamilyId:
    """Canonical single-family helper for call sites that just need the id."""
    return "general_chat"
