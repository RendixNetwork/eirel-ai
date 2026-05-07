"""Typed family selection for the graph-runtime orchestrator.

The orchestrator's first job is to decide which family answers a
user prompt. Today the answer is always ``general_chat``; the type
stays generic so adding a future family is purely additive.

Why a separate module from ``family_selector.py``:
the legacy ``RoutingDecision`` carries route_type / platform_tools /
workflow_template knobs that don't apply to the graph-runtime path —
graph miners answer end-to-end, and the orchestrator just picks one
miner and streams its response. Keeping the new selection type narrow
prevents the legacy DAG semantics from leaking into the new wire.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from eirel.groups import FamilyId

__all__ = [
    "FamilySelection",
    "select_family_for_prompt",
]


@dataclass(frozen=True, slots=True)
class FamilySelection:
    """The orchestrator's chosen family + audit-trail metadata.

    ``confidence`` is a 0–1 self-report; ``rationale`` is a free-text
    explanation eiretes can later score for routing-decision quality.
    """

    family_id: FamilyId
    confidence: float = 1.0
    rationale: str = ""
    available_families: Sequence[FamilyId] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "available_families": list(self.available_families),
            "metadata": dict(self.metadata),
        }


# The active families known to the orchestrator. Single-family
# today; future families plug in here.
_ACTIVE_FAMILIES: tuple[FamilyId, ...] = ("general_chat",)


def select_family_for_prompt(
    *,
    prompt: str,  # noqa: ARG001 — used by future routers
    available_families: Sequence[FamilyId] = _ACTIVE_FAMILIES,
    context: dict[str, Any] | None = None,  # noqa: ARG001
) -> FamilySelection:
    """Pick a family for a user prompt.

    Single-family fast path: returns ``general_chat`` with ``confidence=1.0``.
    Future implementations can examine ``prompt`` + ``context`` to pick
    among ``available_families``.
    """
    if not available_families:
        raise ValueError("at least one family must be available")
    fam = "general_chat" if "general_chat" in available_families else available_families[0]
    return FamilySelection(
        family_id=fam,
        confidence=1.0,
        rationale=(
            "single_family_fast_path"
            if fam == "general_chat" and len(available_families) == 1
            else "fallback_to_first_available"
        ),
        available_families=tuple(available_families),
    )
