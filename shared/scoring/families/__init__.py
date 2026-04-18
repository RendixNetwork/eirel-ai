from __future__ import annotations

"""Pluggable family scorer registry.

Each launch family registers a ``score`` callable here. At this point only
``general_chat`` is live; future families (``deep_research``, ``coding``)
will add themselves to :data:`FAMILY_SCORERS`.
"""

from collections.abc import Callable
from typing import Any

from shared.core.evaluation_models import MinerGeneralChatScore

from shared.scoring.families import general_chat


FamilyScorer = Callable[..., Any]


FAMILY_SCORERS: dict[str, FamilyScorer] = {
    "general_chat": general_chat.score,
}


def get_family_scorer(family_id: str) -> FamilyScorer:
    scorer = FAMILY_SCORERS.get(family_id)
    if scorer is None:
        raise KeyError(f"no scorer registered for family_id={family_id!r}")
    return scorer


def score_general_chat(
    *,
    miner_hotkey: str,
    conversation_results: list[dict[str, Any]],
) -> MinerGeneralChatScore:
    return general_chat.score(
        miner_hotkey=miner_hotkey,
        conversation_results=conversation_results,
    )


__all__ = [
    "FAMILY_SCORERS",
    "FamilyScorer",
    "get_family_scorer",
    "general_chat",
    "score_general_chat",
]
