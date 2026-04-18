from __future__ import annotations

"""general_chat family scorer.

Aggregates a list of per-conversation results for a single miner into the
blended instant/thinking miner score. Conversation results are expected to
already be scored (by ``control_plane.owner_api.evaluation.general_chat_scoring``)
and passed here as plain dicts.
"""

from typing import Any

from shared.core.evaluation_models import ConversationScore, MinerGeneralChatScore


def _coerce_score(item: Any) -> ConversationScore:
    if isinstance(item, ConversationScore):
        return item
    if isinstance(item, dict):
        return ConversationScore.model_validate(item)
    raise TypeError(
        f"conversation_results entries must be ConversationScore or dict, got {type(item).__name__}"
    )


def score(
    *,
    miner_hotkey: str,
    conversation_results: list[dict[str, Any]] | list[ConversationScore],
    run_budget_usd: float = 30.0,
    run_cost_usd_used: float = 0.0,
) -> MinerGeneralChatScore:
    """Aggregate per-conversation scores into a miner-level score."""
    from control_plane.owner_api.evaluation.general_chat_scoring import (
        aggregate_miner_score,
    )

    coerced: list[ConversationScore] = [_coerce_score(item) for item in conversation_results]
    return aggregate_miner_score(
        miner_hotkey=miner_hotkey,
        conversation_scores=coerced,
        run_budget_usd=run_budget_usd,
        run_cost_usd_used=run_cost_usd_used,
    )


__all__ = ["score"]
