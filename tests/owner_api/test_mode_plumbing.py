from __future__ import annotations

"""Regression test for the mode=instant hardcoding bug.

Prior to the fix, every evaluated task was passed into
``build_conversation_score_from_judge`` with ``mode="instant"``, which
meant ``aggregate_miner_score`` saw zero thinking-mode conversations and
blended silently dropped to ``0.6 * instant_mean`` (the 0.4 weight for
thinking_mean was multiplied by zero). This test pins the new behaviour:
thinking tasks must end up in the thinking bucket.
"""

from shared.core.evaluation_models import ConversationScore
from control_plane.owner_api.evaluation.general_chat_scoring import (
    aggregate_miner_score,
)


def _cs(mode: str, total: float) -> ConversationScore:
    return ConversationScore(
        mode=mode,  # type: ignore[arg-type]
        total=total,
        quality=total,
        latency=1.0,
        cost=1.0,
        trace_gate=1.0,
    )


def test_thinking_mode_tasks_contribute_to_thinking_mean():
    """instant 0.8 + thinking 0.6 → blended = 0.6*0.8 + 0.4*0.6 = 0.72."""
    scores = [_cs("instant", 0.8), _cs("thinking", 0.6)]
    result = aggregate_miner_score(
        miner_hotkey="5X",
        conversation_scores=scores,
        run_cost_usd_used=0.0,
        run_budget_usd=30.0,
    )
    # cost_efficiency=1, cost_modifier=1.0, so blended = blended_quality
    # blended_quality = 0.6*0.8 + 0.4*0.6 = 0.72
    assert abs(result.instant_mean - 0.8) < 1e-6
    assert abs(result.thinking_mean - 0.6) < 1e-6
    assert abs(result.blended - 0.72) < 1e-6


def test_all_instant_tasks_do_not_collapse_thinking_to_zero_math():
    """All-instant runs legitimately have thinking_mean=0 — that's the
    expected fallback for a dataset without thinking tasks."""
    scores = [_cs("instant", 0.7), _cs("instant", 0.8)]
    result = aggregate_miner_score(
        miner_hotkey="5X",
        conversation_scores=scores,
    )
    assert result.thinking_mean == 0.0
    # blended = 0.6 * 0.75 = 0.45
    assert abs(result.blended - 0.45) < 1e-6


def test_mode_hardcoding_bug_signature():
    """Cross-check: if mode were hardcoded to 'instant' for a dataset
    that is actually half thinking/half instant, the score formula
    yields blended = 0.6 * mean(all) instead of the correct blend."""
    # A dataset of 0.9 thinking + 0.5 instant
    correct = aggregate_miner_score(
        miner_hotkey="5X",
        conversation_scores=[_cs("instant", 0.5), _cs("thinking", 0.9)],
    )
    buggy = aggregate_miner_score(
        miner_hotkey="5X",
        conversation_scores=[_cs("instant", 0.5), _cs("instant", 0.9)],
    )
    # Correct blended = 0.6*0.5 + 0.4*0.9 = 0.66
    # Buggy blended   = 0.6 * 0.7 = 0.42
    # i.e. 36% lower than reality — matches the pattern we saw in run-24.
    assert abs(correct.blended - 0.66) < 1e-6
    assert abs(buggy.blended - 0.42) < 1e-6
    assert correct.blended > buggy.blended
