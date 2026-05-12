from __future__ import annotations

"""Benchmark orchestration — general_chat 4D scorer entry point.

Replaces the legacy analyst/builder/media/verifier pipeline. The new
orchestration delegates to the 4D conversation scorer and aggregates
miner-level scores via ``shared.scoring.families.general_chat``.
"""

from typing import Any

from shared.core.evaluation_models import (
    BenchmarkRunContext,
    BenchmarkTaskRun,
    ConversationScore,
    FamilyEpochScore,
    MinerBenchmarkTarget,
    ScoreResult,
)


async def score_family_epoch(
    *,
    context: BenchmarkRunContext,
    miners: list[MinerBenchmarkTarget],
    task_runs_by_miner: dict[str, list[BenchmarkTaskRun]] | None = None,
    **_ignored: Any,
) -> FamilyEpochScore:
    """Compute a family epoch score for the general_chat family.

    This is a minimal compatibility wrapper — the real conversation
    scoring happens per-conversation via
    ``control_plane.owner_api.evaluation.general_chat_scoring``. When
    called without conversation-level data, we return an empty epoch
    score so callers can fall back to their own aggregation.
    """
    del task_runs_by_miner
    return FamilyEpochScore(
        run_id=context.run_id,
        family_id=context.family_id,
        benchmark_version=context.benchmark_version,
        rubric_version=context.rubric_version,
        miner_scores={miner.hotkey: 0.0 for miner in miners},
        judge_outputs={},
        miner_responses={},
        metadata={"note": "general_chat scoring runs per-conversation in owner-api"},
    )


def compute_miner_score_from_results(
    *,
    family_id: str,
    benchmark_version: str,
    miner_hotkey: str,
    task_results: list[dict[str, Any]],
    judge_outputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate conversation-level scores for a miner.

    ``task_results`` is expected to be a list of ``ConversationScore``-
    shaped dicts produced by the owner-api 4D scorer. ``judge_outputs``
    is accepted for signature parity with legacy callers but not used.
    """
    del judge_outputs, benchmark_version
    if family_id != "general_chat":
        raise ValueError(
            f"compute_miner_score_from_results only supports general_chat, got {family_id!r}"
        )
    from shared.scoring.families import score_general_chat

    result = score_general_chat(
        miner_hotkey=miner_hotkey,
        conversation_results=task_results,
    )
    score_result = ScoreResult(
        family_id="general_chat",
        miner_hotkey=miner_hotkey,
        overall_score=result.blended,
        components=[],
        metadata={
            "instant_mean": result.instant_mean,
            "thinking_mean": result.thinking_mean,
            "blended": result.blended,
            "official_scoring_version": "general_chat_scoring_v1",
            "scoring_policy_version": "general_chat_scoring_v1",
        },
    )
    evaluation_breakdown = {
        "family_capability_score": result.blended,
        "official_family_score": result.blended,
        "instant_mean": result.instant_mean,
        "thinking_mean": result.thinking_mean,
    }
    return {
        "score_result": score_result.model_dump(mode="json"),
        "evaluation_breakdown": evaluation_breakdown,
        "protocol_gate": {"passed": True, "reason": "general_chat_no_protocol_gate"},
        "rollout_metadata": {
            "family_id": "general_chat",
            "overall_score": result.blended,
            "instant_mean": result.instant_mean,
            "thinking_mean": result.thinking_mean,
            "official_scoring_version": "general_chat_scoring_v1",
        },
    }


def aggregate_conversation_scores(
    conversation_scores: list[ConversationScore],
) -> dict[str, float]:
    """Lightweight mean aggregator used by callers that want a single number."""
    if not conversation_scores:
        return {"overall": 0.0, "count": 0.0}
    total = sum(cs.total for cs in conversation_scores) / len(conversation_scores)
    return {"overall": round(total, 6), "count": float(len(conversation_scores))}
