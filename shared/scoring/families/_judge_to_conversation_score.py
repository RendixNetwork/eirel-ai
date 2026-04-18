from __future__ import annotations

"""Translate judge output into a ConversationScore without requiring a trace.

The layered general_chat scoring path requires the miner to emit both a
``trace`` and ``response_text`` in its response dict so the trace-integrity
gate can run.  Miners that don't opt in (including the example agent)
produce a raw ``AgentInvocationResponse`` with no trace.

``build_conversation_score_from_judge`` fills that gap: it synthesizes a
valid ``ConversationScore`` from the judge sidecar's output plus whatever
latency signal is available in the miner response.  It keeps the same
``total = trace_gate * (QUALITY_WEIGHT * quality + LATENCY_WEIGHT * latency)``
shape as the layered path, so scores stay comparable across miners.

Callers:
    * ``_on_miner_evaluation_complete`` in the evaluation task manager —
      invoked per task before aggregation so every task produces a valid
      ``ConversationScore`` dict the family scorer can consume.
"""

from typing import Any, Literal

from shared.core.evaluation_models import ConversationScore


_QUALITY_WEIGHT = 0.80
_LATENCY_WEIGHT = 0.20

# Mode → soft latency budget (ms).  Matches ``ModeBudget`` used by the
# layered path (20s for instant, 60s for thinking).  Unknown modes default
# to instant; the scoring output never exceeds [0.0, 1.0] anyway.
_LATENCY_BUDGET_MS: dict[str, int] = {
    "instant": 20_000,
    "thinking": 60_000,
}


def _latency_score(latency_ms: int, *, mode: str) -> float:
    budget_ms = _LATENCY_BUDGET_MS.get(mode, _LATENCY_BUDGET_MS["instant"])
    if latency_ms <= 0:
        # Unmeasured latency is neutral, not a free 1.0.  Producing 0.5
        # here keeps an unmeasured miner from out-scoring a measured one
        # that happened to be slightly slow.
        return 0.5
    return max(0.0, min(1.0, 1.0 - (float(latency_ms) / float(budget_ms))))


def _extract_latency_ms(
    miner_response: dict[str, Any],
    judge_output: dict[str, Any],
) -> int:
    # Miner-reported latency (AgentInvocationResponse.latency_ms) is the
    # most accurate — it's measured by the SDK around the handler body.
    resp = miner_response.get("response") if isinstance(miner_response, dict) else None
    if isinstance(resp, dict):
        candidate = resp.get("latency_ms")
        if isinstance(candidate, (int, float)) and candidate > 0:
            return int(candidate)
    # Fall back to the judge's latency_seconds — this measures the judge
    # call, not the miner call, but it's correlated when the miner
    # dominates the judge's input size.  Better than 0.
    judge_latency_s = judge_output.get("latency_seconds")
    if isinstance(judge_latency_s, (int, float)) and judge_latency_s > 0:
        return int(float(judge_latency_s) * 1000.0)
    return 0


def build_conversation_score_from_judge(
    *,
    task_score: float,
    judge_output: dict[str, Any],
    miner_response: dict[str, Any],
    mode: Literal["instant", "thinking"] = "instant",
) -> ConversationScore:
    """Build a ConversationScore from judge output when no trace is available.

    The returned score is shaped identically to one produced by the layered
    path, so the family aggregator (``aggregate_miner_score``) can consume
    it without special-casing.

    ``quality`` prefers ``judge_output["score"]`` (the judge's own 0-1
    overall score) and falls back to ``task_score`` — the latter is the
    value the validator already computed and submitted, which matches the
    judge's output when the layered path didn't run.

    ``trace_gate`` defaults to 1.0 (pass) because there's no trace to gate.
    Constraint flags from the judge are recorded in metadata for audit but
    do NOT zero the gate — they're quality signals, not integrity signals.

    ``cost`` defaults to 0.0.  Per-conversation cost attribution is not
    tracked here; run-level cost lives in ``DeploymentScoreRecord`` via
    ``populate_cost_columns`` and applies a separate multiplicative
    modifier in ``aggregate_miner_score``.
    """
    dimension_scores = (
        judge_output.get("dimension_scores") or {}
        if isinstance(judge_output, dict) else {}
    )

    raw_quality = judge_output.get("score") if isinstance(judge_output, dict) else None
    if not isinstance(raw_quality, (int, float)):
        raw_quality = task_score
    quality = max(0.0, min(1.0, float(raw_quality)))

    latency_ms = _extract_latency_ms(miner_response, judge_output if isinstance(judge_output, dict) else {})
    latency = _latency_score(latency_ms, mode=mode)

    trace_gate = 1.0  # No trace to gate; constraint_flags are quality signals.
    cost = 0.0

    weighted = _QUALITY_WEIGHT * quality + _LATENCY_WEIGHT * latency
    total = round(trace_gate * weighted, 6)

    constraint_flags = (
        judge_output.get("constraint_flags")
        if isinstance(judge_output, dict) else None
    ) or []

    return ConversationScore(
        quality=round(quality, 6),
        latency=round(latency, 6),
        cost=cost,
        trace_gate=trace_gate,
        total=total,
        per_dimension={
            str(k): round(float(v), 6)
            for k, v in dimension_scores.items()
            if isinstance(v, (int, float))
        },
        mode=mode,
        metadata={
            "synthesized_from_judge": True,
            "latency_ms_source": latency_ms,
            "constraint_flags": [str(f) for f in constraint_flags],
        },
    )


__all__ = ["build_conversation_score_from_judge"]
