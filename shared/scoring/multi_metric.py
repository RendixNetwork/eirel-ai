"""Multi-metric per-task scoring for the general_chat eval.

Pure-function helpers that the validator engine uses to compute per-task
scores from (task_definition, miner_response, baseline_response, judge
outputs, tool_call_ledger, latency, cost). No I/O, no LLM calls — those
are upstream. This module is the math.

Per-task formula:

    task_score = w_p · pairwise_preference_score
               + w_g · grounded_correctness
               + w_r · retrieval_quality (or computation_correctness)
               + w_t · tool_routing
               + w_s · instruction_safety
               + w_l · latency_cost

Where the weights re-normalize over whichever dimensions are
``applicable`` for the task type (an N/A dimension drops out of the
sum and shifts its weight to the others proportionally).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# -- Task type taxonomy ----------------------------------------------------


# Bundle category (as published by the eirel-eval-pool renderer) → task
# type used for scoring. Tasks without a recognized category fall back
# to ``no_tool`` so they at least score the always-applicable metrics.
_CATEGORY_TO_TASK_TYPE: dict[str, str] = {
    "live_lookup": "web_required",
    "live_synthesis": "web_required",
    # ``attached_long_doc`` ships the document in-context via
    # ``metadata.attached_files``; the agent answers from context and
    # no retrieval tool is needed. ``rag_required`` (below) is the
    # category for genuine RAG eval where the corpus is indexed
    # server-side and the agent must call ``rag.retrieve``.
    "attached_long_doc": "no_tool",
    "rag_required": "rag_required",
    "multi_turn_agentic_memory": "memory_required",
    "compute_or_orchestrate": "sandbox_required",
    "abstention_probe": "no_tool",
}


def derive_task_type(category: str | None) -> str:
    return _CATEGORY_TO_TASK_TYPE.get(str(category or "").strip(), "no_tool")


# -- Default weights per task type -----------------------------------------


# All six dimensions are listed for every task type; non-applicable
# entries are mapped to ``None`` so the score collector knows to skip
# them. ``computation_correctness`` substitutes for ``retrieval_quality``
# on sandbox tasks; the keys live in the same slot in the dict so the
# re-normalizer treats them as one tile.
_DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "web_required": {
        "pairwise_preference_score": 0.40,
        "grounded_correctness": 0.30,
        "retrieval_quality": 0.15,
        "tool_routing": 0.05,
        "instruction_safety": 0.05,
        "latency_cost": 0.05,
    },
    "rag_required": {
        # retrieval_quality is intentionally absent here — the
        # multi-judge LLM still scores it from the response text rather
        # than from a gold chunk-id ledger. Re-normalization redistributes
        # its share of weight across the remaining dimensions.
        "pairwise_preference_score": 0.40,
        "grounded_correctness": 0.30,
        "tool_routing": 0.05,
        "instruction_safety": 0.05,
        "latency_cost": 0.05,
    },
    "sandbox_required": {
        "pairwise_preference_score": 0.40,
        "grounded_correctness": 0.30,
        "computation_correctness": 0.15,
        "tool_routing": 0.05,
        "instruction_safety": 0.05,
        "latency_cost": 0.05,
    },
    "memory_required": {
        "pairwise_preference_score": 0.40,
        "grounded_correctness": 0.30,
        "tool_routing": 0.05,
        "instruction_safety": 0.05,
        "latency_cost": 0.05,
    },
    "no_tool": {
        # Per the ChatGPT-thread design — pairwise carries more weight
        # when there's no tool/retrieval signal to share weight with.
        "pairwise_preference_score": 0.50,
        "grounded_correctness": 0.25,
        "tool_routing": 0.10,
        "instruction_safety": 0.10,
        "latency_cost": 0.05,
    },
}


def default_weights(task_type: str) -> dict[str, float]:
    return dict(_DEFAULT_WEIGHTS.get(task_type, _DEFAULT_WEIGHTS["no_tool"]))


def applicable_metrics(task_type: str) -> set[str]:
    return set(default_weights(task_type).keys())


# -- Deterministic per-dimension scorers -----------------------------------


def score_tool_routing(
    *,
    task_type: str,
    tools_called: list[str],
    has_citations: bool = False,
) -> float:
    """Did the agent pick the right tool for this task type?

    Primary signal is ``tools_called`` (parsed from ``response.tool_calls``
    in the miner payload). Some SDK runtimes don't surface tool_calls
    even when web tools were used — in that case ``has_citations`` is a
    reasonable proxy for ``web_required``: if the agent returned cited
    URLs, it almost certainly invoked ``web_search`` or ``url_fetch``.

    For ``no_tool`` we reward correctly NOT calling anything; calling a
    tool when none was needed scores 0.5 (not zero — the tool may have
    been a sensible double-check, just not optimal).
    """
    called = {t.strip().lower() for t in tools_called if t}
    web_tools = {"web_search", "url_fetch"}
    if task_type == "web_required":
        if called & web_tools:
            return 1.0
        if has_citations:
            return 1.0  # citations present → web tool used (SDK didn't surface call)
        return 0.0
    if task_type == "rag_required":
        # Tool name matches the SDK ``RagTool.name`` and the
        # ``rag_tool_service`` ledger entry. Older code referenced an
        # imaginary ``query_attachment`` tool that never existed.
        return 1.0 if "rag.retrieve" in called else 0.0
    if task_type == "sandbox_required":
        return 1.0 if "sandbox_python" in called else 0.0
    if task_type == "memory_required":
        # Memory tasks live in `request.turns`; no tool call needed.
        return 1.0
    if task_type == "no_tool":
        if not called and not has_citations:
            return 1.0
        return 0.5
    return 0.5  # unknown task type — neutral


def score_latency_cost(
    *,
    miner_latency_seconds: float,
    mode_budget_seconds: float | None,
    proxy_cost_usd: float,
    cost_budget_usd: float | None,
) -> float:
    """Within latency + cost budget?

    Linear ramp from 1.0 at 0% of budget to 0.0 at 100% of budget. The
    final score is the minimum of the two ramps so a miner can't trade
    speed for cost or vice versa.
    """
    parts: list[float] = []
    if mode_budget_seconds and mode_budget_seconds > 0:
        ratio = max(0.0, miner_latency_seconds) / mode_budget_seconds
        parts.append(max(0.0, 1.0 - ratio))
    if cost_budget_usd and cost_budget_usd > 0:
        ratio = max(0.0, proxy_cost_usd) / cost_budget_usd
        parts.append(max(0.0, 1.0 - ratio))
    if not parts:
        return 1.0
    return min(parts)


# -- Re-normalization + final score ----------------------------------------


@dataclass(slots=True)
class TaskScoreBreakdown:
    """Final per-task score + bookkeeping the validator persists."""

    task_type: str
    dimension_scores: dict[str, float]      # raw score per dimension (only applicable ones)
    applied_weights: dict[str, float]        # post-renormalization weights
    final_task_score: float
    applicable_metrics: list[str]


def renormalize(
    weights: dict[str, float],
    *,
    applicable: set[str],
) -> dict[str, float]:
    """Drop non-applicable dimensions and rescale remaining weights to sum to 1."""
    relevant = {k: v for k, v in weights.items() if k in applicable and v > 0}
    total = sum(relevant.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in relevant.items()}


def assemble_task_score(
    *,
    task_type: str,
    raw_scores: dict[str, float | None],
    weights: dict[str, float] | None = None,
) -> TaskScoreBreakdown:
    """Combine per-dimension scores into a single weighted task_score.

    ``raw_scores`` may contain ``None`` for dimensions that came back
    as N/A (e.g. retrieval_quality on a rag_required task with no
    chunk-id ledger, or grounded_correctness when the judge call
    failed). Those drop out and the remaining weights re-normalize.
    """
    base_weights = dict(weights or default_weights(task_type))
    # Applicable = dimension is in the default weight set AND we have a
    # non-None real-numbered score. If a default-applicable dimension
    # has no score, we drop it (treat as N/A) rather than imputing 0.0.
    applicable: set[str] = set()
    real_scores: dict[str, float] = {}
    for dim, w in base_weights.items():
        score = raw_scores.get(dim)
        if score is None:
            continue
        applicable.add(dim)
        real_scores[dim] = float(score)
    applied = renormalize(base_weights, applicable=applicable)
    final = sum(real_scores[k] * applied.get(k, 0.0) for k in real_scores)
    return TaskScoreBreakdown(
        task_type=task_type,
        dimension_scores=real_scores,
        applied_weights=applied,
        final_task_score=final,
        applicable_metrics=sorted(applicable),
    )


__all__ = [
    "TaskScoreBreakdown",
    "applicable_metrics",
    "assemble_task_score",
    "default_weights",
    "derive_task_type",
    "renormalize",
    "score_latency_cost",
    "score_tool_routing",
]
