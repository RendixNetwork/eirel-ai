"""Validator rank parity (Spearman correlation) helper.

During shadow-mode rollout, the validator computes BOTH the legacy
weighted-sum score and the new multiplicative composite for each
miner. Per the rollout plan, before flipping the default scoring
formula the operator verifies the two paths produce highly correlated
miner ranks (Spearman ρ ≥ 0.85 across 2 evaluation cycles).

This helper is pure-function — no LLM calls, no provider state. Takes
two equal-length sequences of scores (one per miner, same order) and
returns the Spearman rank-correlation coefficient ∈ [-1, 1].

Tests use synthetic score sequences; production reads from
``TaskMinerResult.judge_output.metadata.weighted_sum_score`` and
``...composite_score`` per cycle.
"""

from __future__ import annotations

from collections.abc import Sequence


def rank_parity_spearman(
    weighted_sum_scores: Sequence[float],
    composite_scores: Sequence[float],
) -> float:
    """Spearman rank-correlation coefficient.

    Returns 1.0 when ranks are identical, -1.0 when reversed, 0.0
    when uncorrelated. Empty / single-element / mismatched-length
    inputs return 0.0 (not enough signal for a correlation).

    Computed via Pearson correlation on the rank vectors (with
    average-rank for ties), which equals Spearman's ρ. We avoid
    pulling scipy in just for this — eight lines of stdlib does it.
    """
    if len(weighted_sum_scores) != len(composite_scores):
        return 0.0
    n = len(weighted_sum_scores)
    if n < 2:
        return 0.0

    a = _ranks(weighted_sum_scores)
    b = _ranks(composite_scores)

    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    denom = (var_a * var_b) ** 0.5
    if denom == 0.0:
        # All values tied in at least one input → no rank variance.
        return 0.0
    return cov / denom


def _ranks(values: Sequence[float]) -> list[float]:
    """Average-rank assignment (handles ties).

    Example: [10, 5, 5, 1] → [4.0, 2.5, 2.5, 1.0]
    """
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        # Advance j past tied values.
        while (
            j + 1 < len(indexed)
            and indexed[j + 1][1] == indexed[i][1]
        ):
            j += 1
        # Average rank for the tied group (ranks are 1-indexed).
        avg_rank = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


__all__ = ["rank_parity_spearman"]
