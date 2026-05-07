"""Validator rank parity (Spearman correlation) helper tests."""

from __future__ import annotations

import pytest

from validation.validator.calibration.rank_parity import rank_parity_spearman


# -- Identical / reversed / uncorrelated --------------------------------


def test_identical_ranks_returns_one():
    """Same ordering → Spearman ρ = 1.0 (perfect rank agreement).
    The actual numeric values can differ — Spearman is rank-based."""
    weighted = [0.1, 0.5, 0.9]
    composite = [0.05, 0.40, 0.95]  # different scale, same order
    assert rank_parity_spearman(weighted, composite) == pytest.approx(1.0)


def test_reversed_ranks_returns_minus_one():
    """Opposite ordering → ρ = -1.0."""
    weighted = [0.1, 0.5, 0.9]
    composite = [0.9, 0.5, 0.1]
    assert rank_parity_spearman(weighted, composite) == pytest.approx(-1.0)


def test_uncorrelated_ranks_returns_zero_or_close():
    """Two random rank orderings have ρ near 0.0 over many samples.
    For specific small samples, the value can be exact-zero (no
    monotonic relationship)."""
    weighted = [1, 2, 3, 4]
    composite = [2, 4, 1, 3]
    rho = rank_parity_spearman(weighted, composite)
    # |ρ| ≤ 1; for this specific permutation, ρ = 0
    assert -0.5 <= rho <= 0.5


# -- Production rollout scenarios ---------------------------------------


def test_high_correlation_above_rollout_threshold():
    """Real shadow-mode scenario: 10 miners ranked the same by both
    paths with one slight reordering. ρ should be > 0.85 (the
    rollout threshold from the plan)."""
    weighted = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    composite = [0.1, 0.2, 0.3, 0.5, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]  # 4 ↔ 5 swap
    rho = rank_parity_spearman(weighted, composite)
    assert rho > 0.85, f"ρ={rho} should clear the rollout threshold"


def test_modest_correlation_below_rollout_threshold():
    """A 4-of-10 reorder breaks correlation enough to fail the
    rollout gate. Operator should investigate before flipping the
    default scoring formula."""
    weighted = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # Aggressive reorder
    composite = [0.5, 0.1, 0.7, 0.2, 0.9, 0.3, 0.6, 0.4, 0.8, 1.0]
    rho = rank_parity_spearman(weighted, composite)
    assert rho < 0.85, f"ρ={rho} should NOT clear the rollout threshold"


# -- Tie handling -------------------------------------------------------


def test_ties_in_weighted_sum():
    """Average-rank for ties means a single tie doesn't produce a
    spurious rank disagreement when the composite breaks the tie."""
    weighted = [0.5, 0.5, 0.9]  # 1 and 2 are tied
    composite = [0.4, 0.6, 0.9]  # composite breaks the tie
    rho = rank_parity_spearman(weighted, composite)
    # Two miners with tied weighted scores get rank 1.5; composite
    # ranks them 1 and 2. The third miner is rank 3 in both.
    # Correlation is positive but not 1.0.
    assert 0.5 < rho < 1.0


def test_all_ties_returns_zero():
    """If every miner has the same score in one path, there's no
    rank variance to correlate against → ρ = 0."""
    weighted = [0.5, 0.5, 0.5, 0.5]
    composite = [0.1, 0.2, 0.3, 0.4]
    assert rank_parity_spearman(weighted, composite) == 0.0


# -- Edge cases ---------------------------------------------------------


def test_empty_inputs_return_zero():
    assert rank_parity_spearman([], []) == 0.0


def test_single_element_returns_zero():
    """One miner — no rank correlation possible."""
    assert rank_parity_spearman([0.5], [0.7]) == 0.0


def test_mismatched_lengths_return_zero():
    assert rank_parity_spearman([0.1, 0.2], [0.1, 0.2, 0.3]) == 0.0


def test_average_rank_for_ties():
    """Verify the rank assignment internals: [10, 5, 5, 1] should
    rank as [4.0, 2.5, 2.5, 1.0]. We test this indirectly via the
    correlation result with a known counterpart."""
    weighted = [10, 5, 5, 1]
    # Same ranks → ρ = 1
    composite = [10, 5, 5, 1]
    assert rank_parity_spearman(weighted, composite) == pytest.approx(1.0)
    # Reverse → ρ = -1
    composite_rev = [1, 5, 5, 10]
    assert rank_parity_spearman(weighted, composite_rev) == pytest.approx(-1.0)
