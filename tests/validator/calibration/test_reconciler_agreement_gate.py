"""Reconciler claim-set agreement gate tests."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from validation.validator.calibration.reconciler_agreement import (
    DEFAULT_AGREEMENT_THRESHOLD,
    ReconcilerAgreementFixture,
    ReconcilerAgreementGate,
    claim_set_jaccard,
    measure_reconciler_agreement,
)
from validation.validator.oracles.base import OracleGrounding
from validation.validator.reconciler import ReconciledOracle


pytestmark = pytest.mark.asyncio


class _ScriptedReconciler:
    """Reconciler-protocol stub returning canned ReconciledOracles."""

    def __init__(self, expected_claims_per_call: list[list[str]]) -> None:
        self._queue = list(expected_claims_per_call)

    async def reconcile(
        self,
        *,
        prompt: str,
        groundings: list[OracleGrounding],
        must_not_claim_floor: Iterable[str] = (),
    ) -> ReconciledOracle:
        if not self._queue:
            raise RuntimeError("scripted reconciler exhausted")
        claims = self._queue.pop(0)
        return ReconciledOracle(
            expected_claims=claims,
            must_not_claim=list(must_not_claim_floor),
            oracle_status="consensus",
            consensus_claims=claims,
        )


def _grounding(vendor: str, text: str) -> OracleGrounding:
    return OracleGrounding(vendor=vendor, status="ok", raw_text=text)


def _fixture(prompt: str, golden: list[str]) -> ReconcilerAgreementFixture:
    return ReconcilerAgreementFixture(
        prompt=prompt,
        groundings=[
            _grounding("openai", "stub-a"),
            _grounding("gemini", "stub-b"),
            _grounding("grok", "stub-c"),
        ],
        golden_claims=golden,
    )


# -- Jaccard helper -----------------------------------------------------


def test_jaccard_identical_sets():
    assert claim_set_jaccard(["Paris", "France"], ["Paris", "France"]) == 1.0


def test_jaccard_disjoint_sets():
    assert claim_set_jaccard(["Paris"], ["Berlin"]) == 0.0


def test_jaccard_partial_overlap():
    # 1 intersect / 3 union = 0.333...
    j = claim_set_jaccard(["a", "b"], ["b", "c"])
    assert j == pytest.approx(1 / 3)


def test_jaccard_normalizes_case_and_whitespace():
    j = claim_set_jaccard(["Paris is in France"], ["paris is in france"])
    assert j == 1.0
    j2 = claim_set_jaccard(["  hello world  "], ["hello   world"])
    assert j2 == 1.0


def test_jaccard_both_empty_returns_one():
    """Both empty → no disagreement (trivially identical)."""
    assert claim_set_jaccard([], []) == 1.0


def test_jaccard_one_empty_returns_zero():
    assert claim_set_jaccard([], ["X"]) == 0.0
    assert claim_set_jaccard(["X"], []) == 0.0


def test_jaccard_filters_empty_strings():
    """Empty/whitespace-only strings are ignored."""
    assert claim_set_jaccard(["X", "", "  "], ["X"]) == 1.0


# -- Gate runner --------------------------------------------------------


async def test_perfect_agreement_passes():
    rec = _ScriptedReconciler(
        [["Paris is the capital"], ["1969 was Apollo 11"], ["DNA has four bases"]]
    )
    fixtures = [
        _fixture("capital of france?", ["Paris is the capital"]),
        _fixture("when apollo 11?", ["1969 was Apollo 11"]),
        _fixture("DNA bases?", ["DNA has four bases"]),
    ]
    result = await measure_reconciler_agreement(rec, fixtures)
    assert result.status == "pass"
    assert result.measured_rate == 1.0
    assert result.n_samples == 3
    assert result.details["min_jaccard"] == 1.0
    assert result.details["n_below_threshold"] == 0


async def test_below_threshold_fails():
    """Reconciler outputs disagree with golden on most fixtures →
    mean Jaccard < 0.85 → fail."""
    rec = _ScriptedReconciler(
        [["wrong"], ["also wrong"], ["right answer"]]
    )
    fixtures = [
        _fixture("Q1", ["right answer"]),
        _fixture("Q2", ["right answer"]),
        _fixture("Q3", ["right answer"]),
    ]
    # 2 of 3 disagree (Jaccard=0); 1 agrees (Jaccard=1) → mean ≈ 0.333
    result = await measure_reconciler_agreement(rec, fixtures)
    assert result.status == "fail"
    assert result.measured_rate == pytest.approx(1 / 3)
    assert result.details["n_below_threshold"] == 2


async def test_worst_fixtures_surfaced_in_details():
    """When the gate fails, the details include the worst fixtures so
    the operator can debug curation vs reconciler model."""
    rec = _ScriptedReconciler(
        [["A"], ["WRONG"], ["A"], ["A"], ["A"], ["WRONG"]]
    )
    fixtures = [
        _fixture(f"Q{i}", ["A"]) for i in range(6)
    ]
    result = await measure_reconciler_agreement(rec, fixtures)
    worst = result.details["worst_fixtures"]
    assert len(worst) >= 1
    assert all(w["jaccard"] < DEFAULT_AGREEMENT_THRESHOLD for w in worst)
    assert all("WRONG" in (w["expected_claims"] or [""])[0] for w in worst)


async def test_reconciler_error_counts_as_zero():
    """A reconciler call that raises an exception counts as zero
    Jaccard for that fixture (worst case) and is recorded with the
    error message."""

    class _RaisingReconciler:
        async def reconcile(self, **kwargs):
            raise RuntimeError("simulated reconciler crash")

    fixtures = [_fixture("Q1", ["A"])]
    result = await measure_reconciler_agreement(_RaisingReconciler(), fixtures)
    assert result.measured_rate == 0.0
    assert result.status == "fail"
    worst = result.details["worst_fixtures"]
    assert len(worst) == 1
    assert "reconcile_error" in (worst[0]["error"] or "")


async def test_empty_fixtures_returns_fail():
    rec = _ScriptedReconciler([])
    result = await measure_reconciler_agreement(rec, [])
    assert result.status == "fail"
    assert result.n_samples == 0


async def test_custom_threshold_overrides_default():
    """Operator can lower the bar for less-strict scenarios."""
    rec = _ScriptedReconciler([["A", "B"], ["A"]])
    fixtures = [
        _fixture("Q1", ["A", "B"]),
        _fixture("Q2", ["A", "B"]),  # Jaccard = 1/2 = 0.5
    ]
    # mean = (1.0 + 0.5) / 2 = 0.75
    result_strict = await measure_reconciler_agreement(rec, fixtures)
    assert result_strict.status == "fail"  # 0.75 < 0.85
    rec2 = _ScriptedReconciler([["A", "B"], ["A"]])
    result_loose = await measure_reconciler_agreement(
        rec2, fixtures, threshold=0.7,
    )
    assert result_loose.status == "pass"


async def test_gate_helper_runs_fixtures():
    rec = _ScriptedReconciler([["A"]])
    gate = ReconcilerAgreementGate([_fixture("Q1", ["A"])])
    assert gate.n_fixtures == 1
    result = await gate.run(rec)
    assert result.status == "pass"
