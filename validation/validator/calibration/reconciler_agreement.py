"""Reconciler claim-set agreement calibration gate.

Given a fixture set of 3-oracle scenarios with golden claim sets,
measures Jaccard agreement between the reconciler's output
``expected_claims`` and the golden answer. Operator workflow:

  1. Curate ~30 representative 3-oracle scenarios — each fixture
     carries ``(prompt, [grounding_a, grounding_b, grounding_c],
     golden_claims)``.
  2. Run ``measure_reconciler_agreement(reconciler, fixtures)`` with
     the real reconciler client.
  3. Inspect the returned ``GateResult``: ``status="pass"`` if mean
     Jaccard agreement ≥ 0.85.

Jaccard is computed at the claim-string level after a normalize step
(lowercase + trim) — exact-string match would fail on trivial wording
drift; normalize-then-compare is the right operating point per the
``project_eval_three_oracle_groundedness.md`` memory.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from validation.validator.calibration.gate_result import GateResult
from validation.validator.oracles.base import OracleGrounding
from validation.validator.reconciler import ReconciledOracle


_logger = logging.getLogger(__name__)


DEFAULT_AGREEMENT_THRESHOLD = 0.85


class _ReconcilerLike(Protocol):
    async def reconcile(
        self,
        *,
        prompt: str,
        groundings: list[OracleGrounding],
        must_not_claim_floor: Iterable[str] = (),
    ) -> ReconciledOracle: ...


@dataclass(frozen=True)
class ReconcilerAgreementFixture:
    """One scenario for the reconciler agreement gate.

    ``golden_claims`` is the operator-curated truth set for this
    prompt; the reconciler's ``expected_claims`` is compared via
    normalized Jaccard.
    """

    prompt: str
    groundings: list[OracleGrounding]
    golden_claims: list[str]
    must_not_claim_floor: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Hack: dataclass with default mutable list — convert None → [].
        if self.must_not_claim_floor is None:
            object.__setattr__(self, "must_not_claim_floor", [])


def claim_set_jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity over normalized claim strings.

    Both inputs empty → 1.0 (no disagreement when there's nothing to
    disagree on). One empty + one non-empty → 0.0.
    """
    norm_a = {_normalize_claim(c) for c in a if c}
    norm_b = {_normalize_claim(c) for c in b if c}
    norm_a = {c for c in norm_a if c}
    norm_b = {c for c in norm_b if c}
    if not norm_a and not norm_b:
        return 1.0
    intersection = norm_a & norm_b
    union = norm_a | norm_b
    return len(intersection) / len(union) if union else 0.0


def _normalize_claim(claim: str) -> str:
    return " ".join((claim or "").strip().lower().split())


@dataclass(frozen=True)
class _FixtureOutcome:
    fixture_index: int
    jaccard: float
    expected_claims: list[str]
    golden_claims: list[str]
    error: str | None = None


async def measure_reconciler_agreement(
    reconciler: _ReconcilerLike,
    fixtures: Iterable[ReconcilerAgreementFixture],
    *,
    threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
    name: str = "reconciler_claim_agreement",
) -> GateResult:
    """Run reconciler over each fixture, return mean Jaccard agreement.

    Pass: mean Jaccard ≥ ``threshold`` (default 0.85).
    Fail: below threshold — operator should review fixtures or swap
    the reconciler model.
    """
    outcomes: list[_FixtureOutcome] = []
    for idx, fixture in enumerate(fixtures):
        try:
            reconciled = await reconciler.reconcile(
                prompt=fixture.prompt,
                groundings=fixture.groundings,
                must_not_claim_floor=fixture.must_not_claim_floor,
            )
            agreement = claim_set_jaccard(
                reconciled.expected_claims, fixture.golden_claims,
            )
            outcomes.append(
                _FixtureOutcome(
                    fixture_index=idx,
                    jaccard=agreement,
                    expected_claims=list(reconciled.expected_claims),
                    golden_claims=list(fixture.golden_claims),
                )
            )
        except Exception as exc:
            outcomes.append(
                _FixtureOutcome(
                    fixture_index=idx,
                    jaccard=0.0,
                    expected_claims=[],
                    golden_claims=list(fixture.golden_claims),
                    error=f"reconcile_error: {exc}",
                )
            )

    n = len(outcomes)
    if n == 0:
        return GateResult(
            name=name,
            status="fail",
            measured_rate=0.0,
            threshold=threshold,
            n_samples=0,
            details={"reason": "no_fixtures_provided"},
        )

    mean_agreement = sum(o.jaccard for o in outcomes) / n
    status = "pass" if mean_agreement >= threshold else "fail"
    # Below-threshold runs surface the worst fixtures so the operator
    # can investigate (curation issue vs reconciler model issue).
    sorted_outcomes = sorted(outcomes, key=lambda o: o.jaccard)
    return GateResult(
        name=name,
        status=status,
        measured_rate=mean_agreement,
        threshold=threshold,
        n_samples=n,
        details={
            "min_jaccard": sorted_outcomes[0].jaccard if outcomes else 0.0,
            "max_jaccard": sorted_outcomes[-1].jaccard if outcomes else 0.0,
            "n_below_threshold": sum(
                1 for o in outcomes if o.jaccard < threshold
            ),
            "worst_fixtures": [
                {
                    "fixture_index": o.fixture_index,
                    "jaccard": o.jaccard,
                    "expected_claims": o.expected_claims,
                    "golden_claims": o.golden_claims,
                    "error": o.error,
                }
                for o in sorted_outcomes[:5]
                if o.jaccard < threshold
            ],
        },
    )


class ReconcilerAgreementGate:
    """Operator-facing harness wrapper for the reconciler gate."""

    def __init__(
        self,
        fixtures: Iterable[ReconcilerAgreementFixture],
        *,
        threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
        name: str = "reconciler_claim_agreement",
    ) -> None:
        self._fixtures = list(fixtures)
        self._threshold = threshold
        self._name = name

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def n_fixtures(self) -> int:
        return len(self._fixtures)

    async def run(self, reconciler: _ReconcilerLike) -> GateResult:
        return await measure_reconciler_agreement(
            reconciler,
            self._fixtures,
            threshold=self._threshold,
            name=self._name,
        )


__all__ = [
    "DEFAULT_AGREEMENT_THRESHOLD",
    "ReconcilerAgreementFixture",
    "ReconcilerAgreementGate",
    "claim_set_jaccard",
    "measure_reconciler_agreement",
]
