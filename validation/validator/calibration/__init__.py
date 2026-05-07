"""Validator-side calibration harness.

These tools verify a deployment's LLM provider quality before
flipping production traffic. Operators run them once at deploy time
(and after model swaps) — not in CI, since they call real APIs and
cost real money.

Two gates:

  * **JSON parse-rate gate** — measures what fraction of N calls
    return parseable JSON matching the strict response schema.
    Acceptance: ≥98% parse rate, ≥95% schema-valid.
  * **Reconciler claim-set agreement gate** — given a fixture set of
    3-oracle scenarios with golden claim sets, measures Jaccard
    agreement between the reconciler's output and golden.
    Acceptance: ≥0.85 mean agreement.

Each gate produces a ``GateResult`` (pass/fail + measured rate +
per-fixture details) the operator can dump as JSON for the deploy
record.

Below the gate thresholds, the operator either swaps the model or
wraps the provider client with ``with_json_repair`` (recovers the
90-98% band; below 90% the model isn't ready).
"""

from __future__ import annotations

from validation.validator.calibration.gate_result import (
    GateResult,
    GateStatus,
)
from validation.validator.calibration.json_parse_rate import (
    JsonParseRateGate,
    measure_json_parse_rate,
)
from validation.validator.calibration.rank_parity import (
    rank_parity_spearman,
)
from validation.validator.calibration.reconciler_agreement import (
    ReconcilerAgreementGate,
    claim_set_jaccard,
    measure_reconciler_agreement,
)

__all__ = [
    "GateResult",
    "GateStatus",
    "JsonParseRateGate",
    "ReconcilerAgreementGate",
    "claim_set_jaccard",
    "measure_json_parse_rate",
    "measure_reconciler_agreement",
    "rank_parity_spearman",
]
