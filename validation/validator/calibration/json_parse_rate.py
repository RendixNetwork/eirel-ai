"""JSON parse-rate calibration gate.

Measures the fraction of N provider calls that return valid JSON
parseable by the strict response schema. Used for the eirel-ai
validator's reconciler (``EIREL_VALIDATOR_RECONCILER_*``) — eiretes
runs an analogous gate for its 3 judge roles.

Operator workflow:

  1. Build a fixture file: a list of (system, user, schema) triples
     covering representative reconciler inputs.
  2. Run ``measure_json_parse_rate(client, fixtures)`` with the real
     provider client.
  3. Inspect the returned ``GateResult``: ``status="pass"`` if
     ≥98% of calls returned parseable JSON.

The harness itself is provider-agnostic — it accepts anything with
the ``complete_structured`` shape, so it works with the bare
``OpenAICompatibleClient`` / ``GeminiClient`` AND the
``JsonRepairClient`` wrapper. Running the gate against both lets
operators see the recovery delta.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from validation.validator.calibration.gate_result import GateResult


_logger = logging.getLogger(__name__)


# Default acceptance threshold — drawn from the
# project_eval_judge_input_bundle.md memory's calibration gate.
DEFAULT_PARSE_RATE_THRESHOLD = 0.98
# Below this, JSON-repair retry won't recover; the model isn't ready
# and should be swapped.
SWAP_THRESHOLD = 0.90


class _StructuredCompleter(Protocol):
    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        response_schema: dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        schema_name: str = "response",
    ) -> Any: ...


@dataclass(frozen=True)
class JsonParseRateFixture:
    """One probe input for the parse-rate gate."""

    system: str
    user: str
    response_schema: dict[str, Any]
    schema_name: str = "response"


@dataclass(frozen=True)
class _SampleOutcome:
    fixture_index: int
    parsed: bool
    error: str | None


async def measure_json_parse_rate(
    client: _StructuredCompleter,
    fixtures: Iterable[JsonParseRateFixture],
    *,
    threshold: float = DEFAULT_PARSE_RATE_THRESHOLD,
    name: str = "json_parse_rate",
) -> GateResult:
    """Run every fixture once, return aggregate parse rate.

    Pass: parse rate ≥ ``threshold`` (default 98%).
    Marginal: parse rate ∈ [SWAP_THRESHOLD, threshold).
    Fail: parse rate < SWAP_THRESHOLD (~90%) — model not ready.
    """
    samples: list[_SampleOutcome] = []
    for idx, fixture in enumerate(fixtures):
        try:
            response = await client.complete_structured(
                system=fixture.system,
                user=fixture.user,
                response_schema=fixture.response_schema,
                schema_name=fixture.schema_name,
            )
        except Exception as exc:
            samples.append(
                _SampleOutcome(
                    fixture_index=idx, parsed=False, error=f"call_error: {exc}",
                )
            )
            continue
        try:
            json.loads(getattr(response, "text", str(response)))
        except (TypeError, ValueError) as exc:
            samples.append(
                _SampleOutcome(
                    fixture_index=idx, parsed=False, error=f"parse_error: {exc}",
                )
            )
            continue
        samples.append(
            _SampleOutcome(fixture_index=idx, parsed=True, error=None),
        )

    n = len(samples)
    if n == 0:
        return GateResult(
            name=name,
            status="fail",
            measured_rate=0.0,
            threshold=threshold,
            n_samples=0,
            details={"reason": "no_fixtures_provided"},
        )

    n_parsed = sum(1 for s in samples if s.parsed)
    rate = n_parsed / n
    if rate >= threshold:
        status = "pass"
    elif rate >= SWAP_THRESHOLD:
        status = "marginal"
    else:
        status = "fail"

    return GateResult(
        name=name,
        status=status,
        measured_rate=rate,
        threshold=threshold,
        n_samples=n,
        details={
            "n_parsed": n_parsed,
            "n_failed": n - n_parsed,
            "swap_threshold": SWAP_THRESHOLD,
            "failures": [
                dataclasses.asdict(s) for s in samples if not s.parsed
            ],
        },
    )


class JsonParseRateGate:
    """Operator-facing harness wrapper.

    Holds the threshold + fixtures so the operator can re-run the
    gate idempotently (e.g. on every model swap). The result is
    fully serializable for the deploy record.
    """

    def __init__(
        self,
        fixtures: Iterable[JsonParseRateFixture],
        *,
        threshold: float = DEFAULT_PARSE_RATE_THRESHOLD,
        name: str = "json_parse_rate",
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

    async def run(self, client: _StructuredCompleter) -> GateResult:
        return await measure_json_parse_rate(
            client,
            self._fixtures,
            threshold=self._threshold,
            name=self._name,
        )


__all__ = [
    "DEFAULT_PARSE_RATE_THRESHOLD",
    "JsonParseRateFixture",
    "JsonParseRateGate",
    "SWAP_THRESHOLD",
    "measure_json_parse_rate",
]
