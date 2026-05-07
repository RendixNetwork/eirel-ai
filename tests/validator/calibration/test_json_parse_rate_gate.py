"""JSON parse-rate gate tests.

Verifies the harness logic — pass/marginal/fail thresholds, error
classification, empty-fixture guard. Operators run the same harness
with real provider keys at deploy time.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from validation.validator.calibration.gate_result import GateResult
from validation.validator.calibration.json_parse_rate import (
    DEFAULT_PARSE_RATE_THRESHOLD,
    SWAP_THRESHOLD,
    JsonParseRateFixture,
    JsonParseRateGate,
    measure_json_parse_rate,
)
from validation.validator.providers.types import (
    ProviderError,
    ProviderResponse,
)


pytestmark = pytest.mark.asyncio


class _RotatingClient:
    """Provider-client stub returning a fixed sequence of responses.

    Each ``complete_structured`` call advances through the list,
    raising/returning whatever is at the current index. Lets us
    script a "37 successes, 3 failures" run for threshold tests.
    """

    def __init__(
        self,
        responses: list[str | Exception],
    ) -> None:
        self._responses = list(responses)
        self._idx = 0

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        response_schema: dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        schema_name: str = "response",
    ) -> ProviderResponse:
        item = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        if isinstance(item, Exception):
            raise item
        return ProviderResponse(
            text=item, latency_ms=10, usage_usd=0.0, finish_reason="stop",
        )


def _fixture(n: int = 1) -> list[JsonParseRateFixture]:
    return [
        JsonParseRateFixture(
            system="be terse", user=f"prompt {i}",
            response_schema={"type": "object"},
        )
        for i in range(n)
    ]


# -- pass / marginal / fail thresholds -----------------------------------


async def test_all_parsed_returns_pass():
    """All fixtures return valid JSON → status=pass."""
    client = _RotatingClient([json.dumps({"ok": True})])
    result = await measure_json_parse_rate(client, _fixture(20))
    assert result.status == "pass"
    assert result.measured_rate == 1.0
    assert result.n_samples == 20
    assert result.details["n_parsed"] == 20
    assert result.details["n_failed"] == 0


async def test_just_above_threshold_passes():
    """98 valid + 2 invalid = 98% parse rate, exactly at default
    threshold → pass."""
    responses: list[str | Exception] = [json.dumps({"ok": True})] * 98 + ["not json"] * 2
    client = _RotatingClient(responses)
    result = await measure_json_parse_rate(client, _fixture(100))
    assert result.measured_rate == 0.98
    assert result.status == "pass"


async def test_marginal_zone_returns_marginal():
    """Parse rate ∈ [SWAP_THRESHOLD, default_threshold) → marginal.
    Operator should wrap with JSON-repair retry before promoting."""
    # 92% parse rate: 92 valid + 8 malformed
    responses: list[str | Exception] = (
        [json.dumps({"x": 1})] * 92 + ["malformed"] * 8
    )
    client = _RotatingClient(responses)
    result = await measure_json_parse_rate(client, _fixture(100))
    assert result.measured_rate == 0.92
    assert SWAP_THRESHOLD <= result.measured_rate < DEFAULT_PARSE_RATE_THRESHOLD
    assert result.status == "marginal"


async def test_below_swap_threshold_returns_fail():
    """Parse rate < SWAP_THRESHOLD (~90%) → fail. Model not ready;
    JSON-repair retry can't recover this regime."""
    # 50% parse rate
    responses: list[str | Exception] = (
        [json.dumps({"x": 1})] * 50 + ["bad"] * 50
    )
    client = _RotatingClient(responses)
    result = await measure_json_parse_rate(client, _fixture(100))
    assert result.measured_rate == 0.5
    assert result.status == "fail"


async def test_custom_threshold_overrides_default():
    """Operator-supplied threshold takes precedence over the default."""
    client = _RotatingClient([json.dumps({"ok": True})] * 9 + ["bad"])
    # Default: 0.98 → 0.9 fails. Custom: 0.85 → 0.9 passes.
    result_default = await measure_json_parse_rate(client, _fixture(10))
    assert result_default.status in ("marginal", "fail")
    client2 = _RotatingClient([json.dumps({"ok": True})] * 9 + ["bad"])
    result_custom = await measure_json_parse_rate(
        client2, _fixture(10), threshold=0.85,
    )
    assert result_custom.status == "pass"


# -- error classification ------------------------------------------------


async def test_provider_call_errors_count_as_parse_failures():
    """A ProviderError raised by the wrapped client is counted as a
    failure, not propagated. The gate harness keeps going so a single
    flaky call doesn't tank the run."""
    responses: list[str | Exception] = [
        json.dumps({"ok": True}),
        ProviderError("boom"),
        json.dumps({"ok": True}),
    ]
    client = _RotatingClient(responses)
    result = await measure_json_parse_rate(client, _fixture(3))
    assert result.measured_rate == pytest.approx(2 / 3)
    failure_details = result.details["failures"]
    assert any("call_error" in (f.get("error") or "") for f in failure_details)


async def test_failed_calls_recorded_in_details():
    responses: list[str | Exception] = [
        json.dumps({"ok": True}),
        "totally not json",
        json.dumps({"ok": True}),
    ]
    client = _RotatingClient(responses)
    result = await measure_json_parse_rate(client, _fixture(3))
    failures = result.details["failures"]
    assert len(failures) == 1
    assert failures[0]["fixture_index"] == 1
    assert "parse_error" in failures[0]["error"]


# -- edge cases ----------------------------------------------------------


async def test_empty_fixtures_returns_fail():
    client = _RotatingClient([json.dumps({"ok": True})])
    result = await measure_json_parse_rate(client, [])
    assert result.status == "fail"
    assert result.n_samples == 0
    assert result.details.get("reason") == "no_fixtures_provided"


async def test_gate_helper_runs_fixtures():
    """Operator-facing wrapper: hold fixtures + threshold, call
    ``run(client)``."""
    client = _RotatingClient([json.dumps({"ok": True})] * 10)
    gate = JsonParseRateGate(_fixture(10), threshold=0.95)
    assert gate.threshold == 0.95
    assert gate.n_fixtures == 10
    result = await gate.run(client)
    assert result.status == "pass"


async def test_gate_result_carries_passed_property():
    client = _RotatingClient([json.dumps({"ok": True})])
    result = await measure_json_parse_rate(client, _fixture(5))
    assert isinstance(result, GateResult)
    assert result.passed is True
