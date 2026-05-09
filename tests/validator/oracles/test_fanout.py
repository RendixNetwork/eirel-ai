"""Oracle fanout tests.

Covers:
  * All-3-OK / 1-down / 2-down / all-down combinations → vendor_status_map
    reflects the right shape; successful_groundings drops errors.
  * Vendor order in results matches client construction order.
  * Sequential mode produces same outputs as parallel.
  * Last-resort safety net: an oracle client that *raises* (instead
    of returning ``status="error"``) gets caught.
"""

from __future__ import annotations

import asyncio

import pytest

from validation.validator.oracles.base import (
    OracleClient,
    OracleContext,
    OracleGrounding,
)
from validation.validator.oracles.fanout import (
    OracleFanout,
    successful_groundings,
    vendor_status_map,
)


pytestmark = pytest.mark.asyncio


class _StubOracle(OracleClient):
    """OracleClient that returns a pre-configured grounding."""

    def __init__(self, vendor: str, grounding: OracleGrounding) -> None:
        self._vendor = vendor
        self._grounding = grounding
        self.calls = 0

    @property
    def vendor(self) -> str:
        return self._vendor

    async def produce_grounding(self, context: OracleContext) -> OracleGrounding:
        self.calls += 1
        return self._grounding

    async def aclose(self) -> None:
        pass


class _RaisingOracle(OracleClient):
    """OracleClient that raises an unexpected exception (safety-net path)."""

    def __init__(self, vendor: str) -> None:
        self._vendor = vendor

    @property
    def vendor(self) -> str:
        return self._vendor

    async def produce_grounding(self, context: OracleContext) -> OracleGrounding:
        raise RuntimeError("simulated client bug")

    async def aclose(self) -> None:
        pass


def _ctx() -> OracleContext:
    return OracleContext(task_id="t1", prompt="What's 2+2?")


async def test_all_three_ok():
    a = _StubOracle("openai", OracleGrounding(vendor="openai", status="ok", raw_text="4"))
    b = _StubOracle("gemini", OracleGrounding(vendor="gemini", status="ok", raw_text="4"))
    c = _StubOracle("grok", OracleGrounding(vendor="grok", status="ok", raw_text="4"))
    fan = OracleFanout([a, b, c])
    out = await fan.run(_ctx())

    assert [g.vendor for g in out] == ["openai", "gemini", "grok"]
    assert all(g.status == "ok" for g in out)
    assert vendor_status_map(out) == {"openai": "ok", "gemini": "ok", "grok": "ok"}
    assert len(successful_groundings(out)) == 3


async def test_one_oracle_down_returns_two_ok_one_error():
    a = _StubOracle("openai", OracleGrounding(vendor="openai", status="ok", raw_text="4"))
    b = _StubOracle("gemini", OracleGrounding(vendor="gemini", status="ok", raw_text="4"))
    c = _StubOracle(
        "grok",
        OracleGrounding(vendor="grok", status="error", error_msg="timeout: vendor down"),
    )
    fan = OracleFanout([a, b, c])
    out = await fan.run(_ctx())

    assert vendor_status_map(out) == {"openai": "ok", "gemini": "ok", "grok": "error"}
    ok = successful_groundings(out)
    assert len(ok) == 2
    assert {g.vendor for g in ok} == {"openai", "gemini"}


async def test_two_oracles_down_returns_one_ok():
    a = _StubOracle("openai", OracleGrounding(vendor="openai", status="ok", raw_text="4"))
    b = _StubOracle("gemini", OracleGrounding(vendor="gemini", status="error", error_msg="boom"))
    c = _StubOracle("grok", OracleGrounding(vendor="grok", status="error", error_msg="boom"))
    fan = OracleFanout([a, b, c])
    out = await fan.run(_ctx())

    ok = successful_groundings(out)
    assert len(ok) == 1
    assert ok[0].vendor == "openai"
    assert vendor_status_map(out)["gemini"] == "error"


async def test_all_three_down_returns_no_ok():
    a = _StubOracle("openai", OracleGrounding(vendor="openai", status="error", error_msg="x"))
    b = _StubOracle("gemini", OracleGrounding(vendor="gemini", status="error", error_msg="x"))
    c = _StubOracle("grok", OracleGrounding(vendor="grok", status="error", error_msg="x"))
    fan = OracleFanout([a, b, c])
    out = await fan.run(_ctx())

    assert successful_groundings(out) == []
    assert vendor_status_map(out) == {"openai": "error", "gemini": "error", "grok": "error"}


async def test_blocked_status_isolated_from_error():
    """Gemini's safety-block surface as status=blocked, not error."""
    a = _StubOracle("openai", OracleGrounding(vendor="openai", status="ok", raw_text="x"))
    b = _StubOracle(
        "gemini", OracleGrounding(vendor="gemini", status="blocked", error_msg="SAFETY"),
    )
    c = _StubOracle("grok", OracleGrounding(vendor="grok", status="ok", raw_text="x"))
    fan = OracleFanout([a, b, c])
    out = await fan.run(_ctx())

    statuses = vendor_status_map(out)
    assert statuses == {"openai": "ok", "gemini": "blocked", "grok": "ok"}
    # blocked is NOT an "ok" — successful_groundings drops it.
    assert {g.vendor for g in successful_groundings(out)} == {"openai", "grok"}


async def test_safety_net_catches_unexpected_exception():
    """A buggy oracle client that raises (instead of returning
    status=error) should be caught by the fanout layer's last-resort
    try/except, surfacing as status=error rather than crashing the run."""
    a = _RaisingOracle("openai")
    b = _StubOracle("gemini", OracleGrounding(vendor="gemini", status="ok", raw_text="4"))
    fan = OracleFanout([a, b])
    out = await fan.run(_ctx())

    a_grounding = next(g for g in out if g.vendor == "openai")
    assert a_grounding.status == "error"
    assert "unexpected_exception" in (a_grounding.error_msg or "")
    b_grounding = next(g for g in out if g.vendor == "gemini")
    assert b_grounding.status == "ok"


async def test_parallel_mode_runs_concurrently():
    """Sanity check: parallel mode kicks off all 3 calls before
    any returns. Stub oracles each sleep briefly; total wall-clock
    in parallel ≈ max(per-call), in sequential ≈ sum."""

    class _SlowStub(OracleClient):
        def __init__(self, vendor: str, sleep_s: float) -> None:
            self._vendor = vendor
            self._sleep_s = sleep_s

        @property
        def vendor(self) -> str:
            return self._vendor

        async def produce_grounding(self, context: OracleContext) -> OracleGrounding:
            await asyncio.sleep(self._sleep_s)
            return OracleGrounding(vendor=self._vendor, status="ok", raw_text="x")

        async def aclose(self) -> None:
            pass

    clients = [_SlowStub(v, 0.05) for v in ("openai", "gemini", "grok")]
    parallel_fan = OracleFanout(clients, parallel=True)
    sequential_fan = OracleFanout(clients, parallel=False)

    t0 = asyncio.get_event_loop().time()
    await parallel_fan.run(_ctx())
    t_par = asyncio.get_event_loop().time() - t0

    t0 = asyncio.get_event_loop().time()
    await sequential_fan.run(_ctx())
    t_seq = asyncio.get_event_loop().time() - t0

    # Parallel should finish in ~0.05s vs ~0.15s sequential. Allow
    # generous slack for CI scheduler variance.
    assert t_par < 0.12, f"parallel took {t_par:.3f}s (expected ~0.05s)"
    assert t_seq > 0.12, f"sequential took {t_seq:.3f}s (expected ~0.15s)"


async def test_empty_clients_rejected():
    with pytest.raises(ValueError):
        OracleFanout([])


# -- run_single ------------------------------------------------------------


async def test_run_single_calls_only_named_vendor():
    """``run_single`` invokes ONE client by tag; other clients
    untouched. Used as the pairwise-reference fetch on deterministic
    tasks where calling all 3 oracles would be wasted spend."""
    openai = _StubOracle("openai", OracleGrounding(
        vendor="openai", status="ok", raw_text="Paris is the capital",
        cost_usd=0.012,
    ))
    gemini = _StubOracle("gemini", OracleGrounding(
        vendor="gemini", status="ok", raw_text="Should not be called",
    ))
    grok = _StubOracle("grok", OracleGrounding(
        vendor="grok", status="ok", raw_text="Also should not",
    ))
    fanout = OracleFanout([openai, gemini, grok])

    grounding = await fanout.run_single(
        "openai", OracleContext(task_id="t", prompt="capital of France?"),
    )

    assert grounding is not None
    assert grounding.vendor == "openai"
    assert grounding.raw_text == "Paris is the capital"
    assert grounding.cost_usd == 0.012
    assert openai.calls == 1
    assert gemini.calls == 0
    assert grok.calls == 0


async def test_run_single_unknown_vendor_returns_none():
    """No-op when the requested vendor isn't a configured client.
    Caller falls through to ``expected_claims[0]`` rather than crash."""
    fanout = OracleFanout([
        _StubOracle("openai", OracleGrounding(vendor="openai", status="ok")),
    ])

    grounding = await fanout.run_single(
        "grok", OracleContext(task_id="t", prompt="?"),
    )
    assert grounding is None


async def test_run_single_propagates_oracle_error_status():
    """Provider error inside the oracle still produces a grounding
    object — caller checks status/raw_text and falls through if not
    usable, rather than hitting a network exception."""
    fanout = OracleFanout([
        _StubOracle("openai", OracleGrounding(
            vendor="openai", status="error", error_msg="rate limited",
        )),
    ])
    grounding = await fanout.run_single(
        "openai", OracleContext(task_id="t", prompt="?"),
    )
    assert grounding is not None
    assert grounding.status == "error"
    assert grounding.raw_text == ""
