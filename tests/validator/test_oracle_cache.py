"""TaskOracleCache tests.

Cache is intentionally a plain dict — these tests exist to lock the
contract (set/get/require/clear) so future refactors don't silently
change behavior the validator engine depends on.
"""

from __future__ import annotations

import pytest

from validation.validator.oracle_cache import TaskOracleCache
from validation.validator.reconciler import ReconciledOracle


def _make_reconciled(claims: list[str]) -> ReconciledOracle:
    return ReconciledOracle(
        expected_claims=claims,
        oracle_status="consensus",
    )


def test_set_and_get():
    cache = TaskOracleCache()
    rec = _make_reconciled(["X"])
    cache.set("task-1", rec)
    assert cache.get("task-1") is rec


def test_get_missing_returns_none():
    cache = TaskOracleCache()
    assert cache.get("absent") is None


def test_require_raises_on_missing():
    cache = TaskOracleCache()
    with pytest.raises(KeyError):
        cache.require("absent")


def test_require_returns_value_on_hit():
    cache = TaskOracleCache()
    rec = _make_reconciled(["X"])
    cache.set("t", rec)
    assert cache.require("t") is rec


def test_set_overwrites():
    cache = TaskOracleCache()
    cache.set("t", _make_reconciled(["X"]))
    cache.set("t", _make_reconciled(["Y"]))
    assert cache.get("t").expected_claims == ["Y"]


def test_clear_drops_everything():
    cache = TaskOracleCache()
    cache.set("a", _make_reconciled(["1"]))
    cache.set("b", _make_reconciled(["2"]))
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a") is None
    assert cache.get("b") is None


def test_contains():
    cache = TaskOracleCache()
    cache.set("present", _make_reconciled(["X"]))
    assert "present" in cache
    assert "absent" not in cache


def test_iter_yields_task_ids():
    cache = TaskOracleCache()
    cache.set("a", _make_reconciled(["1"]))
    cache.set("b", _make_reconciled(["2"]))
    assert sorted(cache) == ["a", "b"]


def test_cache_lifetime_simulates_batch_processing():
    """Simulate the validator's batch flow: claim phase populates,
    judge phase consumes (possibly N times for N miners), batch end
    clears."""
    cache = TaskOracleCache()
    # Claim phase: 3 tasks.
    for tid in ("t1", "t2", "t3"):
        cache.set(tid, _make_reconciled([f"answer-{tid}"]))
    # Judge phase: each task consumed multiple times (per miner).
    for _ in range(5):
        for tid in ("t1", "t2", "t3"):
            assert cache.require(tid).expected_claims == [f"answer-{tid}"]
    # Batch end.
    cache.clear()
    assert len(cache) == 0
