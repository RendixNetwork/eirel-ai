from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace

import pytest

from shared.common import chain_queries
from shared.common.chain_queries import (
    ChainQueryError,
    miner_incentive,
    weight_from_validator,
)


def _install_fake_bittensor(
    monkeypatch,
    *,
    hotkeys: list[str],
    incentives: list[float] | None = None,
    weights: list[list[float]] | None = None,
    sleep_seconds: float = 0.0,
) -> None:
    """Install a stub ``bittensor`` module with a configurable metagraph.

    The stub only implements the pieces ``chain_queries`` touches:
    ``Subtensor(network=...).metagraph(netuid=..., lite=...)`` returning
    an object with ``hotkeys``, ``I``, and ``W``.
    """

    class _FakeMetagraph:
        def __init__(self):
            self.hotkeys = list(hotkeys)
            self.I = list(incentives or [0.0] * len(hotkeys))
            self.W = [list(row) for row in (weights or [])]

    class _FakeSubtensor:
        def __init__(self, *, network: str):
            self.network = network

        def metagraph(self, *, netuid: int, lite: bool):
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            return _FakeMetagraph()

    stub = types.ModuleType("bittensor")
    stub.Subtensor = _FakeSubtensor
    monkeypatch.setitem(sys.modules, "bittensor", stub)


def test_miner_incentive_returns_emission_share(monkeypatch):
    _install_fake_bittensor(
        monkeypatch,
        hotkeys=["owner", "miner-a", "miner-b"],
        incentives=[0.0, 1.0, 0.0],
    )
    got = miner_incentive(network="test", netuid=144, hotkey="miner-a")
    assert got == pytest.approx(1.0)


def test_miner_incentive_returns_none_for_unregistered_hotkey(monkeypatch):
    _install_fake_bittensor(
        monkeypatch,
        hotkeys=["owner", "miner-a"],
        incentives=[0.0, 1.0],
    )
    assert miner_incentive(network="test", netuid=144, hotkey="ghost") is None


def test_weight_from_validator_returns_row_entry(monkeypatch):
    _install_fake_bittensor(
        monkeypatch,
        hotkeys=["v1", "m1", "m2"],
        weights=[
            [0.0, 0.7, 0.3],  # v1 weights
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    )
    assert weight_from_validator(
        network="test",
        netuid=144,
        validator_hotkey="v1",
        miner_hotkey="m1",
    ) == pytest.approx(0.7)


def test_weight_from_validator_distinguishes_unregistered_from_zero(monkeypatch):
    _install_fake_bittensor(
        monkeypatch,
        hotkeys=["v1", "m1"],
        weights=[[1.0, 0.0], [0.0, 0.0]],
    )
    # Registered but weight 0 — the self-burn case B4 is about.
    zero = weight_from_validator(
        network="test", netuid=144, validator_hotkey="v1", miner_hotkey="m1",
    )
    assert zero == pytest.approx(0.0)
    # Unregistered — None, not 0.
    missing = weight_from_validator(
        network="test", netuid=144, validator_hotkey="v1", miner_hotkey="ghost",
    )
    assert missing is None


def test_metagraph_timeout_raises_chain_query_error(monkeypatch):
    _install_fake_bittensor(
        monkeypatch,
        hotkeys=["owner"],
        incentives=[0.0],
        sleep_seconds=2.0,
    )
    with pytest.raises(ChainQueryError, match="timed out"):
        miner_incentive(
            network="test", netuid=144, hotkey="owner", timeout_seconds=0.25,
        )


def test_missing_incentive_vector_raises_chain_query_error(monkeypatch):
    class _FakeMetagraph:
        hotkeys = ["miner-a"]
        I = None
        W = None

    class _FakeSubtensor:
        def __init__(self, **_):
            pass

        def metagraph(self, **_):
            return _FakeMetagraph()

    stub = types.ModuleType("bittensor")
    stub.Subtensor = _FakeSubtensor
    monkeypatch.setitem(sys.modules, "bittensor", stub)
    with pytest.raises(ChainQueryError, match="incentive vector"):
        miner_incentive(network="test", netuid=144, hotkey="miner-a")
