from __future__ import annotations

"""Tests for Item 1: Weight submission verification with retry and chain polling."""

import types
from unittest.mock import MagicMock, patch

import pytest

from validation.weight_setter.setter import (
    WeightSubmissionConfig,
    _chain_circuit_breaker,
    build_weight_submission,
    submit_weight_submission,
)
from validation.weight_setter.chain_verifier import verify_weights_on_chain


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset circuit breaker state between tests."""
    _chain_circuit_breaker.reset_all()
    yield
    _chain_circuit_breaker.reset_all()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_config() -> WeightSubmissionConfig:
    return WeightSubmissionConfig(
        network="test",
        netuid=99,
        wallet_name="test_wallet",
        hotkey_name="test_hotkey",
        wallet_path="/tmp/test_wallets",
    )


def _make_payload() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "family_id": "analyst",
        "family_weight": 0.42,
        "weights": {"hk_a": 0.6, "hk_b": 0.4},
        "scaled_weights": {"hk_a": 0.252, "hk_b": 0.168},
        "query_volume_share": 0.0,
        "rubric_version": "v1",
        "allocation_mode": "fixed_family_weights_v1",
    }


def _mock_subtensor(hotkeys=None, set_weights_side_effect=None):
    """Return a mock bt.Subtensor with metagraph hotkeys."""
    hotkeys = hotkeys or ["hk_a", "hk_b"]
    mock = MagicMock()
    metagraph = MagicMock()
    metagraph.hotkeys = hotkeys
    mock.metagraph.return_value = metagraph
    if set_weights_side_effect is not None:
        mock.set_weights.side_effect = set_weights_side_effect
    else:
        mock.set_weights.return_value = True
    return mock


def _mock_wallet(hotkey_address="hk_validator"):
    mock = MagicMock()
    mock.hotkey.ss58_address = hotkey_address
    return mock


# ── build_weight_submission ──────────────────────────────────────────────────


def test_build_weight_submission_produces_valid_payload():
    from shared.contracts.models import FamilyScoreSnapshot

    snapshot = FamilyScoreSnapshot(
        run_id="run-1",
        family_id="general_chat",
        miner_scores={"hk_a": 0.9},
        normalized_weights={"hk_a": 1.0},
        miner_query_volume_shares={"hk_a": 1.0},
        miner_score_breakdowns={},
        miner_robustness_scores={},
        miner_anti_gaming_flags={},
        rubric_version="v1",
    )
    result = build_weight_submission(snapshot)
    assert result["run_id"] == "run-1"
    assert result["family_id"] == "general_chat"
    assert result["allocation_mode"] == "fixed_family_weights_v1"
    assert "scaled_weights" in result


# ── submit_weight_submission: retry on failure ───────────────────────────────


@patch("validation.weight_setter.setter.bt")
def test_retry_on_set_weights_exception(mock_bt):
    mock_sub = _mock_subtensor(set_weights_side_effect=[
        RuntimeError("chain timeout"),
        RuntimeError("chain timeout"),
        True,  # third attempt succeeds
    ])
    mock_bt.Subtensor.return_value = mock_sub
    mock_bt.Wallet.return_value = _mock_wallet()

    result = submit_weight_submission(
        _make_payload(),
        config=_make_config(),
        max_attempts=3,
        backoff_base_seconds=0.01,
        verify=False,
    )
    assert result["submitted"] is True
    assert result["attempt_count"] == 3
    assert result["attempts"][0]["status"] == "failed"
    assert result["attempts"][1]["status"] == "failed"
    assert result["attempts"][2]["status"] == "success"


@patch("validation.weight_setter.setter.bt")
def test_retry_exhaustion_raises(mock_bt):
    mock_sub = _mock_subtensor(set_weights_side_effect=RuntimeError("always fails"))
    mock_bt.Subtensor.return_value = mock_sub
    mock_bt.Wallet.return_value = _mock_wallet()

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        submit_weight_submission(
            _make_payload(),
            config=_make_config(),
            max_attempts=3,
            backoff_base_seconds=0.01,
            verify=False,
        )


@patch("validation.weight_setter.setter.bt")
def test_single_success_no_retry(mock_bt):
    mock_sub = _mock_subtensor()
    mock_bt.Subtensor.return_value = mock_sub
    mock_bt.Wallet.return_value = _mock_wallet()

    result = submit_weight_submission(
        _make_payload(),
        config=_make_config(),
        max_attempts=3,
        backoff_base_seconds=0.01,
        verify=False,
    )
    assert result["submitted"] is True
    assert result["attempt_count"] == 1
    assert mock_sub.set_weights.call_count == 1


# ── submit_weight_submission: wait_for_inclusion=True ────────────────────────


@patch("validation.weight_setter.setter.bt")
def test_wait_for_inclusion_is_true(mock_bt):
    mock_sub = _mock_subtensor()
    mock_bt.Subtensor.return_value = mock_sub
    mock_bt.Wallet.return_value = _mock_wallet()

    submit_weight_submission(
        _make_payload(),
        config=_make_config(),
        verify=False,
        backoff_base_seconds=0.01,
    )
    call_kwargs = mock_sub.set_weights.call_args.kwargs
    assert call_kwargs["wait_for_inclusion"] is True


# ── submit_weight_submission: chain verification ────────────────────────────


@patch("validation.weight_setter.setter.verify_weights_on_chain")
@patch("validation.weight_setter.setter.bt")
def test_verification_called_after_success(mock_bt, mock_verify):
    mock_sub = _mock_subtensor()
    mock_bt.Subtensor.return_value = mock_sub
    mock_bt.Wallet.return_value = _mock_wallet()
    mock_verify.return_value = {"verified": True, "mismatches": [], "poll_attempts": 1}

    result = submit_weight_submission(
        _make_payload(),
        config=_make_config(),
        verify=True,
        backoff_base_seconds=0.01,
    )
    assert result["verified"] is True
    assert result["verification"]["verified"] is True
    mock_verify.assert_called_once()


@patch("validation.weight_setter.setter.verify_weights_on_chain")
@patch("validation.weight_setter.setter.bt")
def test_verification_failure_does_not_raise(mock_bt, mock_verify):
    mock_sub = _mock_subtensor()
    mock_bt.Subtensor.return_value = mock_sub
    mock_bt.Wallet.return_value = _mock_wallet()
    mock_verify.side_effect = RuntimeError("chain unavailable")

    result = submit_weight_submission(
        _make_payload(),
        config=_make_config(),
        verify=True,
        backoff_base_seconds=0.01,
    )
    # Submission succeeded even though verification failed.
    assert result["submitted"] is True
    assert result["verified"] is False


# ── chain_verifier ───────────────────────────────────────────────────────────


def test_verify_weights_match():
    mock_sub = MagicMock()
    metagraph = MagicMock()
    metagraph.hotkeys = ["hk_validator", "hk_a", "hk_b"]
    metagraph.W = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.6, 0.4]]
    # Validator is uid 0, miners are uid 1 and 2.
    # We need to make the validator hotkey resolve to uid 0.
    metagraph.hotkeys = ["hk_validator", "hk_a", "hk_b"]
    metagraph.W = [[0.0, 0.6, 0.4], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    mock_sub.metagraph.return_value = metagraph

    result = verify_weights_on_chain(
        subtensor=mock_sub,
        netuid=99,
        wallet_hotkey="hk_validator",
        expected_uids=[1, 2],
        expected_weights=[0.6, 0.4],
        max_poll_attempts=1,
        poll_interval_seconds=0,
    )
    assert result["verified"] is True
    assert result["mismatches"] == []


def test_verify_weights_mismatch():
    mock_sub = MagicMock()
    metagraph = MagicMock()
    metagraph.hotkeys = ["hk_validator", "hk_a"]
    metagraph.W = [[0.0, 0.3], [0.0, 0.0]]
    mock_sub.metagraph.return_value = metagraph

    result = verify_weights_on_chain(
        subtensor=mock_sub,
        netuid=99,
        wallet_hotkey="hk_validator",
        expected_uids=[1],
        expected_weights=[0.6],
        max_poll_attempts=1,
        poll_interval_seconds=0,
    )
    assert result["verified"] is False
    assert len(result["mismatches"]) == 1


def test_verify_empty_uids():
    result = verify_weights_on_chain(
        subtensor=MagicMock(),
        netuid=99,
        wallet_hotkey="hk_validator",
        expected_uids=[],
        expected_weights=[],
    )
    assert result["verified"] is False
    assert result["reason"] == "no expected uids"


def test_verify_polls_on_hotkey_not_found():
    mock_sub = MagicMock()
    metagraph = MagicMock()
    metagraph.hotkeys = ["other_hotkey"]
    metagraph.W = [[0.5]]
    mock_sub.metagraph.return_value = metagraph

    result = verify_weights_on_chain(
        subtensor=mock_sub,
        netuid=99,
        wallet_hotkey="hk_validator",
        expected_uids=[0],
        expected_weights=[0.5],
        max_poll_attempts=2,
        poll_interval_seconds=0,
    )
    assert result["verified"] is False
    assert result["poll_attempts"] == 2


# ── Backoff timing ───────────────────────────────────────────────────────────


@patch("validation.weight_setter.setter.time.sleep")
@patch("validation.weight_setter.setter.bt")
def test_exponential_backoff_between_retries(mock_bt, mock_sleep):
    mock_sub = _mock_subtensor(set_weights_side_effect=[
        RuntimeError("fail"),
        RuntimeError("fail"),
        True,
    ])
    mock_bt.Subtensor.return_value = mock_sub
    mock_bt.Wallet.return_value = _mock_wallet()

    submit_weight_submission(
        _make_payload(),
        config=_make_config(),
        max_attempts=3,
        backoff_base_seconds=2.0,
        verify=False,
    )
    # Backoff: attempt 1 fails → sleep(2.0 * 2^0 = 2.0), attempt 2 fails → sleep(2.0 * 2^1 = 4.0)
    assert mock_sleep.call_count == 2
    assert mock_sleep.call_args_list[0][0][0] == pytest.approx(2.0)
    assert mock_sleep.call_args_list[1][0][0] == pytest.approx(4.0)


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_missing_wallet_config_raises():
    config = WeightSubmissionConfig(
        network="test", netuid=99,
        wallet_name=None, hotkey_name=None, wallet_path=None,
    )
    with pytest.raises(ValueError, match="wallet configuration is missing"):
        submit_weight_submission(_make_payload(), config=config)


def test_empty_payload_raises():
    config = _make_config()
    with pytest.raises(ValueError, match="weight payload is empty"):
        submit_weight_submission({"weights": {}}, config=config)


@patch("validation.weight_setter.setter.bt")
def test_skipped_hotkeys_tracked(mock_bt):
    mock_sub = _mock_subtensor(hotkeys=["hk_a"])  # hk_b not in metagraph
    mock_bt.Subtensor.return_value = mock_sub
    mock_bt.Wallet.return_value = _mock_wallet()

    result = submit_weight_submission(
        _make_payload(),
        config=_make_config(),
        verify=False,
        backoff_base_seconds=0.01,
    )
    assert "hk_b" in result["skipped_hotkeys"]
