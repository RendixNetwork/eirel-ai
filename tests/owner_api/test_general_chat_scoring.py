"""Tests for agreement aggregation in general_chat_scoring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from control_plane.owner_api.evaluation.general_chat_scoring import (
    MinerRollup,
    aggregate_miner_score,
)


def _result(
    miner_hotkey: str,
    verdict: str,
    *,
    agreement_score: float | None = None,
):
    """Build a TaskMinerResult-shaped object for aggregation tests.

    Uses SimpleNamespace because aggregate_miner_score only reads attributes
    and not ORM-specific behavior.
    """
    if agreement_score is None:
        # Mirror the DB-persisted scalar derived from the verdict.
        agreement_score = {
            "matches": 1.0, "partially_matches": 0.6,
            "not_applicable": 0.7, "contradicts": 0.0, "error": 0.0,
        }.get(verdict, 0.0)
    return SimpleNamespace(
        miner_hotkey=miner_hotkey,
        agreement_verdict=verdict,
        agreement_score=agreement_score,
        judge_output_json={"verdict": verdict},
    )


def test_empty_results_returns_unreliable_zero():
    rollup = aggregate_miner_score([])
    assert rollup.total_judged == 0
    assert rollup.final_score == 0.0
    assert rollup.reliable is False


def test_all_matches_gives_mean_agreement_1():
    results = [_result("hk", "matches") for _ in range(4)]
    rollup = aggregate_miner_score(results)
    assert rollup.mean_agreement == 1.0
    assert rollup.final_score == 1.0
    assert rollup.reliable is True
    assert rollup.matches == 4


def test_all_contradicts_gives_zero():
    results = [_result("hk", "contradicts") for _ in range(4)]
    rollup = aggregate_miner_score(results)
    assert rollup.mean_agreement == 0.0
    assert rollup.final_score == 0.0
    assert rollup.reliable is True


def test_partially_matches_scores_0_6():
    results = [_result("hk", "partially_matches") for _ in range(3)]
    rollup = aggregate_miner_score(results)
    assert rollup.mean_agreement == 0.6
    assert rollup.final_score == 0.6


def test_not_applicable_scores_0_7():
    results = [_result("hk", "not_applicable") for _ in range(2)]
    rollup = aggregate_miner_score(results)
    assert rollup.mean_agreement == 0.7
    assert rollup.final_score == 0.7


def test_mixed_verdicts_compute_correct_mean():
    # matches (1.0) + partially_matches (0.6) + contradicts (0.0) → mean 0.5333
    # not_applicable (0.7) brings mean to (1 + 0.6 + 0 + 0.7) / 4 = 0.575
    results = [
        _result("hk", "matches"),
        _result("hk", "partially_matches"),
        _result("hk", "contradicts"),
        _result("hk", "not_applicable"),
    ]
    rollup = aggregate_miner_score(results)
    assert rollup.mean_agreement == pytest.approx((1.0 + 0.6 + 0.0 + 0.7) / 4)
    assert rollup.reliable is True


def test_errors_excluded_from_mean_but_counted_for_error_rate():
    # 3 matches (score 1.0 each) + 1 error → mean over completed = 1.0
    # error_rate = 1/4 = 25% → below 30% threshold → reliable
    results = [_result("hk", "matches") for _ in range(3)] + [_result("hk", "error")]
    rollup = aggregate_miner_score(results)
    assert rollup.mean_agreement == 1.0
    assert rollup.error_rate == 0.25
    assert rollup.reliable is True
    assert rollup.final_score == 1.0


def test_error_rate_above_threshold_caps_final_score():
    # 2 matches + 2 errors = 50% error rate → unreliable, capped at 0.5
    results = [_result("hk", "matches") for _ in range(2)] + [_result("hk", "error") for _ in range(2)]
    rollup = aggregate_miner_score(results)
    assert rollup.mean_agreement == 1.0
    assert rollup.error_rate == 0.5
    assert rollup.reliable is False
    assert rollup.final_score == 0.5


def test_metadata_export_contains_verdict_counts():
    results = [
        _result("hk", "matches"),
        _result("hk", "partially_matches"),
        _result("hk", "contradicts"),
    ]
    rollup = aggregate_miner_score(results)
    metadata = rollup.to_metadata()
    assert metadata["verdict_counts"]["matches"] == 1
    assert metadata["verdict_counts"]["partially_matches"] == 1
    assert metadata["verdict_counts"]["contradicts"] == 1
    assert metadata["mean_agreement"] == pytest.approx((1.0 + 0.6 + 0.0) / 3)


def test_rollup_is_frozen_dataclass():
    rollup = aggregate_miner_score([_result("hk", "matches")])
    with pytest.raises(Exception):
        rollup.mean_agreement = 0.0  # type: ignore[misc]
