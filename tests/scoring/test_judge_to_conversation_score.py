from __future__ import annotations

import pytest

from shared.core.evaluation_models import ConversationScore
from shared.scoring.families._judge_to_conversation_score import (
    build_conversation_score_from_judge,
)


def _judge(
    score: float = 0.8,
    *,
    latency_seconds: float = 0.0,
    dimension_scores: dict[str, float] | None = None,
    constraint_flags: list[str] | None = None,
) -> dict:
    return {
        "model": "test-judge",
        "rubric_name": "general_chat_v1",
        "score": score,
        "rationale": "",
        "latency_seconds": latency_seconds,
        "dimension_scores": dimension_scores or {},
        "constraint_flags": constraint_flags or [],
        "usage": {},
        "metadata": {},
    }


def _miner_response(latency_ms: int | None = None) -> dict:
    inner = {"content": "hello"}
    if latency_ms is not None:
        inner["latency_ms"] = latency_ms
    return {
        "task_id": "t-1",
        "family_id": "general_chat",
        "response": inner,
        "status": "completed",
    }


def test_build_from_judge_fills_all_required_dimensions():
    cs = build_conversation_score_from_judge(
        task_score=0.7,
        judge_output=_judge(score=0.7),
        miner_response=_miner_response(latency_ms=5000),
    )
    assert isinstance(cs, ConversationScore)
    assert 0.0 <= cs.quality <= 1.0
    assert 0.0 <= cs.latency <= 1.0
    assert 0.0 <= cs.cost <= 1.0
    assert 0.0 <= cs.trace_gate <= 1.0
    assert 0.0 <= cs.total <= 1.0


def test_build_from_judge_is_serializable_and_roundtrips():
    cs = build_conversation_score_from_judge(
        task_score=0.5,
        judge_output=_judge(score=0.5),
        miner_response=_miner_response(latency_ms=1000),
    )
    dumped = cs.model_dump(mode="json")
    roundtrip = ConversationScore.model_validate(dumped)
    assert roundtrip.quality == cs.quality
    assert roundtrip.total == cs.total


def test_quality_prefers_judge_score_over_task_score():
    # When the judge reports 0.9 but the validator submitted a stale task_score
    # of 0.1, the judge's explicit overall score wins.
    cs = build_conversation_score_from_judge(
        task_score=0.1,
        judge_output=_judge(score=0.9),
        miner_response=_miner_response(latency_ms=1000),
    )
    assert cs.quality == pytest.approx(0.9)


def test_quality_falls_back_to_task_score_when_judge_has_no_score():
    cs = build_conversation_score_from_judge(
        task_score=0.42,
        judge_output={"dimension_scores": {}, "constraint_flags": []},
        miner_response=_miner_response(latency_ms=0),
    )
    assert cs.quality == pytest.approx(0.42)


def test_latency_unmeasured_is_neutral_not_free_one():
    # Both miner response and judge output lack latency data.  The result
    # should be 0.5 (neutral), NOT 1.0 — a free 1.0 would let unmeasured
    # miners out-score measured ones.
    cs = build_conversation_score_from_judge(
        task_score=0.7,
        judge_output=_judge(score=0.7, latency_seconds=0.0),
        miner_response=_miner_response(latency_ms=None),
    )
    assert cs.latency == pytest.approx(0.5)


def test_latency_uses_miner_latency_when_present():
    # Instant budget is 20 s; 5 s → 0.75 score.
    cs = build_conversation_score_from_judge(
        task_score=0.7,
        judge_output=_judge(latency_seconds=999.0),  # should be ignored
        miner_response=_miner_response(latency_ms=5000),
    )
    assert cs.latency == pytest.approx(0.75)


def test_latency_falls_back_to_judge_latency_when_miner_omits_it():
    # 2 s from the judge → (1 - 2/20) = 0.9
    cs = build_conversation_score_from_judge(
        task_score=0.7,
        judge_output=_judge(latency_seconds=2.0),
        miner_response=_miner_response(latency_ms=None),
    )
    assert cs.latency == pytest.approx(0.9)


def test_thinking_mode_uses_wider_latency_budget():
    # 30 s in thinking mode → (1 - 30/60) = 0.5; same latency in instant
    # mode would be 0 (clamped).
    thinking = build_conversation_score_from_judge(
        task_score=0.7,
        judge_output=_judge(),
        miner_response=_miner_response(latency_ms=30_000),
        mode="thinking",
    )
    instant = build_conversation_score_from_judge(
        task_score=0.7,
        judge_output=_judge(),
        miner_response=_miner_response(latency_ms=30_000),
        mode="instant",
    )
    assert thinking.latency == pytest.approx(0.5)
    assert instant.latency == pytest.approx(0.0)


def test_total_uses_layered_formula():
    # total = trace_gate * (0.8 * quality + 0.2 * latency)
    cs = build_conversation_score_from_judge(
        task_score=0.0,
        judge_output=_judge(score=1.0),
        miner_response=_miner_response(latency_ms=10_000),  # latency=0.5 in instant
    )
    expected = 1.0 * (0.8 * 1.0 + 0.2 * 0.5)
    assert cs.total == pytest.approx(expected)


def test_trace_gate_defaults_to_pass_and_metadata_records_constraint_flags():
    cs = build_conversation_score_from_judge(
        task_score=0.8,
        judge_output=_judge(score=0.8, constraint_flags=["off_topic", "refusal"]),
        miner_response=_miner_response(latency_ms=1000),
    )
    # The synthesizer does not zero the gate on flags — flags are quality
    # signals, not integrity signals.  They're recorded in metadata for
    # downstream auditing.
    assert cs.trace_gate == 1.0
    assert cs.metadata["synthesized_from_judge"] is True
    assert cs.metadata["constraint_flags"] == ["off_topic", "refusal"]


def test_malformed_judge_output_does_not_crash():
    # Judge output can legitimately be an empty dict on a total judge
    # failure — synthesis must still return a valid ConversationScore.
    cs = build_conversation_score_from_judge(
        task_score=0.3,
        judge_output={},
        miner_response=_miner_response(),
    )
    assert cs.quality == pytest.approx(0.3)
    assert 0.0 <= cs.total <= 1.0
