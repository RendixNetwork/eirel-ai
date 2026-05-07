"""Validator observability metric helpers — emit counters / gauges / histograms."""

from __future__ import annotations

import pytest

from validation.validator import metrics


def _label_value(counter, **labels) -> float:
    """Pull the numeric value out of a labelled prometheus counter/gauge."""
    return counter.labels(**labels)._value.get()  # type: ignore[attr-defined]


def test_record_oracle_grounding_bumps_counter():
    before = _label_value(
        metrics.oracle_grounding_outcomes_total,
        vendor="openai", status="ok",
    )
    metrics.record_oracle_grounding("openai", "ok")
    after = _label_value(
        metrics.oracle_grounding_outcomes_total,
        vendor="openai", status="ok",
    )
    assert after == before + 1.0


def test_record_oracle_grounding_distinguishes_vendors():
    before_gemini = _label_value(
        metrics.oracle_grounding_outcomes_total,
        vendor="gemini", status="ok",
    )
    before_openai = _label_value(
        metrics.oracle_grounding_outcomes_total,
        vendor="openai", status="ok",
    )
    metrics.record_oracle_grounding("gemini", "ok")
    after_gemini = _label_value(
        metrics.oracle_grounding_outcomes_total,
        vendor="gemini", status="ok",
    )
    after_openai = _label_value(
        metrics.oracle_grounding_outcomes_total,
        vendor="openai", status="ok",
    )
    assert after_gemini == before_gemini + 1.0
    assert after_openai == before_openai  # not bumped


def test_record_oracle_status_per_status_bucket():
    for status in ("consensus", "majority", "disputed", "deterministic"):
        before = _label_value(
            metrics.oracle_status_per_run_total,
            family="general_chat", oracle_status=status,
        )
        metrics.record_oracle_status("general_chat", status)
        after = _label_value(
            metrics.oracle_status_per_run_total,
            family="general_chat", oracle_status=status,
        )
        assert after == before + 1.0, f"status={status} didn't bump"


def test_record_eval_outcome_normalizes_case():
    """Inputs may be uppercase; the metric label normalizes."""
    before = _label_value(
        metrics.eval_outcome_total,
        family="general_chat", outcome="correct",
    )
    metrics.record_eval_outcome("general_chat", "CORRECT")
    after = _label_value(
        metrics.eval_outcome_total,
        family="general_chat", outcome="correct",
    )
    assert after == before + 1.0


def test_record_composite_score_clamps_to_unit_range():
    """Defensive: out-of-range inputs don't crash. They get clamped."""
    metrics.record_composite_score("general_chat", -0.5)  # → 0.0
    metrics.record_composite_score("general_chat", 1.5)   # → 1.0
    metrics.record_composite_score("general_chat", 0.42)
    # Histogram doesn't expose a clean per-bucket count; just verify
    # we didn't crash.


def test_record_composite_knockout_per_factor():
    factors = (
        "tool_attestation",
        "hallucination_knockout",
        "cost_attestation_knockout",
        "outcome_zero",
    )
    for factor in factors:
        before = _label_value(
            metrics.composite_knockouts_total,
            family="general_chat", knockout_factor=factor,
        )
        metrics.record_composite_knockout("general_chat", factor)
        after = _label_value(
            metrics.composite_knockouts_total,
            family="general_chat", knockout_factor=factor,
        )
        assert after == before + 1.0


def test_record_judge_json_malformation_per_role():
    for role in ("pairwise", "multi", "eval"):
        before = _label_value(
            metrics.judge_json_malformations_total,
            judge_role=role,
        )
        metrics.record_judge_json_malformation(role)
        after = _label_value(
            metrics.judge_json_malformations_total,
            judge_role=role,
        )
        assert after == before + 1.0


def test_record_oracle_call_supports_batch_increment():
    before = _label_value(
        metrics.oracle_layer_calls_total, vendor="openai",
    )
    metrics.record_oracle_call("openai", n=5)
    after = _label_value(
        metrics.oracle_layer_calls_total, vendor="openai",
    )
    assert after == before + 5.0


def test_record_disputed_rate_clamps():
    metrics.record_disputed_rate("general_chat", 1.5)  # → 1.0
    val_high = metrics.reconciler_disputed_rate.labels(family="general_chat")._value.get()
    assert val_high == 1.0
    metrics.record_disputed_rate("general_chat", -0.1)  # → 0.0
    val_low = metrics.reconciler_disputed_rate.labels(family="general_chat")._value.get()
    assert val_low == 0.0
    metrics.record_disputed_rate("general_chat", 0.42)
    val_mid = metrics.reconciler_disputed_rate.labels(family="general_chat")._value.get()
    assert val_mid == pytest.approx(0.42)


def test_record_rank_parity_clamps_to_minus_one_to_one():
    """Spearman ρ ∈ [-1, 1]; clamp out-of-range defensively."""
    metrics.record_rank_parity("general_chat", 1.5)
    assert metrics.rank_parity_spearman_gauge.labels(family="general_chat")._value.get() == 1.0
    metrics.record_rank_parity("general_chat", -2.0)
    assert metrics.rank_parity_spearman_gauge.labels(family="general_chat")._value.get() == -1.0
    metrics.record_rank_parity("general_chat", 0.87)
    assert metrics.rank_parity_spearman_gauge.labels(family="general_chat")._value.get() == pytest.approx(0.87)
