"""Bundle-schema sync: ``oracle_source`` tag on ``FamilyEvaluationTask``.

The validator's claim-phase enrichment loop branches on
``task.oracle_source`` to decide whether to run the 3-oracle fanout.
This test surface locks the contract: bundles published with the new
enum (``three_oracle`` | ``deterministic``) load cleanly, AND legacy
bundles published before the field existed (or with old gold-
provenance strings like ``live_endpoint``) keep loading without
breaking — they default to ``deterministic`` so the validator skips
enrichment as it does today.
"""

from __future__ import annotations

import pytest

from shared.core.evaluation_models import (
    FamilyEvaluationBundle,
    FamilyEvaluationTask,
)


def _bundle_dict(*tasks: dict) -> dict:
    return {
        "kind": "family_evaluation_bundle",
        "run_id": "test-run",
        "family_id": "general_chat",
        "benchmark_version": "general_chat_eval_pool_v2",
        "rubric_version": "eval_judge_v2",
        "tasks": list(tasks),
    }


def _task_dict(**overrides) -> dict:
    base = {
        "task_id": "t-1",
        "family_id": "general_chat",
        "prompt": "What is 2+2?",
        "expected_output": {"answer": "4"},
    }
    base.update(overrides)
    return base


# -- New enum values ------------------------------------------------------


def test_three_oracle_tag_loads():
    """A bundle published with oracle_source=three_oracle surfaces the
    tag on the loaded task — the validator's claim phase can branch
    on this to run 3-oracle enrichment."""
    bundle = FamilyEvaluationBundle.model_validate(
        _bundle_dict(_task_dict(oracle_source="three_oracle")),
    )
    assert bundle.tasks[0].oracle_source == "three_oracle"


def test_deterministic_tag_loads():
    bundle = FamilyEvaluationBundle.model_validate(
        _bundle_dict(_task_dict(oracle_source="deterministic")),
    )
    assert bundle.tasks[0].oracle_source == "deterministic"


def test_field_missing_defaults_to_none():
    """Back-compat: legacy bundles published before this field existed
    parse cleanly with oracle_source=None. The validator treats None
    as 'deterministic' — skip enrichment, use task.expected_output."""
    bundle = FamilyEvaluationBundle.model_validate(
        _bundle_dict(_task_dict()),
    )
    assert bundle.tasks[0].oracle_source is None


# -- Legacy gold-provenance values normalize to "deterministic" -----------


@pytest.mark.parametrize(
    "legacy_value",
    [
        # eirel-eval-pool render_bundle.py emits these today (one per
        # rendered kind). All represent deterministic gold.
        "live_endpoint",
        "live_endpoint_composed",
        "deterministic_grader",
        # Pre-three-oracle eiretes/eirel-ai conventions.
        "gpt5_oracle",
        "planted_fact",
        "document_span",
        "sandbox_reference",
        "url_fetch_cache",
    ],
)
def test_legacy_oracle_source_normalized_to_deterministic(legacy_value: str):
    """Legacy strings that described the pool's gold provenance map to
    the new enum's ``deterministic`` value so existing pool deployments
    keep loading without a lockstep cutover."""
    task = FamilyEvaluationTask.model_validate(
        _task_dict(oracle_source=legacy_value),
    )
    assert task.oracle_source == "deterministic"


def test_unknown_value_normalizes_to_none_without_crashing():
    """Unrecognized strings are dropped to None rather than crashing
    the load. The validator treats None as 'deterministic'."""
    task = FamilyEvaluationTask.model_validate(
        _task_dict(oracle_source="some_future_value_we_dont_know_yet"),
    )
    assert task.oracle_source is None


def test_empty_string_normalizes_to_none():
    task = FamilyEvaluationTask.model_validate(
        _task_dict(oracle_source=""),
    )
    assert task.oracle_source is None


def test_whitespace_only_normalizes_to_none():
    task = FamilyEvaluationTask.model_validate(
        _task_dict(oracle_source="   "),
    )
    assert task.oracle_source is None


def test_non_string_value_normalizes_to_none():
    """Defensive: non-string input (e.g. a dict from a malformed
    bundle) drops to None instead of crashing."""
    task = FamilyEvaluationTask.model_validate(
        _task_dict(oracle_source=123),
    )
    assert task.oracle_source is None


# -- Mixed-tag bundle -----------------------------------------------------


def test_bundle_with_mixed_oracle_sources():
    """A real bundle ships items with different tags — live_lookup may
    be three_oracle, attached_long_doc is deterministic, etc."""
    bundle = FamilyEvaluationBundle.model_validate(
        _bundle_dict(
            _task_dict(task_id="t-factual", oracle_source="three_oracle"),
            _task_dict(task_id="t-doc", oracle_source="deterministic"),
            _task_dict(task_id="t-legacy", oracle_source="live_endpoint"),
            _task_dict(task_id="t-unset"),  # no field set
        ),
    )
    by_id = {t.task_id: t for t in bundle.tasks}
    assert by_id["t-factual"].oracle_source == "three_oracle"
    assert by_id["t-doc"].oracle_source == "deterministic"
    assert by_id["t-legacy"].oracle_source == "deterministic"
    assert by_id["t-unset"].oracle_source is None


# -- Other expected_output fields preserved -------------------------------


def test_expected_output_unchanged_alongside_oracle_source():
    """Adding oracle_source to the task model must not affect
    expected_output parsing — it carries the deterministic answer +
    must_not_claim floor for items that don't need three_oracle
    enrichment."""
    task = FamilyEvaluationTask.model_validate(
        _task_dict(
            oracle_source="deterministic",
            expected_output={
                "answer": "Paris",
                "must_not_claim": ["London", "Berlin"],
            },
        ),
    )
    assert task.oracle_source == "deterministic"
    assert task.expected_output["answer"] == "Paris"
    assert task.expected_output["must_not_claim"] == ["London", "Berlin"]
