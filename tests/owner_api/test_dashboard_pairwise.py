"""Dashboard metrics shape tests after the agreement-only schema cleanup."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import count
from types import SimpleNamespace

from shared.common.database import Database
from shared.common.models import (
    EvaluationRun,
    MinerEvaluationSummary,
    TaskEvaluation as TaskEvalRow,
    TaskMinerResult,
)
from control_plane.owner_api.dashboard.queries import (
    _merge_summary_metrics,
    _metrics_for_tasks,
    _task_evaluation_from_row,
)
from control_plane.owner_api.dashboard.schemas import MinerMetrics

_task_counter = count()


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'dash.db'}")
    db.create_all()
    return db


def _seed(
    session, *, hotkey: str, verdict: str,
    agreement_score: float | None = None,
    citations: list | None = None,
) -> None:
    run_id = "run-dash"
    if session.get(EvaluationRun, run_id) is None:
        now = datetime.now(UTC).replace(tzinfo=None)
        session.add(EvaluationRun(
            id=run_id, sequence=1, status="open",
            benchmark_version="test", rubric_version="agreement_general_chat_v1",
            judge_model="t", min_scores_json={},
            started_at=now, ends_at=now + timedelta(days=1),
            metadata_json={},
        ))
        session.flush()
    if agreement_score is None:
        agreement_score = {
            "matches": 1.0, "partially_matches": 0.6,
            "not_applicable": 0.7, "contradicts": 0.0, "error": 0.0,
        }[verdict]
    task_id = f"task-{hotkey}-{verdict}-{next(_task_counter)}"
    te = TaskEvalRow(
        run_id=run_id, family_id="general_chat",
        task_id=task_id, task_index=0, status="evaluated",
    )
    session.add(te)
    session.flush()
    session.add(TaskMinerResult(
        task_evaluation_id=te.id,
        run_id=run_id,
        family_id="general_chat",
        task_id=task_id,
        miner_hotkey=hotkey,
        miner_response_json={},
        miner_citations_json=citations or [],
        judge_output_json={"verdict": verdict, "rationale": "ok"},
        agreement_verdict=verdict,
        agreement_score=agreement_score,
        latency_seconds=1.2,
    ))
    session.flush()


def test_fetch_miner_metrics_computes_mean_agreement(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as s:
        _seed(s, hotkey="hk-a", verdict="matches")
        _seed(s, hotkey="hk-a", verdict="partially_matches")
        _seed(s, hotkey="hk-a", verdict="contradicts")
        _seed(s, hotkey="hk-a", verdict="error")
        s.commit()
        metrics = _metrics_for_tasks(
            s, run_id="run-dash", family_id="general_chat", hotkey=None,
        )

    m = metrics["hk-a"]
    # mean over non-error rows = (1.0 + 0.6 + 0.0) / 3 ≈ 0.533
    assert m.mean_agreement == (1.0 + 0.6 + 0.0) / 3
    assert m.matches_count == 1
    assert m.partially_matches_count == 1
    assert m.contradicts_count == 1
    assert m.error_rate == 0.25
    assert m.reliable is True


def test_fetch_miner_metrics_flags_high_error_rate_unreliable(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as s:
        _seed(s, hotkey="hk-broken", verdict="matches")
        _seed(s, hotkey="hk-broken", verdict="error")
        _seed(s, hotkey="hk-broken", verdict="error")
        s.commit()
        metrics = _metrics_for_tasks(
            s, run_id="run-dash", family_id="general_chat", hotkey=None,
        )

    m = metrics["hk-broken"]
    assert m.error_rate > 0.30
    assert m.reliable is False


def test_task_evaluation_from_row_emits_agreement_schema():
    row = SimpleNamespace(
        task_id="t-1",
        miner_hotkey="hk",
        agreement_verdict="matches",
        agreement_score=1.0,
        latency_seconds=1.2,
        created_at=datetime.now(UTC).replace(tzinfo=None),
        miner_response_json={"ok": True},
        miner_citations_json=[
            {"url": "https://bbc.com/a", "title": "BBC A"},
            {"url": "https://nytimes.com/b", "title": "NYT B"},
        ],
        judge_output_json={"rationale": "claims align"},
    )
    bundle_task = {
        "prompt": "What is X?",
        "mode": "instant",
        "category": "factual_web",
        "difficulty": "standard",
        "metadata": {},
    }
    baseline = {
        "citations": [{"url": "https://reuters.com/c", "title": "Reuters"}],
    }
    te = _task_evaluation_from_row(
        row, bundle_task=bundle_task, baseline_response_json=baseline,
    )

    assert te.task_id == "t-1"
    assert te.agreement_verdict == "matches"
    assert te.agreement_score == 1.0
    assert te.task_status == "completed"
    assert [c.url for c in te.miner_citations] == [
        "https://bbc.com/a", "https://nytimes.com/b"
    ]
    assert [c.url for c in te.baseline_citations] == ["https://reuters.com/c"]
    # Legacy fields are gone.
    assert not hasattr(te, "pairwise_verdict")
    assert not hasattr(te, "overall_score")
    assert not hasattr(te, "dimension_scores")


def test_task_evaluation_from_row_renders_error_as_failed():
    row = SimpleNamespace(
        task_id="t-err",
        miner_hotkey="hk",
        agreement_verdict="error",
        agreement_score=0.0,
        latency_seconds=0.0,
        created_at=datetime.now(UTC).replace(tzinfo=None),
        miner_response_json={},
        miner_citations_json=[],
        judge_output_json=None,
    )
    te = _task_evaluation_from_row(row, bundle_task={"prompt": "q"})
    assert te.agreement_verdict == "error"
    assert te.task_status == "failed"
    assert te.agreement_score == 0.0


def test_merge_summary_metrics_overlays_rollup_metadata():
    base = MinerMetrics(mean_agreement=0.5, error_rate=0.1, reliable=True)
    summary = SimpleNamespace(rollout_metadata_json={
        "mean_agreement": 0.75,
        "error_rate": 0.0,
        "reliable": True,
        "verdict_counts": {
            "matches": 8,
            "partially_matches": 2,
            "not_applicable": 0,
            "contradicts": 0,
            "error": 0,
        },
    })
    merged = _merge_summary_metrics(base, summary)
    assert merged.mean_agreement == 0.75
    assert merged.matches_count == 8
    assert merged.partially_matches_count == 2


def test_merge_summary_metrics_noop_when_no_summary():
    base = MinerMetrics(mean_agreement=0.7)
    assert _merge_summary_metrics(base, None) is base
