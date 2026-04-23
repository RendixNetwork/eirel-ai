from __future__ import annotations

"""Live-run metrics tests.

Post-pairwise-redesign the live_run endpoint reads ``TaskMinerResult``
rows (one per landed judgment) instead of ``MinerEvaluationTask``. A row
with ``agreement_verdict="error"`` is rendered as status="failed";
anything else renders as "evaluated".
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from shared.common.database import Database
from shared.common.models import (
    EvaluationRun,
    TaskEvaluation,
    TaskMinerResult,
)
from control_plane.owner_api.routers.health import (
    _collect_live_run_rows,
    _format_live_run_metrics,
)


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'liverun.db'}")
    db.create_all()
    return db


def _seed_run(session, *, sequence: int, status: str, run_id: str | None = None) -> EvaluationRun:
    rid = run_id or f"run-{sequence}"
    now = datetime.now(UTC).replace(tzinfo=None)
    run = EvaluationRun(
        id=rid,
        sequence=sequence,
        status=status,
        benchmark_version="test",
        rubric_version="pairwise_general_chat_v1",
        judge_model="test-judge",
        min_scores_json={},
        started_at=now - timedelta(minutes=5 * sequence),
        ends_at=now + timedelta(days=3),
        metadata_json={},
    )
    session.add(run)
    session.flush()
    return run


def _seed_task_eval(session, *, run_id: str, task_id: str, task_index: int = 0) -> TaskEvaluation:
    te = TaskEvaluation(
        run_id=run_id,
        family_id="general_chat",
        task_id=task_id,
        task_index=task_index,
        status="evaluated",
    )
    session.add(te)
    session.flush()
    return te


def _seed_result(
    session,
    *,
    task_evaluation: TaskEvaluation,
    hotkey: str,
    agreement_verdict: str = "matches",
    agreement_score: float = 0.7,
    latency_seconds: float = 0.5,
    miner_response_json: dict | None = None,
) -> TaskMinerResult:
    r = TaskMinerResult(
        task_evaluation_id=task_evaluation.id,
        run_id=task_evaluation.run_id,
        family_id=task_evaluation.family_id,
        task_id=task_evaluation.task_id,
        miner_hotkey=hotkey,
        miner_response_json=miner_response_json or {},
        miner_citations_json=[],
        judge_output_json={},
        agreement_verdict=agreement_verdict,
        agreement_score=agreement_score,
        latency_seconds=latency_seconds,
    )
    session.add(r)
    session.flush()
    return r


def test_live_run_picks_open_run_only(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as s:
        old_run = _seed_run(s, sequence=1, status="completed", run_id="run-old")
        open_run = _seed_run(s, sequence=2, status="open", run_id="run-now")
        te_old = _seed_task_eval(s, run_id=old_run.id, task_id="t-old")
        te_now = _seed_task_eval(s, run_id=open_run.id, task_id="t-1")
        _seed_result(s, task_evaluation=te_old, hotkey="5Alice")
        _seed_result(s, task_evaluation=te_now, hotkey="5Alice")
        s.commit()

        data = _collect_live_run_rows(s)

    assert data["run_id"] == "run-now"
    task_ids = {row["task_id"] for row in data["tasks"]}
    assert task_ids == {"t-1"}


def test_live_run_empty_when_no_open_run(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as s:
        _seed_run(s, sequence=1, status="completed")
        s.commit()
        data = _collect_live_run_rows(s)
    assert data == {"run_id": None, "tasks": [], "progress": {}}
    assert _format_live_run_metrics(data) == ""


def test_live_run_extracts_tool_calls_and_score(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as s:
        run = _seed_run(s, sequence=1, status="open")
        te_web = _seed_task_eval(s, run_id=run.id, task_id="t-web")
        te_plain = _seed_task_eval(s, run_id=run.id, task_id="t-plain", task_index=1)
        te_err = _seed_task_eval(s, run_id=run.id, task_id="t-err", task_index=2)

        response_with_web = {
            "response": {
                "latency_ms": 924,
                "output": {
                    "tool_calls": [{"tool_name": "web_search", "latency_ms": 900}],
                    "metadata": {"mode": "instant", "web_search_enabled": True},
                },
            }
        }
        _seed_result(
            s, task_evaluation=te_web, hotkey="5Alice",
            agreement_verdict="matches", agreement_score=0.825,
            latency_seconds=0.924,
            miner_response_json=response_with_web,
        )
        _seed_result(
            s, task_evaluation=te_plain, hotkey="5Alice",
            agreement_verdict="contradicts", agreement_score=0.49,
            latency_seconds=0.3,
            miner_response_json={"response": {"latency_ms": 300, "output": {"tool_calls": [], "metadata": {}}}},
        )
        _seed_result(
            s, task_evaluation=te_err, hotkey="5Alice",
            agreement_verdict="error", agreement_score=0.0, latency_seconds=0.0,
        )
        s.commit()
        data = _collect_live_run_rows(s)

    by_task = {row["task_id"]: row for row in data["tasks"]}
    assert by_task["t-web"]["score"] == 0.825
    assert by_task["t-web"]["web_search_used"] == 1
    assert by_task["t-web"]["status"] == "evaluated"
    assert by_task["t-plain"]["web_search_used"] == 0
    # Error verdict surfaces as "failed" status
    assert by_task["t-err"]["status"] == "failed"
    assert by_task["t-err"]["is_evaluated"] == 0


def test_live_run_skips_retired_runs(tmp_path):
    """A run that has ended but isn't the most recent open shouldn't show up."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as s:
        _seed_run(s, sequence=5, status="completed", run_id="run-5")
        _seed_run(s, sequence=6, status="completed", run_id="run-6")
        s.commit()
        data = _collect_live_run_rows(s)
    assert data["run_id"] is None
