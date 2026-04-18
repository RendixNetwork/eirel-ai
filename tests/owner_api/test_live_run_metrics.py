from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from shared.common.database import Database
from shared.common.models import (
    EvaluationRun,
    ManagedDeployment,
    ManagedMinerSubmission,
    MinerEvaluationTask,
    SubmissionArtifact,
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
        rubric_version="test",
        judge_model="test-judge",
        min_scores_json={},
        started_at=now - timedelta(minutes=5 * sequence),
        ends_at=now + timedelta(days=3),
        metadata_json={},
    )
    session.add(run)
    session.flush()
    return run


def _seed_task(
    session,
    *,
    run_id: str,
    hotkey: str,
    task_id: str,
    task_index: int,
    status: str,
    task_score: float | None = None,
    miner_response_json: dict | None = None,
    family_id: str = "general_chat",
) -> MinerEvaluationTask:
    t = MinerEvaluationTask(
        run_id=run_id,
        family_id=family_id,
        miner_hotkey=hotkey,
        task_id=task_id,
        task_index=task_index,
        status=status,
        task_score=task_score or 0.0,
        miner_response_json=miner_response_json or {},
    )
    session.add(t)
    session.flush()
    return t


def test_live_run_picks_open_run_only(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as s:
        _seed_run(s, sequence=1, status="completed", run_id="run-old")
        open_run = _seed_run(s, sequence=2, status="open", run_id="run-now")
        _seed_task(
            s, run_id=open_run.id, hotkey="5Alice",
            task_id="t-1", task_index=0, status="pending",
        )
        _seed_task(
            s, run_id="run-old", hotkey="5Alice",
            task_id="t-old", task_index=0, status="evaluated",
        )
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
        response_with_web = {
            "response": {
                "latency_ms": 924,
                "output": {
                    "tool_calls": [{"tool_name": "web_search", "latency_ms": 900}],
                    "metadata": {"mode": "instant", "web_search_enabled": True},
                },
            }
        }
        _seed_task(
            s, run_id=run.id, hotkey="5Alice", task_id="t-web",
            task_index=0, status="evaluated", task_score=0.825,
            miner_response_json=response_with_web,
        )
        _seed_task(
            s, run_id=run.id, hotkey="5Alice", task_id="t-plain",
            task_index=1, status="evaluated", task_score=0.49,
            miner_response_json={"response": {"latency_ms": 300, "output": {"tool_calls": [], "metadata": {}}}},
        )
        _seed_task(
            s, run_id=run.id, hotkey="5Alice", task_id="t-pending",
            task_index=2, status="pending",
        )
        s.commit()
        data = _collect_live_run_rows(s)

    by_task = {row["task_id"]: row for row in data["tasks"]}
    assert by_task["t-web"]["score"] == 0.825
    assert by_task["t-web"]["web_search_used"] == 1
    assert by_task["t-web"]["latency_ms"] == 924
    assert by_task["t-plain"]["web_search_used"] == 0
    assert by_task["t-pending"]["is_evaluated"] == 0


def test_live_run_formats_all_gauges(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as s:
        run = _seed_run(s, sequence=1, status="open")
        _seed_task(
            s, run_id=run.id, hotkey="5Alice", task_id="t-1",
            task_index=0, status="evaluated", task_score=0.7,
            miner_response_json={
                "response": {"latency_ms": 500, "output": {"tool_calls": [{"tool_name": "web_search"}], "metadata": {}}},
            },
        )
        _seed_task(
            s, run_id=run.id, hotkey="5Bob", task_id="t-1",
            task_index=0, status="claimed",
        )
        s.commit()
        data = _collect_live_run_rows(s)

    body = _format_live_run_metrics(data)
    # Status rows exist for both statuses
    assert 'eirel_owner_task_status{run_id="run-1",family="general_chat",hotkey="5Alice",task_id="t-1",status="evaluated"} 1' in body
    assert 'eirel_owner_task_status{run_id="run-1",family="general_chat",hotkey="5Bob",task_id="t-1",status="claimed"} 1' in body
    # Score only emitted for evaluated
    assert 'eirel_owner_task_score{run_id="run-1",family="general_chat",hotkey="5Alice",task_id="t-1"}' in body
    assert 'eirel_owner_task_score{run_id="run-1",family="general_chat",hotkey="5Bob"' not in body
    # Latency and web_search_used present for evaluated
    assert 'eirel_owner_task_latency_ms{run_id="run-1",family="general_chat",hotkey="5Alice",task_id="t-1"} 500' in body
    assert 'eirel_owner_task_web_search_used{run_id="run-1",family="general_chat",hotkey="5Alice",task_id="t-1"} 1' in body
    # Progress gauge counts each (family,hotkey,status) bucket
    assert 'eirel_owner_task_progress{run_id="run-1",family="general_chat",hotkey="5Alice",state="evaluated"} 1' in body
    assert 'eirel_owner_task_progress{run_id="run-1",family="general_chat",hotkey="5Bob",state="claimed"} 1' in body


def test_live_run_skips_retired_runs(tmp_path):
    """A run that has ended but isn't the most recent open shouldn't show up."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as s:
        _seed_run(s, sequence=5, status="completed", run_id="run-5")
        _seed_run(s, sequence=6, status="completed", run_id="run-6")
        s.commit()
        data = _collect_live_run_rows(s)
    assert data["run_id"] is None
