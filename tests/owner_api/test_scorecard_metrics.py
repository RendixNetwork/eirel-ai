from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from shared.common.database import Database
from shared.common.models import (
    DeploymentScoreRecord,
    EvaluationRun,
    ManagedDeployment,
    ManagedMinerSubmission,
    MinerEvaluationTask,
    RunFamilyResult,
    SubmissionArtifact,
)
from control_plane.owner_api.routers.health import (
    _collect_scorecard_rows,
    _format_scorecard_metrics,
    _SCORECARD_RUNS_PER_FAMILY,
)


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'scorecard.db'}")
    db.create_all()
    return db


def _seed_run(session, *, sequence: int, run_id: str | None = None) -> EvaluationRun:
    rid = run_id or f"run-{sequence}"
    now = datetime.now(UTC).replace(tzinfo=None)
    run = EvaluationRun(
        id=rid,
        sequence=sequence,
        status="open" if sequence == 1 else "closed",
        benchmark_version="test",
        rubric_version="test",
        judge_model="test-judge",
        min_scores_json={},
        started_at=now - timedelta(hours=sequence),
        ends_at=now + timedelta(days=3),
        metadata_json={},
    )
    session.add(run)
    session.flush()
    return run


def _seed_score(
    session,
    *,
    run_id: str,
    hotkey: str,
    family_id: str = "general_chat",
    raw_score: float = 0.5,
    normalized_score: float = 0.5,
    llm_cost_usd: float = 0.01,
    tool_cost_usd: float = 0.0,
) -> None:
    artifact = SubmissionArtifact(
        archive_bytes=b"x",
        sha256="sha-" + uuid4().hex[:8],
        size_bytes=1,
        manifest_json={},
    )
    session.add(artifact)
    session.flush()
    submission = ManagedMinerSubmission(
        miner_hotkey=hotkey,
        submission_seq=1,
        family_id=family_id,
        status="received",
        artifact_id=artifact.id,
        manifest_json={},
        archive_sha256=artifact.sha256,
        submission_block=100,
        introduced_run_id=run_id,
    )
    session.add(submission)
    session.flush()
    deployment = ManagedDeployment(
        submission_id=submission.id,
        miner_hotkey=hotkey,
        family_id=family_id,
        deployment_revision="rev-" + uuid4().hex[:8],
        image_ref="managed://test",
        endpoint="http://test",
        status="active",
        health_status="healthy",
    )
    session.add(deployment)
    session.flush()
    record = DeploymentScoreRecord(
        run_id=run_id,
        family_id=family_id,
        deployment_id=deployment.id,
        submission_id=submission.id,
        miner_hotkey=hotkey,
        deployment_revision=deployment.deployment_revision,
        raw_score=raw_score,
        normalized_score=normalized_score,
        llm_cost_usd=llm_cost_usd,
        tool_cost_usd=tool_cost_usd,
    )
    session.add(record)
    session.flush()


def _seed_tasks(
    session,
    *,
    run_id: str,
    hotkey: str,
    evaluated: int,
    total: int,
    family_id: str = "general_chat",
) -> None:
    for idx in range(total):
        status = "evaluated" if idx < evaluated else "pending"
        session.add(
            MinerEvaluationTask(
                run_id=run_id,
                family_id=family_id,
                miner_hotkey=hotkey,
                task_id=f"task-{idx}",
                task_index=idx,
                status=status,
            )
        )
    session.flush()


def test_scorecard_emits_per_miner_rows(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_run(session, sequence=1)
        _seed_score(session, run_id="run-1", hotkey="5DqP", raw_score=0.37, normalized_score=0.42)
        _seed_score(session, run_id="run-1", hotkey="5EnoN", raw_score=0.33, normalized_score=0.36)
        _seed_tasks(session, run_id="run-1", hotkey="5DqP", evaluated=5, total=5)
        _seed_tasks(session, run_id="run-1", hotkey="5EnoN", evaluated=4, total=5)
        session.add(
            RunFamilyResult(
                run_id="run-1",
                family_id="general_chat",
                winner_hotkey="5DqP",
                best_raw_score=0.37,
                has_winner=True,
            )
        )
        session.commit()

        rows = _collect_scorecard_rows(session)

    assert len(rows) == 2
    top = next(r for r in rows if r["hotkey"] == "5DqP")
    second = next(r for r in rows if r["hotkey"] == "5EnoN")
    assert top["rank"] == 1
    assert top["is_winner"] == 1
    assert top["tasks_completed"] == 5
    assert top["tasks_total"] == 5
    assert second["rank"] == 2
    assert second["is_winner"] == 0
    assert second["tasks_completed"] == 4

    body = _format_scorecard_metrics(rows)
    assert 'eirel_owner_scorecard_is_winner{family="general_chat",run_id="run-1",hotkey="5DqP"} 1' in body
    assert 'eirel_owner_scorecard_rank{family="general_chat",run_id="run-1",hotkey="5EnoN"} 2' in body


def test_scorecard_caps_to_latest_runs_per_family(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        for seq in (1, 2, 3):
            _seed_run(session, sequence=seq)
            _seed_score(session, run_id=f"run-{seq}", hotkey="5DqP", raw_score=0.3 + 0.01 * seq)
        session.commit()

        rows = _collect_scorecard_rows(session)

    emitted_runs = {r["run_id"] for r in rows}
    assert emitted_runs == {"run-2", "run-3"}
    assert len(rows) == _SCORECARD_RUNS_PER_FAMILY


def test_scorecard_empty_when_no_scores(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_run(session, sequence=1)
        session.commit()

        rows = _collect_scorecard_rows(session)

    assert rows == []
    assert _format_scorecard_metrics(rows) == ""


def test_scorecard_per_family_isolation(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_run(session, sequence=1)
        _seed_run(session, sequence=2)
        _seed_run(session, sequence=3)
        _seed_score(session, run_id="run-1", hotkey="5DqP", family_id="general_chat")
        _seed_score(session, run_id="run-2", hotkey="5DqP", family_id="general_chat")
        _seed_score(session, run_id="run-3", hotkey="5DqP", family_id="general_chat")
        _seed_score(session, run_id="run-2", hotkey="5FooF", family_id="analyst")
        _seed_score(session, run_id="run-3", hotkey="5FooF", family_id="analyst")
        session.commit()

        rows = _collect_scorecard_rows(session)

    general_runs = {r["run_id"] for r in rows if r["family_id"] == "general_chat"}
    analyst_runs = {r["run_id"] for r in rows if r["family_id"] == "analyst"}
    assert general_runs == {"run-2", "run-3"}
    assert analyst_runs == {"run-2", "run-3"}
