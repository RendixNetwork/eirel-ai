"""End-to-end integration test for the outcome-only agreement pipeline.

Exercises the full server-side flow through ``ManagedOwnerServices``:

    initialize_evaluation_tasks  →  claim_tasks
        →  submit_task_result (baseline + per-miner agreement results)
            →  aggregate MinerEvaluationSummary
                →  publish AggregateFamilyScoreSnapshot

The OpenAI baseline and miner responses are fabricated here — the test
validates the claim lifecycle, per-miner result persistence,
mean-agreement aggregation, citation preservation for dashboard, and the
final snapshot + DeploymentScoreRecord wiring.
"""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from shared.common.config import Settings, reset_settings
from shared.common.database import Database
from shared.common.models import (
    AggregateFamilyScoreSnapshot,
    DeploymentScoreRecord,
    EpochTargetSnapshot,
    EvaluationRun,
    ManagedDeployment,
    ManagedMinerSubmission,
    MinerEvaluationSummary,
    RegisteredNeuron,
    SubmissionArtifact,
    TaskEvaluation,
    TaskMinerResult,
)
from shared.common.artifacts import create_artifact_store
from control_plane.owner_api.managed import ManagedOwnerServices
from control_plane.owner_api.deployment.runtime_manager import ManagedDeploymentRuntimeManager
from tests.conftest import FIXTURES_ROOT
from tests.owner_api._fake_backend import _FakeBackend


def _utcnow():
    from control_plane.owner_api._helpers import utcnow
    return utcnow()


_BUNDLE = {
    "kind": "family_evaluation_bundle",
    "family_id": "general_chat",
    "benchmark_version": "test_v1",
    "rubric_version": "agreement_general_chat_v1",
    "tasks": [
        {
            "task_id": "e2e-task-1",
            "family_id": "general_chat",
            "prompt": "What is the capital of France?",
            "mode": "instant",
            "category": "factual_web",
            "expected_output": {},
        },
        {
            "task_id": "e2e-task-2",
            "family_id": "general_chat",
            "prompt": "Explain the difference between TCP and UDP.",
            "mode": "thinking",
            "category": "no_tool",
            "expected_output": {},
        },
    ],
    "metadata": {"dataset_generator": {"version": "test"}},
}


@pytest.fixture
def services(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("METAGRAPH_SYNC_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("RESULT_AGGREGATION_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("USE_REDIS_POOL", "0")
    monkeypatch.setenv("METAGRAPH_SNAPSHOT_PATH", str(tmp_path / "metagraph.json"))
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_URL", "http://provider-proxy.test")
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_TOKEN", "provider-token")
    monkeypatch.setenv("EIREL_INTERNAL_SERVICE_TOKEN", "internal-token")
    monkeypatch.setenv("EIREL_ACTIVE_FAMILIES", "general_chat")
    monkeypatch.setenv("EIREL_LAUNCH_MODE", "1")
    # Dataset root is set up by the autouse conftest fixture.
    reset_settings()
    settings = Settings()
    db = Database(settings.database_url)
    db.create_all()
    backend = _FakeBackend()
    runtime_mgr = ManagedDeploymentRuntimeManager(backend=backend)
    artifact_store = create_artifact_store(settings)
    svc = ManagedOwnerServices(
        db=db,
        settings=settings,
        runtime_manager=runtime_mgr,
        artifact_store=artifact_store,
    )
    yield svc
    reset_settings()


def _seed_run(session, miner_hotkeys: list[str]) -> tuple[str, EpochTargetSnapshot]:
    now = _utcnow()
    run = EvaluationRun(
        id="run-e2e-1",
        sequence=1,
        status="open",
        benchmark_version="test_v1",
        rubric_version="agreement_general_chat_v1",
        judge_model="test-judge",
        min_scores_json={},
        started_at=now,
        ends_at=now + timedelta(days=7),
        metadata_json={"evaluation_bundles": {"general_chat": _BUNDLE}},
    )
    session.add(run)
    session.flush()

    session.add(EvaluationRun(
        id="run-e2e-2",
        sequence=2,
        status="scheduled",
        benchmark_version="test_v1",
        rubric_version="agreement_general_chat_v1",
        judge_model="test-judge",
        min_scores_json={},
        started_at=now + timedelta(days=7),
        ends_at=now + timedelta(days=14),
        metadata_json={"evaluation_bundles": {"general_chat": _BUNDLE}},
    ))
    session.flush()

    members = []
    for hotkey in miner_hotkeys:
        if session.get(RegisteredNeuron, hotkey) is None:
            session.add(RegisteredNeuron(hotkey=hotkey, uid=0))
        _seed_deployment(session, run_id=run.id, hotkey=hotkey)
        members.append({
            "hotkey": hotkey,
            "endpoint": f"http://miner-{hotkey[:8]}.test",
            "metadata": {"auth_headers": {}},
        })
    session.flush()

    snapshot = EpochTargetSnapshot(
        run_id=run.id,
        family_id="general_chat",
        rubric_version="agreement_general_chat_v1",
        benchmark_version="test_v1",
        judge_model="test-judge",
        members_json=members,
        status="open",
    )
    session.add(snapshot)
    session.flush()

    return run.id, snapshot


def _seed_deployment(session, *, run_id: str, hotkey: str) -> None:
    artifact = SubmissionArtifact(
        archive_bytes=b"fake",
        sha256="fakehash" + uuid4().hex[:8],
        size_bytes=4,
        manifest_json={},
    )
    session.add(artifact)
    session.flush()

    submission = ManagedMinerSubmission(
        miner_hotkey=hotkey,
        submission_seq=1,
        family_id="general_chat",
        status="deployed_for_eval",
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
        family_id="general_chat",
        deployment_revision="rev-" + uuid4().hex[:8],
        image_ref="managed://test",
        endpoint=f"http://miner-{hotkey[:8]}.test",
        status="deployed_for_eval",
        health_status="healthy",
        health_details_json={},
        placement_status="placed",
        benchmark_version="test_v1",
        rubric_version="agreement_general_chat_v1",
        judge_model="test-judge",
    )
    session.add(deployment)
    session.flush()


def _fabricated_baseline() -> dict:
    return {
        "response_text": "The capital of France is Paris.",
        "citations": [{"url": "https://example.com/paris", "title": "Paris"}],
        "raw_output": [],
        "latency_seconds": 1.2,
        "cost_usd": 0.03,
        "model": "gpt-5",
        "metadata": {},
    }


def _miner_response(text: str) -> dict:
    return {
        "task_id": "whatever",
        "family_id": "general_chat",
        "prompt": "",
        "expected_output": {},
        "response": {"output": {"content": text}},
        "status": "completed",
        "error": None,
        "metadata": {},
    }


def _judge_output(verdict: str, *, agreement_score: float) -> dict:
    return {
        "verdict": verdict,
        "agreement_score": agreement_score,
        "rationale": "test",
        "swap_applied": False,
        "model": "test-judge",
        "rubric_name": "agreement_general_chat_v1:test",
        "metadata": {},
    }


def test_e2e_claim_submit_aggregate(services):
    """Seed → claim → submit × N → aggregate, for 2 miners × 2 tasks.

    Scenario:
      • hk-good matches the baseline on both tasks (mean_agreement = 1.0).
      • hk-bad contradicts the baseline on both tasks (mean_agreement = 0.0).
      • After 2nd submit, family completes → aggregation fires.
    """
    validator_hk = "validator-alpha"

    with services.db.sessionmaker() as session:
        run_id, _ = _seed_run(session, miner_hotkeys=["hk-good", "hk-bad"])
        snapshot = session.execute(
            EpochTargetSnapshot.__table__.select().where(
                EpochTargetSnapshot.run_id == run_id
            )
        ).first()
        snapshot_obj = session.get(EpochTargetSnapshot, snapshot.id)
        created = services.evaluation_tasks.initialize_evaluation_tasks(
            session, run_id=run_id, family_id="general_chat", snapshot=snapshot_obj,
        )
        session.commit()
    assert created == 2

    with services.db.sessionmaker() as session:
        claimed = services.evaluation_tasks.claim_tasks(
            session,
            run_id=run_id,
            family_id="general_chat",
            validator_hotkey=validator_hk,
            batch_size=10,
        )
        assert len(claimed) == 2
        items = services.evaluation_tasks.build_claim_items(
            session,
            claimed_tasks=claimed,
            run_id=run_id,
            family_id="general_chat",
        )
        session.commit()

    for item in items:
        miner_results = [
            {
                "miner_hotkey": "hk-good",
                "miner_response": _miner_response(f"good answer for {item['task_id']}"),
                "miner_citations": [{"url": "https://bbc.com/article-good", "title": "Good"}],
                "judge_output": _judge_output("matches", agreement_score=1.0),
                "agreement_score": 1.0,
                "verdict": "matches",
                "latency_seconds": 2.5,
            },
            {
                "miner_hotkey": "hk-bad",
                "miner_response": _miner_response(f"bad answer for {item['task_id']}"),
                "miner_citations": [],
                "judge_output": _judge_output("contradicts", agreement_score=0.0),
                "agreement_score": 0.0,
                "verdict": "contradicts",
                "latency_seconds": 1.9,
            },
        ]
        with services.db.sessionmaker() as session:
            resp = services.evaluation_tasks.submit_task_result(
                session,
                task_evaluation_id=item["task_evaluation_id"],
                validator_hotkey=validator_hk,
                baseline_response=_fabricated_baseline(),
                miner_results=miner_results,
            )
            session.commit()
        assert resp["status"] == "accepted"

    with services.db.sessionmaker() as session:
        task_evals = session.query(TaskEvaluation).filter_by(run_id=run_id).all()
        assert len(task_evals) == 2
        assert {t.status for t in task_evals} == {"evaluated"}
        assert all(t.baseline_response_json is not None for t in task_evals)

        miner_results = session.query(TaskMinerResult).filter_by(run_id=run_id).all()
        assert len(miner_results) == 4
        good_rows = [r for r in miner_results if r.miner_hotkey == "hk-good"]
        bad_rows = [r for r in miner_results if r.miner_hotkey == "hk-bad"]
        assert len(good_rows) == 2 and len(bad_rows) == 2
        assert all(r.agreement_verdict == "matches" for r in good_rows)
        assert all(r.agreement_verdict == "contradicts" for r in bad_rows)

        # Citations preserved on the row for dashboard display.
        assert good_rows[0].miner_citations_json == [
            {"url": "https://bbc.com/article-good", "title": "Good"}
        ]
        assert bad_rows[0].miner_citations_json == []

        good_summary = session.query(MinerEvaluationSummary).filter_by(
            run_id=run_id, miner_hotkey="hk-good",
        ).one()
        bad_summary = session.query(MinerEvaluationSummary).filter_by(
            run_id=run_id, miner_hotkey="hk-bad",
        ).one()
        assert good_summary.status == "scored"
        assert bad_summary.status == "scored"
        assert good_summary.family_capability_score == pytest.approx(1.0)
        assert bad_summary.family_capability_score == pytest.approx(0.0)
        assert good_summary.official_family_score == pytest.approx(1.0)
        assert bad_summary.official_family_score == pytest.approx(0.0)
        assert good_summary.protocol_gate_passed is True
        assert bad_summary.protocol_gate_passed is True

        aggregate = session.query(AggregateFamilyScoreSnapshot).filter_by(
            run_id=run_id, family_id="general_chat",
        ).one()
        assert aggregate.status == "aggregated"
        assert aggregate.consensus_method == "agreement_against_openai_baseline"
        assert validator_hk in aggregate.validator_hotkeys_json

        score_records = session.query(DeploymentScoreRecord).filter_by(
            run_id=run_id, family_id="general_chat",
        ).all()
        assert len(score_records) == 2
        good_record = next(r for r in score_records if r.miner_hotkey == "hk-good")
        bad_record = next(r for r in score_records if r.miner_hotkey == "hk-bad")
        assert good_record.normalized_score > bad_record.normalized_score
        assert good_record.metadata_json["mean_agreement"] == pytest.approx(1.0)
        assert bad_record.metadata_json["mean_agreement"] == pytest.approx(0.0)


def test_e2e_partial_matches_and_not_applicable(services):
    """Mixed verdicts aggregate to the correct mean."""
    validator_hk = "validator-beta"

    with services.db.sessionmaker() as session:
        bundle = {
            **_BUNDLE,
            "tasks": [
                {"task_id": f"mix-{i}", "family_id": "general_chat",
                 "prompt": f"q{i}", "mode": "instant", "expected_output": {}}
                for i in range(1, 5)
            ],
        }
        now = _utcnow()
        session.add(EvaluationRun(
            id="run-mix-1", sequence=1, status="open",
            benchmark_version="test_v1", rubric_version="agreement_general_chat_v1",
            judge_model="test-judge", min_scores_json={},
            started_at=now, ends_at=now + timedelta(days=7),
            metadata_json={"evaluation_bundles": {"general_chat": bundle}},
        ))
        session.flush()
        if session.get(RegisteredNeuron, "hk-mix") is None:
            session.add(RegisteredNeuron(hotkey="hk-mix", uid=0))
            session.flush()
        _seed_deployment(session, run_id="run-mix-1", hotkey="hk-mix")

        snap = EpochTargetSnapshot(
            run_id="run-mix-1", family_id="general_chat",
            rubric_version="agreement_general_chat_v1",
            benchmark_version="test_v1", judge_model="test-judge",
            members_json=[{"hotkey": "hk-mix", "endpoint": "x", "metadata": {}}],
            status="open",
        )
        session.add(snap)
        session.flush()
        snap_obj = session.get(EpochTargetSnapshot, snap.id)
        services.evaluation_tasks.initialize_evaluation_tasks(
            session, run_id="run-mix-1", family_id="general_chat", snapshot=snap_obj,
        )
        session.commit()

    with services.db.sessionmaker() as session:
        claimed = services.evaluation_tasks.claim_tasks(
            session, run_id="run-mix-1", family_id="general_chat",
            validator_hotkey=validator_hk, batch_size=10,
        )
        items = services.evaluation_tasks.build_claim_items(
            session, claimed_tasks=claimed, run_id="run-mix-1",
            family_id="general_chat",
        )
        session.commit()

    verdicts = ["matches", "partially_matches", "not_applicable", "contradicts"]
    for item, verdict in zip(items, verdicts):
        score = {"matches": 1.0, "partially_matches": 0.6,
                 "not_applicable": 0.7, "contradicts": 0.0}[verdict]
        miner_results = [{
            "miner_hotkey": "hk-mix",
            "miner_response": _miner_response("ok"),
            "miner_citations": [],
            "judge_output": _judge_output(verdict, agreement_score=score),
            "agreement_score": score,
            "verdict": verdict,
            "latency_seconds": 1.0,
        }]
        with services.db.sessionmaker() as session:
            services.evaluation_tasks.submit_task_result(
                session,
                task_evaluation_id=item["task_evaluation_id"],
                validator_hotkey=validator_hk,
                baseline_response=_fabricated_baseline(),
                miner_results=miner_results,
            )
            session.commit()

    with services.db.sessionmaker() as session:
        summary = session.query(MinerEvaluationSummary).filter_by(
            run_id="run-mix-1", miner_hotkey="hk-mix",
        ).one()
        # mean = (1.0 + 0.6 + 0.7 + 0.0) / 4 = 0.575
        assert summary.family_capability_score == pytest.approx(0.575)
        assert summary.official_family_score == pytest.approx(0.575)
        assert summary.protocol_gate_passed is True


def test_e2e_error_rate_cap_fires_above_threshold(services):
    """50% error rate → capped at 0.5."""
    validator_hk = "validator-gamma"

    with services.db.sessionmaker() as session:
        bundle = {
            **_BUNDLE,
            "tasks": [
                {"task_id": f"cap-{i}", "family_id": "general_chat",
                 "prompt": f"q{i}", "mode": "instant", "expected_output": {}}
                for i in range(1, 5)
            ],
        }
        now = _utcnow()
        session.add(EvaluationRun(
            id="run-cap-1", sequence=1, status="open",
            benchmark_version="test_v1", rubric_version="agreement_general_chat_v1",
            judge_model="test-judge", min_scores_json={},
            started_at=now, ends_at=now + timedelta(days=7),
            metadata_json={"evaluation_bundles": {"general_chat": bundle}},
        ))
        session.flush()
        if session.get(RegisteredNeuron, "hk-broken") is None:
            session.add(RegisteredNeuron(hotkey="hk-broken", uid=0))
            session.flush()
        _seed_deployment(session, run_id="run-cap-1", hotkey="hk-broken")

        snap = EpochTargetSnapshot(
            run_id="run-cap-1", family_id="general_chat",
            rubric_version="agreement_general_chat_v1",
            benchmark_version="test_v1", judge_model="test-judge",
            members_json=[{"hotkey": "hk-broken", "endpoint": "x", "metadata": {}}],
            status="open",
        )
        session.add(snap)
        session.flush()
        snap_obj = session.get(EpochTargetSnapshot, snap.id)
        services.evaluation_tasks.initialize_evaluation_tasks(
            session, run_id="run-cap-1", family_id="general_chat", snapshot=snap_obj,
        )
        session.commit()

    with services.db.sessionmaker() as session:
        claimed = services.evaluation_tasks.claim_tasks(
            session, run_id="run-cap-1", family_id="general_chat",
            validator_hotkey=validator_hk, batch_size=10,
        )
        items = services.evaluation_tasks.build_claim_items(
            session, claimed_tasks=claimed, run_id="run-cap-1",
            family_id="general_chat",
        )
        session.commit()

    # 2 matches + 2 errors = 50% error rate > 30% threshold
    verdicts = ["matches", "matches", "error", "error"]
    for item, verdict in zip(items, verdicts):
        if verdict == "error":
            miner_results = [{
                "miner_hotkey": "hk-broken",
                "miner_response": {"status": "failed"},
                "miner_citations": [],
                "judge_output": None,
                "agreement_score": 0.0,
                "verdict": "error",
                "latency_seconds": 0.0,
            }]
        else:
            miner_results = [{
                "miner_hotkey": "hk-broken",
                "miner_response": _miner_response("ok"),
                "miner_citations": [],
                "judge_output": _judge_output("matches", agreement_score=1.0),
                "agreement_score": 1.0,
                "verdict": "matches",
                "latency_seconds": 2.0,
            }]
        with services.db.sessionmaker() as session:
            services.evaluation_tasks.submit_task_result(
                session,
                task_evaluation_id=item["task_evaluation_id"],
                validator_hotkey=validator_hk,
                baseline_response=_fabricated_baseline(),
                miner_results=miner_results,
            )
            session.commit()

    with services.db.sessionmaker() as session:
        summary = session.query(MinerEvaluationSummary).filter_by(
            run_id="run-cap-1", miner_hotkey="hk-broken",
        ).one()
        # Legacy mean_agreement (non-error only): 2 matches → 1.0; the
        # protocol_gate flag still flips false (error_rate > 30%) so the
        # UI can render the UNRELIABLE badge. The canonical
        # official_family_score is the leaderboard-matching all-row
        # mean: (1.0 + 1.0 + 0.0 + 0.0) / 4 = 0.5 — the cap-equivalent
        # falls out naturally because error rows score 0.
        assert summary.protocol_gate_passed is False
        assert summary.family_capability_score == pytest.approx(1.0)
        assert summary.official_family_score == pytest.approx(0.5)


def test_e2e_official_score_uses_multi_metric_final_task_score(services):
    """When the validator stamps Phase-2 ``final_task_score`` per task,
    the summary's ``official_family_score`` must mirror the open-run
    leaderboard formula (mean of ``final_task_score``) — not the legacy
    ``agreement_score`` mean. This is the regression test that locks in
    the alignment between the rank page and the miner detail page.
    """
    validator_hk = "validator-multimetric"

    with services.db.sessionmaker() as session:
        bundle = {
            **_BUNDLE,
            "tasks": [
                {"task_id": f"mm-{i}", "family_id": "general_chat",
                 "prompt": f"q{i}", "mode": "instant", "expected_output": {}}
                for i in range(1, 5)
            ],
        }
        now = _utcnow()
        session.add(EvaluationRun(
            id="run-mm-1", sequence=1, status="open",
            benchmark_version="test_v1", rubric_version="agreement_general_chat_v1",
            judge_model="test-judge", min_scores_json={},
            started_at=now, ends_at=now + timedelta(days=7),
            metadata_json={"evaluation_bundles": {"general_chat": bundle}},
        ))
        session.flush()
        if session.get(RegisteredNeuron, "hk-mm") is None:
            session.add(RegisteredNeuron(hotkey="hk-mm", uid=0))
            session.flush()
        _seed_deployment(session, run_id="run-mm-1", hotkey="hk-mm")

        snap = EpochTargetSnapshot(
            run_id="run-mm-1", family_id="general_chat",
            rubric_version="agreement_general_chat_v1",
            benchmark_version="test_v1", judge_model="test-judge",
            members_json=[{"hotkey": "hk-mm", "endpoint": "x", "metadata": {}}],
            status="open",
        )
        session.add(snap)
        session.flush()
        snap_obj = session.get(EpochTargetSnapshot, snap.id)
        services.evaluation_tasks.initialize_evaluation_tasks(
            session, run_id="run-mm-1", family_id="general_chat", snapshot=snap_obj,
        )
        session.commit()

    with services.db.sessionmaker() as session:
        claimed = services.evaluation_tasks.claim_tasks(
            session, run_id="run-mm-1", family_id="general_chat",
            validator_hotkey=validator_hk, batch_size=10,
        )
        items = services.evaluation_tasks.build_claim_items(
            session, claimed_tasks=claimed, run_id="run-mm-1",
            family_id="general_chat",
        )
        session.commit()

    # Verdict is "matches" on every task (agreement_score=1.0) — i.e. the
    # legacy formula would yield official_family_score=1.0. The Phase-2
    # multi-metric mean is much lower because two tasks were knocked out
    # by composite gates. The test asserts the summary tracks the
    # multi-metric value, not the legacy one.
    final_task_scores = [1.0, 0.8, 0.0, 0.0]
    for item, final_score in zip(items, final_task_scores):
        miner_results = [{
            "miner_hotkey": "hk-mm",
            "miner_response": _miner_response("ok"),
            "miner_citations": [],
            "judge_output": _judge_output("matches", agreement_score=1.0),
            "agreement_score": 1.0,
            "verdict": "matches",
            "latency_seconds": 1.0,
            "final_task_score": final_score,
        }]
        with services.db.sessionmaker() as session:
            services.evaluation_tasks.submit_task_result(
                session,
                task_evaluation_id=item["task_evaluation_id"],
                validator_hotkey=validator_hk,
                baseline_response=_fabricated_baseline(),
                miner_results=miner_results,
            )
            session.commit()

    with services.db.sessionmaker() as session:
        summary = session.query(MinerEvaluationSummary).filter_by(
            run_id="run-mm-1", miner_hotkey="hk-mm",
        ).one()
        # mean_agreement is gate-aware: the two tasks where the gates
        # zeroed final_task_score (0.0) are reclassified as
        # ``gate_knockout`` and contribute 0 to the agreement mean.
        # Two real matches (1.0 + 1.0 = 2.0) over 4 completed rows = 0.5.
        assert summary.family_capability_score == pytest.approx(0.5)
        # Canonical official_family_score = AVG over all 4 rows of
        #   COALESCE(final_task_score, agreement_score)
        #   = (1.0 + 0.8 + 0.0 + 0.0) / 4 = 0.45.
        # This matches the leaderboard's open-run SQL formula exactly.
        assert summary.official_family_score == pytest.approx(0.45)
        # Gate-aware verdict counts: 2 effective matches, 2 gate knockouts.
        verdict_counts = summary.rollout_metadata_json["verdict_counts"]
        assert verdict_counts["matches"] == 2
        assert verdict_counts["gate_knockout"] == 2
        assert (
            summary.rollout_metadata_json["official_score_formula"]
            == "avg(coalesce(final_task_score, agreement_score)) over all rows"
        )
