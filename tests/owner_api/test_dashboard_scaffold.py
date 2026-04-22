from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from httpx import AsyncClient

from control_plane.owner_api.app import app
from control_plane.owner_api.dashboard import queries
from control_plane.owner_api.dashboard.cache import TTLCache
from control_plane.owner_api.routers.dashboard import _reset_cache_for_tests
from shared.common.database import Database
from shared.common.models import (
    DeploymentScoreRecord,
    EvaluationRun,
    ManagedDeployment,
    ManagedMinerSubmission,
    MinerEvaluationSummary,
    MinerEvaluationTask,
    RunFamilyResult,
    SubmissionArtifact,
    ValidatorRecord,
)


EXPECTED_PATHS = {
    ("GET", "/api/v1/dashboard/overview"),
    ("GET", "/api/v1/dashboard/families"),
    ("GET", "/api/v1/dashboard/leaderboard"),
    ("GET", "/api/v1/dashboard/miners/{hotkey}"),
    ("GET", "/api/v1/dashboard/miners/{hotkey}/runs"),
    ("GET", "/api/v1/dashboard/miners/{hotkey}/runs/{run_id}"),
}


def test_dashboard_router_registers_all_six_endpoints():
    registered = {
        (method, route.path)
        for route in app.routes
        if hasattr(route, "path") and route.path.startswith("/api/v1/dashboard")
        for method in getattr(route, "methods", ()) or ()
    }
    missing = EXPECTED_PATHS - registered
    assert not missing, f"missing dashboard routes: {missing}"


def test_hotkey_short_uses_first_four_last_three():
    assert queries.shorten_hotkey("5FHneW46xGXgs5AMrxvJABC") == "5FHn...ABC"
    assert queries.shorten_hotkey("short") == "short"


def test_ttl_cache_expires_entries():
    cache: TTLCache[int] = TTLCache(default_ttl_seconds=1.0)
    cache.set(("k",), 42, ttl=10.0)
    assert cache.get(("k",)) == 42
    cache.set(("k2",), 99, ttl=0.0)
    assert cache.get(("k2",)) is None


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'dash.db'}")
    db.create_all()
    return db


class _FakeSettings:
    active_families = "general_chat"
    family_weights = "general_chat:1.0"
    bittensor_netuid = 144
    bittensor_network = "finney"


class _FakeRunsManager:
    def run_evaluation_bundle(self, session, *, run_id, family_id):
        del session, run_id, family_id
        return {
            "tasks": [
                {"task_id": "task-a", "prompt": "What are the three states of water?",
                 "mode": "instant", "category": "factual", "difficulty": "standard"},
                {"task_id": "task-b", "prompt": "Summarize this article.",
                 "mode": "thinking", "category": "academic_science", "difficulty": "hard"},
            ]
        }


class _FakeServices:
    def __init__(self):
        self.settings = _FakeSettings()
        self.runs = _FakeRunsManager()


def _seed_closed_run(session, *, sequence: int, run_id: str) -> EvaluationRun:
    now = datetime.now(UTC).replace(tzinfo=None)
    run = EvaluationRun(
        id=run_id, sequence=sequence, status="closed",
        benchmark_version="test", rubric_version="test", judge_model="test-judge",
        min_scores_json={}, started_at=now - timedelta(days=1, hours=sequence),
        ends_at=now - timedelta(hours=sequence), closed_at=now - timedelta(hours=sequence),
        metadata_json={},
    )
    session.add(run)
    session.flush()
    return run


def _seed_open_run(session, *, sequence: int, run_id: str) -> EvaluationRun:
    now = datetime.now(UTC).replace(tzinfo=None)
    run = EvaluationRun(
        id=run_id, sequence=sequence, status="open",
        benchmark_version="test", rubric_version="test", judge_model="test-judge",
        min_scores_json={}, started_at=now - timedelta(hours=1),
        ends_at=now + timedelta(days=3), metadata_json={},
    )
    session.add(run)
    session.flush()
    return run


def _seed_deployment_score(session, *, run_id, hotkey, raw, norm, family_id="general_chat"):
    artifact = SubmissionArtifact(
        archive_bytes=b"x", sha256="sha-" + uuid4().hex[:8], size_bytes=1, manifest_json={},
    )
    session.add(artifact)
    session.flush()
    submission = ManagedMinerSubmission(
        miner_hotkey=hotkey, submission_seq=1, family_id=family_id, status="received",
        artifact_id=artifact.id, manifest_json={}, archive_sha256=artifact.sha256,
        submission_block=100, introduced_run_id=run_id,
    )
    session.add(submission)
    session.flush()
    deployment = ManagedDeployment(
        submission_id=submission.id, miner_hotkey=hotkey, family_id=family_id,
        deployment_revision="rev-" + uuid4().hex[:8], image_ref="managed://test",
        endpoint="http://test", status="active", health_status="healthy", is_active=True,
    )
    session.add(deployment)
    session.flush()
    session.add(DeploymentScoreRecord(
        run_id=run_id, family_id=family_id, deployment_id=deployment.id,
        submission_id=submission.id, miner_hotkey=hotkey,
        deployment_revision=deployment.deployment_revision,
        raw_score=raw, normalized_score=norm,
    ))
    session.flush()


def _seed_task(
    session, *, run_id, hotkey, task_id, task_index, score,
    quality=None, dimensions=None, latency=None, latency_ms=None,
    trace_gate_passed=True, honeytoken_cited=False, miner_response=None,
    family_id="general_chat", status="evaluated",
):
    judge_output = {
        "score": quality if quality is not None else score,
        "dimension_scores": dimensions or {
            "goal_fulfillment": 0.9, "correctness": 0.85,
            "grounding": 0.8, "conversation_coherence": 0.88,
        },
        "constraint_flags": [],
        "rationale": "Satisfactory response.",
    }
    result_metadata = {
        "quality_score": quality if quality is not None else score,
        "conversation_score": {
            "quality": quality if quality is not None else score,
            "latency": latency if latency is not None else 0.5,
            "trace_gate": 1.0 if trace_gate_passed else 0.0,
            "total": score,
            "per_dimension": {},
            "metadata": {
                "trace_gate_passed": trace_gate_passed,
                "honeytoken_cited": honeytoken_cited,
                "latency_ms": latency_ms if latency_ms is not None else 5000,
                "claims_extracted": 0,
                "trace_entries": 1,
            },
        },
    }
    session.add(MinerEvaluationTask(
        run_id=run_id, family_id=family_id, miner_hotkey=hotkey, task_id=task_id,
        task_index=task_index, status=status, task_score=score,
        miner_response_json=miner_response or {"response_text": f"response for {task_id}"},
        judge_output_json=judge_output, result_metadata_json=result_metadata,
        evaluated_at=datetime.now(UTC).replace(tzinfo=None),
    ))


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


async def test_metrics_aggregate_quality_latency_trace_gate_honeytoken(tmp_path):
    db = _make_db(tmp_path)
    services = _FakeServices()
    with db.sessionmaker() as session:
        run = _seed_closed_run(session, sequence=1, run_id="run-1")
        _seed_task(session, run_id=run.id, hotkey="hk-1", task_id="t1", task_index=0, score=0.8,
                   quality=0.9, latency=0.7, trace_gate_passed=True, honeytoken_cited=False)
        _seed_task(session, run_id=run.id, hotkey="hk-1", task_id="t2", task_index=1, score=0.6,
                   quality=0.7, latency=0.5, trace_gate_passed=False, honeytoken_cited=False)
        _seed_task(session, run_id=run.id, hotkey="hk-1", task_id="t3", task_index=2, score=0.0,
                   quality=0.6, latency=0.5, trace_gate_passed=True, honeytoken_cited=True)
        session.commit()

    with db.sessionmaker() as session:
        metrics = queries._metrics_for_tasks(
            session, run_id="run-1", family_id="general_chat", hotkey="hk-1",
        )["hk-1"]
    assert metrics.quality_mean == (0.9 + 0.7 + 0.6) / 3
    assert metrics.latency_mean == (0.7 + 0.5 + 0.5) / 3
    assert metrics.trace_gate_pass_rate == 2 / 3
    assert metrics.honeytoken_count == 1


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


async def test_leaderboard_entry_has_general_chat_metrics(tmp_path):
    db = _make_db(tmp_path)
    services = _FakeServices()
    with db.sessionmaker() as session:
        run = _seed_closed_run(session, sequence=1, run_id="run-1")
        _seed_deployment_score(session, run_id=run.id, hotkey="hk-1", raw=0.8, norm=0.5)
        _seed_task(session, run_id=run.id, hotkey="hk-1", task_id="t1", task_index=0, score=0.8,
                   quality=0.9, latency=0.7)
        # Post-close summary with instant/thinking rolled up.
        session.add(MinerEvaluationSummary(
            run_id=run.id, family_id="general_chat", miner_hotkey="hk-1",
            total_tasks=1, completed_tasks=1, failed_tasks=0,
            official_family_score=0.8, status="scored",
            rollout_metadata_json={"general_chat": {
                "instant_mean": 0.72, "thinking_mean": 0.84,
                "blended": 0.77, "cost_efficiency": 0.95,
            }},
        ))
        session.commit()

    with db.sessionmaker() as session:
        result = queries.fetch_leaderboard(
            session, services=services, family_id="general_chat",
            run_id=None, limit=10, offset=0,
        )
    entry = result.entries[0]
    assert entry.metrics.quality_mean == 0.9
    assert entry.metrics.latency_mean == 0.7
    assert entry.metrics.trace_gate_pass_rate == 1.0
    assert entry.metrics.honeytoken_count == 0
    assert entry.metrics.instant_mean == 0.72
    assert entry.metrics.blended == 0.77
    assert entry.metrics.cost_efficiency == 0.95


async def test_leaderboard_in_progress_run_uses_task_avg_marks_is_running(tmp_path):
    db = _make_db(tmp_path)
    services = _FakeServices()
    with db.sessionmaker() as session:
        run = _seed_open_run(session, sequence=1, run_id="run-open")
        _seed_task(session, run_id=run.id, hotkey="hk-a", task_id="t1", task_index=0, score=0.3)
        _seed_task(session, run_id=run.id, hotkey="hk-a", task_id="t2", task_index=1, score=0.5)
        _seed_task(session, run_id=run.id, hotkey="hk-b", task_id="t1", task_index=0, score=0.9)
        _seed_task(session, run_id=run.id, hotkey="hk-b", task_id="t2", task_index=1, score=0.7)
        session.commit()

    with db.sessionmaker() as session:
        result = queries.fetch_leaderboard(
            session, services=services, family_id="general_chat",
            run_id=None, limit=10, offset=0,
        )
    assert [e.hotkey for e in result.entries] == ["hk-b", "hk-a"]
    assert result.entries[0].raw_score == 0.8
    assert all(e.is_running for e in result.entries)
    assert all(e.normalized_score is None for e in result.entries)


async def test_leaderboard_trend_up_down_across_adjacent_runs(tmp_path):
    db = _make_db(tmp_path)
    services = _FakeServices()
    with db.sessionmaker() as session:
        r0 = _seed_closed_run(session, sequence=1, run_id="run-1")
        _seed_deployment_score(session, run_id=r0.id, hotkey="hk-alpha", raw=0.9, norm=0.6)
        _seed_deployment_score(session, run_id=r0.id, hotkey="hk-bravo", raw=0.5, norm=0.3)
        r1 = _seed_closed_run(session, sequence=2, run_id="run-2")
        _seed_deployment_score(session, run_id=r1.id, hotkey="hk-alpha", raw=0.4, norm=0.25)
        _seed_deployment_score(session, run_id=r1.id, hotkey="hk-bravo", raw=0.8, norm=0.55)
        session.commit()

    with db.sessionmaker() as session:
        result = queries.fetch_leaderboard(
            session, services=services, family_id="general_chat",
            run_id=None, limit=10, offset=0,
        )
    by_hk = {e.hotkey: e for e in result.entries}
    assert by_hk["hk-bravo"].trend == "up"
    assert by_hk["hk-alpha"].trend == "down"


# ---------------------------------------------------------------------------
# Miner profile / runs / run detail
# ---------------------------------------------------------------------------


async def test_miner_profile_returns_latest_metrics(tmp_path):
    db = _make_db(tmp_path)
    services = _FakeServices()
    with db.sessionmaker() as session:
        run = _seed_closed_run(session, sequence=1, run_id="run-1")
        _seed_deployment_score(session, run_id=run.id, hotkey="hk-p", raw=0.8, norm=0.5)
        _seed_task(session, run_id=run.id, hotkey="hk-p", task_id="t1", task_index=0, score=0.8,
                   quality=0.88, latency=0.6)
        session.commit()

    with db.sessionmaker() as session:
        profile = queries.fetch_miner_profile(
            session, services=services, hotkey="hk-p", family_id="general_chat",
        )
    assert profile.current_rank == 1
    assert profile.latest_metrics.quality_mean == 0.88
    assert profile.latest_metrics.latency_mean == 0.6


async def test_miner_runs_ranks_highest_norm_score_first(tmp_path):
    db = _make_db(tmp_path)
    services = _FakeServices()
    with db.sessionmaker() as session:
        for seq in (1, 2, 3):
            run = _seed_closed_run(session, sequence=seq, run_id=f"run-{seq}")
            _seed_deployment_score(session, run_id=run.id, hotkey="hk-focus", raw=0.5 + seq * 0.1, norm=0.3 + seq * 0.05)
            _seed_deployment_score(session, run_id=run.id, hotkey="hk-other", raw=0.9, norm=0.8)
        session.commit()

    with db.sessionmaker() as session:
        result = queries.fetch_miner_runs(
            session, services=services, hotkey="hk-focus",
            family_id="general_chat", limit=10, offset=0,
        )
    assert [r.epoch_sequence for r in result.runs] == [3, 2, 1]
    assert [r.rank for r in result.runs] == [2, 2, 2]


async def test_run_detail_surfaces_judge_dimensions_and_conversation_score(tmp_path):
    db = _make_db(tmp_path)
    services = _FakeServices()
    with db.sessionmaker() as session:
        run = _seed_closed_run(session, sequence=1, run_id="run-1")
        _seed_task(session, run_id=run.id, hotkey="hk-focus", task_id="task-a",
                   task_index=0, score=0.82,
                   quality=0.9,
                   dimensions={"goal_fulfillment": 0.95, "correctness": 0.9,
                               "grounding": 0.85, "conversation_coherence": 0.88},
                   latency=0.7, latency_ms=6000, trace_gate_passed=True, honeytoken_cited=False,
                   miner_response={"response_text": "H2O has solid, liquid, gas forms."})
        _seed_task(session, run_id=run.id, hotkey="hk-focus", task_id="task-b",
                   task_index=1, score=0.0,
                   quality=0.4, latency=0.3, trace_gate_passed=False, honeytoken_cited=True)
        session.commit()

    with db.sessionmaker() as session:
        detail = queries.fetch_run_detail(
            session, services=services, hotkey="hk-focus",
            family_id="general_chat", run_id="run-1",
        )

    assert len(detail.tasks) == 2
    by_id = {t.task_id: t for t in detail.tasks}

    ta = by_id["task-a"]
    assert ta.prompt == "What are the three states of water?"
    assert ta.mode == "instant"
    assert ta.category == "factual"
    assert ta.difficulty == "standard"
    assert ta.quality_score == 0.9
    assert ta.dimension_scores.goal_fulfillment == 0.95
    assert ta.latency_score == 0.7
    assert ta.latency_ms == 6000
    assert ta.trace_gate_passed is True
    assert ta.honeytoken_cited is False
    assert ta.miner_response == {"response_text": "H2O has solid, liquid, gas forms."}
    assert ta.judge_rationale == "Satisfactory response."

    tb = by_id["task-b"]
    assert tb.trace_gate_passed is False
    assert tb.honeytoken_cited is True
    assert tb.mode == "thinking"

    # Aggregate metrics reflect both tasks.
    assert detail.metrics.quality_mean == (0.9 + 0.4) / 2
    assert detail.metrics.honeytoken_count == 1
    assert detail.metrics.trace_gate_pass_rate == 0.5


# ---------------------------------------------------------------------------
# End-to-end via ASGI client
# ---------------------------------------------------------------------------


async def test_dashboard_endpoints_return_200_on_empty_db(client: AsyncClient):
    _reset_cache_for_tests()
    r = await client.get("/api/v1/dashboard/overview")
    assert r.status_code == 200, r.text
    assert r.json()["total_miners"] == 0
    assert r.json()["current_run_id"] is None

    r = await client.get("/api/v1/dashboard/families")
    assert r.status_code == 200
    assert any(f["id"] == "general_chat" for f in r.json()["families"])

    r = await client.get("/api/v1/dashboard/leaderboard", params={"family_id": "general_chat"})
    assert r.status_code == 200
    assert r.json()["entries"] == []

    r = await client.get(
        "/api/v1/dashboard/miners/hk-none",
        params={"family_id": "general_chat"},
    )
    assert r.status_code == 200
    assert r.json()["lifetime_wins"] == 0
    assert "latest_metrics" in r.json()

    r = await client.get(
        "/api/v1/dashboard/miners/hk-none/runs",
        params={"family_id": "general_chat"},
    )
    assert r.status_code == 200
    assert r.json()["runs"] == []

    r = await client.get(
        "/api/v1/dashboard/miners/hk-none/runs/unknown-run",
        params={"family_id": "general_chat"},
    )
    assert r.status_code == 404


async def test_dashboard_rejects_invalid_family_id(client: AsyncClient):
    _reset_cache_for_tests()
    r = await client.get("/api/v1/dashboard/leaderboard", params={"family_id": "not-a-family"})
    assert r.status_code == 400
    assert "unsupported family_id" in r.json()["detail"]


async def test_overview_counts_validators_and_miners(tmp_path):
    db = _make_db(tmp_path)
    services = _FakeServices()
    with db.sessionmaker() as session:
        _seed_closed_run(session, sequence=1, run_id="run-1")
        _seed_deployment_score(session, run_id="run-1", hotkey="hk-1", raw=0.5, norm=0.3)
        _seed_deployment_score(session, run_id="run-1", hotkey="hk-2", raw=0.6, norm=0.4)
        session.add(ValidatorRecord(hotkey="v1", uid=1, is_active=True))
        session.add(ValidatorRecord(hotkey="v2", uid=2, is_active=False))
        session.commit()

    with db.sessionmaker() as session:
        r = queries.fetch_overview(session, services=services)
    assert r.total_miners == 2
    assert r.active_validators == 1
    assert r.netuid == 144
    assert r.network == "finney"


async def test_leaderboard_marks_winner_from_run_family_result(tmp_path):
    db = _make_db(tmp_path)
    services = _FakeServices()
    with db.sessionmaker() as session:
        run = _seed_closed_run(session, sequence=1, run_id="run-p")
        _seed_deployment_score(session, run_id=run.id, hotkey="hk-p", raw=0.7, norm=0.5)
        session.add(RunFamilyResult(
            run_id=run.id, family_id="general_chat",
            winner_hotkey="hk-p", best_raw_score=0.7, has_winner=True,
        ))
        session.commit()

    with db.sessionmaker() as session:
        result = queries.fetch_leaderboard(
            session, services=services, family_id="general_chat",
            run_id=None, limit=10, offset=0,
        )
    entry = result.entries[0]
    assert entry.is_serving_winner is True
    assert entry.win_count == 1
