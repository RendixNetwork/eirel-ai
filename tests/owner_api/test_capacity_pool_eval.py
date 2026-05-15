"""Capacity-aware continuous-pool evaluation (Option B).

Covers the pure/unit-testable core: the oracle-baseline TTL migration,
the reconciled-oracle cache round-trip, the recency-ordered pool
selection with done-detection + K slicing, and build_claim_items'
healthy-deployment + cached-baseline behavior.
"""

from __future__ import annotations

import tempfile
from datetime import timedelta
from types import SimpleNamespace

from sqlalchemy import create_engine, inspect, text

from shared.common.database import Database
from shared.common.migrations import run_migrations
from shared.common.models import (
    Base,
    EpochTargetSnapshot,
    ManagedDeployment,
    ManagedMinerSubmission,
    TaskEvaluation,
    TaskMinerResult,
    utcnow,
)
from control_plane.owner_api.evaluation.run_manager import RunManager
from control_plane.owner_api.evaluation.evaluation_task_manager import (
    EvaluationTaskManager,
    _ORACLE_BASELINE_TTL,
)
from validation.validator.engine import (
    _reconciled_from_payload,
    _reconciled_to_payload,
)
from validation.validator.reconciler import ReconciledOracle


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'pool.db'}")
    db.create_all()
    return db


# -- migration -------------------------------------------------------------


def test_baseline_cached_at_present_on_fresh_db():
    e = create_engine(f"sqlite:///{tempfile.mktemp(suffix='.db')}")
    run_migrations(e)
    Base.metadata.create_all(e)
    cols = {c["name"] for c in inspect(e).get_columns("task_evaluations")}
    assert "baseline_cached_at" in cols


def test_migration_alters_preexisting_db_and_is_idempotent():
    path = tempfile.mktemp(suffix=".db")
    e = create_engine(f"sqlite:///{path}")
    # Simulate a pre-existing DB: table without the new column.
    with e.begin() as c:
        c.execute(text(
            "CREATE TABLE task_evaluations (id TEXT PRIMARY KEY, "
            "epoch_id TEXT, family_id TEXT, task_id TEXT, task_index INT, "
            "status TEXT, claim_attempt_count INT, oracle_cost_usd REAL, "
            "created_at TIMESTAMP, updated_at TIMESTAMP)"
        ))
    first = run_migrations(e)
    assert "oracle_baseline_ttl_cache" in first
    cols = {c["name"] for c in inspect(e).get_columns("task_evaluations")}
    assert "baseline_cached_at" in cols
    # Re-running applies nothing new (idempotent).
    second = run_migrations(e)
    assert "oracle_baseline_ttl_cache" not in second


# -- reconciled-oracle cache round-trip ------------------------------------


def test_reconciled_payload_round_trips():
    original = ReconciledOracle(
        expected_claims=["a", "b"],
        must_not_claim=["x"],
        oracle_status="consensus",
        vendor_costs={"openai": 0.0021},
        vendor_citations={"openai": ["http://e.g/1"]},
    )
    restored = _reconciled_from_payload(_reconciled_to_payload(original))
    assert restored.expected_claims == ["a", "b"]
    assert restored.must_not_claim == ["x"]
    assert restored.oracle_status == "consensus"
    assert restored.vendor_costs == {"openai": 0.0021}


def test_reconciled_from_payload_drops_unknown_keys():
    # A future validator may add fields; replaying an old cache must not
    # explode.
    payload = _reconciled_to_payload(ReconciledOracle(expected_claims=["q"]))
    payload["some_future_field"] = 123
    restored = _reconciled_from_payload(payload)
    assert restored.expected_claims == ["q"]


# -- pool_keep_deployment_ids ---------------------------------------------


def _snap_member(hk: str, dep_id: str, sub_id: str) -> dict:
    return {"hotkey": hk, "metadata": {"deployment_id": dep_id, "submission_id": sub_id}}


def _seed_pool(session, *, run_id="run-1", family="general_chat", n=4, n_tasks=3):
    """n miners, n_tasks tasks. Submissions created oldest→newest by index."""
    base = utcnow()
    members = []
    for i in range(n):
        sub = ManagedMinerSubmission(
            id=f"sub-{i}", miner_hotkey=f"hk{i}", family_id=family,
            archive_sha256=f"sha{i}", submission_block=i, submission_seq=i,
            artifact_id=f"art-{i}", manifest_json={}, status="received",
            created_at=base + timedelta(minutes=i),  # i=n-1 is newest
        )
        session.add(sub)
        members.append(_snap_member(f"hk{i}", f"dep-{i}", f"sub-{i}"))
    session.add(EpochTargetSnapshot(
        run_id=run_id, family_id=family, benchmark_version="v1",
        rubric_version="v1", judge_model="j", status="open",
        frozen_validator_stakes_json={}, members_json=members,
    ))
    for t in range(n_tasks):
        session.add(TaskEvaluation(
            run_id=run_id, family_id=family, task_id=f"t{t}",
            task_index=t, status="pending",
        ))
    session.commit()


def test_pool_selects_top_k_by_recency(tmp_path):
    db = _make_db(tmp_path)
    rm = RunManager(SimpleNamespace())
    with db.sessionmaker() as s:
        _seed_pool(s, n=4, n_tasks=3)
        out = rm.pool_keep_deployment_ids(s, family_id="general_chat", capacity_k=2)
    # Newest two submissions (sub-3, sub-2) kept; all 4 eligible.
    assert out["keep"] == {"dep-3", "dep-2"}
    assert out["eligible_total"] == 4
    assert out["done"] == set()
    assert out["snapshot_total"] == 4


def test_pool_marks_fully_evaluated_miner_done(tmp_path):
    db = _make_db(tmp_path)
    rm = RunManager(SimpleNamespace())
    with db.sessionmaker() as s:
        _seed_pool(s, n=3, n_tasks=2)
        # hk0 has a row for both tasks → done (even all-error counts).
        for t in ("t0", "t1"):
            s.add(TaskMinerResult(
                task_evaluation_id=f"te-{t}", run_id="run-1",
                family_id="general_chat", task_id=t, miner_hotkey="hk0",
                miner_response_json={}, miner_citations_json=[],
                agreement_verdict="error", agreement_score=0.0,
                miner_latency_seconds=0.0, latency_seconds=0.0,
                proxy_cost_usd=0.0, judge_cost_usd=0.0,
            ))
        s.commit()
        out = rm.pool_keep_deployment_ids(s, family_id="general_chat", capacity_k=10)
    assert "dep-0" in out["done"]
    assert out["eligible_total"] == 2  # hk1, hk2 still owe tasks
    assert "dep-0" not in out["keep"]


def test_pool_empty_when_no_open_snapshot(tmp_path):
    db = _make_db(tmp_path)
    rm = RunManager(SimpleNamespace())
    with db.sessionmaker() as s:
        out = rm.pool_keep_deployment_ids(s, family_id="general_chat", capacity_k=5)
    assert out == {
        "keep": set(), "done": set(), "eligible_total": 0, "snapshot_total": 0,
    }


# -- build_claim_items: healthy predicate + cached baseline ---------------


def _owner_stub(tasks):
    return SimpleNamespace(
        run_evaluation_bundle=lambda *a, **k: {"tasks": tasks},
    )


def test_claim_excludes_miners_without_healthy_deployment(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as s:
        members = [
            _snap_member("hkA", "depA", "subA"),
            _snap_member("hkB", "depB", "subB"),
        ]
        s.add(EpochTargetSnapshot(
            run_id="run-1", family_id="general_chat", benchmark_version="v1",
            rubric_version="v1", judge_model="j", status="open",
            frozen_validator_stakes_json={}, members_json=members,
        ))
        s.add(ManagedDeployment(
            id="depA", submission_id="subA", miner_hotkey="hkA",
            family_id="general_chat", deployment_revision="revA",
            image_ref="img", endpoint="http://a",
            status="deployed_for_eval", health_status="healthy",
        ))
        s.add(ManagedDeployment(
            id="depB", submission_id="subB", miner_hotkey="hkB",
            family_id="general_chat", deployment_revision="revB",
            image_ref="img", endpoint="http://b",
            status="building", health_status="starting",
        ))
        te = TaskEvaluation(
            run_id="run-1", family_id="general_chat", task_id="t0",
            task_index=0, status="claimed",
        )
        s.add(te)
        s.commit()
        mgr = EvaluationTaskManager(_owner_stub([{"task_id": "t0"}]))
        items = mgr.build_claim_items(
            s, claimed_tasks=[te], run_id="run-1", family_id="general_chat",
        )
    assert len(items) == 1
    hks = {m["hotkey"] for m in items[0]["miners"]}
    assert hks == {"hkA"}  # hkB excluded — pod not healthy yet


def test_claim_attaches_cached_baseline_within_ttl(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as s:
        s.add(EpochTargetSnapshot(
            run_id="run-1", family_id="general_chat", benchmark_version="v1",
            rubric_version="v1", judge_model="j", status="open",
            frozen_validator_stakes_json={},
            members_json=[_snap_member("hkA", "depA", "subA")],
        ))
        s.add(ManagedDeployment(
            id="depA", submission_id="subA", miner_hotkey="hkA",
            family_id="general_chat", deployment_revision="r1",
            image_ref="img", endpoint="http://a",
            status="deployed_for_eval", health_status="healthy",
        ))
        fresh = TaskEvaluation(
            run_id="run-1", family_id="general_chat", task_id="tf",
            task_index=0, status="claimed",
            baseline_response_json={"response_text": "cached!"},
            baseline_cached_at=utcnow() - timedelta(hours=1),
        )
        stale = TaskEvaluation(
            run_id="run-1", family_id="general_chat", task_id="ts",
            task_index=1, status="claimed",
            baseline_response_json={"response_text": "old"},
            baseline_cached_at=utcnow() - _ORACLE_BASELINE_TTL
            - timedelta(minutes=1),
        )
        s.add_all([fresh, stale])
        s.commit()
        mgr = EvaluationTaskManager(
            _owner_stub([{"task_id": "tf"}, {"task_id": "ts"}])
        )
        items = {
            it["task_id"]: it
            for it in mgr.build_claim_items(
                s, claimed_tasks=[fresh, stale],
                run_id="run-1", family_id="general_chat",
            )
        }
    assert items["tf"].get("cached_baseline") == {"response_text": "cached!"}
    assert "cached_baseline" not in items["ts"]  # past 12h TTL → recompute
