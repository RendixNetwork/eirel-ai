"""Tests for the eval-winner → product promotion job."""
from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest

from shared.common.database import Database
from shared.common.models import (
    DeploymentScoreRecord,
    ManagedDeployment,
    ManagedMinerSubmission,
    ServingPromotion,
    ServingRelease,
)
from control_plane.owner_api.promotions import (
    PromotionResult,
    promote_winners_for_run,
    select_winners_for_run,
)


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'promo.db'}")
    db.create_all()
    return db


def _seed_eval_record(
    session,
    *,
    deployment_id: str,
    family_id: str,
    run_id: str,
    miner_hotkey: str,
    raw_score: float,
    is_eligible: bool = True,
    health: str = "healthy",
) -> str:
    submission_id = str(uuid4())
    session.add(
        ManagedMinerSubmission(
            id=submission_id,
            miner_hotkey=miner_hotkey,
            submission_seq=1,
            family_id=family_id,
            status="deployed",
            artifact_id=str(uuid4()),
            manifest_json={"runtime": {"kind": "graph"}},
            archive_sha256="0" * 64,
            submission_block=0,
        )
    )
    revision = str(uuid4())
    session.add(
        ManagedDeployment(
            id=deployment_id,
            submission_id=submission_id,
            miner_hotkey=miner_hotkey,
            family_id=family_id,
            deployment_revision=revision,
            image_ref="img:test",
            endpoint=f"http://eval-{deployment_id}.test:8080",
            status="active",
            health_status=health,
            placement_status="placed",
        )
    )
    session.add(
        DeploymentScoreRecord(
            run_id=run_id,
            family_id=family_id,
            deployment_id=deployment_id,
            submission_id=submission_id,
            miner_hotkey=miner_hotkey,
            deployment_revision=revision,
            raw_score=raw_score,
            is_eligible=is_eligible,
        )
    )
    session.commit()
    return submission_id


def _seed_published_release(session) -> str:
    release_id = str(uuid4())
    session.add(ServingRelease(
        id=release_id,
        trigger_type="test",
        status="published",
        published_at=datetime.utcnow(),
    ))
    session.commit()
    return release_id


# -- select_winners_for_run --------------------------------------------------


def test_select_winners_picks_highest_raw_score(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_eval_record(
            session, deployment_id="d-low", family_id="general_chat",
            run_id="run-1", miner_hotkey="hk1", raw_score=0.5,
        )
        _seed_eval_record(
            session, deployment_id="d-high", family_id="general_chat",
            run_id="run-1", miner_hotkey="hk2", raw_score=0.9,
        )
    decisions = select_winners_for_run(database=db, run_id="run-1")
    assert len(decisions) == 1
    assert decisions[0].source_deployment_id == "d-high"
    assert decisions[0].family_id == "general_chat"


def test_select_winners_skips_ineligible(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_eval_record(
            session, deployment_id="d-bad", family_id="general_chat",
            run_id="run-1", miner_hotkey="hk1", raw_score=1.0,
            is_eligible=False,
        )
        _seed_eval_record(
            session, deployment_id="d-ok", family_id="general_chat",
            run_id="run-1", miner_hotkey="hk2", raw_score=0.5,
        )
    decisions = select_winners_for_run(database=db, run_id="run-1")
    assert len(decisions) == 1
    assert decisions[0].source_deployment_id == "d-ok"


def test_select_winners_skips_unhealthy(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_eval_record(
            session, deployment_id="d-sick", family_id="general_chat",
            run_id="run-1", miner_hotkey="hk1", raw_score=1.0,
            health="degraded",
        )
        _seed_eval_record(
            session, deployment_id="d-fit", family_id="general_chat",
            run_id="run-1", miner_hotkey="hk2", raw_score=0.5,
        )
    decisions = select_winners_for_run(database=db, run_id="run-1")
    assert len(decisions) == 1
    assert decisions[0].source_deployment_id == "d-fit"


def test_select_winners_returns_empty_when_no_records(tmp_path):
    db = _make_db(tmp_path)
    decisions = select_winners_for_run(database=db, run_id="run-empty")
    assert decisions == []


def test_select_winners_one_per_family(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_eval_record(
            session, deployment_id="gc-1", family_id="general_chat",
            run_id="run-1", miner_hotkey="hk1", raw_score=0.9,
        )
        _seed_eval_record(
            session, deployment_id="gc-2", family_id="general_chat",
            run_id="run-1", miner_hotkey="hk2", raw_score=0.5,
        )
        _seed_eval_record(
            session, deployment_id="dr-1", family_id="deep_research",
            run_id="run-1", miner_hotkey="hk3", raw_score=0.7,
        )
    decisions = select_winners_for_run(database=db, run_id="run-1")
    families = {d.family_id: d.source_deployment_id for d in decisions}
    assert families == {"general_chat": "gc-1", "deep_research": "dr-1"}


# -- promote_winners_for_run -------------------------------------------------


def test_promote_writes_serving_promotion_row(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_eval_record(
            session, deployment_id="d-win", family_id="general_chat",
            run_id="run-1", miner_hotkey="hk1", raw_score=0.9,
        )
        release_id = _seed_published_release(session)

    result = promote_winners_for_run(database=db, run_id="run-1")
    assert isinstance(result, PromotionResult)
    assert len(result.promoted) == 1
    assert len(result.skipped) == 0
    promoted = result.promoted[0]
    assert promoted["source_deployment_id"] == "d-win"
    assert promoted["serving_release_id"] == release_id

    with db.sessionmaker() as session:
        rows = session.query(ServingPromotion).all()
        assert len(rows) == 1
        assert rows[0].family_id == "general_chat"
        assert rows[0].run_id == "run-1"
        assert rows[0].source_deployment_id == "d-win"


def test_promote_is_idempotent_on_family_run(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_eval_record(
            session, deployment_id="d-win", family_id="general_chat",
            run_id="run-1", miner_hotkey="hk1", raw_score=0.9,
        )
        _seed_published_release(session)

    first = promote_winners_for_run(database=db, run_id="run-1")
    assert len(first.promoted) == 1

    # Re-run — should skip with reason already_promoted, no new row.
    second = promote_winners_for_run(database=db, run_id="run-1")
    assert len(second.promoted) == 0
    assert len(second.skipped) == 1
    assert second.skipped[0]["reason"] == "already_promoted"
    with db.sessionmaker() as session:
        assert session.query(ServingPromotion).count() == 1


def test_promote_skips_when_no_published_release(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_eval_record(
            session, deployment_id="d-win", family_id="general_chat",
            run_id="run-1", miner_hotkey="hk1", raw_score=0.9,
        )
        # no ServingRelease seeded
    result = promote_winners_for_run(database=db, run_id="run-1")
    assert len(result.promoted) == 0
    assert len(result.skipped) == 1
    assert result.skipped[0]["reason"] == "no_published_serving_release"


def test_promote_returns_empty_when_no_winners(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_published_release(session)
    result = promote_winners_for_run(database=db, run_id="run-none")
    assert result.promoted == []
    assert result.skipped == []
