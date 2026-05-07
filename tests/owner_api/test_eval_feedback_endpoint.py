"""Tests for the EvalFeedback read endpoint and the server-side upsert helper.

EvalFeedback rows are written *server-side* by the evaluation_task_manager
when accepting the validator's task-result POST (see
``_upsert_eval_feedback``). Miners read their own rows via the hotkey-
signed ``GET /v1/eval/feedback`` — no proxy, no internal token.

Coverage:
  * ``_upsert_eval_feedback`` — write + idempotency on (run, miner, task)
  * ``GET /v1/eval/feedback`` — miner reads their own rows; the filter
    ``miner_hotkey`` is derived from the signature.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from shared.common.database import Database
from shared.common.models import EvalFeedback
from control_plane.owner_api.evaluation.evaluation_task_manager import (
    _upsert_eval_feedback,
)
from control_plane.owner_api.routers.internal_eval import (
    EvalFeedbackListResponse,
    read_eval_feedback,
)


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'feedback.db'}")
    db.create_all()
    return db


def _make_request(*, db) -> SimpleNamespace:
    services = SimpleNamespace(db=db)
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(services=services)),
        headers={},
    )


def _meta(**overrides: Any) -> dict[str, Any]:
    base = {
        "eval_outcome": "correct",
        "eval_failure_mode": None,
        "eval_guidance": "ok",
        "eval_prompt_excerpt": "What is 2+2?",
        "eval_response_excerpt": "The answer is 4.",
        "eval_knockout_reasons": [],
        "composite_score": 1.0,
        "oracle_status": "deterministic",
    }
    base.update(overrides)
    return base


def _seed(
    db: Database,
    *,
    run_id: str = "run-1",
    miner_hotkey: str = "hk_alpha",
    task_id: str = "task-1",
    **meta_overrides: Any,
) -> None:
    with db.sessionmaker() as session:
        _upsert_eval_feedback(
            session,
            run_id=run_id,
            miner_hotkey=miner_hotkey,
            task_id=task_id,
            judge_meta=_meta(**meta_overrides),
        )
        session.commit()


# -- _upsert_eval_feedback ------------------------------------------------


def test_upsert_persists_row(tmp_path):
    db = _make_db(tmp_path)
    _seed(db)

    with db.sessionmaker() as session:
        rows = session.query(EvalFeedback).all()
        assert len(rows) == 1
        row = rows[0]
        assert row.run_id == "run-1"
        assert row.miner_hotkey == "hk_alpha"
        assert row.task_id == "task-1"
        assert row.outcome == "correct"
        assert row.guidance == "ok"
        assert row.composite_score == 1.0
        assert row.oracle_status == "deterministic"


def test_upsert_carries_failure_mode_and_knockouts(tmp_path):
    db = _make_db(tmp_path)
    _seed(
        db,
        eval_outcome="wrong",
        eval_failure_mode="wrong_fact",
        eval_guidance="check the date",
        composite_score=0.0,
        eval_knockout_reasons=["hallucination_knockout=0"],
    )

    with db.sessionmaker() as session:
        row = session.query(EvalFeedback).one()
        assert row.outcome == "wrong"
        assert row.failure_mode == "wrong_fact"
        assert row.knockout_reasons_json == ["hallucination_knockout=0"]


def test_upsert_idempotent_on_run_miner_task_key(tmp_path):
    """Re-running upsert with the same (run, miner, task) UPDATEs in
    place rather than tripping the unique constraint."""
    db = _make_db(tmp_path)
    _seed(db, eval_outcome="partial", composite_score=0.5)

    # Same identity, different values.
    _seed(
        db,
        eval_outcome="correct",
        composite_score=1.0,
        eval_guidance="actually correct on review",
    )

    with db.sessionmaker() as session:
        rows = session.query(EvalFeedback).all()
        assert len(rows) == 1
        row = rows[0]
        assert row.outcome == "correct"
        assert row.composite_score == 1.0
        assert row.guidance == "actually correct on review"


def test_upsert_distinct_tasks_create_separate_rows(tmp_path):
    db = _make_db(tmp_path)
    for tid in ("t-1", "t-2", "t-3"):
        _seed(db, task_id=tid)

    with db.sessionmaker() as session:
        rows = session.query(EvalFeedback).all()
        assert len(rows) == 3
        assert {r.task_id for r in rows} == {"t-1", "t-2", "t-3"}


def test_upsert_separates_miners_within_same_run(tmp_path):
    db = _make_db(tmp_path)
    _seed(db, miner_hotkey="hk_alpha", eval_outcome="correct")
    _seed(db, miner_hotkey="hk_beta", eval_outcome="wrong")

    with db.sessionmaker() as session:
        rows = session.query(EvalFeedback).all()
        assert len(rows) == 2
        by_hk = {r.miner_hotkey: r for r in rows}
        assert by_hk["hk_alpha"].outcome == "correct"
        assert by_hk["hk_beta"].outcome == "wrong"


# -- read_eval_feedback ---------------------------------------------------


async def test_read_returns_all_rows_for_miner_run(tmp_path):
    db = _make_db(tmp_path)
    for tid in ("t-1", "t-2", "t-3"):
        _seed(db, miner_hotkey="hk_alpha", task_id=tid)
    _seed(db, miner_hotkey="hk_beta", task_id="t-1")

    req = _make_request(db=db)
    response = await read_eval_feedback(
        req, run_id="run-1", hotkey="hk_alpha",
    )
    assert isinstance(response, EvalFeedbackListResponse)
    assert response.n_items == 3
    assert {item.task_id for item in response.items} == {"t-1", "t-2", "t-3"}
    assert all(item.miner_hotkey == "hk_alpha" for item in response.items)


async def test_read_returns_only_signers_rows(tmp_path):
    """Hotkey filter is derived from the signature — calling with
    ``hk_beta`` cannot pull ``hk_alpha``'s rows even though they share
    the same run."""
    db = _make_db(tmp_path)
    _seed(db, miner_hotkey="hk_alpha", task_id="t-1")
    _seed(db, miner_hotkey="hk_beta", task_id="t-1")

    req = _make_request(db=db)
    response = await read_eval_feedback(req, run_id="run-1", hotkey="hk_beta")
    assert response.n_items == 1
    assert response.items[0].miner_hotkey == "hk_beta"


async def test_read_empty_when_no_rows(tmp_path):
    """A miner that hasn't been judged in this run gets an empty list,
    not a 404."""
    db = _make_db(tmp_path)
    req = _make_request(db=db)
    response = await read_eval_feedback(
        req, run_id="run-1", hotkey="hk_unknown",
    )
    assert response.n_items == 0
    assert response.items == []


async def test_read_isolates_by_run_id(tmp_path):
    db = _make_db(tmp_path)
    _seed(db, run_id="run-old", task_id="t-1")
    _seed(db, run_id="run-new", task_id="t-1")
    _seed(db, run_id="run-new", task_id="t-2")

    req = _make_request(db=db)
    response = await read_eval_feedback(
        req, run_id="run-new", hotkey="hk_alpha",
    )
    assert response.n_items == 2
    assert {item.task_id for item in response.items} == {"t-1", "t-2"}
    assert all(item.run_id == "run-new" for item in response.items)


async def test_read_orders_by_created_at(tmp_path):
    """Rows return in arrival order so the per-miner doc renders items
    chronologically."""
    db = _make_db(tmp_path)
    for tid in ("t-3", "t-1", "t-2"):
        _seed(db, task_id=tid)

    req = _make_request(db=db)
    response = await read_eval_feedback(
        req, run_id="run-1", hotkey="hk_alpha",
    )
    assert [item.task_id for item in response.items] == ["t-3", "t-1", "t-2"]
