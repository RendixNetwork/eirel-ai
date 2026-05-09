"""Validator-cost dashboard endpoint + aggregation tests.

Covers ``queries.validator_run_costs`` and the
``GET /api/v1/dashboard/runs/{run_id}/validator-costs`` route.

Setup uses the test DB pattern shared with ``test_dashboard_pairwise.py``:
ephemeral SQLite, hand-seeded TaskEvaluation + TaskMinerResult rows
across two validators so the cross-validator aggregate has bite.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import count

import pytest

from shared.common.database import Database
from shared.common.models import (
    EvaluationRun,
    TaskEvaluation,
    TaskMinerResult,
)
from control_plane.owner_api.dashboard.queries import validator_run_costs


_task_counter = count()
_run_seq_counter = count(1)
_RUN_ID = "run-vcost"
_RUN_ID_OTHER = "run-other"


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'vcost.db'}")
    db.create_all()
    return db


def _ensure_run(session, run_id: str) -> None:
    if session.get(EvaluationRun, run_id) is not None:
        return
    now = datetime.now(UTC).replace(tzinfo=None)
    session.add(EvaluationRun(
        id=run_id, sequence=next(_run_seq_counter), status="open",
        benchmark_version="test", rubric_version="agreement_general_chat_v1",
        judge_model="t", min_scores_json={},
        started_at=now, ends_at=now + timedelta(days=1),
        metadata_json={},
    ))
    session.flush()


def _seed_task(
    session, *,
    run_id: str, validator: str | None,
    oracle_cost_usd: float,
    miners: list[tuple[str, float]],
    status: str = "evaluated",
) -> None:
    """Seed one TaskEvaluation row + a TaskMinerResult per miner.

    ``miners`` is a list of ``(miner_hotkey, judge_cost_usd)`` pairs so
    each test can mix multiple miners on a single task.
    """
    _ensure_run(session, run_id)
    task_id = f"task-{run_id}-{next(_task_counter)}"
    te = TaskEvaluation(
        run_id=run_id, family_id="general_chat",
        task_id=task_id, task_index=0, status=status,
        claimed_by_validator=validator,
        oracle_cost_usd=oracle_cost_usd,
    )
    session.add(te)
    session.flush()
    for miner_hotkey, judge_cost in miners:
        session.add(TaskMinerResult(
            task_evaluation_id=te.id,
            run_id=run_id,
            family_id="general_chat",
            task_id=task_id,
            miner_hotkey=miner_hotkey,
            miner_response_json={},
            miner_citations_json=[],
            judge_output_json={},
            agreement_verdict="matches",
            agreement_score=1.0,
            latency_seconds=1.0,
            judge_cost_usd=judge_cost,
            proxy_cost_usd=0.0,
        ))
    session.flush()


def test_validator_run_costs_aggregates_per_validator(tmp_path):
    """Two validators each claim two tasks with two miners. Costs roll
    up by claimed_by_validator with both oracle + judge components."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as s:
        # Validator A: 2 tasks × $1 oracle + 2 miners × $0.05 judge each
        # → oracle $2, judge $0.20, total $2.20.
        _seed_task(s, run_id=_RUN_ID, validator="val-a",
                   oracle_cost_usd=1.0,
                   miners=[("miner-1", 0.05), ("miner-2", 0.05)])
        _seed_task(s, run_id=_RUN_ID, validator="val-a",
                   oracle_cost_usd=1.0,
                   miners=[("miner-1", 0.05), ("miner-2", 0.05)])
        # Validator B: 1 task × $0.50 oracle + 3 miners × $0.10 judge
        # → oracle $0.50, judge $0.30, total $0.80.
        _seed_task(s, run_id=_RUN_ID, validator="val-b",
                   oracle_cost_usd=0.50,
                   miners=[("miner-1", 0.10), ("miner-2", 0.10), ("miner-3", 0.10)])

        result = validator_run_costs(s, run_id=_RUN_ID)

    assert result.run_id == _RUN_ID
    by_hotkey = {v.validator_hotkey: v for v in result.validators}
    assert set(by_hotkey) == {"val-a", "val-b"}

    a = by_hotkey["val-a"]
    assert a.tasks_claimed == 2
    assert a.tasks_evaluated == 2
    assert a.oracle_cost_usd == pytest.approx(2.0)
    assert a.judge_cost_usd == pytest.approx(0.20)
    assert a.total_cost_usd == pytest.approx(2.20)

    b = by_hotkey["val-b"]
    assert b.tasks_claimed == 1
    assert b.oracle_cost_usd == pytest.approx(0.50)
    assert b.judge_cost_usd == pytest.approx(0.30)
    assert b.total_cost_usd == pytest.approx(0.80)

    # Sort order: highest total first.
    assert result.validators[0].validator_hotkey == "val-a"

    # Run-wide totals.
    assert result.total_oracle_cost_usd == pytest.approx(2.50)
    assert result.total_judge_cost_usd == pytest.approx(0.50)
    assert result.total_cost_usd == pytest.approx(3.00)


def test_validator_run_costs_excludes_other_runs(tmp_path):
    """Cost aggregation must scope to the requested run_id only."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as s:
        _seed_task(s, run_id=_RUN_ID, validator="val-a",
                   oracle_cost_usd=1.0, miners=[("m1", 0.05)])
        _seed_task(s, run_id=_RUN_ID_OTHER, validator="val-a",
                   oracle_cost_usd=99.0, miners=[("m1", 99.0)])

        result = validator_run_costs(s, run_id=_RUN_ID)

    assert len(result.validators) == 1
    assert result.validators[0].oracle_cost_usd == pytest.approx(1.0)
    assert result.validators[0].judge_cost_usd == pytest.approx(0.05)


def test_validator_run_costs_skips_unclaimed_tasks(tmp_path):
    """Tasks with claimed_by_validator IS NULL are pending — they have
    no validator to attribute cost to and shouldn't appear in the
    aggregate at all (they'd surface as NULL hotkey rows otherwise)."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as s:
        _seed_task(s, run_id=_RUN_ID, validator=None,
                   oracle_cost_usd=0.0, miners=[], status="pending")
        _seed_task(s, run_id=_RUN_ID, validator="val-a",
                   oracle_cost_usd=1.0, miners=[("m1", 0.05)])

        result = validator_run_costs(s, run_id=_RUN_ID)

    assert len(result.validators) == 1
    assert result.validators[0].validator_hotkey == "val-a"


def test_validator_run_costs_counts_evaluated_only(tmp_path):
    """``tasks_evaluated`` counts only ``status='evaluated'`` rows;
    claimed-but-not-yet-judged tasks bump ``tasks_claimed`` only."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as s:
        _seed_task(s, run_id=_RUN_ID, validator="val-a",
                   oracle_cost_usd=1.0,
                   miners=[("m1", 0.05)],
                   status="evaluated")
        _seed_task(s, run_id=_RUN_ID, validator="val-a",
                   oracle_cost_usd=0.0,
                   miners=[],
                   status="claimed")

        result = validator_run_costs(s, run_id=_RUN_ID)

    a = result.validators[0]
    assert a.tasks_claimed == 2
    assert a.tasks_evaluated == 1


def test_validator_run_costs_empty_run(tmp_path):
    """A run with no claimed tasks returns an empty list and zero
    totals — frontend uses this to show 'no validators claimed'."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as s:
        _ensure_run(s, _RUN_ID)
        result = validator_run_costs(s, run_id=_RUN_ID)

    assert result.validators == []
    assert result.total_cost_usd == 0.0
    assert result.total_oracle_cost_usd == 0.0
    assert result.total_judge_cost_usd == 0.0


def test_submit_task_result_persists_oracle_cost(tmp_path):
    """Going through the manager method (not just direct DB writes)
    proves the validator-submitted oracle_cost_usd lands on the
    TaskEvaluation row used by the dashboard aggregate."""
    from types import SimpleNamespace

    from control_plane.owner_api.evaluation.evaluation_task_manager import (
        EvaluationTaskManager,
    )

    # The manager only touches ``owner.settings`` lazily via a property,
    # never during ``submit_task_result`` — so a SimpleNamespace stub
    # is enough. Wrapping ManagedOwnerServices would require a fake
    # k8s client / runtime registry / etc which this test doesn't need.
    db = _make_db(tmp_path)
    manager = EvaluationTaskManager(SimpleNamespace(settings=None))
    with db.sessionmaker() as s:
        _ensure_run(s, _RUN_ID)
        te = TaskEvaluation(
            run_id=_RUN_ID, family_id="general_chat",
            task_id="task-cost-1", task_index=0, status="claimed",
            claimed_by_validator="val-x",
            oracle_cost_usd=0.0,
        )
        s.add(te)
        s.flush()
        task_evaluation_id = te.id
        s.commit()

    with db.sessionmaker() as s:
        result = manager.submit_task_result(
            s,
            task_evaluation_id=task_evaluation_id,
            validator_hotkey="val-x",
            baseline_response={"response_text": "ok"},
            miner_results=[{
                "miner_hotkey": "miner-1",
                "miner_response": {},
                "miner_citations": [],
                "judge_output": {"verdict": "matches", "rationale": "ok"},
                "verdict": "matches",
                "agreement_score": 1.0,
                "miner_latency_seconds": 1.0,
                "latency_seconds": 0.5,
                "judge_cost_usd": 0.07,
            }],
            oracle_cost_usd=1.234,
        )
        s.commit()

    assert result["status"] == "accepted"

    with db.sessionmaker() as s:
        row = s.get(TaskEvaluation, task_evaluation_id)
        assert row is not None
        assert row.oracle_cost_usd == pytest.approx(1.234)

        # And the dashboard aggregate sees it now.
        agg = validator_run_costs(s, run_id=_RUN_ID)
        assert len(agg.validators) == 1
        assert agg.validators[0].oracle_cost_usd == pytest.approx(1.234)
        assert agg.validators[0].judge_cost_usd == pytest.approx(0.07)
        assert agg.validators[0].total_cost_usd == pytest.approx(1.304)
