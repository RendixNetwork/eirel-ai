from __future__ import annotations

from datetime import timedelta

from control_plane.owner_api.app import app
from control_plane.owner_api._helpers import utcnow
from shared.common.models import EvaluationRun, RunFamilyResult, ValidatorRecord
from tests.conftest import signed_headers


def _register_validator(hotkey: str) -> None:
    services = app.state.services
    with services.db.sessionmaker() as session:
        if session.get(ValidatorRecord, hotkey) is None:
            session.add(ValidatorRecord(hotkey=hotkey, uid=1))
            session.commit()


def _make_run(run_id: str, *, sequence: int, status: str) -> None:
    services = app.state.services
    with services.db.sessionmaker() as session:
        if session.get(EvaluationRun, run_id) is not None:
            return
        now = utcnow()
        session.add(
            EvaluationRun(
                id=run_id,
                sequence=sequence,
                status=status,
                benchmark_version="test_v1",
                rubric_version="test_v1",
                judge_model="test-judge",
                min_scores_json={},
                started_at=now,
                ends_at=now + timedelta(days=1),
                metadata_json={},
            )
        )
        session.commit()


def _make_family_result(
    run_id: str,
    *,
    family_id: str,
    has_winner: bool,
    winner_hotkey: str | None = None,
    best_raw_score: float = 0.0,
) -> None:
    services = app.state.services
    with services.db.sessionmaker() as session:
        session.add(
            RunFamilyResult(
                run_id=run_id,
                family_id=family_id,
                winner_hotkey=winner_hotkey,
                best_raw_score=best_raw_score,
                min_score=0.0,
                has_winner=has_winner,
            )
        )
        session.commit()


async def test_weights_endpoint_returns_not_ready_when_no_completed_run(
    client, identities
):
    signer = identities["validator-1"]["signer"]
    _register_validator(signer.hotkey)

    hdrs = signed_headers(signer, method="GET", path="/v1/weights", body=b"")
    resp = await client.get("/v1/weights", headers=hdrs)

    assert resp.status_code == 200
    body = resp.json()
    # Fresh subnet: no completed run yet. The validator's weight-setter
    # loop treats this case the same as "owner-api unreachable" — it
    # publishes ``UID 0 → 1.0`` so vtrust doesn't decay during the
    # warm-up window. The 180-block cadence is longer than the chain's
    # set_weights rate-limit, so each cycle's burn succeeds.
    assert body["ready"] is False
    assert body["weights"] == {}
    assert body["family_winners"] == []


async def test_weights_endpoint_returns_winner_weights_when_run_completed(
    client, identities
):
    signer = identities["validator-1"]["signer"]
    _register_validator(signer.hotkey)
    miner_hotkey = identities["miner"]["signer"].hotkey

    _make_run("run-weights-1", sequence=1, status="completed")
    _make_family_result(
        "run-weights-1",
        family_id="general_chat",
        has_winner=True,
        winner_hotkey=miner_hotkey,
        best_raw_score=0.75,
    )

    hdrs = signed_headers(signer, method="GET", path="/v1/weights", body=b"")
    resp = await client.get("/v1/weights", headers=hdrs)

    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["run_id"] == "run-weights-1"
    assert body["weights"] == {miner_hotkey: 1.0}
    assert len(body["family_winners"]) == 1
    winner = body["family_winners"][0]
    assert winner["family_id"] == "general_chat"
    assert winner["winner_hotkey"] == miner_hotkey
    assert winner["family_weight"] == 1.0


async def test_weights_endpoint_returns_empty_winner_when_family_has_no_winner(
    client, identities
):
    signer = identities["validator-1"]["signer"]
    _register_validator(signer.hotkey)

    _make_run("run-weights-2", sequence=2, status="completed")
    _make_family_result(
        "run-weights-2",
        family_id="general_chat",
        has_winner=False,
        winner_hotkey=None,
    )

    hdrs = signed_headers(signer, method="GET", path="/v1/weights", body=b"")
    resp = await client.get("/v1/weights", headers=hdrs)

    assert resp.status_code == 200
    body = resp.json()
    # A completed run with no winning miner is a legitimate "burn to UID 0"
    # signal — still ``ready`` so the loop publishes once, but the validator
    # loop's last-published-run-id guard prevents repeated burns on the same
    # run.
    assert body["ready"] is True
    assert body["run_id"] == "run-weights-2"
    assert body["weights"] == {}
    assert len(body["family_winners"]) == 1
    assert body["family_winners"][0]["winner_hotkey"] is None
