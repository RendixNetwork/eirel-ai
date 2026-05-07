from __future__ import annotations

from datetime import timedelta

import pytest

from control_plane.owner_api.app import app
from control_plane.owner_api._helpers import utcnow
from shared.common.models import EvaluationRun
from tests.conftest import make_submission_archive, signed_headers


# -- Helpers ---------------------------------------------------------------


def _ensure_run() -> None:
    """Ensure an evaluation run exists so create_submission can target it."""
    services = app.state.services
    with services.db.sessionmaker() as session:
        existing = session.get(EvaluationRun, "run-access-test")
        if existing is not None:
            return
        now = utcnow()
        run = EvaluationRun(
            id="run-access-test",
            sequence=1,
            status="open",
            benchmark_version="test_v1",
            rubric_version="test_v1",
            judge_model="test-judge",
            min_scores_json={},
            started_at=now,
            ends_at=now + timedelta(days=7),
            metadata_json={},
        )
        session.add(run)
        session.commit()


def _make_submission(hotkey: str) -> str:
    from shared.common.models import RegisteredNeuron
    _ensure_run()
    services = app.state.services
    archive = make_submission_archive()
    with services.db.sessionmaker() as session:
        if session.get(RegisteredNeuron, hotkey) is None:
            session.add(RegisteredNeuron(hotkey=hotkey, uid=0))
            session.flush()
        sub, _dep = services.create_submission(
            session,
            miner_hotkey=hotkey,
            submission_block=100,
            archive_bytes=archive,
            base_url="http://testserver",
        )
        sub_id = sub.id
        session.commit()
    return sub_id


def _multipart_body(
    archive_bytes: bytes, boundary: str = "testboundary"
) -> bytes:
    """Build a minimal multipart/form-data body with a fixed boundary."""
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="archive"; filename="sub.tar.gz"\r\n'
        f"Content-Type: application/gzip\r\n\r\n"
    ).encode() + archive_bytes + f"\r\n--{boundary}--\r\n".encode()


# -- Fix 1: Access control ------------------------------------------------


async def test_pool_rejects_miner_signature(client, identities):
    signer = identities["miner"]["signer"]
    hdrs = signed_headers(
        signer, method="GET", path="/v1/submissions/pool", body=b""
    )
    resp = await client.get("/v1/submissions/pool", headers=hdrs)
    assert resp.status_code in (401, 403)


async def test_pool_accepts_internal_token(client):
    resp = await client.get(
        "/v1/submissions/pool",
        headers={"Authorization": "Bearer internal-token"},
    )
    assert resp.status_code == 200


async def test_get_submission_returns_own(client, identities):
    signer_a = identities["miner"]["signer"]
    sub_id = _make_submission(signer_a.hotkey)

    hdrs = signed_headers(
        signer_a, method="GET", path=f"/v1/submissions/{sub_id}", body=b""
    )
    resp = await client.get(f"/v1/submissions/{sub_id}", headers=hdrs)
    assert resp.status_code == 200


async def test_get_submission_denies_other(client, identities):
    signer_a = identities["miner"]["signer"]
    signer_b = identities["validator-1"]["signer"]
    sub_id = _make_submission(signer_a.hotkey)

    hdrs = signed_headers(
        signer_b, method="GET", path=f"/v1/submissions/{sub_id}", body=b""
    )
    resp = await client.get(f"/v1/submissions/{sub_id}", headers=hdrs)
    assert resp.status_code == 403


async def test_artifact_denies_validator(client, identities):
    # Validators no longer need miner source — owner-api builds the
    # miner pod itself via the internal-token path. Only the submitter
    # may pull their own archive.
    signer_a = identities["miner"]["signer"]
    signer_b = identities["validator-1"]["signer"]
    sub_id = _make_submission(signer_a.hotkey)

    hdrs = signed_headers(
        signer_b, method="GET", path=f"/v1/submissions/{sub_id}/artifact", body=b""
    )
    resp = await client.get(f"/v1/submissions/{sub_id}/artifact", headers=hdrs)
    assert resp.status_code == 403


async def test_artifact_allows_owner(client, identities):
    signer_a = identities["miner"]["signer"]
    sub_id = _make_submission(signer_a.hotkey)

    hdrs = signed_headers(
        signer_a, method="GET", path=f"/v1/submissions/{sub_id}/artifact", body=b""
    )
    resp = await client.get(f"/v1/submissions/{sub_id}/artifact", headers=hdrs)
    assert resp.status_code == 200


async def test_scorecards_denies_validator(client, identities):
    signer_a = identities["miner"]["signer"]
    signer_b = identities["validator-1"]["signer"]
    sub_id = _make_submission(signer_a.hotkey)

    hdrs = signed_headers(
        signer_b, method="GET", path=f"/v1/submissions/{sub_id}/scorecards", body=b""
    )
    resp = await client.get(f"/v1/submissions/{sub_id}/scorecards", headers=hdrs)
    assert resp.status_code == 403


# -- Fix 2: Rate limiting -------------------------------------------------


async def test_submission_rate_limit_blocks_over_threshold(
    client, identities, monkeypatch
):
    monkeypatch.setenv("EIREL_SUBMISSION_RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("EIREL_SUBMISSION_RATE_LIMIT_WINDOW_SECONDS", "3600")

    signer = identities["miner"]["signer"]
    hotkey = signer.hotkey
    archive = make_submission_archive()
    boundary = "ratelimitboundary"
    body = _multipart_body(archive, boundary)

    # Pre-fill the rate limiter for this hotkey so the next call triggers 429.
    limiter = app.state.submission_rate_limiter
    limiter._hits.pop(hotkey, None)  # ensure clean state
    limiter.max_requests = 2
    limiter.check(hotkey)  # hit 1
    limiter.check(hotkey)  # hit 2

    hdrs = signed_headers(signer, method="POST", path="/v1/submissions", body=body)
    hdrs["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    resp = await client.post("/v1/submissions", content=body, headers=hdrs)
    assert resp.status_code == 429


async def test_submission_rate_limit_per_hotkey(client, identities, monkeypatch):
    monkeypatch.setenv("EIREL_SUBMISSION_RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setenv("EIREL_SUBMISSION_RATE_LIMIT_WINDOW_SECONDS", "3600")

    limiter = app.state.submission_rate_limiter
    limiter.max_requests = 1

    signer_a = identities["miner"]["signer"]
    signer_b = identities["validator-1"]["signer"]
    archive = make_submission_archive()
    boundary = "perhkboundary"
    body = _multipart_body(archive, boundary)

    # Miner A uses their one allowed slot
    limiter._hits.pop(signer_a.hotkey, None)
    limiter._hits.pop(signer_b.hotkey, None)
    limiter.check(signer_a.hotkey)  # fills miner A's quota

    # Miner A should now be rate-limited
    hdrs_a = signed_headers(signer_a, method="POST", path="/v1/submissions", body=body)
    hdrs_a["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    resp_a = await client.post("/v1/submissions", content=body, headers=hdrs_a)
    assert resp_a.status_code == 429

    # Miner B should still be allowed (per-hotkey isolation).
    # Pre-create a run so create_submission can proceed past the limiter.
    _ensure_run()
    hdrs_b = signed_headers(signer_b, method="POST", path="/v1/submissions", body=body)
    hdrs_b["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    resp_b = await client.post("/v1/submissions", content=body, headers=hdrs_b)
    assert resp_b.status_code != 429


# -- Fix 4: Mandatory Redis in production ---------------------------------


async def test_production_requires_redis_url(monkeypatch, tmp_path):
    monkeypatch.setenv("LAUNCH_MODE", "production")
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.setenv(
        "DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    )

    from shared.common.config import reset_settings

    reset_settings()

    with pytest.raises(RuntimeError, match="REDIS_URL is required"):
        async with app.router.lifespan_context(app):
            pass
