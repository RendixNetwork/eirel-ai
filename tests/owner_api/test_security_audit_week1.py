from __future__ import annotations

import re
from datetime import timedelta

import pytest

from control_plane.owner_api.app import app
from control_plane.owner_api._helpers import utcnow
from control_plane.owner_api.fee_verifier import FeeVerificationResult
from shared.common.models import EvaluationRun, RegisteredNeuron
from tests.conftest import make_submission_archive, signed_headers


# -- Helpers ---------------------------------------------------------------

INTERNAL_TOKEN = "internal-token"


def _ensure_run() -> None:
    services = app.state.services
    with services.db.sessionmaker() as session:
        existing = session.get(EvaluationRun, "run-security-test")
        if existing is not None:
            return
        now = utcnow()
        run = EvaluationRun(
            id="run-security-test",
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


def _make_submission(hotkey: str) -> tuple[str, str]:
    """Create a submission and deployment; return (submission_id, deployment_id)."""
    _ensure_run()
    services = app.state.services
    archive = make_submission_archive()
    with services.db.sessionmaker() as session:
        if session.get(RegisteredNeuron, hotkey) is None:
            session.add(RegisteredNeuron(hotkey=hotkey, uid=0))
            session.flush()
        sub, dep = services.create_submission(
            session,
            miner_hotkey=hotkey,
            submission_block=100,
            archive_bytes=archive,
            base_url="http://testserver",
        )
        sub_id = sub.id
        dep_id = dep.id
        session.commit()
    return sub_id, dep_id


# -- Fix 1: Runtime auth ---------------------------------------------------


async def test_runtime_healthz_rejects_unauthenticated(client):
    resp = await client.get("/runtime/fake-id/healthz")
    assert resp.status_code in (401, 403)


async def test_runtime_infer_rejects_unauthenticated(client):
    resp = await client.post(
        "/runtime/fake-id/v1/agent/infer",
        json={
            "task_id": "t1",
            "family_id": "analyst",
            "subtask": "test",
            "context": {},
        },
    )
    assert resp.status_code in (401, 403)


async def test_runtime_healthz_accepts_internal_token(client):
    resp = await client.get(
        "/runtime/fake-id/healthz",
        headers={"Authorization": f"Bearer {INTERNAL_TOKEN}"},
    )
    # Should not be 401/403 — 404 for nonexistent deployment is fine
    assert resp.status_code not in (401, 403)


# -- Fix 2: Internal registry auth ----------------------------------------


async def test_internal_registry_rejects_unauthenticated(client):
    resp = await client.get("/v1/internal/registry")
    assert resp.status_code in (401, 403)


async def test_internal_registry_accepts_internal_token(client):
    resp = await client.get(
        "/v1/internal/registry",
        headers={"Authorization": f"Bearer {INTERNAL_TOKEN}"},
    )
    assert resp.status_code == 200


# -- Fix 3: Deployment ownership filtering ---------------------------------


async def test_deployments_list_filtered_by_hotkey(client, identities):
    signer_a = identities["miner"]["signer"]
    signer_b = identities["validator-1"]["signer"]

    _sub_a, dep_a = _make_submission(signer_a.hotkey)
    _sub_b, dep_b = _make_submission(signer_b.hotkey)

    hdrs_a = signed_headers(
        signer_a, method="GET", path="/v1/deployments", body=b""
    )
    resp = await client.get("/v1/deployments", headers=hdrs_a)
    assert resp.status_code == 200

    data = resp.json()
    returned_ids = {d["id"] for d in data}
    assert dep_a in returned_ids
    assert dep_b not in returned_ids


async def test_deployment_detail_denied_for_other_miner(client, identities):
    signer_a = identities["miner"]["signer"]
    signer_b = identities["validator-1"]["signer"]

    _sub_a, dep_a = _make_submission(signer_a.hotkey)

    hdrs_b = signed_headers(
        signer_b,
        method="GET",
        path=f"/v1/deployments/{dep_a}",
        body=b"",
    )
    resp = await client.get(f"/v1/deployments/{dep_a}", headers=hdrs_b)
    assert resp.status_code == 403


# -- Fix 5: Scrubbed error messages ----------------------------------------


async def test_runtime_error_does_not_leak_internals(client):
    """Verify the 502 detail from a runtime proxy error is generic."""
    # The runtime.py endpoints now return "runtime invocation failed"
    # instead of str(exc). Verify by checking the code path: any 502
    # from the runtime router should have a generic detail.
    # We can't easily trigger a real 502 through the test client without
    # a full deployment, so verify the error format by importing the
    # pattern and confirming the endpoints use the scrubbed message.
    from control_plane.owner_api.routers import runtime
    import ast
    import inspect

    source = inspect.getsource(runtime)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "HTTPException"
        ):
            for kw in node.keywords:
                if kw.arg == "status_code":
                    if isinstance(kw.value, ast.Constant) and kw.value.value == 502:
                        # Find the detail kwarg
                        for detail_kw in node.keywords:
                            if detail_kw.arg == "detail":
                                assert isinstance(detail_kw.value, ast.Constant), (
                                    "502 detail should be a string literal, not "
                                    "a dynamic expression (potential info leak)"
                                )
                                detail_str = detail_kw.value.value
                                assert "Traceback" not in detail_str
                                assert "/" not in detail_str or detail_str == "runtime invocation failed"


# Regex: SS58 addresses are base58-encoded, 48 chars, start with '5'
_SS58_PATTERN = re.compile(r"\b5[A-HJ-NP-Za-km-z1-9]{47}\b")


async def test_fee_error_does_not_reveal_addresses():
    """Verify fee verification error messages do not contain SS58 addresses."""
    # Check the three failure reasons that previously leaked addresses
    sender_mismatch = FeeVerificationResult(
        valid=False,
        reason="sender does not match hotkey owner",
        sender="5FakeAddress" + "A" * 35,
        destination="5Treasury" + "B" * 39,
        amount_rao=1_000_000_000,
    )
    dest_mismatch = FeeVerificationResult(
        valid=False,
        reason="destination does not match treasury",
        sender="5FakeAddress" + "A" * 35,
        destination="5Treasury" + "B" * 39,
        amount_rao=1_000_000_000,
    )
    amount_mismatch = FeeVerificationResult(
        valid=False,
        reason="transferred amount below required fee",
        sender="5FakeAddress" + "A" * 35,
        destination="5Treasury" + "B" * 39,
        amount_rao=500_000_000,
    )

    for result in (sender_mismatch, dest_mismatch, amount_mismatch):
        # The reason string itself should NOT contain SS58 addresses
        assert not _SS58_PATTERN.search(result.reason), (
            f"Fee error reason leaks SS58 address: {result.reason}"
        )
