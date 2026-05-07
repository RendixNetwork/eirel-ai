"""Tests for the server-attested tool-call ledger endpoints.

Two routes:
  * POST /v1/internal/eval/tool_calls  — write one row
  * GET  /v1/internal/eval/job_ledger  — read all rows for a job_id
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from shared.common.database import Database
from shared.common.models import OrchestratorToolCallLog
from control_plane.owner_api.routers.internal_eval import (
    JobLedgerResponse,
    ToolCallLogWriteRequest,
    read_job_ledger,
    write_tool_call,
)


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'ledger.db'}")
    db.create_all()
    return db


def _make_request(*, db) -> SimpleNamespace:
    services = SimpleNamespace(db=db)
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(services=services)),
        headers={},
    )


# -- write_tool_call --------------------------------------------------------


async def test_write_tool_call_persists_row(tmp_path):
    db = _make_db(tmp_path)
    req = _make_request(db=db)
    body = ToolCallLogWriteRequest(
        job_id="job-1",
        tool_name="web_search",
        args_hash="deadbeef" * 8,
        args_json={"query": "claude code", "top_k": 5},
        result_digest="brave: 3 results",
        latency_ms=120,
        cost_usd=0.003,
        status="ok",
    )
    response = await write_tool_call(req, body, _token=None)
    assert response.job_id == "job-1"
    assert response.tool_name == "web_search"
    assert response.args_json == {"query": "claude code", "top_k": 5}

    with db.sessionmaker() as session:
        rows = session.query(OrchestratorToolCallLog).all()
        assert len(rows) == 1
        assert rows[0].job_id == "job-1"
        assert rows[0].cost_usd == 0.003


async def test_write_tool_call_records_error_when_status_failed(tmp_path):
    db = _make_db(tmp_path)
    req = _make_request(db=db)
    body = ToolCallLogWriteRequest(
        job_id="job-2",
        tool_name="url_fetch",
        args_json={"url": "https://example.com"},
        status="error",
        error="upstream timeout",
    )
    response = await write_tool_call(req, body, _token=None)
    assert response.status == "error"
    assert response.error == "upstream timeout"


# -- read_job_ledger --------------------------------------------------------


async def test_read_job_ledger_returns_all_rows_for_job(tmp_path):
    db = _make_db(tmp_path)
    req = _make_request(db=db)
    for i, tool in enumerate(["web_search", "url_fetch", "sandbox"]):
        body = ToolCallLogWriteRequest(
            job_id="job-X",
            tool_name=tool,
            args_json={"i": i},
            cost_usd=0.001 * (i + 1),
        )
        await write_tool_call(req, body, _token=None)

    response: JobLedgerResponse = await read_job_ledger(
        req, job_id="job-X", validator_hotkey="hk_validator",
    )
    assert response.job_id == "job-X"
    assert response.n_calls == 3
    tools = [r.tool_name for r in response.tool_calls]
    assert tools == ["web_search", "url_fetch", "sandbox"]


async def test_read_job_ledger_isolates_by_job_id(tmp_path):
    db = _make_db(tmp_path)
    req = _make_request(db=db)
    for job_id in ("alpha", "beta"):
        body = ToolCallLogWriteRequest(
            job_id=job_id, tool_name="web_search", args_json={},
        )
        await write_tool_call(req, body, _token=None)

    alpha = await read_job_ledger(req, job_id="alpha", validator_hotkey="hk_validator")
    beta = await read_job_ledger(req, job_id="beta", validator_hotkey="hk_validator")
    assert alpha.n_calls == 1
    assert beta.n_calls == 1
    assert alpha.tool_calls[0].job_id == "alpha"
    assert beta.tool_calls[0].job_id == "beta"


async def test_read_job_ledger_returns_empty_for_unknown_job(tmp_path):
    db = _make_db(tmp_path)
    req = _make_request(db=db)
    response = await read_job_ledger(
        req, job_id="never-existed", validator_hotkey="hk_validator",
    )
    assert response.n_calls == 0
    assert response.tool_calls == []
