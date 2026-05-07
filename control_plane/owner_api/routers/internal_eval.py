"""HTTP API for the server-attested tool-call ledger and
per-(run, miner, task) EvalFeedback persistence.

Tool services (web_search, url_fetch, sandbox, rag) POST one row per
call to the ledger via the internal-token write path; the validator
GETs the unified ledger for a job (hotkey-signed) to compute tool-use
KPIs without trusting miner-emitted trace frames.

The EvalFeedback row is written *server-side* as a side-effect of the
validator's task-result POST — there is no separate write endpoint.
Miners read their own feedback rows via the hotkey-signed GET below;
the signing hotkey must equal the requested ``miner_hotkey`` (or be
omitted, in which case it's derived from the signature).

Routes:

    POST  /v1/internal/eval/tool_calls   (internal token)
        Tool-platform services write one ledger row each. Fire-and-forget.

    GET   /v1/internal/eval/job_ledger?job_id=...   (validator hotkey)
        Validator reads the unified ledger for a job to compute the
        composite ``tool_attestation`` factor.

    GET   /v1/eval/feedback?run_id=...   (miner hotkey)
        Miner reads their own per-(run, miner, task) feedback rows.
        Signing hotkey is the miner_hotkey filter — no proxy or
        internal-token leakage to validators.

Note: prepared-task distribution to validators reads the bundle from R2
by convention (``s3://${EIREL_EVAL_POOL_BUCKET}/${family_id}/pool-run-${run_id}.json``)
and serves through ``/v1/families/{family_id}/tasks/claim``. This router
only owns the orthogonal tool-call attestation ledger and EvalFeedback
read path.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from control_plane.owner_api.dependencies import (
    require_internal_service_token,
    signature_dependency,
    validator_dependency,
)
from control_plane.owner_api.managed import ManagedOwnerServices
from shared.common.models import EvalFeedback, OrchestratorToolCallLog

logger = logging.getLogger(__name__)

router = APIRouter(tags=["internal_eval"])


class ToolCallLogWriteRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=64)
    tool_name: str = Field(min_length=1, max_length=64)
    args_hash: str = Field(default="", max_length=64)
    args_json: dict[str, Any] = Field(default_factory=dict)
    result_digest: str = Field(default="")
    latency_ms: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    status: str = Field(default="ok", max_length=16)
    error: str | None = Field(default=None)


class ToolCallLogRow(BaseModel):
    id: str
    job_id: str
    tool_name: str
    args_hash: str
    args_json: dict[str, Any]
    result_digest: str
    latency_ms: int
    cost_usd: float
    status: str
    error: str | None
    ts: datetime


class JobLedgerResponse(BaseModel):
    job_id: str
    n_calls: int
    tool_calls: list[ToolCallLogRow]


def _row_to_model(row: OrchestratorToolCallLog) -> ToolCallLogRow:
    return ToolCallLogRow(
        id=row.id,
        job_id=row.job_id,
        tool_name=row.tool_name,
        args_hash=row.args_hash,
        args_json=dict(row.args_json or {}),
        result_digest=row.result_digest or "",
        latency_ms=row.latency_ms,
        cost_usd=row.cost_usd,
        status=row.status,
        error=row.error,
        ts=row.ts,
    )


@router.post(
    "/v1/internal/eval/tool_calls",
    response_model=ToolCallLogRow,
    status_code=201,
)
async def write_tool_call(
    request: Request,
    body: ToolCallLogWriteRequest,
    _token: None = Depends(require_internal_service_token),
) -> ToolCallLogRow:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        row = OrchestratorToolCallLog(
            job_id=body.job_id,
            tool_name=body.tool_name,
            args_hash=body.args_hash,
            args_json=body.args_json,
            result_digest=body.result_digest,
            latency_ms=body.latency_ms,
            cost_usd=body.cost_usd,
            status=body.status,
            error=body.error,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return _row_to_model(row)


@router.get(
    "/v1/internal/eval/job_ledger",
    response_model=JobLedgerResponse,
)
async def read_job_ledger(
    request: Request,
    job_id: str = Query(min_length=1, max_length=64),
    validator_hotkey: str = Depends(validator_dependency),
) -> JobLedgerResponse:
    del validator_hotkey  # auth-only — registered active validators may read
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        rows = list(
            session.scalars(
                select(OrchestratorToolCallLog)
                .where(OrchestratorToolCallLog.job_id == job_id)
                .order_by(OrchestratorToolCallLog.ts)
            )
        )
    if not rows:
        # Empty ledger is a valid answer (job had no tool calls).
        return JobLedgerResponse(job_id=job_id, n_calls=0, tool_calls=[])
    return JobLedgerResponse(
        job_id=job_id,
        n_calls=len(rows),
        tool_calls=[_row_to_model(r) for r in rows],
    )


# -- EvalFeedback (read-only — rows are written server-side by the
# evaluation_task_manager when accepting the validator's task-result
# POST; see ``_upsert_eval_feedback``)
# -----------------------------------------------------------------------


class EvalFeedbackRow(BaseModel):
    id: str
    run_id: str
    miner_hotkey: str
    task_id: str
    outcome: str
    failure_mode: str | None
    guidance: str
    prompt_excerpt: str
    response_excerpt: str
    composite_score: float
    knockout_reasons: list[str]
    oracle_status: str | None
    created_at: datetime


class EvalFeedbackListResponse(BaseModel):
    run_id: str
    miner_hotkey: str
    n_items: int
    items: list[EvalFeedbackRow]


def _eval_feedback_row_to_model(row: EvalFeedback) -> EvalFeedbackRow:
    knockout_reasons = row.knockout_reasons_json
    if not isinstance(knockout_reasons, list):
        knockout_reasons = []
    return EvalFeedbackRow(
        id=row.id,
        run_id=row.run_id,
        miner_hotkey=row.miner_hotkey,
        task_id=row.task_id,
        outcome=row.outcome,
        failure_mode=row.failure_mode,
        guidance=row.guidance or "",
        prompt_excerpt=row.prompt_excerpt or "",
        response_excerpt=row.response_excerpt or "",
        composite_score=float(row.composite_score),
        knockout_reasons=[str(x) for x in knockout_reasons],
        oracle_status=row.oracle_status,
        created_at=row.created_at,
    )


@router.get(
    "/v1/eval/feedback",
    response_model=EvalFeedbackListResponse,
)
async def read_eval_feedback(
    request: Request,
    run_id: str = Query(min_length=1, max_length=64),
    hotkey: str = Depends(signature_dependency),
) -> EvalFeedbackListResponse:
    """Miner reads their own EvalFeedback rows for a run.

    The ``miner_hotkey`` filter is **derived from the signature** —
    callers cannot read another miner's feedback by passing a different
    hotkey in a query param. No proxy / no internal token.
    """
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        rows = list(
            session.scalars(
                select(EvalFeedback)
                .where(
                    EvalFeedback.run_id == run_id,
                    EvalFeedback.miner_hotkey == hotkey,
                )
                .order_by(EvalFeedback.created_at)
            )
        )
    return EvalFeedbackListResponse(
        run_id=run_id,
        miner_hotkey=hotkey,
        n_items=len(rows),
        items=[_eval_feedback_row_to_model(r) for r in rows],
    )
