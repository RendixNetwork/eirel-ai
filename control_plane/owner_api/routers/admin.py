from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from shared.common.models import (
    EvaluationRun,
    ManagedDeployment,
    ManagedMinerSubmission,
    RegisteredNeuron,
    ValidatorRecord,
)
from control_plane.owner_api.dependencies import require_owner_signature
from control_plane.owner_api.managed import ManagedOwnerServices

logger = logging.getLogger(__name__)

router = APIRouter(tags=["admin"])


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ------------------------------------------------------------------
# Validator whitelist
# ------------------------------------------------------------------


class AddValidatorRequest(BaseModel):
    hotkey: str = Field(min_length=40, max_length=128)
    uid: int = 0
    stake: int = 0


@router.post("/v1/admin/validators")
async def admin_add_validator(
    request: Request,
    payload: AddValidatorRequest,
    _owner: str = Depends(require_owner_signature),
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        existing = session.get(ValidatorRecord, payload.hotkey)
        if existing is not None:
            existing.uid = payload.uid
            existing.stake = payload.stake
            existing.is_active = True
            existing.last_synced_at = _utcnow()
        else:
            session.add(
                ValidatorRecord(
                    hotkey=payload.hotkey,
                    uid=payload.uid,
                    stake=payload.stake,
                    is_active=True,
                    last_synced_at=_utcnow(),
                )
            )
        session.commit()
    logger.info("admin: added validator %s", payload.hotkey[:16])
    return {"status": "added", "hotkey": payload.hotkey}


@router.delete("/v1/admin/validators/{hotkey}")
async def admin_remove_validator(
    request: Request,
    hotkey: str,
    _owner: str = Depends(require_owner_signature),
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        existing = session.get(ValidatorRecord, hotkey)
        if existing is None:
            raise HTTPException(status_code=404, detail="validator not found")
        session.delete(existing)
        session.commit()
    logger.info("admin: removed validator %s", hotkey[:16])
    return {"status": "removed", "hotkey": hotkey}


@router.get("/v1/admin/validators")
async def admin_list_validators(
    request: Request,
    _owner: str = Depends(require_owner_signature),
) -> list[dict[str, Any]]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        rows = list(session.execute(select(ValidatorRecord).order_by(ValidatorRecord.uid)).scalars())
    return [
        {
            "hotkey": v.hotkey,
            "uid": v.uid,
            "stake": v.stake,
            "is_active": v.is_active,
            "last_synced_at": v.last_synced_at.isoformat(),
        }
        for v in rows
    ]


class SetValidatorActiveRequest(BaseModel):
    is_active: bool


@router.patch("/v1/admin/validators/{hotkey}")
async def admin_set_validator_active(
    request: Request,
    hotkey: str,
    payload: SetValidatorActiveRequest,
    _owner: str = Depends(require_owner_signature),
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        existing = session.get(ValidatorRecord, hotkey)
        if existing is None:
            raise HTTPException(status_code=404, detail="validator not found")
        existing.is_active = payload.is_active
        existing.last_synced_at = _utcnow()
        session.commit()
    return {"status": "updated", "hotkey": hotkey, "is_active": payload.is_active}


# ------------------------------------------------------------------
# Runs (read-only)
# ------------------------------------------------------------------


@router.get("/v1/admin/runs")
async def admin_list_runs(
    request: Request,
    _owner: str = Depends(require_owner_signature),
) -> list[dict[str, Any]]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        rows = list(
            session.execute(
                select(EvaluationRun).order_by(EvaluationRun.sequence.desc()).limit(20)
            ).scalars()
        )
    return [
        {
            "id": r.id,
            "sequence": r.sequence,
            "status": r.status,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "ends_at": r.ends_at.isoformat() if r.ends_at else None,
            "closed_at": r.closed_at.isoformat() if getattr(r, "closed_at", None) else None,
        }
        for r in rows
    ]


@router.get("/v1/admin/runs/current")
async def admin_current_run(
    request: Request,
    _owner: str = Depends(require_owner_signature),
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        run = services.runs.current_run(session)
        if run is None:
            return {"status": "no_open_run"}
        return {
            "id": run.id,
            "sequence": run.sequence,
            "status": run.status,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "ends_at": run.ends_at.isoformat() if run.ends_at else None,
        }


async def _reset_tool_service_usage(services: ManagedOwnerServices) -> None:
    """Best-effort POST to the web-search tool service's /v1/usage/reset.

    Failure is logged, never raised — a reset miss is far less bad than
    blocking the run advance.
    """
    base_url = (services.settings.web_search_tool_service_url or "").rstrip("/")
    if not base_url:
        return
    token = services.settings.web_search_tool_service_token or ""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"{base_url}/v1/usage/reset"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, headers=headers)
            resp.raise_for_status()
            body = resp.json() if resp.content else {}
            logger.info(
                "admin: web-search tool usage reset (cleared_job_count=%s)",
                body.get("cleared_job_count"),
            )
    except Exception as exc:
        logger.warning("admin: web-search tool usage reset failed: %s", exc)


@router.post("/v1/admin/runs/advance")
async def admin_advance_run(
    request: Request,
    _owner: str = Depends(require_owner_signature),
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        previous = services.runs.current_run(session)
        previous_id = previous.id if previous else None
        previous_sequence = previous.sequence if previous else None
        if previous is not None:
            # Set ends_at to the past so the rotation branch in
            # ensure_current_run fires.
            previous.ends_at = _utcnow()
            previous.updated_at = _utcnow()
            session.commit()
    # Use the async wrapper so queued deployments buffered by
    # start_queued_deployments actually get scheduled (the sync
    # runs.ensure_current_run only transitions DB rows to status=received +
    # placement_status=pending; the runtime pods are spun up here).
    new_run = await services.deployments.ensure_current_run_and_reconcile()
    if previous_id is not None and new_run.id == previous_id:
        raise HTTPException(
            status_code=500,
            detail="advance failed: ensure_current_run did not rotate the run",
        )
    logger.info(
        "admin: advanced run %s (seq %s) → %s (seq %s)",
        previous_id, previous_sequence, new_run.id, new_run.sequence,
    )
    # Reset the web-search tool service's per-job request budgets so each
    # miner deployment starts the new run with a fresh quota. Miners use a
    # sticky job_id (miner-<deployment_id>) for cost attribution, so
    # without this periodic reset their counters grow monotonically and
    # eventually 429 for the rest of the pod's lifetime.
    await _reset_tool_service_usage(services)
    return {
        "result": "advanced",
        "previous_run_id": previous_id,
        "previous_sequence": previous_sequence,
        "id": new_run.id,
        "sequence": new_run.sequence,
        "status": new_run.status,
        "started_at": new_run.started_at.isoformat() if new_run.started_at else None,
        "ends_at": new_run.ends_at.isoformat() if new_run.ends_at else None,
    }


# ------------------------------------------------------------------
# Metagraph (read-only; sync is on the listener service)
# ------------------------------------------------------------------


@router.get("/v1/admin/neurons")
async def admin_list_neurons(
    request: Request,
    _owner: str = Depends(require_owner_signature),
) -> list[dict[str, Any]]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        rows = list(
            session.execute(select(RegisteredNeuron).order_by(RegisteredNeuron.uid)).scalars()
        )
    return [
        {"hotkey": n.hotkey, "uid": n.uid, "last_synced_at": n.last_synced_at.isoformat()}
        for n in rows
    ]


# ------------------------------------------------------------------
# Submissions / deployments (read-only; full details via /v1/submissions/pool)
# ------------------------------------------------------------------


@router.get("/v1/admin/submissions")
async def admin_list_submissions(
    request: Request,
    limit: int = 50,
    _owner: str = Depends(require_owner_signature),
) -> list[dict[str, Any]]:
    services: ManagedOwnerServices = request.app.state.services
    limit = max(1, min(200, limit))
    with services.db.sessionmaker() as session:
        rows = list(
            session.execute(
                select(ManagedMinerSubmission)
                .order_by(ManagedMinerSubmission.created_at.desc())
                .limit(limit)
            ).scalars()
        )
    return [
        {
            "id": s.id,
            "miner_hotkey": s.miner_hotkey,
            "submission_seq": s.submission_seq,
            "family_id": s.family_id,
            "status": s.status,
            "introduced_run_id": s.introduced_run_id,
            "created_at": s.created_at.isoformat(),
        }
        for s in rows
    ]


@router.get("/v1/admin/deployments")
async def admin_list_deployments(
    request: Request,
    limit: int = 50,
    _owner: str = Depends(require_owner_signature),
) -> list[dict[str, Any]]:
    services: ManagedOwnerServices = request.app.state.services
    limit = max(1, min(200, limit))
    with services.db.sessionmaker() as session:
        rows = list(
            session.execute(
                select(ManagedDeployment)
                .order_by(ManagedDeployment.updated_at.desc())
                .limit(limit)
            ).scalars()
        )
    return [
        {
            "id": d.id,
            "submission_id": d.submission_id,
            "miner_hotkey": d.miner_hotkey,
            "family_id": d.family_id,
            "status": d.status,
            "health_status": d.health_status,
            "placement_status": d.placement_status,
            "assigned_node_name": d.assigned_node_name,
            "is_active": d.is_active,
            "updated_at": d.updated_at.isoformat(),
        }
        for d in rows
    ]


# ------------------------------------------------------------------
# Identity / config
# ------------------------------------------------------------------


@router.get("/v1/admin/whoami")
async def admin_whoami(
    request: Request,
    owner_hotkey: str = Depends(require_owner_signature),
) -> dict[str, Any]:
    """Lightweight check that the caller is recognised as the owner."""
    services: ManagedOwnerServices = request.app.state.services
    return {
        "role": "owner",
        "hotkey": owner_hotkey,
        "network": services.settings.bittensor_network,
        "netuid": services.settings.bittensor_netuid,
    }
