from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select

from shared.common.models import ManagedDeployment
from control_plane.owner_api.dependencies import signature_dependency
from control_plane.owner_api.managed import ManagedOwnerServices

router = APIRouter(tags=["deployments"])


@router.get("/v1/deployments")
async def list_deployments(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    hotkey: str = Depends(signature_dependency),
) -> list[dict[str, Any]]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        return [
            services.deployment_payload(item)
            for item in session.execute(
                select(ManagedDeployment)
                .where(ManagedDeployment.miner_hotkey == hotkey)
                .order_by(ManagedDeployment.created_at.desc())
                .offset(offset)
                .limit(limit)
            ).scalars()
        ]


@router.get("/v1/deployments/{deployment_id}")
async def get_deployment(
    request: Request,
    deployment_id: str,
    hotkey: str = Depends(signature_dependency),
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        if deployment.miner_hotkey != hotkey:
            raise HTTPException(status_code=403, detail="access denied")
        return services.deployment_payload(deployment) or {}


@router.get("/v1/deployments/{deployment_id}/health-history")
async def deployment_health_history(
    request: Request,
    deployment_id: str,
    hotkey: str = Depends(signature_dependency),
) -> list[dict[str, Any]]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        return services.list_health_events(session, deployment_id=deployment_id)
