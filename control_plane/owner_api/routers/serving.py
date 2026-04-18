from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select

from shared.common.models import ServingRelease
from control_plane.owner_api.dependencies import signature_dependency
from control_plane.owner_api.managed import ManagedOwnerServices

router = APIRouter(tags=["serving"])


@router.get("/v1/serving/releases")
async def list_serving_releases(
    request: Request,
    hotkey: str = Depends(signature_dependency),
) -> list[dict[str, Any]]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        releases = list(
            session.execute(select(ServingRelease).order_by(ServingRelease.created_at.desc())).scalars()
        )
        return [services.serving_release_payload(item) for item in releases]


@router.get("/v1/serving/releases/current")
async def current_serving_release(
    request: Request,
    hotkey: str = Depends(signature_dependency),
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        release = services.latest_published_release(session)
        fleet = services.current_serving_fleet(session)
        return {
            "release": services.serving_release_payload(release),
            "deployments": [services.serving_deployment_payload(item) for item in fleet],
        }


@router.get("/v1/serving/fleet")
async def serving_fleet(
    request: Request,
    hotkey: str = Depends(signature_dependency),
) -> list[dict[str, Any]]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        return [
            services.serving_deployment_payload(item)
            for item in services.current_serving_fleet(session)
        ]
