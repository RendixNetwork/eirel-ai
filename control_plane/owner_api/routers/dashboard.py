from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from control_plane.owner_api.dashboard import queries
from control_plane.owner_api.dashboard.schemas import (
    FamiliesResponse,
    LeaderboardResponse,
    MinerProfileResponse,
    MinerRunsResponse,
    OverviewResponse,
    RunDetailResponse,
    WindowLiteral,
)
from control_plane.owner_api.managed import ManagedOwnerServices


router = APIRouter(tags=["dashboard"], prefix="/api/v1/dashboard")


def _not_implemented() -> HTTPException:
    return HTTPException(status_code=501, detail="dashboard endpoint not yet implemented (Phase 1b)")


@router.get("/overview", response_model=OverviewResponse)
async def get_overview(request: Request) -> OverviewResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            return queries.fetch_overview(session, services=services)
        except NotImplementedError:
            raise _not_implemented()


@router.get("/families", response_model=FamiliesResponse)
async def get_families(request: Request) -> FamiliesResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            return queries.fetch_families(session, services=services)
        except NotImplementedError:
            raise _not_implemented()


@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(
    request: Request,
    family_id: str = Query(...),
    window: WindowLiteral = Query(default="latest"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> LeaderboardResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            return queries.fetch_leaderboard(
                session,
                services=services,
                family_id=family_id,
                window=window,
                limit=limit,
                offset=offset,
            )
        except NotImplementedError:
            raise _not_implemented()


@router.get("/miners/{hotkey}", response_model=MinerProfileResponse)
async def get_miner_profile(
    request: Request,
    hotkey: str,
    family_id: str = Query(...),
) -> MinerProfileResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            return queries.fetch_miner_profile(
                session, services=services, hotkey=hotkey, family_id=family_id,
            )
        except NotImplementedError:
            raise _not_implemented()


@router.get("/miners/{hotkey}/runs", response_model=MinerRunsResponse)
async def get_miner_runs(
    request: Request,
    hotkey: str,
    family_id: str = Query(...),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> MinerRunsResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            return queries.fetch_miner_runs(
                session,
                services=services,
                hotkey=hotkey,
                family_id=family_id,
                limit=limit,
                offset=offset,
            )
        except NotImplementedError:
            raise _not_implemented()


@router.get("/miners/{hotkey}/runs/{run_id}", response_model=RunDetailResponse)
async def get_miner_run_detail(
    request: Request,
    hotkey: str,
    run_id: str,
    family_id: str = Query(...),
) -> RunDetailResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            return queries.fetch_run_detail(
                session,
                services=services,
                hotkey=hotkey,
                family_id=family_id,
                run_id=run_id,
            )
        except NotImplementedError:
            raise _not_implemented()
