from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from eirel.groups import ensure_family_id
from control_plane.owner_api.dashboard import queries
from control_plane.owner_api.dashboard.cache import TTLCache
from control_plane.owner_api.dashboard.schemas import (
    FamiliesResponse,
    LeaderboardResponse,
    MinerProfileResponse,
    MinerRunsResponse,
    OverviewResponse,
    RunDetailResponse,
    RunListResponse,
)
from control_plane.owner_api.managed import ManagedOwnerServices


router = APIRouter(tags=["dashboard"], prefix="/api/v1/dashboard")


# Process-local response cache. Closed runs get 30s TTL; anything involving
# the open run falls back to 5s TTL so the live view stays fresh.
_CACHE: TTLCache[object] = TTLCache(default_ttl_seconds=30.0)
_OPEN_RUN_TTL = 5.0
_CLOSED_RUN_TTL = 30.0


def _validate_family_id(family_id: str) -> str:
    try:
        return ensure_family_id(family_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _ttl_for(response: object) -> float:
    # LeaderboardResponse with is_running on any entry → short TTL.
    entries = getattr(response, "entries", None)
    if entries and any(getattr(e, "is_running", False) for e in entries):
        return _OPEN_RUN_TTL
    # RunDetailResponse or OverviewResponse for an open run.
    if getattr(response, "status", None) == "open":
        return _OPEN_RUN_TTL
    if getattr(response, "current_run_status", None) == "open":
        return _OPEN_RUN_TTL
    # LeaderboardResponse targeting an open run.
    if getattr(response, "run_status", None) == "open":
        return _OPEN_RUN_TTL
    return _CLOSED_RUN_TTL


@router.get("/overview", response_model=OverviewResponse)
async def get_overview(request: Request) -> OverviewResponse:
    services: ManagedOwnerServices = request.app.state.services
    key = ("overview",)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    with services.db.sessionmaker() as session:
        result = queries.fetch_overview(session, services=services)
    _CACHE.set(key, result, ttl=_ttl_for(result))
    return result


@router.get("/families", response_model=FamiliesResponse)
async def get_families(request: Request) -> FamiliesResponse:
    services: ManagedOwnerServices = request.app.state.services
    key = ("families",)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    with services.db.sessionmaker() as session:
        result = queries.fetch_families(session, services=services)
    _CACHE.set(key, result, ttl=_CLOSED_RUN_TTL)
    return result


@router.get("/runs", response_model=RunListResponse)
async def get_runs(request: Request) -> RunListResponse:
    services: ManagedOwnerServices = request.app.state.services
    key = ("runs",)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    with services.db.sessionmaker() as session:
        result = queries.fetch_runs(session, services=services)
    # Short TTL if any run is open, otherwise 30s.
    ttl = _OPEN_RUN_TTL if any(r.status == "open" for r in result.runs) else _CLOSED_RUN_TTL
    _CACHE.set(key, result, ttl=ttl)
    return result


@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(
    request: Request,
    family_id: str = Query(...),
    run_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> LeaderboardResponse:
    family_id = _validate_family_id(family_id)
    services: ManagedOwnerServices = request.app.state.services
    key = ("leaderboard", family_id, run_id or "latest", limit, offset)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    with services.db.sessionmaker() as session:
        result = queries.fetch_leaderboard(
            session,
            services=services,
            family_id=family_id,
            run_id=run_id,
            limit=limit,
            offset=offset,
        )
    _CACHE.set(key, result, ttl=_ttl_for(result))
    return result


@router.get("/miners/{hotkey}", response_model=MinerProfileResponse)
async def get_miner_profile(
    request: Request,
    hotkey: str,
    family_id: str = Query(...),
) -> MinerProfileResponse:
    family_id = _validate_family_id(family_id)
    services: ManagedOwnerServices = request.app.state.services
    key = ("miner_profile", hotkey, family_id)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    with services.db.sessionmaker() as session:
        result = queries.fetch_miner_profile(
            session, services=services, hotkey=hotkey, family_id=family_id,
        )
    _CACHE.set(key, result, ttl=_OPEN_RUN_TTL)  # profile reflects live rank; keep fresh.
    return result


@router.get("/miners/{hotkey}/runs", response_model=MinerRunsResponse)
async def get_miner_runs(
    request: Request,
    hotkey: str,
    family_id: str = Query(...),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> MinerRunsResponse:
    family_id = _validate_family_id(family_id)
    services: ManagedOwnerServices = request.app.state.services
    key = ("miner_runs", hotkey, family_id, limit, offset)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    with services.db.sessionmaker() as session:
        result = queries.fetch_miner_runs(
            session,
            services=services,
            hotkey=hotkey,
            family_id=family_id,
            limit=limit,
            offset=offset,
        )
    _CACHE.set(key, result, ttl=_CLOSED_RUN_TTL)
    return result


@router.get("/miners/{hotkey}/runs/{run_id}", response_model=RunDetailResponse)
async def get_miner_run_detail(
    request: Request,
    hotkey: str,
    run_id: str,
    family_id: str = Query(...),
) -> RunDetailResponse:
    family_id = _validate_family_id(family_id)
    services: ManagedOwnerServices = request.app.state.services
    key = ("run_detail", hotkey, family_id, run_id)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    with services.db.sessionmaker() as session:
        try:
            result = queries.fetch_run_detail(
                session,
                services=services,
                hotkey=hotkey,
                family_id=family_id,
                run_id=run_id,
            )
        except HTTPException:
            raise
    _CACHE.set(key, result, ttl=_ttl_for(result))
    return result


def _reset_cache_for_tests() -> None:
    _CACHE.clear()
