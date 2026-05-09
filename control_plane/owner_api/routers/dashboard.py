from __future__ import annotations

import io
import tarfile

from fastapi import APIRouter, HTTPException, Query, Request, Response
from sqlalchemy import select

from eirel.groups import ensure_family_id
from shared.common.models import (
    DeploymentScoreRecord,
    EvaluationRun,
    ManagedMinerSubmission,
    SubmissionArtifact,
)
from control_plane.owner_api.dashboard import queries
from control_plane.owner_api.dashboard.cache import TTLCache
from control_plane.owner_api.dashboard.schemas import (
    FamiliesResponse,
    LeaderboardResponse,
    MinerProfileResponse,
    MinerRunsResponse,
    OverviewResponse,
    QueuedSubmissionsResponse,
    RunDetailResponse,
    RunListResponse,
    SubmissionFile,
    SubmissionFilesResponse,
    ValidatorRunCostsResponse,
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


@router.get("/submissions/queued", response_model=QueuedSubmissionsResponse)
async def get_queued_submissions(
    request: Request,
    limit: int = Query(default=200, ge=1, le=500),
) -> QueuedSubmissionsResponse:
    services: ManagedOwnerServices = request.app.state.services
    key = ("queued_submissions", limit)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    with services.db.sessionmaker() as session:
        result = queries.fetch_queued_submissions(
            session, services=services, limit=limit,
        )
    # Submissions move through states quickly enough that a 5s TTL keeps
    # the queue page feeling live without hammering the DB.
    _CACHE.set(key, result, ttl=_OPEN_RUN_TTL)
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


@router.get(
    "/runs/{run_id}/validator-costs",
    response_model=ValidatorRunCostsResponse,
)
async def get_validator_costs_for_run(
    request: Request, run_id: str,
) -> ValidatorRunCostsResponse:
    """Per-validator validator-paid cost breakdown for one run.

    Aggregates ``task_evaluations.oracle_cost_usd`` (oracle layer) +
    ``task_miner_results.judge_cost_usd`` (eiretes-judge) joined on
    ``claimed_by_validator``. Returned validators are sorted by
    total descending. Public endpoint — no auth, same policy as
    other ``/api/v1/dashboard`` reads.
    """
    services: ManagedOwnerServices = request.app.state.services
    key = ("validator_costs", run_id)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    with services.db.sessionmaker() as session:
        result = queries.validator_run_costs(session, run_id=run_id)
    _CACHE.set(key, result, ttl=_OPEN_RUN_TTL)
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


# ── Public submission viewer (closed runs only) ────────────────────────────
#
# Once a run closes, every submission scored in that run becomes publicly
# downloadable + viewable on the leaderboard. The gate below enforces:
#   1. The run exists and its status is "completed" (the canonical
#      post-run state in this codebase — "closed" is not used).
#   2. The submission was actually scored in this run (DeploymentScoreRecord
#      lookup) — prevents probing a stranger submission_id against a closed
#      run id you happen to know.
# Any failure → 404 (not 403) so we don't leak existence of the submission.

_MAX_VIEWABLE_FILE_BYTES = 5 * 1024 * 1024


def _load_archive_for_closed_run(
    request: Request, *, run_id: str, submission_id: str
) -> bytes:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        run = session.get(EvaluationRun, run_id)
        if run is None or run.status != "completed":
            raise HTTPException(status_code=404, detail="not found")
        score_rec = session.execute(
            select(DeploymentScoreRecord).where(
                DeploymentScoreRecord.run_id == run_id,
                DeploymentScoreRecord.submission_id == submission_id,
            ).limit(1)
        ).scalar_one_or_none()
        if score_rec is None:
            raise HTTPException(status_code=404, detail="not found")
        submission = session.get(ManagedMinerSubmission, submission_id)
        if submission is None:
            raise HTTPException(status_code=404, detail="not found")
        artifact = session.get(SubmissionArtifact, submission.artifact_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="not found")
        return artifact.archive_bytes


@router.get("/runs/{run_id}/submissions/{submission_id}/artifact")
async def public_download_submission_artifact(
    request: Request, run_id: str, submission_id: str
):
    archive = _load_archive_for_closed_run(
        request, run_id=run_id, submission_id=submission_id
    )
    return Response(content=archive, media_type="application/gzip")


@router.get(
    "/runs/{run_id}/submissions/{submission_id}/files",
    response_model=SubmissionFilesResponse,
)
async def public_list_submission_files(
    request: Request, run_id: str, submission_id: str
) -> SubmissionFilesResponse:
    archive = _load_archive_for_closed_run(
        request, run_id=run_id, submission_id=submission_id
    )
    files: list[SubmissionFile] = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            files.append(
                SubmissionFile(path=member.name, size_bytes=member.size)
            )
    files.sort(key=lambda f: f.path)
    return SubmissionFilesResponse(files=files)


@router.get("/runs/{run_id}/submissions/{submission_id}/files/{path:path}")
async def public_get_submission_file(
    request: Request, run_id: str, submission_id: str, path: str
) -> Response:
    archive = _load_archive_for_closed_run(
        request, run_id=run_id, submission_id=submission_id
    )
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        try:
            member = tar.getmember(path)
        except KeyError:
            raise HTTPException(status_code=404, detail="file not found")
        if not member.isfile():
            raise HTTPException(status_code=404, detail="not a file")
        if member.size > _MAX_VIEWABLE_FILE_BYTES:
            raise HTTPException(status_code=413, detail="file too large to view")
        extracted = tar.extractfile(member)
        if extracted is None:
            raise HTTPException(status_code=404, detail="file not readable")
        raw = extracted.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=415, detail="binary file")
    return Response(content=text, media_type="text/plain; charset=utf-8")
