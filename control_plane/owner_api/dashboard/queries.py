from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from .schemas import (
    FamiliesResponse,
    LeaderboardResponse,
    MinerProfileResponse,
    MinerRunsResponse,
    OverviewResponse,
    RunDetailResponse,
    WindowLiteral,
)


# ---------------------------------------------------------------------------
# Phase 1a: stubs. Implementations land in Phase 1b.
# ---------------------------------------------------------------------------


def fetch_overview(session: Session, *, services: Any) -> OverviewResponse:
    raise NotImplementedError("fetch_overview — implemented in Phase 1b")


def fetch_families(session: Session, *, services: Any) -> FamiliesResponse:
    raise NotImplementedError("fetch_families — implemented in Phase 1b")


def fetch_leaderboard(
    session: Session,
    *,
    services: Any,
    family_id: str,
    window: WindowLiteral,
    limit: int,
    offset: int,
) -> LeaderboardResponse:
    raise NotImplementedError("fetch_leaderboard — implemented in Phase 1b")


def fetch_miner_profile(
    session: Session,
    *,
    services: Any,
    hotkey: str,
    family_id: str,
) -> MinerProfileResponse:
    raise NotImplementedError("fetch_miner_profile — implemented in Phase 1b")


def fetch_miner_runs(
    session: Session,
    *,
    services: Any,
    hotkey: str,
    family_id: str,
    limit: int,
    offset: int,
) -> MinerRunsResponse:
    raise NotImplementedError("fetch_miner_runs — implemented in Phase 1b")


def fetch_run_detail(
    session: Session,
    *,
    services: Any,
    hotkey: str,
    family_id: str,
    run_id: str,
) -> RunDetailResponse:
    raise NotImplementedError("fetch_run_detail — implemented in Phase 1b")


def shorten_hotkey(hk: str) -> str:
    # first-4 + last-3; plan §11A locked.
    if len(hk) <= 7:
        return hk
    return f"{hk[:4]}...{hk[-3:]}"
