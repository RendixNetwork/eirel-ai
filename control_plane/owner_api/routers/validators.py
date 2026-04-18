from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from shared.common.models import ValidatorRecord
from control_plane.owner_api.dependencies import validator_dependency
from control_plane.owner_api.managed import ManagedOwnerServices

router = APIRouter(tags=["validators"])


@router.get("/v1/validators/me")
async def validator_me(
    request: Request,
    validator_hotkey: str = Depends(validator_dependency),
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        validator = session.get(ValidatorRecord, validator_hotkey)
        if validator is None:
            raise HTTPException(status_code=404, detail="validator not found")
        return {
            "hotkey": validator.hotkey,
            "uid": validator.uid,
            "stake": validator.stake,
            "is_active": validator.is_active,
            "last_synced_at": validator.last_synced_at.isoformat(),
        }
