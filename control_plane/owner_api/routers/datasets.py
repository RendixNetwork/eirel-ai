"""Owner-signed admin endpoints for dataset bindings.

Operators register out-of-band-generated bundles in the
``OwnerDatasetBinding`` table and flip their status via these endpoints.
Validators read bundles back through the normal loader path.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from shared.common.models import OwnerDatasetBinding

from control_plane.owner_api.dependencies import require_owner_signature
from control_plane.owner_api.managed import ManagedOwnerServices
from control_plane.owner_api.schemas import (
    DatasetBindingListResponse,
    DatasetBindingResponse,
)


router = APIRouter(tags=["operator", "datasets"])


# -- helpers ------------------------------------------------------------


def _binding_to_response(row: OwnerDatasetBinding) -> DatasetBindingResponse:
    return DatasetBindingResponse(
        id=row.id,
        family_id=row.family_id,
        run_id=row.run_id,
        bundle_uri=row.bundle_uri,
        bundle_sha256=row.bundle_sha256,
        generator_version=row.generator_version,
        generated_by=row.generated_by,
        signature_hex=row.signature_hex,
        generator_provider=row.generator_provider,
        generator_model=row.generator_model,
        status=row.status,
        provenance=dict(row.provenance_json or {}),
        created_at=row.created_at,
        activated_at=row.activated_at,
    )


# -- endpoints ----------------------------------------------------------


@router.get(
    "/v1/operators/datasets/{family_id}/bindings",
    response_model=DatasetBindingListResponse,
)
async def list_dataset_bindings(
    family_id: str,
    request: Request,
    status: str | None = Query(default=None),
    _: str = Depends(require_owner_signature),
) -> DatasetBindingListResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        query = session.query(OwnerDatasetBinding).filter_by(family_id=family_id)
        if status:
            query = query.filter_by(status=status)
        rows = query.order_by(OwnerDatasetBinding.created_at.desc()).all()
        return DatasetBindingListResponse(
            family_id=family_id,
            bindings=[_binding_to_response(row) for row in rows],
        )


@router.post(
    "/v1/operators/datasets/{family_id}/bindings/{binding_id}/activate",
    response_model=DatasetBindingResponse,
)
async def activate_dataset_binding(
    family_id: str,
    binding_id: str,
    request: Request,
    _: str = Depends(require_owner_signature),
) -> DatasetBindingResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        target = (
            session.query(OwnerDatasetBinding)
            .filter_by(id=binding_id, family_id=family_id)
            .one_or_none()
        )
        if target is None:
            raise HTTPException(status_code=404, detail="binding not found")
        if target.status == "active":
            return _binding_to_response(target)
        if target.status not in {"pending", "superseded"}:
            raise HTTPException(
                status_code=409,
                detail=f"cannot activate binding in status={target.status!r}",
            )
        session.query(OwnerDatasetBinding).filter_by(
            family_id=family_id, status="active"
        ).update({"status": "superseded"})
        target.status = "active"
        target.activated_at = datetime.now(UTC).replace(tzinfo=None)
        session.commit()
        session.refresh(target)
        return _binding_to_response(target)


@router.post(
    "/v1/operators/datasets/{family_id}/bindings/{binding_id}/supersede",
    response_model=DatasetBindingResponse,
)
async def supersede_dataset_binding(
    family_id: str,
    binding_id: str,
    request: Request,
    _: str = Depends(require_owner_signature),
) -> DatasetBindingResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        target = (
            session.query(OwnerDatasetBinding)
            .filter_by(id=binding_id, family_id=family_id)
            .one_or_none()
        )
        if target is None:
            raise HTTPException(status_code=404, detail="binding not found")
        if target.status == "superseded":
            return _binding_to_response(target)
        target.status = "superseded"
        session.commit()
        session.refresh(target)
        return _binding_to_response(target)
