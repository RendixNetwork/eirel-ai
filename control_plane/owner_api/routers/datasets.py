"""Owner-signed admin endpoints for dataset bindings.

Post-refactor the only supported family is ``general_chat``. The forge
generates a ``GeneralChatBundle`` deterministically from a seed and
serializes it into an ``OwnerDatasetBinding`` row so validators can read
it back through the normal loader path.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from shared.common.models import OwnerDatasetBinding
from shared.dataset_forge import (
    FORGE_GENERATOR_VERSION,
    ForgeError,
    forge_general_chat_bundle,
)
from shared.dataset_forge.validator import (
    BundleValidationError,
    validate_general_chat_bundle,
)

from control_plane.owner_api.dependencies import require_owner_signature
from control_plane.owner_api.managed import ManagedOwnerServices
from control_plane.owner_api.schemas import (
    DatasetBindingListResponse,
    DatasetBindingResponse,
    DatasetForgeRequest,
    DatasetForgeResponse,
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


def _derive_rng_seed(run_id: str) -> int:
    digest = hashlib.sha256(f"gc-forge:{run_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


# -- endpoints ----------------------------------------------------------


@router.post(
    "/v1/operators/datasets/{family_id}/forge",
    response_model=DatasetForgeResponse,
)
async def forge_dataset(
    family_id: str,
    payload: DatasetForgeRequest,
    request: Request,
    _: str = Depends(require_owner_signature),
) -> DatasetForgeResponse:
    if family_id != "general_chat":
        raise HTTPException(
            status_code=400,
            detail=f"family={family_id!r} is not supported (general_chat only at launch)",
        )
    services: ManagedOwnerServices = request.app.state.services

    try:
        bundle = forge_general_chat_bundle(
            size=200,
            rng_seed=_derive_rng_seed(payload.run_id),
        )
    except ForgeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        report = validate_general_chat_bundle(bundle)
    except BundleValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    canonical_bytes = json.dumps(
        bundle.model_dump(mode="json"), sort_keys=True
    ).encode("utf-8")
    bundle_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
    bundle_uri = f"memory://general_chat/{payload.run_id}"

    activate = payload.activate
    activated_at = datetime.now(UTC).replace(tzinfo=None) if activate else None
    with services.db.sessionmaker() as session:
        existing = (
            session.query(OwnerDatasetBinding)
            .filter_by(family_id=family_id, run_id=payload.run_id)
            .one_or_none()
        )
        if existing is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"binding for ({family_id}, {payload.run_id}) already exists; "
                    f"supersede it before re-forging"
                ),
            )
        if activate:
            session.query(OwnerDatasetBinding).filter_by(
                family_id=family_id, status="active"
            ).update({"status": "superseded"})
        binding = OwnerDatasetBinding(
            family_id=family_id,
            run_id=payload.run_id,
            bundle_uri=bundle_uri,
            bundle_sha256=bundle_sha256,
            generator_version=FORGE_GENERATOR_VERSION,
            generated_by="owner",
            signature_hex="",
            generator_provider="general_chat_forge",
            generator_model="deterministic",
            status="active" if activate else "pending",
            provenance_json={
                "bundle_id": bundle.bundle_id,
                "metadata": dict(bundle.metadata or {}),
            },
            activated_at=activated_at,
        )
        session.add(binding)
        session.commit()
        session.refresh(binding)
        binding_response = _binding_to_response(binding)

    return DatasetForgeResponse(
        binding=binding_response,
        validation={
            "total_fixtures": report.total_fixtures,
            "scripted_count": report.scripted_count,
            "simulated_count": report.simulated_count,
            "category_distribution": report.category_distribution,
            "difficulty_distribution": report.difficulty_distribution,
        },
        bundle_uri=bundle_uri,
        history_record_uri=f"memory://general_chat/history/{payload.run_id}",
    )


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
