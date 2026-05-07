"""Internal HTTP API for graph-runtime checkpoints.

Miner pods running ``runtime.kind == graph`` post their checkpoints
here through eirel SDK's ``PostgresCheckpointer``, keeping the miner
image free of database credentials. Eirel-ai owns the shared Postgres
and writes rows to the ``graph_checkpoints`` table; the
``conversation_threads`` table tracks the latest checkpoint per
``thread_id`` so the orchestrator can pin multi-turn flows to the same
deployment.

Routes:

    POST    /v1/internal/checkpoints/{thread_id}            (write)
    GET     /v1/internal/checkpoints/{thread_id}/latest     (read latest)
    GET     /v1/internal/checkpoints/{thread_id}/history    (paginated list)
    DELETE  /v1/internal/checkpoints/{thread_id}            (purge thread)

All routes are guarded by ``require_internal_service_token`` so only
co-located eirel-ai components (the miner pod, the orchestrator) can
write or read.
"""
from __future__ import annotations

import base64
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from control_plane.owner_api.dependencies import require_internal_service_token
from control_plane.owner_api.managed import ManagedOwnerServices
from shared.common.models import (
    ConversationThread,
    GraphCheckpoint,
    ManagedDeployment,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["checkpoints"])


# Hard cap; mirrors eirel.checkpoint.base.MAX_CHECKPOINT_BLOB_BYTES so a
# misbehaving miner can't fill the shared store with one giant write.
_MAX_BLOB_BYTES = 256 * 1024


class CheckpointWriteRequest(BaseModel):
    checkpoint_id: str = Field(min_length=1, max_length=64)
    parent_id: str | None = Field(default=None, max_length=64)
    node: str | None = Field(default=None, max_length=128)
    state: str = Field(description="Base64-encoded JSON blob (capped at 256KB).")
    pending_writes: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointResponse(BaseModel):
    thread_id: str
    checkpoint_id: str
    parent_id: str | None = None
    created_at: datetime
    node: str | None = None
    state: str = Field(description="Base64-encoded JSON blob.")
    pending_writes: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointHistoryResponse(BaseModel):
    items: list[CheckpointResponse]


def _resolve_deployment_id(
    services: ManagedOwnerServices,
    *,
    namespace: str | None,
) -> tuple[str, str]:
    """Resolve a checkpoint namespace header to (deployment_id, family_id).

    Owner-api stamps ``EIREL_CHECKPOINT_NAMESPACE=miner-{deployment_id}``
    into the pod env, so the miner SDK passes
    ``X-Eirel-Checkpoint-Namespace: miner-{deployment_id}`` on writes.
    The router validates that the namespace points at a known
    deployment and returns its family_id for the row.
    """
    if not namespace:
        raise HTTPException(
            status_code=400,
            detail="X-Eirel-Checkpoint-Namespace header is required",
        )
    if not namespace.startswith("miner-"):
        raise HTTPException(
            status_code=400,
            detail="checkpoint namespace must start with 'miner-'",
        )
    deployment_id = namespace[len("miner-"):]
    if not deployment_id:
        raise HTTPException(status_code=400, detail="empty deployment id in namespace")
    with services.db.sessionmaker() as session:
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        return deployment_id, deployment.family_id


def _row_to_response(row: GraphCheckpoint) -> CheckpointResponse:
    return CheckpointResponse(
        thread_id=row.thread_id,
        checkpoint_id=row.checkpoint_id,
        parent_id=row.parent_checkpoint_id,
        created_at=row.created_at,
        node=row.node,
        state=base64.b64encode(row.state_blob).decode("ascii"),
        pending_writes=list(row.pending_writes_json or []),
        metadata=dict(row.metadata_json or {}),
    )


@router.post(
    "/v1/internal/checkpoints/{thread_id}",
    response_model=CheckpointResponse,
)
async def write_checkpoint(
    request: Request,
    thread_id: str,
    body: CheckpointWriteRequest,
    _token: None = Depends(require_internal_service_token),
) -> CheckpointResponse:
    services: ManagedOwnerServices = request.app.state.services
    namespace = request.headers.get("X-Eirel-Checkpoint-Namespace")
    deployment_id, family_id = _resolve_deployment_id(services, namespace=namespace)

    try:
        blob = base64.b64decode(body.state.encode("ascii"))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"state must be base64-encoded: {exc}"
        ) from None
    if len(blob) > _MAX_BLOB_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"checkpoint blob is {len(blob)} bytes; limit is {_MAX_BLOB_BYTES}",
        )

    with services.db.sessionmaker() as session:
        # Idempotency: if (thread_id, checkpoint_id) already exists,
        # return the existing row rather than violating the unique
        # constraint. A retried write from a flaky network is a normal
        # case for HTTP write-through.
        existing = session.scalar(
            select(GraphCheckpoint).where(
                GraphCheckpoint.thread_id == thread_id,
                GraphCheckpoint.checkpoint_id == body.checkpoint_id,
            )
        )
        if existing is not None:
            return _row_to_response(existing)

        row = GraphCheckpoint(
            thread_id=thread_id,
            checkpoint_id=body.checkpoint_id,
            parent_checkpoint_id=body.parent_id,
            deployment_id=deployment_id,
            family_id=family_id,
            checkpoint_namespace=namespace,
            node=body.node,
            state_blob=blob,
            pending_writes_json=list(body.pending_writes or []),
            metadata_json=dict(body.metadata or {}),
            blob_size_bytes=len(blob),
        )
        session.add(row)

        # Upsert the conversation_threads anchor so the orchestrator can
        # quickly find which deployment owns a given thread.
        thread_row = session.get(ConversationThread, thread_id)
        if thread_row is None:
            session.add(
                ConversationThread(
                    thread_id=thread_id,
                    user_id=None,
                    deployment_id=deployment_id,
                    family_id=family_id,
                    last_checkpoint_id=body.checkpoint_id,
                )
            )
        else:
            thread_row.last_checkpoint_id = body.checkpoint_id
            thread_row.deployment_id = deployment_id
            thread_row.family_id = family_id
        session.commit()
        session.refresh(row)
        return _row_to_response(row)


@router.get(
    "/v1/internal/checkpoints/{thread_id}/latest",
    response_model=CheckpointResponse,
)
async def read_latest_checkpoint(
    request: Request,
    thread_id: str,
    _token: None = Depends(require_internal_service_token),
) -> CheckpointResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        row = session.scalar(
            select(GraphCheckpoint)
            .where(GraphCheckpoint.thread_id == thread_id)
            .order_by(GraphCheckpoint.created_at.desc())
            .limit(1)
        )
        if row is None:
            raise HTTPException(status_code=404, detail="no checkpoint for thread")
        return _row_to_response(row)


@router.get(
    "/v1/internal/checkpoints/{thread_id}/history",
    response_model=CheckpointHistoryResponse,
)
async def read_checkpoint_history(
    request: Request,
    thread_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    checkpoint_id: str | None = Query(default=None),
    _token: None = Depends(require_internal_service_token),
) -> CheckpointHistoryResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        stmt = select(GraphCheckpoint).where(GraphCheckpoint.thread_id == thread_id)
        if checkpoint_id is not None:
            stmt = stmt.where(GraphCheckpoint.checkpoint_id == checkpoint_id)
        stmt = stmt.order_by(GraphCheckpoint.created_at.desc()).limit(limit)
        rows = list(session.scalars(stmt))
    return CheckpointHistoryResponse(items=[_row_to_response(r) for r in rows])


@router.delete(
    "/v1/internal/checkpoints/{thread_id}",
    response_model=dict,
)
async def delete_thread_checkpoints(
    request: Request,
    thread_id: str,
    _token: None = Depends(require_internal_service_token),
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        rows = list(
            session.scalars(
                select(GraphCheckpoint).where(GraphCheckpoint.thread_id == thread_id)
            )
        )
        for row in rows:
            session.delete(row)
        thread = session.get(ConversationThread, thread_id)
        if thread is not None:
            session.delete(thread)
        session.commit()
    return {"deleted": len(rows), "thread_id": thread_id}
