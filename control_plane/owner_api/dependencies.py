from __future__ import annotations

"""Shared FastAPI dependencies and helper functions for the owner API routers."""

import asyncio
import hmac
from typing import Any

import httpx
from fastapi import HTTPException, Request

from shared.common.models import ValidatorRecord
from shared.common.security import SignatureVerifier, authenticate_request
from shared.common.exceptions import WorkflowEpisodeCancelledError, WorkflowEpisodeLeaseFencedError
from control_plane.owner_api.managed import ManagedOwnerServices


async def signature_dependency(request: Request) -> str:
    settings = request.app.state.services.settings
    verifier = SignatureVerifier()
    return await authenticate_request(
        request,
        verifier=verifier,
        ttl_seconds=settings.signature_ttl_seconds,
        replay_protector=request.app.state.replay_protector,
    )


async def validator_dependency(request: Request) -> str:
    hotkey = await signature_dependency(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        validator = session.get(ValidatorRecord, hotkey)
        if validator is None or not validator.is_active:
            raise HTTPException(status_code=403, detail="validator access denied")
    return hotkey


async def require_owner_signature(request: Request) -> str:
    """Authenticate the request as the subnet owner.

    Wraps ``signature_dependency`` and additionally asserts the signing
    hotkey matches ``settings.owner_hotkey_ss58``. Endpoints that mutate
    dataset bindings (forge / activate / supersede) gate on this so only
    the owner can change which evaluation bundle is active.
    """
    hotkey = await signature_dependency(request)
    services: ManagedOwnerServices = request.app.state.services
    expected = (services.settings.owner_hotkey_ss58 or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="owner authentication is not configured (set EIREL_OWNER_HOTKEY_SS58)",
        )
    if not hmac.compare_digest(hotkey, expected):
        raise HTTPException(status_code=403, detail="owner access denied")
    return hotkey


def require_internal_service_token(request: Request) -> None:
    expected = request.app.state.services.settings.internal_service_token
    provided = request.headers.get("Authorization", "")
    if not expected or not hmac.compare_digest(provided, f"Bearer {expected}"):
        raise HTTPException(status_code=401, detail="invalid internal authorization")


def get_services(request: Request) -> ManagedOwnerServices:
    return request.app.state.services


async def post_execution_worker_action(
    request: Request,
    *,
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    client_factory = getattr(request.app.state, "execution_worker_client_factory", None)
    base_url = services.settings.execution_worker_internal_url.rstrip("/")
    try:
        if client_factory is not None:
            async with client_factory() as client:
                response = await client.post(path, json=payload)
        else:
            async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
                response = await client.post(path, json=payload)
        response.raise_for_status()
        return {
            "ok": True,
            "url": f"{base_url}{path}",
            "payload": response.json(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "url": f"{base_url}{path}",
            "error": str(exc),
        }


async def post_weight_setter_action(
    request: Request,
    *,
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    client_factory = getattr(request.app.state, "weight_setter_client_factory", None)
    base_url = services.settings.weight_setter_internal_url.rstrip("/")
    try:
        if client_factory is not None:
            async with client_factory() as client:
                response = await client.post(path, json=payload)
        else:
            async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
                response = await client.post(path, json=payload)
        response.raise_for_status()
        return {
            "ok": True,
            "url": f"{base_url}{path}",
            "payload": response.json(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "url": f"{base_url}{path}",
            "error": str(exc),
        }


def workflow_episode_http_error(exc: ValueError) -> HTTPException:
    if isinstance(exc, WorkflowEpisodeLeaseFencedError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, WorkflowEpisodeCancelledError):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


_deployment_locks: dict[str, asyncio.Lock] = {}
_deployment_locks_guard = asyncio.Lock()


async def ensure_candidate_runtime_available(
    *,
    services: ManagedOwnerServices,
    deployment_id: str,
) -> None:
    existing = services.runtime_manager.runtime_handle(deployment_id)
    if existing is not None and existing.state == "healthy":
        return

    # Serialize per-deployment to avoid concurrent ensure_deployment_runtime
    # calls that race on docker run with the same container name.
    async with _deployment_locks_guard:
        if deployment_id not in _deployment_locks:
            _deployment_locks[deployment_id] = asyncio.Lock()
        lock = _deployment_locks[deployment_id]
    async with lock:
        # Re-check after acquiring lock — another coroutine may have started it
        existing = services.runtime_manager.runtime_handle(deployment_id)
        if existing is not None and existing.state == "healthy":
            return
        # Try to recover handle for an already-running container before
        # attempting a full (re)deploy which would fail if the container exists.
        with services.db.sessionmaker() as session:
            from shared.common.manifest import SubmissionManifest
            from shared.common.models import ManagedDeployment, ManagedMinerSubmission
            deployment = session.get(ManagedDeployment, deployment_id)
            if deployment is not None:
                submission = session.get(ManagedMinerSubmission, deployment.submission_id)
                if submission is not None:
                    manifest = SubmissionManifest.model_validate(submission.manifest_json)
                    recovered = await services.runtime_manager.backend.recover_runtime_handle(
                        submission_id=deployment_id, manifest=manifest,
                    )
                    if recovered is not None and recovered.state == "healthy":
                        return
        await services.ensure_deployment_runtime(deployment_id=deployment_id)


async def ensure_serving_runtime_available(
    *,
    services: ManagedOwnerServices,
    serving_deployment_id: str,
) -> None:
    if services.runtime_manager.runtime_handle(serving_deployment_id) is None:
        await services.ensure_serving_runtime(serving_deployment_id=serving_deployment_id)
