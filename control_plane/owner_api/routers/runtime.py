from __future__ import annotations

import json as _json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

from shared.common.manifest import SubmissionManifest
from shared.common.models import ManagedDeployment, ManagedMinerSubmission, ServingDeployment
from eirel.schemas import AgentInvocationRequest, AgentInvocationResponse
from control_plane.owner_api.dependencies import (
    ensure_candidate_runtime_available,
    ensure_serving_runtime_available,
    require_internal_service_token,
    validator_dependency,
)
from control_plane.owner_api.managed import ManagedOwnerServices

router = APIRouter(tags=["runtime"])


# Outer wall-clock budget for streaming proxy requests. Set above the
# largest configured downstream timeout (validator's fan-out wrapper is
# 660s) so a slow miner doesn't trip this before the upstream client's
# own timeout fires.
_STREAM_TIMEOUT_SECONDS = 660.0

# Graph-runtime miners can run multi-turn cycles with reflection +
# parallel tool dispatch — set a longer wall-clock budget for them so a
# legitimately long graph doesn't get cut off mid-cycle. Triggered by
# manifest.runtime.kind == "graph" via the per-call cost_tag context.
_GRAPH_STREAM_TIMEOUT_SECONDS = 1800.0

# Event taxonomy carried over the miner-pod NDJSON stream. The graph
# rollout introduces ``tool_result``, ``trace``, and ``checkpoint``
# alongside the existing ``delta``/``tool_call``/``citation``/``done``.
# ``trace`` is teed to eiretes for KPI computation but never reaches
# downstream consumers.
_TRACE_EVENT = "trace"
_TERMINAL_EVENT = "done"

# Provider-proxy URL used for cost reconciliation after a miner stream
# completes. Owner-api stamps a per-task ``X-Eirel-Job-Id`` header on
# the way to the miner; the miner SDK forwards that as the proxy
# job_id; provider-proxy ledgers cost per job_id; owner-api queries
# this URL after the stream closes and injects the result into the
# final ``done`` chunk's metadata so the validator sees per-task
# proxy_cost_usd directly.
_PROVIDER_PROXY_URL = os.getenv(
    "EIREL_PROVIDER_PROXY_URL", "http://provider-proxy:8092"
).rstrip("/")
_PROVIDER_PROXY_TOKEN = os.getenv("EIREL_PROVIDER_PROXY_TOKEN", "")


def _build_cost_tag(*, deployment_id: str, payload: dict[str, Any]) -> str:
    """Stable per-task tag used as the provider-proxy job_id.

    Validator-issued requests carry ``turn_id`` (slim 0.3.0 contract);
    consumer-chat-api requests do too. Falls back to ``task_id``
    (legacy 0.2.x) for older callers, then to deployment_id for any
    request that has neither (e.g. orchestrator-internal probes —
    those keep the deployment-sticky semantics they had before).
    """
    turn_id = payload.get("turn_id") or payload.get("task_id") or ""
    turn_id = str(turn_id).strip()
    if turn_id:
        return f"task-eval={turn_id};deployment={deployment_id}"
    return f"miner-{deployment_id}"


async def _fetch_proxy_cost(job_id: str) -> dict[str, Any] | None:
    """Best-effort cost lookup against provider-proxy.

    Returns None on any failure — cost reporting must never break the
    eval pipeline. Caller renders a missing dict as "cost unknown".
    """
    if not job_id:
        return None
    url = f"{_PROVIDER_PROXY_URL}/v1/jobs/{job_id}/cost"
    headers = (
        {"Authorization": f"Bearer {_PROVIDER_PROXY_TOKEN}"}
        if _PROVIDER_PROXY_TOKEN
        else {}
    )
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 404:
                # No ledger entry — miner didn't make any LLM calls
                # under that job_id (or didn't forward the header).
                return {"llm_cost_usd": 0.0, "tool_cost_usd": 0.0, "absent": True}
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning(
            "proxy_cost lookup failed for job=%s url=%s: %s",
            job_id, url, exc,
        )
        return None


@router.get("/runtime/{deployment_id}/healthz")
async def runtime_health(
    request: Request,
    deployment_id: str,
    _token: None = Depends(require_internal_service_token),
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        handle = services.runtime_manager.runtime_handle(deployment_id)
        return {
            "status": deployment.health_status,
            "family_id": deployment.family_id,
            "deployment_revision": deployment.deployment_revision,
            "is_active": deployment.is_active,
            "runtime_endpoint_url": handle.endpoint_url if handle else None,
        }


@router.post("/runtime/{deployment_id}/v1/agent/infer", response_model=AgentInvocationResponse)
async def runtime_infer(
    request: Request,
    deployment_id: str,
    payload: AgentInvocationRequest,
    _token: None = Depends(require_internal_service_token),
) -> AgentInvocationResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        submission = session.get(ManagedMinerSubmission, deployment.submission_id)
        if submission is None:
            raise HTTPException(status_code=404, detail="submission not found")
        manifest = SubmissionManifest.model_validate(submission.manifest_json)
        if deployment.status == "retired":
            raise HTTPException(status_code=409, detail="deployment retired")
        if deployment.health_status != "healthy":
            raise HTTPException(status_code=503, detail="deployment unhealthy")
        if deployment.placement_status not in {"running", "ready"}:
            raise HTTPException(status_code=409, detail="deployment runtime not ready")
        candidate_registry = services.get_candidate_registry(session)
        candidate_ids = {
            str((item.metadata or {}).get("deployment_id"))
            for item in candidate_registry.get(deployment.family_id, [])
        }
        if deployment.id not in candidate_ids:
            raise HTTPException(status_code=409, detail="deployment is not callable on the consumer path")
    await ensure_candidate_runtime_available(
        services=services,
        deployment_id=deployment_id,
    )
    try:
        response = await services.runtime_manager.invoke_runtime(
            deployment_id=deployment_id,
            manifest=manifest,
            request=payload,
        )
    except Exception as exc:
        logger.warning("runtime invocation failed for %s: %s", deployment_id, exc)
        raise HTTPException(status_code=502, detail="runtime invocation failed") from exc
    response.latency_ms = deployment.latency_ms_p50
    response.metadata = {
        **response.metadata,
        "managed_execution": True,
        "deployment_id": deployment.id,
        "deployment_revision": deployment.deployment_revision,
    }
    return response


@router.get("/runtime/serving/{serving_deployment_id}/healthz")
async def serving_runtime_health(
    request: Request,
    serving_deployment_id: str,
    _token: None = Depends(require_internal_service_token),
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        serving = session.get(ServingDeployment, serving_deployment_id)
        if serving is None:
            raise HTTPException(status_code=404, detail="serving deployment not found")
        handle = services.runtime_manager.runtime_handle(serving_deployment_id)
        return {
            "status": serving.health_status,
            "family_id": serving.family_id,
            "source_deployment_revision": serving.source_deployment_revision,
            "runtime_endpoint_url": handle.endpoint_url if handle else None,
        }


@router.post(
    "/runtime/serving/{serving_deployment_id}/v1/agent/infer",
    response_model=AgentInvocationResponse,
)
async def serving_runtime_infer(
    request: Request,
    serving_deployment_id: str,
    payload: AgentInvocationRequest,
    _token: None = Depends(require_internal_service_token),
) -> AgentInvocationResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        serving = session.get(ServingDeployment, serving_deployment_id)
        if serving is None:
            raise HTTPException(status_code=404, detail="serving deployment not found")
        if serving.status != "healthy" or serving.health_status != "healthy":
            raise HTTPException(status_code=409, detail="serving deployment is not healthy")
        source = session.get(ManagedDeployment, serving.source_deployment_id)
        if source is None:
            raise HTTPException(status_code=404, detail="source deployment not found")
        submission = session.get(ManagedMinerSubmission, serving.source_submission_id)
        if submission is None:
            raise HTTPException(status_code=404, detail="submission not found")
        manifest = SubmissionManifest.model_validate(submission.manifest_json)
    await ensure_serving_runtime_available(
        services=services,
        serving_deployment_id=serving_deployment_id,
    )
    try:
        response = await services.runtime_manager.invoke_runtime(
            deployment_id=serving_deployment_id,
            manifest=manifest,
            request=payload,
        )
    except Exception as exc:
        logger.warning("runtime invocation failed for %s: %s", serving_deployment_id, exc)
        raise HTTPException(status_code=502, detail="runtime invocation failed") from exc
    response.metadata = {
        **response.metadata,
        "managed_execution": True,
        "serving_deployment_id": serving.id,
        "source_deployment_id": serving.source_deployment_id,
        "deployment_revision": serving.source_deployment_revision,
        "serving_release_id": serving.release_id,
    }
    return response


async def _validator_run_infer_impl(
    request: Request,
    *,
    run_id: str,
    deployment_id: str,
    payload: AgentInvocationRequest,
) -> AgentInvocationResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        snapshot, member = services.resolve_run_member(
            session,
            run_id=run_id,
            deployment_id=deployment_id,
        )
        if snapshot is None or member is None:
            raise HTTPException(status_code=404, detail="run deployment not found")
        if snapshot.status != "open":
            raise HTTPException(status_code=409, detail="run snapshot is closed")
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        submission = session.get(ManagedMinerSubmission, deployment.submission_id)
        if submission is None:
            raise HTTPException(status_code=404, detail="submission not found")
        manifest = SubmissionManifest.model_validate(submission.manifest_json)
    await ensure_candidate_runtime_available(
        services=services,
        deployment_id=deployment_id,
    )
    try:
        response = await services.runtime_manager.invoke_runtime(
            deployment_id=deployment_id,
            manifest=manifest,
            request=payload,
        )
    except Exception as exc:
        logger.warning("runtime invocation failed for %s: %s", deployment_id, exc)
        raise HTTPException(status_code=502, detail="runtime invocation failed") from exc
    response.latency_ms = deployment.latency_ms_p50
    response.metadata = {
        **response.metadata,
        "managed_execution": True,
        "deployment_id": deployment.id,
        "deployment_revision": deployment.deployment_revision,
        "run_id": run_id,
        "validator_pinned": True,
    }
    return response


@router.get("/v1/validator/runs/{run_id}/deployments/{deployment_id}/healthz")
async def validator_runtime_health(
    request: Request,
    run_id: str,
    deployment_id: str,
    validator_hotkey: str = Depends(validator_dependency),
) -> dict[str, Any]:
    del validator_hotkey
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        snapshot, member = services.resolve_run_member(
            session,
            run_id=run_id,
            deployment_id=deployment_id,
        )
        if snapshot is None or member is None:
            raise HTTPException(status_code=404, detail="run deployment not found")
        if snapshot.status != "open":
            raise HTTPException(status_code=409, detail="run snapshot is closed")
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        handle = services.runtime_manager.runtime_handle(deployment_id)
        return {
            "status": deployment.health_status,
            "family_id": deployment.family_id,
            "deployment_revision": deployment.deployment_revision,
            "run_id": snapshot.run_id,
            "snapshot_status": snapshot.status,
            "runtime_endpoint_url": handle.endpoint_url if handle else None,
        }


@router.get("/v1/validator/epochs/{epoch_id}/deployments/{deployment_id}/healthz")
async def validator_runtime_health_epoch_alias(
    request: Request,
    epoch_id: str,
    deployment_id: str,
    validator_hotkey: str = Depends(validator_dependency),
) -> dict[str, Any]:
    del validator_hotkey
    return await validator_runtime_health(request, run_id=epoch_id, deployment_id=deployment_id)


@router.get("/v1/internal/runs/{run_id}/deployments/{deployment_id}/healthz")
async def internal_runtime_health(
    request: Request,
    run_id: str,
    deployment_id: str,
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        snapshot, member = services.resolve_run_member(
            session,
            run_id=run_id,
            deployment_id=deployment_id,
        )
        if snapshot is None or member is None:
            raise HTTPException(status_code=404, detail="run deployment not found")
        if snapshot.status != "open":
            raise HTTPException(status_code=409, detail="run snapshot is closed")
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        handle = services.runtime_manager.runtime_handle(deployment_id)
        return {
            "status": deployment.health_status,
            "family_id": deployment.family_id,
            "deployment_revision": deployment.deployment_revision,
            "run_id": snapshot.run_id,
            "snapshot_status": snapshot.status,
            "runtime_endpoint_url": handle.endpoint_url if handle else None,
        }


@router.get("/v1/internal/epochs/{epoch_id}/deployments/{deployment_id}/healthz")
async def internal_runtime_health_epoch_alias(
    request: Request,
    epoch_id: str,
    deployment_id: str,
) -> dict[str, Any]:
    return await internal_runtime_health(request, run_id=epoch_id, deployment_id=deployment_id)


@router.post(
    "/v1/validator/runs/{run_id}/deployments/{deployment_id}/v1/agent/infer",
    response_model=AgentInvocationResponse,
)
async def validator_runtime_infer(
    request: Request,
    run_id: str,
    deployment_id: str,
    payload: AgentInvocationRequest,
    validator_hotkey: str = Depends(validator_dependency),
) -> AgentInvocationResponse:
    del validator_hotkey
    return await _validator_run_infer_impl(
        request,
        run_id=run_id,
        deployment_id=deployment_id,
        payload=payload,
    )


@router.post(
    "/v1/validator/runs/{run_id}/deployments/{deployment_id}/infer",
    response_model=AgentInvocationResponse,
)
async def validator_runtime_infer_alias(
    request: Request,
    run_id: str,
    deployment_id: str,
    payload: AgentInvocationRequest,
    validator_hotkey: str = Depends(validator_dependency),
) -> AgentInvocationResponse:
    del validator_hotkey
    return await _validator_run_infer_impl(
        request,
        run_id=run_id,
        deployment_id=deployment_id,
        payload=payload,
    )


@router.post(
    "/v1/validator/epochs/{epoch_id}/deployments/{deployment_id}/v1/agent/infer",
    response_model=AgentInvocationResponse,
)
async def validator_runtime_infer_epoch_alias(
    request: Request,
    epoch_id: str,
    deployment_id: str,
    payload: AgentInvocationRequest,
    validator_hotkey: str = Depends(validator_dependency),
) -> AgentInvocationResponse:
    del validator_hotkey
    return await _validator_run_infer_impl(
        request,
        run_id=epoch_id,
        deployment_id=deployment_id,
        payload=payload,
    )


@router.post(
    "/v1/internal/runs/{run_id}/deployments/{deployment_id}/v1/agent/infer",
    response_model=AgentInvocationResponse,
)
async def internal_runtime_infer(
    request: Request,
    run_id: str,
    deployment_id: str,
    payload: AgentInvocationRequest,
) -> AgentInvocationResponse:
    require_internal_service_token(request)
    return await _validator_run_infer_impl(
        request,
        run_id=run_id,
        deployment_id=deployment_id,
        payload=payload,
    )


@router.post(
    "/v1/internal/epochs/{epoch_id}/deployments/{deployment_id}/v1/agent/infer",
    response_model=AgentInvocationResponse,
)
async def internal_runtime_infer_epoch_alias(
    request: Request,
    epoch_id: str,
    deployment_id: str,
    payload: AgentInvocationRequest,
) -> AgentInvocationResponse:
    require_internal_service_token(request)
    return await _validator_run_infer_impl(
        request,
        run_id=epoch_id,
        deployment_id=deployment_id,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Streaming proxy routes
# ---------------------------------------------------------------------------
# Pipe NDJSON from the miner pod's /v1/agent/infer/stream straight through
# to the caller (validator / consumer-chat-api). Each line is forwarded
# byte-for-byte (with a re-attached newline) so FastAPI's StreamingResponse
# controls every chunk boundary and never gets confused by partial reads.
#
# Note: this used to be broken because RawBodyCaptureMiddleware was lying
# to FastAPI's StreamingResponse about client disconnects, killing any
# streaming response that awaited between yields. The middleware was
# fixed in app.py to forward real `receive()` after the body is replayed.

async def _proxy_stream_lines_to_pod(
    *,
    pod_endpoint: str,
    payload: dict[str, Any],
    cost_tag: str | None = None,
    runtime_kind: str = "base_agent",
    run_id: str | None = None,
    deployment_hotkey: str | None = None,
) -> AsyncIterator[bytes]:
    """Forward the pod's NDJSON line-by-line.

    Each yielded chunk is one complete NDJSON frame ending with `\\n`.
    Failures emerge as a synthetic ``done`` chunk so callers always see
    a terminator they can act on.

    When ``cost_tag`` is non-empty, owner-api injects it as
    ``X-Eirel-Job-Id`` so the miner SDK forwards it to the
    provider-proxy as the LLM-call attribution key. After the miner's
    terminal ``done`` arrives, we look up the cost ledger for that
    tag and merge ``{proxy_cost_usd, proxy_request_count, ...}`` into
    the chunk's ``metadata`` before re-emitting. Cost data is the
    server-side ground truth; miner self-report is never trusted.

    Event taxonomy
    --------------
    Event taxonomy: ``delta`` / ``tool_call`` / ``tool_result`` /
    ``citation`` / ``trace`` / ``checkpoint`` / ``done``. All
    non-``done`` events pass through byte-for-byte to the caller.
    """
    url = f"{pod_endpoint.rstrip('/')}/v1/agent/infer/stream"
    headers = {"X-Eirel-Job-Id": cost_tag} if cost_tag else {}
    pending_done: dict[str, Any] | None = None
    trace_buffer: list[dict[str, Any]] = []
    timeout = (
        _GRAPH_STREAM_TIMEOUT_SECONDS
        if runtime_kind == "graph"
        else _STREAM_TIMEOUT_SECONDS
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        parsed = _json.loads(stripped)
                    except _json.JSONDecodeError:
                        # Non-JSON garbage: pass through so the consumer
                        # sees exactly what the pod emitted.
                        yield (stripped + "\n").encode("utf-8")
                        continue
                    if isinstance(parsed, dict):
                        event = parsed.get("event")
                        if event == _TERMINAL_EVENT:
                            # Hold the terminal ``done`` so we can
                            # augment it with cost metadata before
                            # forwarding.
                            pending_done = parsed
                            continue
                        if event == _TRACE_EVENT:
                            # Tee trace frames to eiretes; never forward
                            # them downstream — consumers expect only
                            # the wire-stable event vocabulary.
                            trace_buffer.append(parsed)
                            continue
                    # All other events (delta, tool_call, tool_result,
                    # citation, checkpoint, plus any future additions)
                    # pass through byte-for-byte.
                    yield (stripped + "\n").encode("utf-8")
    except httpx.HTTPStatusError as exc:
        logger.warning("stream pod returned %d for %s", exc.response.status_code, url)
        yield (
            ('{"event":"done","status":"failed","error":"pod returned '
             + str(exc.response.status_code) + '"}\n').encode("utf-8")
        )
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning("stream pod connect failed for %s: %s", url, exc)
        msg = str(exc).replace('"', '\\"')
        yield ('{"event":"done","status":"failed","error":"' + msg + '"}\n').encode("utf-8")
        return

    # Stream closed cleanly. Reconcile cost from the ledger, augment
    # ``done`` metadata, emit. If the miner never sent ``done`` (rare,
    # malformed stream) synthesize one so the validator's wire contract
    # is preserved.
    if pending_done is None:
        pending_done = {"event": "done", "status": "completed"}
    if cost_tag:
        cost = await _fetch_proxy_cost(cost_tag)
        if cost is not None:
            meta = pending_done.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            meta["proxy_cost_tag"] = cost_tag
            meta["proxy_cost_usd"] = round(float(cost.get("llm_cost_usd") or 0.0), 8)
            meta["proxy_tool_cost_usd"] = round(
                float(cost.get("tool_cost_usd") or 0.0), 8,
            )
            meta["proxy_cost_absent"] = bool(cost.get("absent", False))
            # Per-graph-span cost attribution: surfaces the roll-up
            # so eiretes' trace KPIs and the leaderboard can attribute
            # cost to the specific span that emitted it.
            per_span = cost.get("per_span") or {}
            if isinstance(per_span, dict) and per_span:
                meta["proxy_cost_by_span"] = {
                    str(k): round(float(v), 8) for k, v in per_span.items()
                }
            pending_done["metadata"] = meta

    # Advertise the runtime kind so eiretes knows the pod shape.
    if runtime_kind:
        meta = pending_done.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("runtime_kind", runtime_kind)
        pending_done["metadata"] = meta

    yield (_json.dumps(pending_done, separators=(",", ":")) + "\n").encode("utf-8")


def _runtime_kind_from_manifest(manifest_json: dict[str, Any] | None) -> str:
    """Extract ``runtime.kind`` from a submission manifest.

    Defaults to ``base_agent`` for old manifests that predate the field.
    """
    if not isinstance(manifest_json, dict):
        return "base_agent"
    runtime_section = manifest_json.get("runtime")
    if isinstance(runtime_section, dict):
        kind = runtime_section.get("kind")
        if isinstance(kind, str) and kind:
            return kind
    return "base_agent"


async def _eval_stream_impl(
    request: Request,
    *,
    run_id: str | None,
    deployment_id: str,
    payload: dict[str, Any],
) -> StreamingResponse:
    services: ManagedOwnerServices = request.app.state.services
    runtime_kind = "base_agent"
    deployment_hotkey: str | None = None
    with services.db.sessionmaker() as session:
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        if deployment.status == "retired":
            raise HTTPException(status_code=409, detail="deployment retired")
        deployment_hotkey = deployment.miner_hotkey
        submission = session.get(ManagedMinerSubmission, deployment.submission_id)
        if submission is not None:
            runtime_kind = _runtime_kind_from_manifest(submission.manifest_json)
        if run_id is not None:
            snapshot, member = services.resolve_run_member(
                session, run_id=run_id, deployment_id=deployment_id,
            )
            if snapshot is None or member is None:
                raise HTTPException(status_code=404, detail="run deployment not found")
            if snapshot.status != "open":
                raise HTTPException(status_code=409, detail="run snapshot is closed")
    await ensure_candidate_runtime_available(
        services=services, deployment_id=deployment_id,
    )
    handle = services.runtime_manager.runtime_handle(deployment_id)
    if handle is None:
        raise HTTPException(status_code=503, detail="runtime not ready")
    cost_tag = _build_cost_tag(deployment_id=deployment_id, payload=payload)
    return StreamingResponse(
        _proxy_stream_lines_to_pod(
            pod_endpoint=handle.endpoint_url,
            payload=payload,
            cost_tag=cost_tag,
            runtime_kind=runtime_kind,
            run_id=run_id,
            deployment_hotkey=deployment_hotkey,
        ),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.post("/runtime/{deployment_id}/v1/agent/infer/stream")
async def runtime_infer_stream(
    request: Request,
    deployment_id: str,
    payload: dict[str, Any],
    _token: None = Depends(require_internal_service_token),
) -> StreamingResponse:
    return await _eval_stream_impl(
        request, run_id=None, deployment_id=deployment_id, payload=payload,
    )


@router.post("/v1/internal/runs/{run_id}/deployments/{deployment_id}/v1/agent/infer/stream")
async def internal_runtime_infer_stream(
    request: Request,
    run_id: str,
    deployment_id: str,
    payload: dict[str, Any],
) -> StreamingResponse:
    require_internal_service_token(request)
    return await _eval_stream_impl(
        request, run_id=run_id, deployment_id=deployment_id, payload=payload,
    )


@router.post("/v1/internal/epochs/{epoch_id}/deployments/{deployment_id}/v1/agent/infer/stream")
async def internal_runtime_infer_stream_epoch_alias(
    request: Request,
    epoch_id: str,
    deployment_id: str,
    payload: dict[str, Any],
) -> StreamingResponse:
    require_internal_service_token(request)
    return await _eval_stream_impl(
        request, run_id=epoch_id, deployment_id=deployment_id, payload=payload,
    )


@router.post("/v1/validator/runs/{run_id}/deployments/{deployment_id}/v1/agent/infer/stream")
async def validator_runtime_infer_stream(
    request: Request,
    run_id: str,
    deployment_id: str,
    payload: dict[str, Any],
    validator_hotkey: str = Depends(validator_dependency),
) -> StreamingResponse:
    del validator_hotkey
    return await _eval_stream_impl(
        request, run_id=run_id, deployment_id=deployment_id, payload=payload,
    )


@router.post("/runtime/serving/{serving_deployment_id}/v1/agent/infer/stream")
async def serving_runtime_infer_stream(
    request: Request,
    serving_deployment_id: str,
    payload: dict[str, Any],
    _token: None = Depends(require_internal_service_token),
) -> StreamingResponse:
    services: ManagedOwnerServices = request.app.state.services
    runtime_kind = "base_agent"
    serving_hotkey: str | None = None
    with services.db.sessionmaker() as session:
        serving = session.get(ServingDeployment, serving_deployment_id)
        if serving is None:
            raise HTTPException(status_code=404, detail="serving deployment not found")
        if serving.status != "healthy" or serving.health_status != "healthy":
            raise HTTPException(status_code=409, detail="serving deployment is not healthy")
        serving_hotkey = getattr(serving, "miner_hotkey", None)
        submission = session.get(ManagedMinerSubmission, serving.submission_id)
        if submission is not None:
            runtime_kind = _runtime_kind_from_manifest(submission.manifest_json)
    await ensure_serving_runtime_available(
        services=services, serving_deployment_id=serving_deployment_id,
    )
    handle = services.runtime_manager.runtime_handle(serving_deployment_id)
    if handle is None:
        raise HTTPException(status_code=503, detail="serving runtime not ready")
    cost_tag = _build_cost_tag(
        deployment_id=serving_deployment_id, payload=payload,
    )
    return StreamingResponse(
        _proxy_stream_lines_to_pod(
            pod_endpoint=handle.endpoint_url,
            payload=payload,
            cost_tag=cost_tag,
            runtime_kind=runtime_kind,
            run_id=None,
            deployment_hotkey=serving_hotkey,
        ),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
