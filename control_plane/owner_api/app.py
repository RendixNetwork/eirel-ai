from __future__ import annotations

"""Owner API composition root.

Lifespan management, middleware, and router mounting.
All endpoint handlers live in ``owner_api.routers.*``.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import uvicorn
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

from shared.common.config import get_settings
from shared.common.artifacts import create_artifact_store
from shared.common.http_control import SlidingWindowRateLimiter
from shared.common.database import Database
from shared.common.migrations import run_migrations
from control_plane.owner_api.managed import ManagedOwnerServices
from control_plane.owner_api.deployment import ManagedDeploymentRuntimeManager
from control_plane.owner_api.routers import (
    health,
    submissions,
    deployments,
    datasets,
    dashboard,
    evaluation_tasks,
    runs,
    scoring,
    serving,
    workflows,
    operator,
    runtime,
    validators,
    internal,
)
from shared.common.request_context import RequestIdMiddleware
from shared.common.security import create_replay_protector
from infra.miner_runtime.runtime_manager import DockerMinerRuntimeManager, KubernetesMinerRuntimeManager
from control_plane.owner_api.operations.cleanup_tasks import (
    _reap_pending_runtime_stops,
    _retry_pending_capacity,
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


_MAX_REQUEST_BODY = 250 * 1024 * 1024  # 250 MB


class RawBodyCaptureMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        body = b""
        disconnected = False
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
                break
            body += message.get("body", b"")
            if len(body) > _MAX_REQUEST_BODY:
                await send({
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [[b"content-type", b"text/plain"]],
                })
                await send({
                    "type": "http.response.body",
                    "body": b"Request body too large",
                })
                return
            if not message.get("more_body", False):
                break
        scope["cached_body"] = body
        delivered = False

        async def replay_receive() -> Message:
            nonlocal delivered, disconnected
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            if disconnected:
                return {"type": "http.disconnect"}
            disconnected = True
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------


async def _runtime_capacity_refresh_loop(app: FastAPI) -> None:
    services: ManagedOwnerServices = app.state.services
    interval = max(5.0, services.settings.owner_runtime_capacity_refresh_interval_seconds)
    while True:
        try:
            await services.refresh_runtime_node_inventory()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("runtime capacity refresh failed")
        await asyncio.sleep(interval)


async def _runtime_remediation_loop(app: FastAPI) -> None:
    services: ManagedOwnerServices = app.state.services
    interval = max(
        5.0,
        float(services.settings.workflow_runtime_auto_remediation_interval_seconds),
    )
    while True:
        try:
            if services.settings.workflow_runtime_auto_remediation_enabled:
                await operator._execute_runtime_remediation_policy(
                    app,
                    dry_run=False,
                    reason="auto_policy_loop",
                    run_id=None,
                    workflow_spec_id=None,
                    cooldown_seconds=None,
                    max_actions=None,
                    trigger_worker_recover=True,
                    trigger_worker_run_once=True,
                    run_once_non_blocking=True,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            app.state.runtime_remediation_policy_state["last_run_at"] = (
                datetime.now(UTC).replace(tzinfo=None).isoformat()
            )
            app.state.runtime_remediation_policy_state["last_error"] = str(exc)
        await asyncio.sleep(interval)


async def _container_health_loop(app: FastAPI) -> None:
    """Periodically check that containers for active deployments are still
    running and trigger recovery for any that have stopped unexpectedly."""
    services: ManagedOwnerServices = app.state.services
    interval = max(30.0, float(services.settings.owner_runtime_health_timeout_seconds))
    while True:
        try:
            await services.deployments.proactive_health_check()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("container health check loop failed")
        try:
            services.check_and_open_pending_snapshots()
        except Exception as exc:
            logger.warning("snapshot readiness check error: %s", exc)
        await asyncio.sleep(interval)


async def _retired_runtime_reaper_loop(app: FastAPI) -> None:
    services: ManagedOwnerServices = app.state.services
    interval = max(10.0, float(services.settings.owner_runtime_reaper_interval_seconds))
    while True:
        try:
            await _reap_pending_runtime_stops(services)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("reaper loop error: %s", exc)
        await asyncio.sleep(interval)


async def _pending_capacity_retry_loop(app: FastAPI) -> None:
    services: ManagedOwnerServices = app.state.services
    interval = max(10.0, float(services.settings.owner_pending_capacity_retry_interval_seconds))
    while True:
        try:
            await _retry_pending_capacity(services)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("pending_capacity retry loop error: %s", exc)
        await asyncio.sleep(interval)


async def _claim_expiry_sweeper_loop(app: FastAPI) -> None:
    """Periodically release evaluation task claims that have expired.

    Without this, if a validator crashes mid-evaluation its claimed tasks
    stay stuck until another validator happens to call claim_tasks() for
    the same run+family.
    """
    from sqlalchemy import update
    from shared.common.models import MinerEvaluationTask
    from control_plane.owner_api._helpers import utcnow

    services: ManagedOwnerServices = app.state.services
    interval = max(30.0, float(services.settings.task_claim_timeout_seconds) / 4)
    while True:
        try:
            with services.db.sessionmaker() as session:
                now = utcnow()
                result = session.execute(
                    update(MinerEvaluationTask)
                    .where(MinerEvaluationTask.status == "claimed")
                    .where(MinerEvaluationTask.claim_expires_at <= now)
                    .values(
                        status="pending",
                        claimed_by_validator=None,
                        claimed_at=None,
                        claim_expires_at=None,
                        updated_at=now,
                    )
                )
                released = result.rowcount
                session.commit()
                if released > 0:
                    logger.info("claim sweeper: released %d expired claims", released)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("claim expiry sweeper failed")
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    db = Database(settings.database_url)
    db.create_all()
    run_migrations(db.engine)
    logger.info(
        "owner-api startup: backend=%s, database_url=%s, redis_url=%s, "
        "provider_proxy_url=%s, web_search_tool_url=%s, x_tool_url=%s, "
        "semantic_scholar_tool_url=%s, sandbox_tool_url=%s, "
        "namespace=%s, system_namespace=%s, control_plane_namespace=%s",
        settings.owner_runtime_backend,
        settings.database_url,
        settings.redis_url,
        settings.provider_proxy_url,
        settings.web_search_tool_service_url,
        settings.x_tool_service_url,
        settings.semantic_scholar_tool_service_url,
        settings.sandbox_tool_service_url,
        settings.owner_runtime_namespace,
        settings.owner_runtime_system_namespace,
        settings.owner_runtime_control_plane_namespace,
    )
    if settings.owner_runtime_backend == "kubernetes":
        backend = KubernetesMinerRuntimeManager(
            kubeconfig_path=settings.owner_kubeconfig_path or None,
            namespace=settings.owner_runtime_namespace,
            system_namespace=settings.owner_runtime_system_namespace,
            control_plane_namespace=settings.owner_runtime_control_plane_namespace,
            runtime_image=settings.owner_runtime_image,
            shared_secret_name=settings.owner_runtime_shared_secret_name,
            service_domain=settings.owner_runtime_service_domain,
            health_timeout_seconds=settings.owner_runtime_health_timeout_seconds,
            probe_period_seconds=settings.owner_runtime_probe_period_seconds,
        )
    elif settings.owner_runtime_backend == "baremetal":
        from infra.miner_runtime.baremetal_runtime_manager import BaremetalDockerMinerRuntimeManager

        backend = BaremetalDockerMinerRuntimeManager(
            inventory_path=settings.owner_baremetal_inventory_path,
            docker_binary=settings.owner_docker_binary_path,
            runtime_image=settings.owner_miner_runtime_image,
            sdk_root=settings.owner_sdk_repo_root,
            work_root=settings.owner_runtime_work_root,
            health_timeout_seconds=settings.owner_runtime_health_timeout_seconds,
            storage_root=settings.owner_baremetal_storage_root,
            provider_proxy_url_override=settings.owner_baremetal_provider_proxy_url,
            research_tool_url_override=settings.owner_baremetal_research_tool_url,
        )
    else:
        backend = DockerMinerRuntimeManager(
            docker_binary=settings.owner_docker_binary_path,
            runtime_image=settings.owner_miner_runtime_image,
            sdk_root=settings.owner_sdk_repo_root,
            work_root=settings.owner_runtime_work_root,
            endpoint_host=settings.owner_runtime_endpoint_host,
            bind_host=settings.owner_runtime_bind_host,
            health_timeout_seconds=settings.owner_runtime_health_timeout_seconds,
            docker_network=settings.owner_runtime_docker_network or None,
        )
    logger.info("trace store backend=%s", settings.trace_store_backend)
    app.state.replay_protector = create_replay_protector(settings.redis_url)
    if not settings.redis_url:
        if settings.launch_mode == "production":
            raise RuntimeError(
                "REDIS_URL is required in production for replay protection. "
                "Set REDIS_URL or change LAUNCH_MODE."
            )
        logger.warning(
            "REDIS_URL not set — using in-memory replay protection (dev only)"
        )
    app.state.submission_rate_limiter = SlidingWindowRateLimiter(
        max_requests=settings.submission_rate_limit_requests,
        window_seconds=settings.submission_rate_limit_window_seconds,
    )
    # Resolve owner hotkey SS58 from the wallet files. Required for
    # /v1/admin/* endpoints (require_owner_signature verifies against this).
    if settings.owner_wallet_name and settings.owner_hotkey_name:
        try:
            import bittensor as bt
            wallet = bt.Wallet(
                name=settings.owner_wallet_name,
                hotkey=settings.owner_hotkey_name,
            )
            settings.owner_hotkey_ss58 = wallet.hotkey.ss58_address
            logger.info("owner hotkey: %s", settings.owner_hotkey_ss58)
        except Exception as exc:
            logger.warning(
                "could not resolve owner wallet %s/%s: %s — admin endpoints will return 503",
                settings.owner_wallet_name, settings.owner_hotkey_name, exc,
            )
    else:
        logger.warning(
            "EIREL_OWNER_WALLET_NAME / EIREL_OWNER_HOTKEY_NAME not set — "
            "admin endpoints will return 503"
        )
    app.state.artifact_store = create_artifact_store(settings)
    app.state.services = ManagedOwnerServices(
        db=db,
        settings=settings,
        runtime_manager=ManagedDeploymentRuntimeManager(backend=backend),
        artifact_store=app.state.artifact_store,
        top_k_per_group=3,
    )
    if settings.submission_treasury_address:
        from control_plane.owner_api.fee_verifier import FeeVerifier

        app.state.services._fee_verifier = FeeVerifier(
            network=settings.bittensor_network,
            treasury_address=settings.submission_treasury_address,
            fee_tao=settings.submission_fee_tao,
        )
    app.state.execution_worker_client_factory = None
    app.state.weight_setter_client_factory = None
    app.state.runtime_remediation_policy_state = {
        "enabled": bool(settings.workflow_runtime_auto_remediation_enabled),
        "interval_seconds": float(settings.workflow_runtime_auto_remediation_interval_seconds),
        "cooldown_seconds": int(settings.workflow_runtime_auto_remediation_cooldown_seconds),
        "max_actions": int(settings.workflow_runtime_auto_remediation_max_actions),
        "requeue_limit": int(settings.workflow_runtime_auto_remediation_requeue_limit),
        "escalation_window_seconds": int(
            settings.workflow_runtime_auto_remediation_escalation_window_seconds
        ),
        "worker_failure_backoff_seconds": int(
            settings.workflow_runtime_auto_remediation_worker_failure_backoff_seconds
        ),
        "last_run_at": None,
        "last_error": None,
        "last_dry_run": None,
        "last_applied_count": 0,
        "worker_action_failure_count": 0,
        "last_worker_action_failure": None,
        "last_worker_action_failure_at": None,
        "next_worker_action_retry_at": None,
        "worker_action_backoff_active": False,
        "active_suppression_count": 0,
        "total_runs": 0,
        "total_applied_count": 0,
        "last_result": None,
    }
    await app.state.services.refresh_runtime_node_inventory()
    await app.state.services.reconcile_all_active_deployments()
    refresh_task = asyncio.create_task(_runtime_capacity_refresh_loop(app))
    remediation_task = asyncio.create_task(_runtime_remediation_loop(app))
    sweeper_task = asyncio.create_task(_claim_expiry_sweeper_loop(app))
    health_task = asyncio.create_task(_container_health_loop(app))
    reaper_task = asyncio.create_task(_retired_runtime_reaper_loop(app))
    retry_task = asyncio.create_task(_pending_capacity_retry_loop(app))
    yield
    for task in (refresh_task, remediation_task, sweeper_task, health_task, reaper_task, retry_task):
        task.cancel()
    for task in (refresh_task, remediation_task, sweeper_task, health_task, reaper_task, retry_task):
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    await app.state.replay_protector.close()
    app.state.services.db.engine.dispose()
    logger.info("database engine disposed")


# ---------------------------------------------------------------------------
# App construction
# ---------------------------------------------------------------------------

app = FastAPI(title="owner-api", lifespan=lifespan)
app.add_middleware(RawBodyCaptureMiddleware)
app.add_middleware(RequestIdMiddleware)

app.include_router(health.router)
app.include_router(submissions.router)
app.include_router(deployments.router)
app.include_router(runs.router)
app.include_router(scoring.router)
app.include_router(evaluation_tasks.router)
app.include_router(serving.router)
app.include_router(workflows.router)
app.include_router(operator.router)
app.include_router(datasets.router)
app.include_router(runtime.router)
app.include_router(validators.router)
app.include_router(internal.router)
app.include_router(dashboard.router)

from control_plane.owner_api.routers import admin as admin_router
app.include_router(admin_router.router)


def main() -> None:
    uvicorn.run("control_plane.owner_api.app:app", host="0.0.0.0", port=8000, reload=False)
