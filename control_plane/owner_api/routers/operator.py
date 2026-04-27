from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import select

from shared.common.models import RuntimeNodeSnapshot
from eirel.groups import ensure_active_family_id
from control_plane.owner_api.dependencies import (
    post_execution_worker_action,
    require_internal_service_token,
)
from control_plane.owner_api.managed import ManagedOwnerServices
from control_plane.owner_api.schemas import (
    RolloutFreezeRequest,
    RuntimeRemediationPolicyRequest,
    RuntimeRemediationRequest,
    RuntimeRemediationSuppressionRequest,
    ServingReleaseTriggerRequest,
    WorkflowIncidentRemediationRequest,
)

router = APIRouter(tags=["operator"])


async def _execute_runtime_remediation_policy(
    app,
    *,
    dry_run: bool,
    reason: str | None,
    run_id: str | None,
    workflow_spec_id: str | None,
    cooldown_seconds: int | None,
    max_actions: int | None,
    trigger_worker_recover: bool,
    trigger_worker_run_once: bool,
    run_once_non_blocking: bool,
) -> dict[str, Any]:
    services: ManagedOwnerServices = app.state.services
    applied_count = 0
    persistent_policy_state: dict[str, Any] = {}
    with services.db.sessionmaker() as session:
        policy_result = services.auto_remediate_workflow_incidents(
            session,
            dry_run=dry_run,
            cooldown_seconds=(
                services.settings.workflow_runtime_auto_remediation_cooldown_seconds
                if cooldown_seconds is None
                else cooldown_seconds
            ),
            max_actions=(
                services.settings.workflow_runtime_auto_remediation_max_actions
                if max_actions is None
                else max_actions
            ),
            reason=reason,
            run_id=run_id,
            workflow_spec_id=workflow_spec_id,
        )
        applied_count = int(policy_result.get("applied_count", 0) or 0)
        session.commit()
    with services.db.sessionmaker() as session:
        persistent_policy_state = services.runtime_remediation_policy_state_payload(session)

    if dry_run:
        recover_result: dict[str, Any] = {
            "skipped": True,
            "planned": trigger_worker_recover,
            "reason": "dry_run",
        }
        run_once_result: dict[str, Any] = {
            "skipped": True,
            "planned": trigger_worker_run_once,
            "reason": "dry_run",
        }
    elif applied_count <= 0:
        recover_result = {
            "skipped": True,
            "planned": trigger_worker_recover,
            "reason": "no_policy_actions_applied",
        }
        run_once_result = {
            "skipped": True,
            "planned": trigger_worker_run_once,
            "reason": "no_policy_actions_applied",
        }
    elif persistent_policy_state.get("worker_action_backoff_active"):
        backoff_reason = {
            "reason": "worker_action_backoff_active",
            "next_worker_action_retry_at": persistent_policy_state.get("next_worker_action_retry_at"),
            "last_worker_action_failure": persistent_policy_state.get("last_worker_action_failure"),
        }
        recover_result = {
            "skipped": True,
            "planned": trigger_worker_recover,
            **backoff_reason,
        }
        run_once_result = {
            "skipped": True,
            "planned": trigger_worker_run_once,
            **backoff_reason,
        }
    else:
        request = type("PolicyRequest", (), {"app": app})()
        if trigger_worker_recover:
            recover_result = await post_execution_worker_action(
                request,
                path="/v1/operators/runtime-actions/recover",
                payload={},
            )
        else:
            recover_result = {
                "skipped": True,
                "planned": False,
                "reason": "disabled",
            }
        if trigger_worker_run_once:
            run_once_result = await post_execution_worker_action(
                request,
                path="/v1/operators/runtime-actions/run-once",
                payload={"non_blocking": run_once_non_blocking},
            )
        else:
            run_once_result = {
                "skipped": True,
                "planned": False,
                "reason": "disabled",
            }
        worker_error = None
        for result in (recover_result, run_once_result):
            if result.get("skipped"):
                continue
            if not result.get("ok", False):
                worker_error = str(result.get("error") or "execution_worker_action_failed")
                break
        with services.db.sessionmaker() as session:
            if worker_error:
                persistent_policy_state = services.record_runtime_policy_worker_failure(
                    session,
                    error_text=worker_error,
                    now=datetime.now(UTC).replace(tzinfo=None),
                )
            else:
                persistent_policy_state = services._clear_runtime_policy_worker_failure(
                    session,
                    now=datetime.now(UTC).replace(tzinfo=None),
                )
            session.commit()
            persistent_policy_state = {
                **persistent_policy_state,
                **services.runtime_remediation_policy_state_payload(session),
            }

    policy_state = app.state.runtime_remediation_policy_state
    policy_state["last_run_at"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
    policy_state["last_error"] = None
    policy_state["last_dry_run"] = dry_run
    policy_state["last_applied_count"] = applied_count
    policy_state["last_result"] = {
        "owner_policy": {
            "matched_count": policy_result.get("matched_count"),
            "eligible_count": policy_result.get("eligible_count"),
            "applied_count": policy_result.get("applied_count"),
            "skipped_count": policy_result.get("skipped_count"),
        },
        "worker_actions": {
            "recover": recover_result,
            "run_once": run_once_result,
        },
    }
    policy_state["worker_action_failure_count"] = int(
        persistent_policy_state.get("worker_action_failure_count", 0) or 0
    )
    policy_state["last_worker_action_failure"] = persistent_policy_state.get(
        "last_worker_action_failure"
    )
    policy_state["last_worker_action_failure_at"] = persistent_policy_state.get(
        "last_worker_action_failure_at"
    )
    policy_state["next_worker_action_retry_at"] = persistent_policy_state.get(
        "next_worker_action_retry_at"
    )
    policy_state["worker_action_backoff_active"] = bool(
        persistent_policy_state.get("worker_action_backoff_active", False)
    )
    policy_state["active_suppression_count"] = int(
        persistent_policy_state.get("active_suppression_count", 0) or 0
    )
    policy_state["total_runs"] = int(policy_state.get("total_runs", 0) or 0) + 1
    policy_state["total_applied_count"] = int(policy_state.get("total_applied_count", 0) or 0) + applied_count

    return {
        "service": "owner",
        "action": "runtime_remediation_policy",
        "dry_run": dry_run,
        "owner_policy": policy_result,
        "worker_actions": {
            "recover": recover_result,
            "run_once": run_once_result,
        },
    }


@router.get("/v1/operators/summary")
async def operator_summary(request: Request) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        payload = services.operator_summary(session)
        payload["runtime_remediation_policy"] = {
            **dict(request.app.state.runtime_remediation_policy_state),
            **services.runtime_remediation_policy_state_payload(session),
        }
    return payload


@router.get("/v1/operators/workflow-incidents")
async def operator_workflow_incidents(
    request: Request,
    incident_state: str | None = None,
    workflow_spec_id: str | None = None,
    task_id: str | None = None,
    lease_owner: str | None = None,
    run_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    if incident_state is not None and incident_state not in {
        "stale",
        "retry_wait",
        "retryable",
        "dead_lettered",
        "failed",
    }:
        raise HTTPException(status_code=400, detail="unsupported incident_state")
    with services.db.sessionmaker() as session:
        return services.workflow_incidents(
            session,
            incident_state=incident_state,
            workflow_spec_id=workflow_spec_id,
            task_id=task_id,
            lease_owner=lease_owner,
            run_id=run_id,
            limit=limit,
            offset=offset,
        )


@router.post("/v1/operators/workflow-incidents/remediate")
async def remediate_operator_workflow_incidents(
    request: Request,
    payload: WorkflowIncidentRemediationRequest,
) -> dict[str, Any]:
    require_internal_service_token(request)
    if payload.incident_state is not None and payload.incident_state not in {
        "stale",
        "retry_wait",
        "retryable",
        "dead_lettered",
        "failed",
    }:
        raise HTTPException(status_code=400, detail="unsupported incident_state")
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            result = services.remediate_workflow_incidents(
                session,
                action=payload.action,
                dry_run=payload.dry_run,
                reason=payload.reason,
                episode_ids=payload.episode_ids,
                incident_state=payload.incident_state,
                workflow_spec_id=payload.workflow_spec_id,
                task_id=payload.task_id,
                lease_owner=payload.lease_owner,
                run_id=payload.run_id,
                limit=payload.limit,
                offset=payload.offset,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        session.commit()
        return result


@router.post("/v1/operators/runtime-remediation")
async def operator_runtime_remediation(
    request: Request,
    payload: RuntimeRemediationRequest,
) -> dict[str, Any]:
    require_internal_service_token(request)
    if payload.incident_state is not None and payload.incident_state not in {
        "stale",
        "retry_wait",
        "retryable",
        "dead_lettered",
        "failed",
    }:
        raise HTTPException(status_code=400, detail="unsupported incident_state")
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            remediation = services.remediate_workflow_incidents(
                session,
                action=payload.action,
                dry_run=payload.dry_run,
                reason=payload.reason,
                episode_ids=payload.episode_ids,
                incident_state=payload.incident_state,
                workflow_spec_id=payload.workflow_spec_id,
                task_id=payload.task_id,
                lease_owner=payload.lease_owner,
                run_id=payload.run_id,
                limit=payload.limit,
                offset=payload.offset,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        session.commit()

    should_trigger_workers = not payload.dry_run and remediation.get("applied_count", 0) > 0

    if payload.dry_run:
        recover_result: dict[str, Any] = {
            "skipped": True,
            "planned": payload.trigger_worker_recover,
            "reason": "dry_run",
        }
        run_once_result: dict[str, Any] = {
            "skipped": True,
            "planned": payload.trigger_worker_run_once,
            "reason": "dry_run",
        }
    elif not should_trigger_workers:
        recover_result = {
            "skipped": True,
            "planned": payload.trigger_worker_recover,
            "reason": "no_owner_actions_applied",
        }
        run_once_result = {
            "skipped": True,
            "planned": payload.trigger_worker_run_once,
            "reason": "no_owner_actions_applied",
        }
    else:
        if payload.trigger_worker_recover:
            recover_result = await post_execution_worker_action(
                request,
                path="/v1/operators/runtime-actions/recover",
                payload={},
            )
        else:
            recover_result = {
                "skipped": True,
                "planned": False,
                "reason": "disabled",
            }
        if payload.trigger_worker_run_once:
            run_once_result = await post_execution_worker_action(
                request,
                path="/v1/operators/runtime-actions/run-once",
                payload={"non_blocking": payload.run_once_non_blocking},
            )
        else:
            run_once_result = {
                "skipped": True,
                "planned": False,
                "reason": "disabled",
            }

    return {
        "service": "owner",
        "action": "runtime_remediation",
        "dry_run": payload.dry_run,
        "owner_remediation": remediation,
        "worker_actions": {
            "recover": recover_result,
            "run_once": run_once_result,
        },
    }


@router.post("/v1/operators/runtime-remediation/policy")
async def operator_runtime_remediation_policy(
    request: Request,
    payload: RuntimeRemediationPolicyRequest,
) -> dict[str, Any]:
    require_internal_service_token(request)
    return await _execute_runtime_remediation_policy(
        request.app,
        dry_run=payload.dry_run,
        reason=payload.reason,
        run_id=payload.run_id,
        workflow_spec_id=payload.workflow_spec_id,
        cooldown_seconds=payload.cooldown_seconds,
        max_actions=payload.max_actions,
        trigger_worker_recover=payload.trigger_worker_recover,
        trigger_worker_run_once=payload.trigger_worker_run_once,
        run_once_non_blocking=payload.run_once_non_blocking,
    )


@router.get("/v1/operators/runtime-remediation/suppressions")
async def operator_runtime_remediation_suppressions(
    request: Request,
    active_only: bool = False,
) -> list[dict[str, Any]]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        return services.list_runtime_remediation_suppressions(
            session,
            active_only=active_only,
        )


@router.post("/v1/operators/runtime-remediation/suppressions")
async def create_operator_runtime_remediation_suppression(
    request: Request,
    payload: RuntimeRemediationSuppressionRequest,
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            created = services.create_runtime_remediation_suppression(
                session,
                target_kind=payload.target_kind,
                target_value=payload.target_value,
                reason=payload.reason,
                created_by=payload.created_by,
                expires_at=payload.expires_at.replace(tzinfo=None)
                if payload.expires_at is not None
                else None,
                metadata=payload.metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        session.commit()
        return created


@router.delete("/v1/operators/runtime-remediation/suppressions/{suppression_id}")
async def delete_operator_runtime_remediation_suppression(
    request: Request,
    suppression_id: str,
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        deleted = services.delete_runtime_remediation_suppression(
            session,
            suppression_id=suppression_id,
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="suppression not found")
        session.commit()
        return {
            "deleted": True,
            "suppression_id": suppression_id,
        }


@router.get("/v1/operators/runtime-nodes")
async def operator_runtime_nodes(request: Request) -> list[dict[str, Any]]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        return services.list_runtime_nodes(session)


@router.get("/v1/operators/runtime-capacity")
async def operator_runtime_capacity(request: Request) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        return services.runtime_capacity_summary(session)


@router.get("/v1/operators/prometheus-targets")
async def operator_prometheus_targets(request: Request) -> list[dict[str, Any]]:
    """Return node_exporter targets in Prometheus ``file_sd_configs`` JSON format.

    Prometheus can poll this endpoint via ``http_sd_configs`` or an operator
    can curl it into a file for ``file_sd_configs``.
    """
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        nodes = list(session.execute(select(RuntimeNodeSnapshot)).scalars())

    targets: list[dict[str, Any]] = []
    for node in nodes:
        meta = node.metadata_json or {}
        ssh_host = meta.get("ssh_host", "")
        if not ssh_host:
            continue
        targets.append({
            "targets": [f"{ssh_host}:9100"],
            "labels": {
                "job": "runtime_node",
                "node": node.node_name,
                "backend": meta.get("backend", "unknown"),
            },
        })
    return targets


@router.post("/v1/operators/runtime-nodes/refresh")
async def operator_refresh_runtime_nodes(request: Request) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    snapshots = await services.refresh_runtime_node_inventory()
    return {
        "status": "refreshed",
        "node_count": len(snapshots),
    }


@router.get("/v1/metagraph/status")
async def metagraph_status(request: Request) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        return services.latest_metagraph_sync_status(session)


@router.post("/v1/operators/families/{family_id}/freeze")
async def operator_freeze_family(
    request: Request,
    family_id: str,
    payload: RolloutFreezeRequest,
) -> dict[str, Any]:
    require_internal_service_token(request)
    family_id = ensure_active_family_id(family_id)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        state = services.set_rollout_freeze(
            session,
            family_id=family_id,
            rollout_frozen=payload.rollout_frozen,
            reason=payload.reason,
        )
        return {
            "family_id": state.family_id,
            "rollout_frozen": state.rollout_frozen,
            "freeze_reason": state.freeze_reason,
            "updated_at": state.updated_at.isoformat(),
        }


@router.post("/v1/operators/serving-releases/trigger")
async def operator_trigger_serving_release(
    request: Request,
    payload: ServingReleaseTriggerRequest,
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    try:
        return await services.publish_serving_release(
            base_url=str(request.base_url).rstrip("/"),
            trigger_type="manual",
            candidate_overrides=payload.candidate_overrides,
            force=payload.force,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/operators/serving-releases/run-scheduled")
async def operator_run_scheduled_serving_release(request: Request) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    result = await services.publish_due_serving_release(base_url=str(request.base_url).rstrip("/"))
    if result is None:
        return {"status": "skipped", "reason": "release_not_due"}
    return result


@router.post("/v1/operators/serving-releases/{release_id}/cancel")
async def operator_cancel_serving_release(request: Request, release_id: str) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            release = services.cancel_serving_release(session, release_id=release_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return services.serving_release_payload(release) or {}


@router.post("/v1/operators/deployments/{deployment_id}/promote")
async def operator_promote_deployment(request: Request, deployment_id: str) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            deployment = services.manual_promote_deployment(session, deployment_id=deployment_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return services.deployment_payload(deployment) or {}


@router.post("/v1/operators/deployments/{deployment_id}/drain")
async def operator_drain_deployment(request: Request, deployment_id: str) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            deployment = services.manual_drain_deployment(session, deployment_id=deployment_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await services.reconcile_family_deployments(family_id=deployment.family_id)
        return services.deployment_payload(deployment) or {}


@router.post("/v1/operators/deployments/{deployment_id}/retire")
async def operator_retire_deployment(request: Request, deployment_id: str) -> dict[str, Any]:
    require_internal_service_token(request)
    from shared.common.models import ManagedDeployment
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        family_id = deployment.family_id
    await services.stop_deployment_runtime(
        deployment_id=deployment_id,
        reason="operator_retire",
        retire=True,
    )
    await services.reconcile_family_deployments(family_id=family_id)
    with services.db.sessionmaker() as session:
        deployment = session.get(ManagedDeployment, deployment_id)
        return services.deployment_payload(deployment) or {}


@router.post("/v1/operators/serving-deployments/{serving_deployment_id}/drain")
async def operator_drain_serving_deployment(
    request: Request, serving_deployment_id: str
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            serving = services.manual_drain_serving_deployment(
                session, serving_deployment_id=serving_deployment_id
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    await services.reconcile_runtime_pool(family_id=serving.family_id)
    return services.serving_deployment_payload(serving) or {}


@router.post("/v1/operators/serving-deployments/{serving_deployment_id}/retire")
async def operator_retire_serving_deployment(
    request: Request, serving_deployment_id: str
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            serving = services.manual_retire_serving_deployment(
                session, serving_deployment_id=serving_deployment_id
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    await services.reconcile_runtime_pool(family_id=serving.family_id)
    return services.serving_deployment_payload(serving) or {}


@router.get("/v1/operators/runtime-status")
async def operator_runtime_status(request: Request) -> dict[str, Any]:
    """Placeholder for execution worker runtime status link."""
    return {"status": "not_implemented"}
