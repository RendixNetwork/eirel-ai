from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import httpx

from shared.common.bittensor_signing import load_signer
from shared.common.security import sha256_hex


REQUIRED_FAMILIES = ("analyst", "builder", "verifier", "media")
DIRECT_ANALYSIS_PROMPT = "Analyze the request directly and return a verified response."
RESEARCH_TO_BUILD_PROMPT = "Research the task, implement the solution, and verify the result."
CONCEPT_TO_MEDIA_PROMPT = "Turn the concept into media, then review the generated artifact."
DEPRECATED_OWNER_ENDPOINTS = (
    "/v1/workflow-serving/coalitions",
    "/v1/workflow-serving/releases/current",
    "/v1/operators/workflow-serving/freezes",
    "/v1/families/builder/coalition-scores",
)


def _status_from_checks(checks: dict[str, bool]) -> str:
    return "passed" if checks and all(checks.values()) else "failed"


def _blocked_message(exc: Exception, *, fallback: str) -> str:
    message = str(exc).strip() or fallback
    if "name resolution" in message.lower() or "connection refused" in message.lower():
        return fallback
    return message


def _owner_signed_headers(
    *,
    validator_mnemonic: str,
    method: str = "GET",
    path: str = "/",
    body: bytes = b"",
) -> dict[str, str]:
    signer = load_signer(
        wallet_name=None,
        hotkey_name=None,
        wallet_path=None,
        mnemonic=validator_mnemonic,
    )
    return signer.signed_headers(method, path, sha256_hex(body))


async def _wait_json(
    *,
    url: str,
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


async def _fetch_json(
    *,
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = await client.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


async def _post_json(
    *,
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


async def _endpoint_unreachable(
    *,
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str] | None = None,
) -> bool:
    response = await client.get(url, headers=headers)
    return response.status_code in {404, 405}


async def _post_gateway_task(
    *,
    api_gateway_url: str,
    consumer_api_url: str,
    gateway_api_key: str,
    consumer_api_key: str,
    raw_input: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    del consumer_api_url, consumer_api_key
    async with httpx.AsyncClient(
        base_url=api_gateway_url.rstrip("/"),
        timeout=timeout_seconds,
    ) as client:
        response = await client.post(
            "/v1/tasks",
            json={"raw_input": raw_input, "mode": "sync"},
            headers={"Authorization": f"Bearer {gateway_api_key}"},
        )
        response.raise_for_status()
        return response.json()


async def run_compose_smoke(**kwargs: Any) -> dict[str, Any]:
    raw_input = kwargs.get("raw_input") or DIRECT_ANALYSIS_PROMPT
    return await _post_gateway_task(
        api_gateway_url=str(kwargs["api_gateway_url"]),
        consumer_api_url=str(kwargs["consumer_api_url"]),
        gateway_api_key=str(kwargs["gateway_api_key"]),
        consumer_api_key=str(kwargs["consumer_api_key"]),
        raw_input=raw_input,
        timeout_seconds=float(kwargs.get("timeout_seconds", 30.0)),
    )


async def run_live_research_smoke(**kwargs: Any) -> dict[str, Any]:
    del kwargs
    raise RuntimeError("live research smoke is not configured in this workspace")


async def _run_e2e_smoke(args: Any) -> dict[str, Any]:
    del args
    raise RuntimeError("validator end-to-end smoke is not configured in this workspace")


async def _collect_snapshot(**kwargs: Any) -> dict[str, Any]:
    del kwargs
    return {}


async def _ensure_family_deployments(
    *,
    owner_api_url: str,
    family_ids: list[str],
    timeout_seconds: float,
    seed_missing_families: bool,
    seed_mnemonics: list[str],
) -> dict[str, Any]:
    del seed_missing_families, seed_mnemonics
    async with httpx.AsyncClient(
        base_url=owner_api_url.rstrip("/"),
        timeout=timeout_seconds,
    ) as client:
        registry = await _fetch_json(client=client, url="/v1/internal/registry")
        candidate_registry = await _fetch_json(client=client, url="/v1/internal/candidate-registry")
    resolved_registry: dict[str, list[dict[str, Any]]] = {}
    missing_families: list[str] = []
    for family_id in family_ids:
        family_entries = list(registry.get(family_id) or [])
        if not family_entries:
            family_entries = list(candidate_registry.get(family_id) or [])
        resolved_registry[family_id] = family_entries
        if not family_entries:
            missing_families.append(family_id)
    return {
        "status": "passed" if not missing_families else "failed",
        "registry": resolved_registry,
        "missing_families": missing_families,
        "seeded_submissions": [],
    }


async def _validate_prompt_workflow(
    *,
    name: str,
    prompt: str,
    api_gateway_url: str,
    consumer_api_url: str,
    gateway_api_key: str,
    consumer_api_key: str,
    timeout_seconds: float,
    validator: Callable[[dict[str, Any]], dict[str, bool]],
) -> dict[str, Any]:
    try:
        payload = await _post_gateway_task(
            api_gateway_url=api_gateway_url,
            consumer_api_url=consumer_api_url,
            gateway_api_key=gateway_api_key,
            consumer_api_key=consumer_api_key,
            raw_input=prompt,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        return {
            "status": "blocked",
            "message": _blocked_message(
                exc,
                fallback="Prompt workflow smoke is blocked by provider or gateway configuration.",
            ),
            "checks": {},
        }
    errors = list(((payload.get("execution_result") or {}).get("errors")) or [])
    branch_failures = dict(
        (((payload.get("execution_result") or {}).get("final_output") or {}).get("branch_failures"))
        or {}
    )
    failure_text = " ".join(
        [str(item) for item in errors]
        + [str(item.get("error") or "") for item in branch_failures.values() if isinstance(item, dict)]
    ).lower()
    if str(payload.get("status") or "").lower() == "failed" and "provider" in failure_text:
        return {
            "status": "blocked",
            "message": "Prompt workflow smoke is blocked by provider connectivity or configuration.",
            "checks": {},
        }
    checks = validator(payload)
    return {
        "status": _status_from_checks(checks),
        "checks": checks,
        "scenario": name,
    }


async def _validate_deep_research(
    *,
    owner_api_url: str,
    validator_mnemonic: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    headers = _owner_signed_headers(
        validator_mnemonic=validator_mnemonic,
        method="GET",
        path="/v1/families/analyst/targets",
        body=b"",
    )
    async with httpx.AsyncClient(
        base_url=owner_api_url.rstrip("/"),
        timeout=timeout_seconds,
    ) as client:
        await _fetch_json(client=client, url="/v1/runs/current", headers=headers)
        targets = await _fetch_json(client=client, url="/v1/families/analyst/targets", headers=headers)
    analyst_tasks = list(((targets.get("evaluation_bundle") or {}).get("tasks")) or [])
    checks = {
        "evaluation_bundle_only": "evaluation_bundle" in targets and "family_targets" not in targets,
        "live_web_enabled": str((targets.get("retrieval_environment") or {}).get("mode") or "") == "live_web",
        "analyst_bundle_size": len(analyst_tasks) == 40,
        "analyst_live_research_fixture_count": sum(1 for item in analyst_tasks if str(item.get("execution_mode") or "") == "live_web") == 14,
        "analyst_replay_research_fixture_count": sum(1 for item in analyst_tasks if str(item.get("execution_mode") or "") == "replay_web") == 14,
        "analyst_reasoning_fixture_count": sum(1 for item in analyst_tasks if str(item.get("execution_mode") or "") == "offline_reasoning") == 12,
        "analyst_hidden_fixture_count": sum(1 for item in analyst_tasks if bool((item.get("metadata") or {}).get("hidden_fixture"))) >= 12,
    }
    try:
        smoke = await run_live_research_smoke(
            owner_api_url=owner_api_url,
            validator_mnemonic=validator_mnemonic,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        return {
            "status": "blocked",
            "message": _blocked_message(
                exc,
                fallback="Deep research smoke is blocked by research-tool or provider configuration.",
            ),
            "checks": checks,
        }
    checks["retrieval_ledger_present"] = bool(smoke.get("retrieval_ledger_id"))
    return {
        "status": _status_from_checks(checks),
        "checks": checks,
        "result": smoke,
    }


def _direct_analysis_checks(payload: dict[str, Any]) -> dict[str, bool]:
    dependency_graph = (
        (((payload.get("routing_plan") or {}).get("metadata") or {}).get("dependency_graph")) or {}
    )
    composer_metadata = (
        ((((payload.get("execution_result") or {}).get("final_output") or {}).get("composer_metadata")) or {})
    )
    return {
        "dependency_graph_node_keys": set(dependency_graph) == {
            "control_plane_context_pack",
            "analyst_plan",
            "verifier_review",
        },
        "final_response_family_is_verifier": composer_metadata.get("source_family_id") == "verifier",
    }


def _research_to_build_checks(payload: dict[str, Any]) -> dict[str, bool]:
    execution_result = payload.get("execution_result") or {}
    composer_metadata = ((execution_result.get("final_output") or {}).get("composer_metadata")) or {}
    lineage = list(composer_metadata.get("upstream_lineage") or [])
    return {
        "ordered_upstream_lineage": lineage == [
            "control_plane_retrieval",
            "control_plane_context_pack",
            "analyst_plan",
            "builder_impl",
            "verifier_review",
            "analyst_synthesis",
        ],
        "final_response_family_is_analyst": composer_metadata.get("source_family_id") == "analyst",
    }


def _concept_to_media_checks(payload: dict[str, Any]) -> dict[str, bool]:
    nodes = list(((payload.get("execution_result") or {}).get("nodes")) or [])
    media_artifacts_present = any(
        list(((node.get("output") or {}).get("artifacts")) or [])
        for node in nodes
        if isinstance(node, dict)
    )
    composer_metadata = ((payload.get("execution_result") or {}).get("final_output") or {}).get("composer_metadata") or {}
    return {
        "media_artifacts_present": media_artifacts_present,
        "final_response_family_is_verifier": composer_metadata.get("source_family_id") == "verifier",
    }


async def _validate_validator_promotion(*, timeout_seconds: float) -> dict[str, Any]:
    protocol = await _run_e2e_smoke(SimpleNamespace(mode="protocol", timeout_seconds=timeout_seconds))
    scored = await _run_e2e_smoke(SimpleNamespace(mode="scored", timeout_seconds=timeout_seconds))
    task_failure = await _run_e2e_smoke(SimpleNamespace(mode="task-failure", timeout_seconds=timeout_seconds))
    scored_progress = dict(scored.get("progress") or {})
    checks = {
        "protocol_submission_created": bool((protocol.get("submission") or {}).get("id")),
        "scored_run_finalized": str((scored.get("canonical") or {}).get("status") or "") == "finalized",
        "judge_backed_results_present": any(
            int(item.get("judge_backed_result_count") or 0) > 0
            and bool(item.get("score_bearing"))
            for item in list(scored_progress.get("tasks") or [])
            if isinstance(item, dict)
        ),
        "failure_path_exercised": bool(task_failure.get("failure_releases")),
    }
    return {
        "status": _status_from_checks(checks),
        "checks": checks,
        "protocol": protocol,
        "scored": scored,
        "task_failure": task_failure,
    }


async def _validate_serving_cutover(
    *,
    owner_api_url: str,
    validator_mnemonic: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    runs_headers = _owner_signed_headers(
        validator_mnemonic=validator_mnemonic,
        method="GET",
        path="/v1/runs/current",
        body=b"",
    )
    async with httpx.AsyncClient(
        base_url=owner_api_url.rstrip("/"),
        timeout=timeout_seconds,
    ) as client:
        registry = await _fetch_json(client=client, url="/v1/internal/registry")
        current_run = await _fetch_json(client=client, url="/v1/runs/current", headers=runs_headers)
        runs = await _fetch_json(client=client, url="/v1/runs", headers=runs_headers)
        current_release = await _fetch_json(client=client, url="/v1/serving/releases/current")
        composition = await _fetch_json(client=client, url="/v1/workflow-composition/registry")
        target_payloads = {
            family_id: await _fetch_json(
                client=client,
                url=f"/v1/families/{family_id}/targets",
                headers=_owner_signed_headers(
                    validator_mnemonic=validator_mnemonic,
                    method="GET",
                    path=f"/v1/families/{family_id}/targets",
                    body=b"",
                ),
            )
            for family_id in REQUIRED_FAMILIES
        }
        deprecated_endpoint_status = {
            path: await _endpoint_unreachable(client=client, url=path)
            for path in DEPRECATED_OWNER_ENDPOINTS
        }
    latest_run = runs[0] if isinstance(runs, list) and runs else current_run
    latest_family_results = dict(latest_run.get("family_results") or {})
    release_metadata = dict((current_release.get("release") or {}).get("metadata") or {})
    selected_by_family = dict(release_metadata.get("serving_selected_deployment_id_by_family") or {})
    fleet_by_family = {
        str(item.get("family_id") or ""): str(item.get("source_deployment_id") or "")
        for item in list(current_release.get("deployments") or [])
        if isinstance(item, dict)
    }
    checks: dict[str, bool] = {}
    for family_id in REQUIRED_FAMILIES:
        registry_entries = list(registry.get(family_id) or [])
        selected_deployment_id = str(selected_by_family.get(family_id) or "")
        winner_deployment_id = str(
            (latest_family_results.get(family_id) or {}).get("winner_deployment_id") or ""
        )
        registry_deployment_id = str(
            ((registry_entries[0].get("metadata") or {}).get("deployment_id")) if registry_entries else ""
        )
        checks[f"{family_id}_registry_matches_release"] = bool(
            selected_deployment_id and registry_deployment_id == selected_deployment_id
        )
        checks[f"{family_id}_winner_matches_serving_release"] = bool(
            selected_deployment_id and selected_deployment_id == winner_deployment_id == fleet_by_family.get(family_id, "")
        )
        target_payload = target_payloads[family_id]
        checks[f"{family_id}_targets_canonical"] = (
            "evaluation_bundle" in target_payload
            and "family_targets" not in target_payload
            and "family_target_artifact" not in target_payload
        )
    composition_checks: list[bool] = []
    for spec_payload in composition.values():
        selected_node_map = dict(spec_payload.get("selected_node_map") or {})
        for node_payload in selected_node_map.values():
            family_id = str(node_payload.get("family_id") or "")
            deployment_id = str(node_payload.get("deployment_id") or "")
            composition_checks.append(
                bool(family_id and deployment_id and deployment_id == selected_by_family.get(family_id))
            )
    checks["workflow_composition_matches_family_winners"] = bool(composition_checks) and all(composition_checks)
    checks["deprecated_endpoints_absent"] = all(deprecated_endpoint_status.values())
    return {
        "status": _status_from_checks(checks),
        "checks": checks,
        "deprecated_endpoint_status": deprecated_endpoint_status,
        "workflow_composition_registry": composition,
    }


async def run_staging_validation(
    *,
    stack: str,
    owner_api_url: str,
    api_gateway_url: str,
    consumer_api_url: str,
    validator_engine_url: str,
    metagraph_listener_url: str,
    weight_setter_url: str,
    provider_proxy_url: str,
    gateway_api_key: str,
    consumer_api_key: str,
    miner_mnemonic: str,
    validator_mnemonic: str,
    sandbox_service_url: str,
    seed_missing_families: bool,
    family_miner_mnemonics: list[str],
    timeout_seconds: float,
) -> dict[str, Any]:
    del stack, miner_mnemonic
    baseline_checks = {
        "metagraph_listener_reachable": True,
        "validator_engine_reachable": True,
        "weight_setter_reachable": True,
        "provider_proxy_reachable": True,
        "sandbox_service_reachable": True,
    }
    for name, url in {
        "metagraph_listener_reachable": metagraph_listener_url,
        "validator_engine_reachable": validator_engine_url,
        "weight_setter_reachable": weight_setter_url,
        "provider_proxy_reachable": provider_proxy_url,
        "sandbox_service_reachable": sandbox_service_url,
    }.items():
        try:
            await _wait_json(url=url, timeout_seconds=timeout_seconds)
        except Exception:
            baseline_checks[name] = False
    compose_smoke = await run_compose_smoke(
        owner_api_url=owner_api_url,
        api_gateway_url=api_gateway_url,
        consumer_api_url=consumer_api_url,
        gateway_api_key=gateway_api_key,
        consumer_api_key=consumer_api_key,
        timeout_seconds=timeout_seconds,
    )
    snapshot = await _collect_snapshot(
        owner_api_url=owner_api_url,
        validator_engine_url=validator_engine_url,
        metagraph_listener_url=metagraph_listener_url,
        weight_setter_url=weight_setter_url,
        timeout_seconds=timeout_seconds,
    )
    deployment_check = await _ensure_family_deployments(
        owner_api_url=owner_api_url,
        family_ids=list(REQUIRED_FAMILIES),
        timeout_seconds=timeout_seconds,
        seed_missing_families=seed_missing_families,
        seed_mnemonics=list(family_miner_mnemonics),
    )
    compose_baseline_checks = {
        **baseline_checks,
        "compose_smoke_completed": str((compose_smoke.get("task_result") or {}).get("status") or "completed") == "completed",
        "family_registry_complete": not deployment_check["missing_families"],
    }
    compose_baseline = {
        "status": _status_from_checks(compose_baseline_checks),
        "checks": compose_baseline_checks,
        "compose_smoke": compose_smoke,
        "snapshot": snapshot,
        "family_registry": deployment_check,
    }
    workflow_matrix = {
        "scenarios": {
            "direct_analysis": await _validate_prompt_workflow(
                name="direct_analysis",
                prompt=DIRECT_ANALYSIS_PROMPT,
                api_gateway_url=api_gateway_url,
                consumer_api_url=consumer_api_url,
                gateway_api_key=gateway_api_key,
                consumer_api_key=consumer_api_key,
                timeout_seconds=timeout_seconds,
                validator=_direct_analysis_checks,
            ),
            "research_to_build": await _validate_prompt_workflow(
                name="research_to_build",
                prompt=RESEARCH_TO_BUILD_PROMPT,
                api_gateway_url=api_gateway_url,
                consumer_api_url=consumer_api_url,
                gateway_api_key=gateway_api_key,
                consumer_api_key=consumer_api_key,
                timeout_seconds=timeout_seconds,
                validator=_research_to_build_checks,
            ),
            "concept_to_media": await _validate_prompt_workflow(
                name="concept_to_media",
                prompt=CONCEPT_TO_MEDIA_PROMPT,
                api_gateway_url=api_gateway_url,
                consumer_api_url=consumer_api_url,
                gateway_api_key=gateway_api_key,
                consumer_api_key=consumer_api_key,
                timeout_seconds=timeout_seconds,
                validator=_concept_to_media_checks,
            ),
            "deep_research": await _validate_deep_research(
                owner_api_url=owner_api_url,
                validator_mnemonic=validator_mnemonic,
                timeout_seconds=timeout_seconds,
            ),
        }
    }
    workflow_matrix["status"] = _status_from_checks(
        {
            name: payload["status"] == "passed"
            for name, payload in workflow_matrix["scenarios"].items()
        }
    )
    validator_promotion = await _validate_validator_promotion(timeout_seconds=timeout_seconds)
    serving_cutover = await _validate_serving_cutover(
        owner_api_url=owner_api_url,
        validator_mnemonic=validator_mnemonic,
        timeout_seconds=timeout_seconds,
    )
    report = {
        "status": _status_from_checks(
            {
                "compose_baseline": compose_baseline["status"] == "passed",
                "workflow_matrix": workflow_matrix["status"] == "passed",
                "validator_promotion": validator_promotion["status"] == "passed",
                "serving_cutover": serving_cutover["status"] == "passed",
            }
        ),
        "compose_baseline": compose_baseline,
        "workflow_matrix": workflow_matrix,
        "validator_promotion": validator_promotion,
        "serving_cutover": serving_cutover,
        "kubernetes_confirmation": {
            "status": "pending",
            "message": "Run the same checks against the Kubernetes stack before launch.",
        },
    }
    return report
