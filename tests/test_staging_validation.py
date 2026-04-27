from __future__ import annotations

from typing import Any

import pytest

from tests import staging_validation as sv


def _task_payload(
    *,
    workflow_template: str,
    dependency_graph_keys: list[str],
    source_node_id: str,
    source_family_id: str,
    specialist_node_ids: list[str],
    node_results: list[dict[str, Any]] | None = None,
    upstream_lineage: list[str] | None = None,
) -> dict[str, Any]:
    node_results = list(node_results or [])
    return {
        "routing_plan": {
            "metadata": {
                "workflow_template": workflow_template,
                "dependency_graph": {
                    node_id: [] for node_id in dependency_graph_keys
                },
            }
        },
        "execution_dag": {
            "metadata": {
                "parallel_stages": [],
            }
        },
        "execution_result": {
            "metadata": {
                "execution_trace": {
                    "final_response_node_id": source_node_id,
                }
            },
            "nodes": node_results,
            "final_output": {
                "specialist_outputs": {
                    node_id: {"summary": node_id} for node_id in specialist_node_ids
                },
                "composer_metadata": {
                    "source_node_id": source_node_id,
                    "source_family_id": source_family_id,
                    "upstream_lineage": list(upstream_lineage or []),
                },
                "branch_failures": {},
            },
            "errors": [],
        },
    }


@pytest.mark.asyncio
async def test_run_staging_validation_reports_passed_matrix(monkeypatch):
    async def fake_wait_json(*, url: str, timeout_seconds: float, headers: dict[str, str] | None = None) -> dict[str, Any]:
        del timeout_seconds, headers
        return {"status": "ok", "url": url}

    async def fake_fetch_json(
        *,
        client: Any,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del client, headers, params
        if url == "/v1/operators/summary":
            return {
                "service": "owner",
                "scoring_readiness": {"ok": True, "ready": True},
            }
        if url == "/v1/internal/registry":
            return {
                "analyst": [{"metadata": {"deployment_id": "analyst-selected"}}],
                "builder": [{"metadata": {"deployment_id": "builder-selected"}}],
                "verifier": [{"metadata": {"deployment_id": "verifier-selected"}}],
                "media": [{"metadata": {"deployment_id": "media-selected"}}],
            }
        if url == "/v1/internal/candidate-registry":
            return {family_id: [{"hotkey": f"{family_id}-1"}] for family_id in sv.REQUIRED_FAMILIES}
        if url == "/v1/runs/current":
            return {
                "run_id": "run-1",
                "evaluation_bundle_summaries": {
                    family_id: {"kind": "family_evaluation_bundle"} for family_id in sv.REQUIRED_FAMILIES
                },
                "family_score_summaries": {
                    family_id: {"scored_miner_count": 1} for family_id in sv.REQUIRED_FAMILIES
                },
                "family_results": {
                    "analyst": {"winner_hotkey": "analyst-1", "winner_deployment_id": "analyst-selected"},
                    "builder": {"winner_hotkey": "builder-1", "winner_deployment_id": "builder-selected"},
                    "verifier": {"winner_hotkey": "verifier-1", "winner_deployment_id": "verifier-selected"},
                    "media": {"winner_hotkey": "media-1", "winner_deployment_id": "media-selected"},
                },
            }
        if url == "/v1/runs":
            return [
                {
                    "run_id": "run-1",
                    "family_results": {
                        "analyst": {"winner_deployment_id": "analyst-selected"},
                        "builder": {"winner_deployment_id": "builder-selected"},
                        "verifier": {"winner_deployment_id": "verifier-selected"},
                        "media": {"winner_deployment_id": "media-selected"},
                    },
                }
            ]
        if url == "/v1/workflow-composition/registry":
            return {
                "analysis_verify_v1": {
                    "workflow_spec_id": "analysis_verify_v1",
                    "selection_reason": "derived_family_winners",
                    "source_serving_release_id": "release-1",
                    "selected_node_map": {
                        "analyst_plan": {
                            "node_id": "analyst_plan",
                            "family_id": "analyst",
                            "deployment_id": "analyst-selected",
                        },
                        "verifier_review": {
                            "node_id": "verifier_review",
                            "family_id": "verifier",
                            "deployment_id": "verifier-selected",
                        },
                    },
                },
                "research_build_verify_v1": {
                    "workflow_spec_id": "research_build_verify_v1",
                    "selection_reason": "derived_family_winners",
                    "source_serving_release_id": "release-1",
                    "selected_node_map": {
                        "analyst_plan": {
                            "node_id": "analyst_plan",
                            "family_id": "analyst",
                            "deployment_id": "analyst-selected",
                        },
                        "builder_impl": {
                            "node_id": "builder_impl",
                            "family_id": "builder",
                            "deployment_id": "builder-selected",
                        },
                        "verifier_review": {
                            "node_id": "verifier_review",
                            "family_id": "verifier",
                            "deployment_id": "verifier-selected",
                        },
                    },
                },
                "concept_media_review_v1": {
                    "workflow_spec_id": "concept_media_review_v1",
                    "selection_reason": "derived_family_winners",
                    "source_serving_release_id": "release-1",
                    "selected_node_map": {
                        "analyst_brief": {
                            "node_id": "analyst_brief",
                            "family_id": "analyst",
                            "deployment_id": "analyst-selected",
                        },
                        "media_generate": {
                            "node_id": "media_generate",
                            "family_id": "media",
                            "deployment_id": "media-selected",
                        },
                        "verifier_review": {
                            "node_id": "verifier_review",
                            "family_id": "verifier",
                            "deployment_id": "verifier-selected",
                        },
                    },
                },
            }
        if url == "/v1/serving/releases/current":
            return {
                "release": {
                    "id": "release-1",
                    "metadata": {
                        "run_id": "run-1",
                        "selection_reason_by_family": {
                            "analyst": "family_protocol",
                            "builder": "family_protocol",
                            "verifier": "family_protocol",
                            "media": "family_protocol",
                        },
                        "serving_selected_deployment_id_by_family": {
                            "analyst": "analyst-selected",
                            "builder": "builder-selected",
                            "verifier": "verifier-selected",
                            "media": "media-selected",
                        },
                        "candidate_shortlists_by_family": {
                            "analyst": [
                                {
                                    "deployment_id": "analyst-selected",
                                    "serving_selection_score": 0.82,
                                    "official_family_score": 0.9,
                                    "reliability_score": 0.8,
                                }
                            ],
                            "builder": [
                                {
                                    "deployment_id": "builder-selected",
                                    "serving_selection_score": 0.88,
                                    "official_family_score": 0.92,
                                    "reliability_score": 0.83,
                                }
                            ],
                            "verifier": [
                                {
                                    "deployment_id": "verifier-selected",
                                    "serving_selection_score": 0.84,
                                    "official_family_score": 0.89,
                                    "reliability_score": 0.82,
                                }
                            ],
                            "media": [
                                {
                                    "deployment_id": "media-selected",
                                    "serving_selection_score": 0.86,
                                    "official_family_score": 0.91,
                                    "reliability_score": 0.81,
                                }
                            ],
                        },
                    },
                },
                "deployments": [
                    {"family_id": "analyst", "source_deployment_id": "analyst-selected"},
                    {"family_id": "builder", "source_deployment_id": "builder-selected"},
                    {"family_id": "verifier", "source_deployment_id": "verifier-selected"},
                    {"family_id": "media", "source_deployment_id": "media-selected"},
                ],
            }
        if url in {
            "/v1/families/analyst/targets",
            "/v1/families/builder/targets",
            "/v1/families/verifier/targets",
            "/v1/families/media/targets",
        }:
            family_id = url.split("/")[-2]
            return {
                "benchmark_version": f"{family_id}_family_v2",
                "rubric_version": f"{family_id}_family_rubric_v2",
                "retrieval_environment": {"mode": "live_web"} if family_id == "analyst" else {},
                "judge_config": {"model": "judge-model"} if family_id == "analyst" else {},
                "evaluation_bundle": {
                    "kind": "family_evaluation_bundle",
                    "tasks": (
                        [{"execution_mode": "live_web", "metadata": {}}] * 9
                        + [{"execution_mode": "live_web", "metadata": {"hidden_fixture": True}}] * 5
                        + [{"execution_mode": "replay_web", "metadata": {}}] * 9
                        + [{"execution_mode": "replay_web", "metadata": {"hidden_fixture": True}}] * 5
                        + [{"execution_mode": "offline_reasoning", "metadata": {}}] * 8
                        + [{"execution_mode": "offline_reasoning", "metadata": {"hidden_fixture": True}}] * 4
                        if family_id == "analyst"
                        else []
                    ),
                },
            }
        raise AssertionError(f"unexpected fetch url: {url}")

    async def fake_run_compose_smoke(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"submission_id": "sub-1", "deployment_id": "dep-1", "task_result": {"status": "completed"}}

    async def fake_collect_snapshot(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"services": {"owner": {"summary": {"ok": True}}}}

    async def fake_run_live_research_smoke(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"status": "completed", "retrieval_ledger_id": "ledger-1"}

    async def fake_run_e2e_smoke(args: Any) -> dict[str, Any]:
        if args.mode == "protocol":
            return {"submission": {"id": "sub-1"}, "validator": {"hotkey": "validator-1"}, "canonical": None}
        if args.mode == "scored":
            return {
                "canonical": {"status": "finalized"},
                "progress": {
                    "failure_classification_counts": {"infra_failed_results": 0},
                    "tasks": [{"judge_backed_result_count": 1, "score_bearing": True}],
                },
            }
        if args.mode == "task-failure":
            return {
                "canonical": {"status": "failed"},
                "failure_releases": [{"assignment_id": "assignment-1"}],
            }
        raise AssertionError(f"unexpected mode: {args.mode}")

    async def fake_post_json(
        *,
        client: Any,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del client, payload, headers
        if url == "/v1/operators/serving-releases/trigger":
            return {"status": "published"}
        raise AssertionError(f"unexpected post url: {url}")

    async def fake_post_gateway_task(
        *,
        api_gateway_url: str,
        consumer_api_url: str,
        gateway_api_key: str,
        consumer_api_key: str,
        raw_input: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        del api_gateway_url, consumer_api_url, gateway_api_key, consumer_api_key, timeout_seconds
        if raw_input == sv.DIRECT_ANALYSIS_PROMPT:
            return _task_payload(
                workflow_template="direct_analysis",
                dependency_graph_keys=[
                    "control_plane_context_pack",
                    "analyst_plan",
                    "verifier_review",
                ],
                source_node_id="verifier_review",
                source_family_id="verifier",
                specialist_node_ids=["analyst_plan", "verifier_review"],
            )
        if raw_input == sv.RESEARCH_TO_BUILD_PROMPT:
            return _task_payload(
                workflow_template="research_to_build",
                dependency_graph_keys=[
                    "control_plane_retrieval",
                    "control_plane_context_pack",
                    "analyst_plan",
                    "builder_impl",
                    "verifier_review",
                    "analyst_synthesis",
                ],
                source_node_id="analyst_synthesis",
                source_family_id="analyst",
                specialist_node_ids=["analyst_plan", "builder_impl", "verifier_review"],
                node_results=[
                    {
                        "node_id": "builder_impl",
                        "metadata": {
                            "upstream_node_ids": ["analyst_plan", "control_plane_context_pack"],
                        },
                    },
                    {
                        "node_id": "analyst_synthesis",
                        "metadata": {
                            "upstream_node_ids": [
                                "control_plane_context_pack",
                                "analyst_plan",
                                "builder_impl",
                                "verifier_review",
                            ],
                        },
                    },
                ],
                upstream_lineage=[
                    "control_plane_retrieval",
                    "control_plane_context_pack",
                    "analyst_plan",
                    "builder_impl",
                    "verifier_review",
                    "analyst_synthesis",
                ],
            )
        if raw_input == sv.CONCEPT_TO_MEDIA_PROMPT:
            return _task_payload(
                workflow_template="concept_to_media",
                dependency_graph_keys=[
                    "control_plane_context_pack",
                    "analyst_brief",
                    "media_generate",
                    "verifier_review",
                ],
                source_node_id="verifier_review",
                source_family_id="verifier",
                specialist_node_ids=["analyst_brief", "media_generate", "verifier_review"],
                node_results=[
                    {
                        "node_id": "media_generate",
                        "metadata": {"context_bundle_keys": ["artifacts", "constraints"]},
                        "output": {
                            "artifacts": [
                                {"artifact_id": "image-1", "kind": "image", "uri": "https://example.invalid/poster.png"}
                            ]
                        },
                    }
                ],
            )
        raise AssertionError(f"unexpected prompt: {raw_input}")

    async def fake_ensure_family_deployments(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {
            "status": "passed",
            "registry": {family_id: [{"hotkey": f"{family_id}-1"}] for family_id in sv.REQUIRED_FAMILIES},
            "missing_families": [],
            "seeded_submissions": [],
        }

    async def fake_endpoint_unreachable(**kwargs: Any) -> bool:
        del kwargs
        return True

    monkeypatch.setattr(sv, "_wait_json", fake_wait_json)
    monkeypatch.setattr(sv, "_fetch_json", fake_fetch_json)
    monkeypatch.setattr(sv, "run_compose_smoke", fake_run_compose_smoke)
    monkeypatch.setattr(sv, "_collect_snapshot", fake_collect_snapshot)
    monkeypatch.setattr(sv, "run_live_research_smoke", fake_run_live_research_smoke)
    monkeypatch.setattr(sv, "_run_e2e_smoke", fake_run_e2e_smoke)
    monkeypatch.setattr(sv, "_post_gateway_task", fake_post_gateway_task)
    monkeypatch.setattr(sv, "_post_json", fake_post_json)
    monkeypatch.setattr(sv, "_ensure_family_deployments", fake_ensure_family_deployments)
    monkeypatch.setattr(sv, "_endpoint_unreachable", fake_endpoint_unreachable)
    monkeypatch.setattr(sv, "_owner_signed_headers", lambda **kwargs: {"X-Validator": "ok"})

    report = await sv.run_staging_validation(
        stack="compose",
        owner_api_url="http://owner.local",
        api_gateway_url="http://gateway.local",
        consumer_api_url="http://consumer.local",
        validator_engine_url="http://validator.local",
        metagraph_listener_url="http://metagraph.local",
        provider_proxy_url="http://provider.local",
        gateway_api_key="gateway-key",
        consumer_api_key="consumer-key",
        miner_mnemonic="miner mnemonic",
        validator_mnemonic="validator mnemonic",
        sandbox_service_url="http://sandbox.local",
        seed_missing_families=False,
        family_miner_mnemonics=[],
        timeout_seconds=5.0,
    )

    assert report["status"] == "passed"
    assert report["compose_baseline"]["status"] == "passed"
    assert report["workflow_matrix"]["status"] == "passed"
    assert report["validator_promotion"]["status"] == "passed"
    assert report["serving_cutover"]["status"] == "passed"
    assert report["kubernetes_confirmation"]["status"] == "pending"
    assert report["workflow_matrix"]["scenarios"]["direct_analysis"]["checks"]["dependency_graph_node_keys"] is True
    assert report["workflow_matrix"]["scenarios"]["research_to_build"]["checks"]["ordered_upstream_lineage"] is True
    assert report["workflow_matrix"]["scenarios"]["concept_to_media"]["checks"]["media_artifacts_present"] is True
    assert report["serving_cutover"]["checks"]["builder_registry_matches_release"] is True
    assert report["serving_cutover"]["checks"]["builder_winner_matches_serving_release"] is True
    assert report["serving_cutover"]["checks"]["workflow_composition_matches_family_winners"] is True
    assert report["serving_cutover"]["checks"]["deprecated_endpoints_absent"] is True


@pytest.mark.asyncio
async def test_validate_deep_research_marks_dns_failure_blocked(monkeypatch):
    async def fake_fetch_json(
        *,
        client: Any,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del client, headers, params
        if url == "/v1/runs/current":
            return {"run_id": "run-1"}
        if url == "/v1/families/analyst/targets":
            return {"benchmark_version": "analyst_family_v3"}
        raise AssertionError(f"unexpected fetch url: {url}")

    async def fake_run_live_research_smoke(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise RuntimeError("[Errno -3] Temporary failure in name resolution")

    monkeypatch.setattr(sv, "_fetch_json", fake_fetch_json)
    monkeypatch.setattr(sv, "run_live_research_smoke", fake_run_live_research_smoke)
    monkeypatch.setattr(sv, "_owner_signed_headers", lambda **kwargs: {"X-Validator": "ok"})

    result = await sv._validate_deep_research(
        owner_api_url="http://owner.local",
        validator_mnemonic="validator mnemonic",
        timeout_seconds=5.0,
    )

    assert result["status"] == "blocked"
    assert "research-tool or provider configuration" in result["message"]


@pytest.mark.asyncio
async def test_validate_prompt_workflow_marks_provider_failure_blocked(monkeypatch):
    async def fake_post_gateway_task(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {
            "status": "failed",
            "execution_result": {
                "errors": ["provider proxy connection refused"],
                "final_output": {
                    "branch_failures": {
                        "media_generate": {"error": "provider proxy connection refused"}
                    }
                },
            },
        }

    monkeypatch.setattr(sv, "_post_gateway_task", fake_post_gateway_task)

    result = await sv._validate_prompt_workflow(
        name="concept_to_media",
        prompt=sv.CONCEPT_TO_MEDIA_PROMPT,
        api_gateway_url="http://gateway.local",
        consumer_api_url="http://consumer.local",
        gateway_api_key="gateway-key",
        consumer_api_key="consumer-key",
        timeout_seconds=5.0,
        validator=lambda payload: payload,
    )

    assert result["status"] == "blocked"
    assert "provider" in result["message"]


@pytest.mark.asyncio
async def test_ensure_family_deployments_uses_candidate_registry_before_seeding(monkeypatch):
    class FakeResponse:
        def __init__(self, payload: dict[str, Any]):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any):
            del args, kwargs

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        async def get(self, url: str, **kwargs: Any) -> FakeResponse:
            if url == "/v1/internal/registry":
                return FakeResponse({family_id: [] for family_id in sv.REQUIRED_FAMILIES})
            if url == "/v1/internal/candidate-registry":
                return FakeResponse({family_id: [{"hotkey": f"{family_id}-candidate"}] for family_id in sv.REQUIRED_FAMILIES})
            raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(sv.httpx, "AsyncClient", FakeClient)

    result = await sv._ensure_family_deployments(
        owner_api_url="http://owner.local",
        family_ids=list(sv.REQUIRED_FAMILIES),
        timeout_seconds=5.0,
        seed_missing_families=False,
        seed_mnemonics=[],
    )

    assert result["status"] == "passed"
    assert result["missing_families"] == []
    assert result["registry"]["analyst"][0]["hotkey"] == "analyst-candidate"
