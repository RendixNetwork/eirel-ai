"""Tests for the graph-runtime checkpoint env injection in pod manifests."""
from __future__ import annotations

import os
from types import SimpleNamespace

from infra.miner_runtime.runtime_manager import _deployment_manifest_common


def _fake_manifest(*, kind: str = "base_agent", invoke_path: str = "/v1/agent/infer"):
    """Build a SimpleNamespace mimicking SubmissionManifest for the helper."""
    return SimpleNamespace(
        runtime=SimpleNamespace(
            kind=kind,
            port=8080,
            health_path="/healthz",
            invoke_path=invoke_path,
        ),
        sdk_runtime=SimpleNamespace(
            package_mode="package",
            entry_module="app",
            app_object="app",
            dependency_group="",
        ),
        inference=SimpleNamespace(
            providers=["chutes"],
            model="test-model",
        ),
    )


def _env_dict(manifest_objs: list[dict]) -> dict[str, str]:
    """Pull the container env vars out of the deployment manifest list."""
    for obj in manifest_objs:
        if obj.get("kind") == "Deployment":
            containers = obj["spec"]["template"]["spec"]["containers"]
            return {entry["name"]: entry["value"] for entry in containers[0].get("env", [])}
    raise AssertionError("no Deployment object in the manifest list")


def test_graph_runtime_injects_checkpoint_env(monkeypatch):
    monkeypatch.setenv("OWNER_API_URL", "http://owner-api:8080")
    monkeypatch.delenv("EIREL_CHECKPOINT_BACKEND_URL", raising=False)
    monkeypatch.setenv("EIREL_RESUME_TOKEN_SECRET", "rotation-key")

    objs = _deployment_manifest_common(
        deployment_name="d-1",
        service_name="svc-d-1",
        submission_id="sub-1",
        artifact_url="img:test",
        manifest=_fake_manifest(kind="graph"),
        internal_service_token="svc-token",
        provider_proxy_url="http://proxy:8082",
        provider_proxy_token="proxy-tok",
        assigned_node_name=None,
        requested_cpu_millis=500,
        requested_memory_bytes=512 * 1024 * 1024,
        deployment_id="deploy-42",
    )
    env = _env_dict(objs)

    assert env.get("EIREL_CHECKPOINT_BACKEND_URL") == "http://owner-api:8080"
    assert env.get("EIREL_CHECKPOINT_NAMESPACE") == "miner-deploy-42"
    assert env.get("EIREL_CHECKPOINT_BACKEND_TOKEN") == "svc-token"
    assert env.get("EIREL_RESUME_TOKEN_SECRET") == "rotation-key"


def test_base_agent_runtime_skips_checkpoint_env():
    objs = _deployment_manifest_common(
        deployment_name="d-2",
        service_name="svc-d-2",
        submission_id="sub-2",
        artifact_url="img:test",
        manifest=_fake_manifest(kind="base_agent"),
        internal_service_token="svc-token",
        provider_proxy_url="http://proxy:8082",
        provider_proxy_token="proxy-tok",
        assigned_node_name=None,
        requested_cpu_millis=500,
        requested_memory_bytes=512 * 1024 * 1024,
        deployment_id="deploy-43",
    )
    env = _env_dict(objs)
    assert "EIREL_CHECKPOINT_BACKEND_URL" not in env
    assert "EIREL_CHECKPOINT_NAMESPACE" not in env
    assert "EIREL_CHECKPOINT_BACKEND_TOKEN" not in env


def test_graph_runtime_no_deployment_id_skips_env():
    """Without deployment_id we can't form a namespace; env is skipped."""
    objs = _deployment_manifest_common(
        deployment_name="d-3",
        service_name="svc-d-3",
        submission_id="sub-3",
        artifact_url="img:test",
        manifest=_fake_manifest(kind="graph"),
        internal_service_token="svc-token",
        provider_proxy_url="http://proxy:8082",
        provider_proxy_token="proxy-tok",
        assigned_node_name=None,
        requested_cpu_millis=500,
        requested_memory_bytes=512 * 1024 * 1024,
        deployment_id=None,
    )
    env = _env_dict(objs)
    assert "EIREL_CHECKPOINT_BACKEND_URL" not in env
    assert "EIREL_CHECKPOINT_NAMESPACE" not in env
