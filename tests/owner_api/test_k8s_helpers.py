from __future__ import annotations

import io
import tarfile
from types import SimpleNamespace

import pytest

from infra.miner_runtime._k8s_helpers import (
    _CODE_ARCHIVE_KEY,
    _build_code_configmap,
    _build_network_policy,
    _check_archive_configmap_size,
    _extract_archive_to_dict,
    _repack_clean_archive,
    parse_cpu_to_millis,
    parse_memory_to_bytes,
)
from infra.miner_runtime.runtime_manager import (
    RuntimeManagerError,
    _deployment_manifest_common,
)


def _make_targz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path, data in files.items():
            info = tarfile.TarInfo(name=path)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _stub_manifest(port: int = 8080, health_path: str = "/healthz"):
    return SimpleNamespace(
        runtime=SimpleNamespace(
            port=port,
            health_path=health_path,
            invoke_path="/v1/chat/completions",
        ),
    )


def _base_kwargs(**overrides):
    defaults = {
        "deployment_name": "miner-test-123",
        "service_name": "miner-test-123",
        "submission_id": "test-123",
        "artifact_url": "registry.local/eirel-miner-runtime:latest",
        "manifest": _stub_manifest(),
        "internal_service_token": "tok",
        "provider_proxy_url": "http://proxy:8092",
        "provider_proxy_token": "ptok",
        "assigned_node_name": None,
        "requested_cpu_millis": 500,
        "requested_memory_bytes": 256 * 1024 * 1024,
    }
    defaults.update(overrides)
    return defaults


# -- parse_cpu_to_millis -----------


def test_parse_cpu_to_millis_handles_m_suffix_and_plain_int():
    assert parse_cpu_to_millis("500m") == 500
    assert parse_cpu_to_millis("2") == 2000
    assert parse_cpu_to_millis(2) == 2000
    assert parse_cpu_to_millis("1.5") == 1500
    assert parse_cpu_to_millis("0m") == 0


def test_parse_cpu_to_millis_handles_empty_and_none():
    assert parse_cpu_to_millis("") == 0
    assert parse_cpu_to_millis(None) == 0
    assert parse_cpu_to_millis("  ") == 0


# -- parse_memory_to_bytes -----------


def test_parse_memory_to_bytes_handles_mi_gi_binary_suffixes():
    assert parse_memory_to_bytes("256Mi") == 256 * 2**20
    assert parse_memory_to_bytes("1Gi") == 2**30
    assert parse_memory_to_bytes("2Ti") == 2 * 2**40
    assert parse_memory_to_bytes("4Ki") == 4 * 2**10


def test_parse_memory_to_bytes_handles_m_g_decimal_suffixes():
    assert parse_memory_to_bytes("512M") == 512 * 10**6
    assert parse_memory_to_bytes("1G") == 10**9
    assert parse_memory_to_bytes("2T") == 2 * 10**12
    assert parse_memory_to_bytes("8K") == 8 * 10**3


def test_parse_memory_to_bytes_handles_plain_bytes():
    assert parse_memory_to_bytes("1000000") == 1000000
    assert parse_memory_to_bytes(1073741824) == 1073741824
    assert parse_memory_to_bytes("") == 0
    assert parse_memory_to_bytes(None) == 0


# -- _deployment_manifest_common -----------


def _get_deployment_and_container(manifests):
    deployment = next(m for m in manifests if m.get("kind") == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    return deployment, container


def _env_value(container, name: str) -> str | None:
    for entry in container.get("env", []) or []:
        if entry.get("name") == name:
            return entry.get("value")
    return None


def test_manifest_sets_job_id_from_deployment_id_when_provided():
    manifests = _deployment_manifest_common(
        **_base_kwargs(deployment_id="dep-uuid-abc"),
    )
    _, container = _get_deployment_and_container(manifests)
    job_id = _env_value(container, "EIREL_PROVIDER_PROXY_JOB_ID")
    # This key is the load-bearing one for cost attribution: it must
    # match the path owner-api queries at
    # ``ScoringManager.fetch_deployment_cost`` (``/v1/jobs/miner-<dep>/cost``).
    assert job_id == "miner-dep-uuid-abc"


def test_manifest_falls_back_to_submission_id_when_deployment_id_absent():
    manifests = _deployment_manifest_common(**_base_kwargs())
    _, container = _get_deployment_and_container(manifests)
    job_id = _env_value(container, "EIREL_PROVIDER_PROXY_JOB_ID")
    # Back-compat: callers that don't thread deployment_id yet get the
    # old submission-id behaviour, not a silent empty job_id.
    assert job_id == "miner-test-123"


def test_manifest_always_emits_readiness_and_liveness_probes():
    manifests = _deployment_manifest_common(**_base_kwargs())
    _, container = _get_deployment_and_container(manifests)
    assert "readinessProbe" in container
    assert container["readinessProbe"]["httpGet"]["path"] == "/healthz"
    assert "livenessProbe" in container
    assert container["livenessProbe"]["initialDelaySeconds"] == 60
    assert container["livenessProbe"]["failureThreshold"] == 4


def test_manifest_always_emits_runtime_pool_hard_selector():
    for node_name in [None, "worker-1"]:
        manifests = _deployment_manifest_common(
            **_base_kwargs(assigned_node_name=node_name),
        )
        deployment, _ = _get_deployment_and_container(manifests)
        ns = deployment["spec"]["template"]["spec"]["nodeSelector"]
        assert ns["eirel.dev/runtime-pool"] == "true"
        assert ns["eirel.dev/runtime-class"] == "miner"


def test_manifest_has_no_hostname_nodeselector_when_node_assigned():
    manifests = _deployment_manifest_common(
        **_base_kwargs(assigned_node_name="worker-1"),
    )
    deployment, _ = _get_deployment_and_container(manifests)
    ns = deployment["spec"]["template"]["spec"]["nodeSelector"]
    assert "kubernetes.io/hostname" not in ns


def test_manifest_emits_soft_affinity_when_node_assigned():
    manifests = _deployment_manifest_common(
        **_base_kwargs(assigned_node_name="worker-1"),
    )
    deployment, _ = _get_deployment_and_container(manifests)
    pod_spec = deployment["spec"]["template"]["spec"]
    assert "affinity" in pod_spec
    prefs = pod_spec["affinity"]["nodeAffinity"][
        "preferredDuringSchedulingIgnoredDuringExecution"
    ]
    assert prefs[0]["weight"] == 100
    expr = prefs[0]["preference"]["matchExpressions"][0]
    assert expr["key"] == "kubernetes.io/hostname"
    assert expr["values"] == ["worker-1"]


def test_manifest_emits_envfrom_shared_secret_when_provided():
    manifests = _deployment_manifest_common(
        **_base_kwargs(shared_secret_name="eirel-runtime-shared"),
    )
    _, container = _get_deployment_and_container(manifests)
    assert "envFrom" in container
    assert container["envFrom"] == [
        {"secretRef": {"name": "eirel-runtime-shared"}},
    ]
    env_names = {e["name"] for e in container["env"]}
    assert "INTERNAL_SERVICE_TOKEN" not in env_names
    assert "MINER_SUBMISSION_ID" in env_names


def test_manifest_omits_envfrom_when_shared_secret_none():
    manifests = _deployment_manifest_common(**_base_kwargs())
    _, container = _get_deployment_and_container(manifests)
    assert "envFrom" not in container
    env_names = {e["name"] for e in container["env"]}
    assert "INTERNAL_SERVICE_TOKEN" in env_names
    assert "PROVIDER_PROXY_URL" in env_names


def test_manifest_emits_configmap_volume_when_code_configmap_name_set():
    manifests = _deployment_manifest_common(
        **_base_kwargs(code_configmap_name="miner-test-123-code"),
    )
    deployment, container = _get_deployment_and_container(manifests)
    pod_spec = deployment["spec"]["template"]["spec"]

    # Runtime container imports from the extracted emptyDir, never the
    # raw ConfigMap.
    assert any(
        vm["mountPath"] == "/submission"
        and vm["name"] == "submission-code-extracted"
        for vm in container["volumeMounts"]
    )

    # ConfigMap (single archive key) is mounted only into the init
    # container, which unpacks it into the shared emptyDir.
    init = pod_spec["initContainers"][0]
    assert init["image"] == container["image"]
    assert init["command"][:2] == ["python", "-c"]
    assert "filter='data'" in init["command"][2]
    init_mounts = {vm["name"]: vm["mountPath"] for vm in init["volumeMounts"]}
    assert init_mounts["submission-code"] == "/submission-archive"
    assert init_mounts["submission-code-extracted"] == "/submission"

    vols = {v["name"]: v for v in pod_spec["volumes"]}
    assert vols["submission-code"]["configMap"]["name"] == "miner-test-123-code"
    assert "emptyDir" in vols["submission-code-extracted"]


def test_manifest_emits_network_policy_when_emit_network_policy_true():
    manifests = _deployment_manifest_common(
        **_base_kwargs(
            emit_network_policy=True,
            system_namespace="eirel-system",
            control_plane_namespace="eirel-control-plane",
        ),
    )
    np = next(m for m in manifests if m.get("kind") == "NetworkPolicy")
    assert np["spec"]["podSelector"] == {
        "matchLabels": {"app": "miner-test-123"},
    }
    assert "Ingress" in np["spec"]["policyTypes"]
    assert "Egress" in np["spec"]["policyTypes"]
    ingress_from = np["spec"]["ingress"][0]["from"]
    assert ingress_from[0]["namespaceSelector"] == {"matchLabels": {"name": "eirel-system"}}
    assert ingress_from[1]["namespaceSelector"] == {"matchLabels": {"name": "eirel-control-plane"}}
    egress_dns = np["spec"]["egress"][1]
    assert egress_dns["ports"] == [{"port": 53, "protocol": "UDP"}]


# -- _build_network_policy (k8s backend) -----------


def test_build_network_policy_includes_system_and_control_plane_namespaces():
    np = _build_network_policy(
        name="miner-abc",
        submission_id="abc",
        system_namespace="eirel-system",
        control_plane_namespace="eirel-control-plane",
        port=8080,
    )
    ingress_from = np["spec"]["ingress"][0]["from"]
    assert len(ingress_from) == 2
    assert ingress_from[0]["namespaceSelector"] == {"matchLabels": {"name": "eirel-system"}}
    assert ingress_from[1]["namespaceSelector"] == {"matchLabels": {"name": "eirel-control-plane"}}


def test_build_network_policy_uses_custom_control_plane_namespace():
    np = _build_network_policy(
        name="miner-xyz",
        submission_id="xyz",
        system_namespace="custom-sys",
        control_plane_namespace="custom-cp",
        port=9090,
    )
    ingress_from = np["spec"]["ingress"][0]["from"]
    assert ingress_from[0]["namespaceSelector"] == {"matchLabels": {"name": "custom-sys"}}
    assert ingress_from[1]["namespaceSelector"] == {"matchLabels": {"name": "custom-cp"}}


def test_build_network_policy_blocks_direct_https_egress():
    """Miners must not have a blanket 0.0.0.0/0:443 rule — otherwise they
    could call api.openai.com directly and bypass the provider-proxy
    allowlist. Egress must be scoped to provider-proxy + DNS (+ host tool
    services when host_ip is provided).
    """
    np = _build_network_policy(
        name="miner-abc",
        submission_id="abc",
        system_namespace="eirel-system",
        control_plane_namespace="eirel-control-plane",
        port=8080,
    )
    for rule in np["spec"]["egress"]:
        for peer in rule.get("to", []) or []:
            cidr = (peer.get("ipBlock") or {}).get("cidr")
            assert cidr != "0.0.0.0/0", (
                "egress rule allows unscoped outbound HTTPS — miners would "
                "bypass provider-proxy and reach api.openai.com directly"
            )


# -- single-archive ConfigMap ------


def test_configmap_has_single_archive_key_for_nested_submission():
    """A submission with subdirectories must NOT produce per-path
    ConfigMap keys (those fail k8s validation with 422). One opaque
    archive key only."""
    archive = _make_targz({
        "app.py": b"x = 1\n",
        "bench/tasks.py": b"# nested\n",
        "tests/test_middleware.py": b"# deep/nested\n",
        "pkg/sub/mod.py": b"# very/deep\n",
    })
    files = _extract_archive_to_dict(archive)
    clean = _repack_clean_archive(files)
    cm = _build_code_configmap(
        name="miner-x-code", submission_id="x", archive_bytes=clean,
    )
    assert list(cm["binaryData"].keys()) == [_CODE_ARCHIVE_KEY]
    # No '/' in any key — the whole class of 422 failures is gone.
    assert all("/" not in k for k in cm["binaryData"])


def test_repacked_archive_round_trips_and_drops_junk():
    archive = _make_targz({
        "app.py": b"MAIN\n",
        "pkg/util.py": b"UTIL\n",
        "__pycache__/app.cpython-312.pyc": b"\x00junk",
        ".git/config": b"[core]\n",
    })
    files = _extract_archive_to_dict(archive)
    clean = _repack_clean_archive(files)
    extracted: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(clean), mode="r:gz") as tar:
        for m in tar.getmembers():
            f = tar.extractfile(m)
            if f is not None:
                extracted[m.name] = f.read()
    assert extracted == {"app.py": b"MAIN\n", "pkg/util.py": b"UTIL\n"}


def test_repacked_archive_is_deterministic():
    archive = _make_targz({"b.py": b"B", "a/c.py": b"C", "a.py": b"A"})
    files = _extract_archive_to_dict(archive)
    assert _repack_clean_archive(files) == _repack_clean_archive(files)


def test_archive_size_check_rejects_oversized():
    big = b"\x00" * (2 * 1024 * 1024)
    with pytest.raises(RuntimeManagerError, match="too large for ConfigMap"):
        _check_archive_configmap_size(big)


def test_archive_size_check_allows_small():
    _check_archive_configmap_size(b"small archive")  # no raise


def test_extract_archive_rejects_path_escape():
    escape = _make_targz({"../evil.py": b"PWN"})
    with pytest.raises(ValueError, match="escape root"):
        _extract_archive_to_dict(escape)
