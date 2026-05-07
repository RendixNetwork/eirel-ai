from __future__ import annotations

from pathlib import Path

import pytest

from shared.common.config import Settings, _resolve_fixture_path


def test_resolve_fixture_path_prefers_explicit_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    container = tmp_path / "container"
    container.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setenv("EIREL_TEST_FIXTURE_PATH", str(explicit))

    resolved = _resolve_fixture_path(
        env_name="EIREL_TEST_FIXTURE_PATH",
        container_default=str(container),
        workspace_default=str(workspace),
        repo_relative_default="does/not/matter",
        expected_kind="directory",
    )

    assert resolved == str(explicit.resolve())


def test_resolve_fixture_path_prefers_container_path_before_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.delenv("EIREL_TEST_FIXTURE_PATH", raising=False)
    container = tmp_path / "container.json"
    workspace = tmp_path / "workspace.json"
    container.write_text("{}", encoding="utf-8")
    workspace.write_text("{}", encoding="utf-8")

    resolved = _resolve_fixture_path(
        env_name="EIREL_TEST_FIXTURE_PATH",
        container_default=str(container),
        workspace_default=str(workspace),
        repo_relative_default="does/not/matter.json",
        expected_kind="file",
    )

    assert resolved == str(container.resolve())


def test_resolve_fixture_path_falls_back_to_workspace_when_container_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.delenv("EIREL_TEST_FIXTURE_PATH", raising=False)
    container = tmp_path / "missing-container"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    resolved = _resolve_fixture_path(
        env_name="EIREL_TEST_FIXTURE_PATH",
        container_default=str(container),
        workspace_default=str(workspace),
        repo_relative_default="does/not/matter",
        expected_kind="directory",
    )

    assert resolved == str(workspace.resolve())


def test_resolve_fixture_path_raises_clear_value_error_for_missing_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    missing = tmp_path / "missing.json"
    monkeypatch.setenv("EIREL_TEST_FIXTURE_PATH", str(missing))

    with pytest.raises(ValueError, match="invalid EIREL_TEST_FIXTURE_PATH: expected an existing file") as exc:
        _resolve_fixture_path(
            env_name="EIREL_TEST_FIXTURE_PATH",
            container_default=str(tmp_path / "container.json"),
            workspace_default=str(tmp_path / "workspace.json"),
            repo_relative_default="missing/repo-default.json",
            expected_kind="file",
        )

    message = str(exc.value)
    assert str(missing.resolve()) in message
    assert "EIREL_TEST_FIXTURE_PATH=" in message


def test_settings_honors_explicit_fixture_root_envs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    datasets = tmp_path / "owner_datasets"
    datasets.mkdir(exist_ok=True)

    monkeypatch.setenv("EIREL_OWNER_DATASET_ROOT_PATH", str(datasets))

    settings = Settings()

    assert settings.owner_dataset_root_path == str(datasets.resolve())


def test_settings_rejects_example_owner_dataset_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    examples_root = tmp_path / "examples" / "family_datasets"
    examples_root.mkdir(parents=True)
    monkeypatch.setenv("EIREL_OWNER_DATASET_ROOT_PATH", str(examples_root))

    with pytest.raises(
        ValueError,
        match="invalid EIREL_OWNER_DATASET_ROOT_PATH: official owner dataset roots may not live under examples/\\*\\*",
    ):
        Settings()
