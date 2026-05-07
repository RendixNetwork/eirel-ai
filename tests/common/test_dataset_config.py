from __future__ import annotations

import pytest

from shared.common.config import _validate_dataset_source_type


def test_validate_dataset_source_type_accepts_filesystem():
    _validate_dataset_source_type("filesystem")


def test_validate_dataset_source_type_accepts_s3():
    _validate_dataset_source_type("s3")


def test_validate_dataset_source_type_rejects_unknown():
    with pytest.raises(ValueError, match="EIREL_OWNER_DATASET_SOURCE_TYPE"):
        _validate_dataset_source_type("gcs")


def test_settings_accepts_filesystem_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    from shared.common.config import Settings, reset_settings

    reset_settings()
    datasets = tmp_path / "owner_datasets"
    (datasets / "families").mkdir(parents=True, exist_ok=True)
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    workflow_corpus = tmp_path / "workflow_corpus"
    workflow_corpus.mkdir()
    web_search_catalog = tmp_path / "web_search_tool_catalog.json"
    web_search_catalog.write_text("{}", encoding="utf-8")
    retrieval_snapshot = tmp_path / "snapshot.json"
    retrieval_snapshot.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("EIREL_OWNER_DATASET_ROOT_PATH", str(datasets / "families"))
    monkeypatch.setenv("EIREL_CALIBRATION_FIXTURES_ROOT_PATH", str(calibration))
    monkeypatch.setenv("EIREL_WORKFLOW_CORPUS_ROOT_PATH", str(workflow_corpus))
    monkeypatch.setenv("EIREL_WEB_SEARCH_TOOL_CATALOG_PATH", str(web_search_catalog))
    monkeypatch.setenv("EIREL_RETRIEVAL_SNAPSHOT_PATH", str(retrieval_snapshot))
    monkeypatch.setenv("EIREL_OWNER_DATASET_SOURCE_TYPE", "filesystem")

    settings = Settings()
    assert settings.owner_dataset_source_type == "filesystem"
