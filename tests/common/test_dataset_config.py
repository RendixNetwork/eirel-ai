from __future__ import annotations

import pytest

from shared.common.config import (
    _validate_dataset_source_type,
    _validate_forge_cross_vendor,
)


def test_validate_dataset_source_type_accepts_filesystem():
    _validate_dataset_source_type("filesystem")


def test_validate_dataset_source_type_accepts_s3():
    _validate_dataset_source_type("s3")


def test_validate_dataset_source_type_rejects_unknown():
    with pytest.raises(ValueError, match="EIREL_OWNER_DATASET_SOURCE_TYPE"):
        _validate_dataset_source_type("gcs")


def test_validate_forge_cross_vendor_accepts_when_both_empty():
    _validate_forge_cross_vendor(
        generator_provider="",
        generator_base_url="",
        judge_base_url="",
        must_differ=True,
    )


def test_validate_forge_cross_vendor_accepts_distinct_base_urls():
    _validate_forge_cross_vendor(
        generator_provider="openai",
        generator_base_url="https://api.openai.com/v1",
        judge_base_url="https://api.anthropic.com/v1",
        must_differ=True,
    )


def test_validate_forge_cross_vendor_rejects_identical_base_urls():
    with pytest.raises(ValueError, match="must differ from EIREL_JUDGE_BASE_URL"):
        _validate_forge_cross_vendor(
            generator_provider="openai",
            generator_base_url="https://api.openai.com/v1",
            judge_base_url="https://api.openai.com/v1",
            must_differ=True,
        )


def test_validate_forge_cross_vendor_skips_when_override_disabled():
    _validate_forge_cross_vendor(
        generator_provider="openai",
        generator_base_url="https://api.openai.com/v1",
        judge_base_url="https://api.openai.com/v1",
        must_differ=False,
    )


def test_settings_rejects_s3_mode_without_owner_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """H1: in S3 mode the owner secret is mandatory.

    Without it, the forge's hidden allocation RNG and the C1 task-ID hash
    are predictable from run_id alone, which would defeat C1/C2/C3 in any
    attacker who knows the run_id schedule.
    """
    from shared.common.config import Settings, reset_settings

    reset_settings()
    # Need the filesystem paths to resolve successfully so __post_init__
    # reaches the S3/owner_secret check rather than failing earlier.
    datasets = tmp_path / "owner_datasets"
    (datasets / "families").mkdir(parents=True)
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    workflow_corpus = tmp_path / "workflow_corpus"
    workflow_corpus.mkdir()
    research_catalog = tmp_path / "research_tool_catalog.json"
    research_catalog.write_text("{}", encoding="utf-8")
    retrieval_snapshot = tmp_path / "snapshot.json"
    retrieval_snapshot.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("EIREL_OWNER_DATASET_ROOT_PATH", str(datasets / "families"))
    monkeypatch.setenv("EIREL_CALIBRATION_FIXTURES_ROOT_PATH", str(calibration))
    monkeypatch.setenv("EIREL_WORKFLOW_CORPUS_ROOT_PATH", str(workflow_corpus))
    monkeypatch.setenv("EIREL_RESEARCH_TOOL_CATALOG_PATH", str(research_catalog))
    monkeypatch.setenv("EIREL_RETRIEVAL_SNAPSHOT_PATH", str(retrieval_snapshot))
    monkeypatch.setenv("EIREL_OWNER_DATASET_SOURCE_TYPE", "s3")
    monkeypatch.delenv("EIREL_DATASET_FORGE_OWNER_SECRET", raising=False)

    with pytest.raises(ValueError, match="EIREL_DATASET_FORGE_OWNER_SECRET must be set"):
        Settings()


def test_settings_accepts_s3_mode_with_owner_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    from shared.common.config import Settings, reset_settings

    reset_settings()
    datasets = tmp_path / "owner_datasets"
    (datasets / "families").mkdir(parents=True)
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    workflow_corpus = tmp_path / "workflow_corpus"
    workflow_corpus.mkdir()
    research_catalog = tmp_path / "research_tool_catalog.json"
    research_catalog.write_text("{}", encoding="utf-8")
    retrieval_snapshot = tmp_path / "snapshot.json"
    retrieval_snapshot.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("EIREL_OWNER_DATASET_ROOT_PATH", str(datasets / "families"))
    monkeypatch.setenv("EIREL_CALIBRATION_FIXTURES_ROOT_PATH", str(calibration))
    monkeypatch.setenv("EIREL_WORKFLOW_CORPUS_ROOT_PATH", str(workflow_corpus))
    monkeypatch.setenv("EIREL_RESEARCH_TOOL_CATALOG_PATH", str(research_catalog))
    monkeypatch.setenv("EIREL_RETRIEVAL_SNAPSHOT_PATH", str(retrieval_snapshot))
    monkeypatch.setenv("EIREL_OWNER_DATASET_SOURCE_TYPE", "s3")
    monkeypatch.setenv("EIREL_DATASET_FORGE_OWNER_SECRET", "not-a-real-secret-test-only")

    settings = Settings()
    assert settings.owner_dataset_source_type == "s3"
    assert settings.dataset_forge_owner_secret == "not-a-real-secret-test-only"


def test_settings_allows_empty_owner_secret_in_filesystem_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """Dev/test mode (filesystem) does not require the secret."""
    from shared.common.config import Settings, reset_settings

    reset_settings()
    datasets = tmp_path / "owner_datasets"
    (datasets / "families").mkdir(parents=True)
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    workflow_corpus = tmp_path / "workflow_corpus"
    workflow_corpus.mkdir()
    research_catalog = tmp_path / "research_tool_catalog.json"
    research_catalog.write_text("{}", encoding="utf-8")
    retrieval_snapshot = tmp_path / "snapshot.json"
    retrieval_snapshot.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("EIREL_OWNER_DATASET_ROOT_PATH", str(datasets / "families"))
    monkeypatch.setenv("EIREL_CALIBRATION_FIXTURES_ROOT_PATH", str(calibration))
    monkeypatch.setenv("EIREL_WORKFLOW_CORPUS_ROOT_PATH", str(workflow_corpus))
    monkeypatch.setenv("EIREL_RESEARCH_TOOL_CATALOG_PATH", str(research_catalog))
    monkeypatch.setenv("EIREL_RETRIEVAL_SNAPSHOT_PATH", str(retrieval_snapshot))
    monkeypatch.setenv("EIREL_OWNER_DATASET_SOURCE_TYPE", "filesystem")
    monkeypatch.delenv("EIREL_DATASET_FORGE_OWNER_SECRET", raising=False)

    settings = Settings()
    assert settings.owner_dataset_source_type == "filesystem"
    assert settings.dataset_forge_owner_secret == ""
