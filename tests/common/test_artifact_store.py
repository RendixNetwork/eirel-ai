from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

from shared.common.artifacts import (
    FilesystemArtifactStore,
    S3CompatibleArtifactStore,
    create_artifact_store,
    sha256_hex,
)


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[(Bucket, Key)] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, BytesIO]:
        return {"Body": BytesIO(self.objects[(Bucket, Key)])}


def test_filesystem_artifact_store_roundtrip(tmp_path):
    store = FilesystemArtifactStore(root=str(tmp_path), bucket="managed-local")
    content = b"artifact-payload"

    stored = store.put_bytes(storage_key="builder/image/test.bin", content=content)

    assert stored.storage_key == "builder/image/test.bin"
    assert stored.storage_uri == "s3://managed-local/builder/image/test.bin"
    assert stored.sha256 == sha256_hex(content)
    assert store.get_bytes(storage_key=stored.storage_key) == content


def test_s3_compatible_artifact_store_roundtrip():
    client = FakeS3Client()
    store = S3CompatibleArtifactStore(
        bucket="managed-prod",
        prefix="production",
        client=client,
    )
    content = b"artifact-payload"

    stored = store.put_bytes(storage_key="builder/image/test.bin", content=content)

    assert stored.storage_key == "production/builder/image/test.bin"
    assert stored.storage_uri == "s3://managed-prod/production/builder/image/test.bin"
    assert stored.sha256 == sha256_hex(content)
    assert store.get_bytes(storage_key=stored.storage_key) == content


def test_create_artifact_store_selects_s3_backend():
    settings = SimpleNamespace(
        object_storage_backend="s3",
        artifact_storage_bucket="managed-prod",
        object_storage_endpoint_url="https://s3.example.com",
        object_storage_region="us-east-1",
        object_storage_access_key_id="key",
        object_storage_secret_access_key="secret",
        object_storage_prefix="production",
        object_storage_use_ssl=True,
        object_storage_addressing_style="path",
    )

    try:
        store = create_artifact_store(settings)
    except Exception as exc:
        # boto3 may not be installed in the active test environment; the package dependency is declared
        assert "boto3" in str(exc).lower()
    else:
        assert isinstance(store, S3CompatibleArtifactStore)
