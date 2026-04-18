from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class ArtifactStoreError(RuntimeError):
    pass


@dataclass(slots=True)
class StoredArtifact:
    storage_key: str
    storage_uri: str
    sha256: str
    size_bytes: int


class ArtifactStore(Protocol):
    def put_bytes(self, *, storage_key: str, content: bytes) -> StoredArtifact:
        ...

    def get_bytes(self, *, storage_key: str) -> bytes:
        ...


class FilesystemArtifactStore:
    def __init__(self, *, root: str, bucket: str = "eirel-managed") -> None:
        self.root = Path(root)
        self.bucket = bucket
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, *, storage_key: str, content: bytes) -> StoredArtifact:
        normalized_key = _normalize_storage_key(storage_key)
        target = self.root / normalized_key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        digest = sha256_hex(content)
        return StoredArtifact(
            storage_key=normalized_key,
            storage_uri=f"s3://{self.bucket}/{normalized_key}",
            sha256=digest,
            size_bytes=len(content),
        )

    def get_bytes(self, *, storage_key: str) -> bytes:
        normalized_key = _normalize_storage_key(storage_key)
        return (self.root / normalized_key).read_bytes()


class S3CompatibleArtifactStore:
    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str = "",
        region: str = "",
        access_key_id: str = "",
        secret_access_key: str = "",
        prefix: str = "",
        use_ssl: bool = True,
        addressing_style: str = "auto",
        client: Any | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if client is None:
            try:
                import boto3
                from botocore.client import Config as BotoConfig
            except ImportError as exc:
                raise ArtifactStoreError(
                    "boto3 is required for the S3-compatible artifact backend"
                ) from exc
            session = boto3.session.Session()
            client = session.client(
                "s3",
                endpoint_url=endpoint_url or None,
                region_name=region or None,
                aws_access_key_id=access_key_id or None,
                aws_secret_access_key=secret_access_key or None,
                use_ssl=use_ssl,
                config=BotoConfig(s3={"addressing_style": addressing_style}),
            )
        self.client = client

    def put_bytes(self, *, storage_key: str, content: bytes) -> StoredArtifact:
        normalized_key = self._full_key(storage_key)
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=normalized_key,
                Body=content,
            )
        except Exception as exc:
            raise ArtifactStoreError(f"failed to upload artifact '{normalized_key}'") from exc
        digest = sha256_hex(content)
        return StoredArtifact(
            storage_key=normalized_key,
            storage_uri=f"s3://{self.bucket}/{normalized_key}",
            sha256=digest,
            size_bytes=len(content),
        )

    def get_bytes(self, *, storage_key: str) -> bytes:
        normalized_key = self._full_key(storage_key, normalize_only=True)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=normalized_key)
        except Exception as exc:
            raise ArtifactStoreError(f"failed to fetch artifact '{normalized_key}'") from exc
        body = response.get("Body", b"")
        if hasattr(body, "read"):
            return body.read()
        if isinstance(body, bytes):
            return body
        raise ArtifactStoreError("unsupported S3 response body type")

    def _full_key(self, storage_key: str, *, normalize_only: bool = False) -> str:
        normalized = _normalize_storage_key(storage_key)
        if not self.prefix:
            return normalized
        if normalized == self.prefix or normalized.startswith(f"{self.prefix}/"):
            return normalized
        return f"{self.prefix}/{normalized}"


def create_artifact_store(settings: Any) -> ArtifactStore:
    backend = str(getattr(settings, "object_storage_backend", "filesystem")).strip().lower()
    if backend == "s3":
        return S3CompatibleArtifactStore(
            bucket=getattr(settings, "artifact_storage_bucket", "eirel-managed"),
            endpoint_url=getattr(settings, "object_storage_endpoint_url", ""),
            region=getattr(settings, "object_storage_region", ""),
            access_key_id=getattr(settings, "object_storage_access_key_id", ""),
            secret_access_key=getattr(settings, "object_storage_secret_access_key", ""),
            prefix=getattr(settings, "object_storage_prefix", ""),
            use_ssl=bool(getattr(settings, "object_storage_use_ssl", True)),
            addressing_style=getattr(settings, "object_storage_addressing_style", "auto"),
        )
    return FilesystemArtifactStore(
        root=getattr(settings, "artifact_storage_root", "/tmp/eirel-managed-artifacts"),
        bucket=getattr(settings, "artifact_storage_bucket", "eirel-managed"),
    )


def _normalize_storage_key(storage_key: str) -> str:
    key = storage_key.strip().lstrip("/")
    if not key:
        raise ArtifactStoreError("artifact storage key must be non-empty")
    return key
