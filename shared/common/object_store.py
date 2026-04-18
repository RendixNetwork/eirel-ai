"""URI-native object store for dataset bundles, seeds, and manifests.

This module is complementary to ``shared.common.artifacts``. ``ArtifactStore`` is
a key-addressable content store for miner submission archives; ``ObjectStore``
is a URI-native fetch/put/list abstraction used by the dataset forge and the
owner-api dataset loader. It understands two URI schemes:

- ``file://`` — local filesystem, used in dev, tests (via ``file://`` fixtures),
  and as a disk cache layer in front of S3.
- ``s3://bucket/key`` — S3-compatible storage; synchronous boto3 client wrapped
  in ``asyncio.to_thread`` so callers can ``await`` without dragging in aioboto3.

Typical use::

    store = ObjectStore.from_settings(get_settings())
    payload = await store.fetch("s3://eirel-owner-private/datasets/bundles/run-001/analyst.json")
    await store.put("s3://eirel-owner-private/datasets/bundles/run-001/analyst.json", payload)
    async for uri in store.list("s3://eirel-owner-private/datasets/seeds/analyst/"):
        ...
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class ObjectStoreError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ParsedUri:
    scheme: str
    bucket: str
    key: str

    @property
    def is_s3(self) -> bool:
        return self.scheme == "s3"

    @property
    def is_file(self) -> bool:
        return self.scheme == "file"


def parse_uri(uri: str) -> ParsedUri:
    if not uri:
        raise ObjectStoreError("object store URI must be non-empty")
    parsed = urlparse(uri)
    scheme = (parsed.scheme or "").lower()
    if scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket:
            raise ObjectStoreError(f"s3 URI missing bucket: {uri!r}")
        return ParsedUri(scheme="s3", bucket=bucket, key=key)
    if scheme == "file":
        return ParsedUri(scheme="file", bucket="", key=parsed.path)
    if scheme == "":
        # Bare path: treat as file://
        return ParsedUri(scheme="file", bucket="", key=uri)
    raise ObjectStoreError(f"unsupported object store URI scheme: {scheme!r} ({uri!r})")


class ObjectStore:
    """URI-native object store supporting ``file://`` and ``s3://`` schemes."""

    def __init__(self, *, s3_client: Any | None = None) -> None:
        self._s3_client = s3_client

    @classmethod
    def from_settings(cls, settings: Any) -> ObjectStore:
        return cls(s3_client=None)

    # -- fetch -----------------------------------------------------------

    async def fetch(self, uri: str) -> bytes:
        parsed = parse_uri(uri)
        if parsed.is_file:
            return await asyncio.to_thread(_read_file, parsed.key)
        if parsed.is_s3:
            client = self._require_s3_client()
            return await asyncio.to_thread(_s3_get_object, client, parsed.bucket, parsed.key)
        raise ObjectStoreError(f"unsupported URI: {uri!r}")

    # -- put -------------------------------------------------------------

    async def put(
        self,
        uri: str,
        content: bytes,
        *,
        metadata: dict[str, str] | None = None,
    ) -> None:
        parsed = parse_uri(uri)
        if parsed.is_file:
            await asyncio.to_thread(_write_file, parsed.key, content)
            return
        if parsed.is_s3:
            client = self._require_s3_client()
            await asyncio.to_thread(
                _s3_put_object, client, parsed.bucket, parsed.key, content, metadata or {}
            )
            return
        raise ObjectStoreError(f"unsupported URI: {uri!r}")

    # -- list ------------------------------------------------------------

    async def list(self, prefix_uri: str) -> AsyncIterator[str]:
        parsed = parse_uri(prefix_uri)
        if parsed.is_file:
            for entry in await asyncio.to_thread(_list_files, parsed.key):
                yield f"file://{entry}"
            return
        if parsed.is_s3:
            client = self._require_s3_client()
            keys = await asyncio.to_thread(_s3_list_objects, client, parsed.bucket, parsed.key)
            for key in keys:
                yield f"s3://{parsed.bucket}/{key}"
            return
        raise ObjectStoreError(f"unsupported URI: {prefix_uri!r}")

    # -- exists ----------------------------------------------------------

    async def exists(self, uri: str) -> bool:
        parsed = parse_uri(uri)
        if parsed.is_file:
            return await asyncio.to_thread(_file_exists, parsed.key)
        if parsed.is_s3:
            client = self._require_s3_client()
            return await asyncio.to_thread(_s3_object_exists, client, parsed.bucket, parsed.key)
        return False

    # -- sync helpers ---------------------------------------------------
    #
    # Owner-api lives inside an asyncio event loop, but its bundle loader
    # path runs inside the sync ``RunManager`` (which holds a sync SQLAlchemy
    # session). Calling ``asyncio.run()`` from inside a running loop raises,
    # so we expose direct sync versions of the read paths. The underlying
    # boto3 / file IO is already synchronous — the async API is just a
    # convenience for orchestrator-side callers.

    def fetch_sync(self, uri: str) -> bytes:
        parsed = parse_uri(uri)
        if parsed.is_file:
            return _read_file(parsed.key)
        if parsed.is_s3:
            client = self._require_s3_client()
            return _s3_get_object(client, parsed.bucket, parsed.key)
        raise ObjectStoreError(f"unsupported URI: {uri!r}")

    def exists_sync(self, uri: str) -> bool:
        parsed = parse_uri(uri)
        if parsed.is_file:
            return _file_exists(parsed.key)
        if parsed.is_s3:
            client = self._require_s3_client()
            return _s3_object_exists(client, parsed.bucket, parsed.key)
        return False

    # -- internals -------------------------------------------------------

    def _require_s3_client(self) -> Any:
        if self._s3_client is None:
            self._s3_client = _build_default_s3_client()
        return self._s3_client


# -- filesystem helpers --------------------------------------------------


def _read_file(path: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise ObjectStoreError(f"file not found: {path!r}") from exc
    except OSError as exc:
        raise ObjectStoreError(f"failed to read file {path!r}: {exc}") from exc


def _write_file(path: str, content: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def _list_files(root: str) -> list[str]:
    base = Path(root)
    if not base.exists():
        return []
    if base.is_file():
        return [str(base)]
    return sorted(str(p) for p in base.rglob("*") if p.is_file())


def _file_exists(path: str) -> bool:
    return Path(path).exists()


# -- s3 helpers (sync; wrapped via to_thread) ----------------------------


def _build_default_s3_client() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise ObjectStoreError(
            "boto3 is required for s3:// URIs but is not installed"
        ) from exc
    session = boto3.session.Session()
    return session.client("s3")


def _s3_get_object(client: Any, bucket: str, key: str) -> bytes:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        raise ObjectStoreError(f"failed to fetch s3://{bucket}/{key}: {exc}") from exc
    body = response.get("Body", b"")
    if hasattr(body, "read"):
        return body.read()
    if isinstance(body, bytes):
        return body
    raise ObjectStoreError(f"unexpected s3 response body type for s3://{bucket}/{key}")


def _s3_put_object(
    client: Any,
    bucket: str,
    key: str,
    content: bytes,
    metadata: dict[str, str],
) -> None:
    kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key, "Body": content}
    if metadata:
        kwargs["Metadata"] = {str(k): str(v) for k, v in metadata.items()}
    try:
        client.put_object(**kwargs)
    except Exception as exc:
        raise ObjectStoreError(f"failed to put s3://{bucket}/{key}: {exc}") from exc


def _s3_list_objects(client: Any, bucket: str, prefix: str) -> list[str]:
    try:
        paginator = client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for entry in page.get("Contents", []) or []:
                key = entry.get("Key")
                if key:
                    keys.append(str(key))
        return keys
    except Exception as exc:
        raise ObjectStoreError(f"failed to list s3://{bucket}/{prefix}: {exc}") from exc


def _s3_object_exists(client: Any, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False
