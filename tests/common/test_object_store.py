from __future__ import annotations

import pytest

from shared.common.object_store import (
    ObjectStore,
    ObjectStoreError,
    parse_uri,
)


# -- parse_uri -----------------------------------------------------------


def test_parse_uri_s3():
    parsed = parse_uri("s3://my-bucket/datasets/bundles/run-001/analyst.json")
    assert parsed.is_s3
    assert parsed.bucket == "my-bucket"
    assert parsed.key == "datasets/bundles/run-001/analyst.json"


def test_parse_uri_file_scheme():
    parsed = parse_uri("file:///var/lib/eirel/analyst.json")
    assert parsed.is_file
    assert parsed.key == "/var/lib/eirel/analyst.json"


def test_parse_uri_bare_path_treated_as_file():
    parsed = parse_uri("/var/lib/eirel/analyst.json")
    assert parsed.is_file
    assert parsed.key == "/var/lib/eirel/analyst.json"


def test_parse_uri_rejects_empty():
    with pytest.raises(ObjectStoreError):
        parse_uri("")


def test_parse_uri_rejects_unknown_scheme():
    with pytest.raises(ObjectStoreError):
        parse_uri("ftp://example.com/file.json")


def test_parse_uri_rejects_s3_without_bucket():
    with pytest.raises(ObjectStoreError):
        parse_uri("s3:///key-only")


# -- file:// roundtrip ---------------------------------------------------


async def test_file_put_and_fetch_roundtrip(tmp_path):
    store = ObjectStore()
    target = tmp_path / "nested" / "analyst.json"
    payload = b'{"family_id": "analyst"}'

    await store.put(f"file://{target}", payload)
    assert target.exists()

    fetched = await store.fetch(f"file://{target}")
    assert fetched == payload


async def test_file_fetch_missing_raises(tmp_path):
    store = ObjectStore()
    with pytest.raises(ObjectStoreError):
        await store.fetch(f"file://{tmp_path / 'missing.json'}")


async def test_file_exists(tmp_path):
    store = ObjectStore()
    target = tmp_path / "exists.json"
    target.write_bytes(b"x")

    assert await store.exists(f"file://{target}") is True
    assert await store.exists(f"file://{tmp_path / 'nope.json'}") is False


async def test_file_list(tmp_path):
    store = ObjectStore()
    (tmp_path / "seeds" / "analyst").mkdir(parents=True)
    (tmp_path / "seeds" / "analyst" / "a.yaml").write_bytes(b"x")
    (tmp_path / "seeds" / "analyst" / "b.yaml").write_bytes(b"y")
    (tmp_path / "seeds" / "analyst" / "sub" / "c.yaml").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "seeds" / "analyst" / "sub" / "c.yaml").write_bytes(b"z")

    entries = [uri async for uri in store.list(f"file://{tmp_path / 'seeds' / 'analyst'}")]
    assert len(entries) == 3
    assert all(uri.startswith("file://") for uri in entries)
    assert all(uri.endswith(".yaml") for uri in entries)


# -- s3:// roundtrip (moto) ---------------------------------------------


@pytest.fixture
def s3_bucket():
    from moto import mock_aws
    import boto3

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-bucket")
        yield client


async def test_s3_put_and_fetch_roundtrip(s3_bucket):
    store = ObjectStore(s3_client=s3_bucket)
    uri = "s3://test-bucket/datasets/bundles/run-001/analyst.json"
    payload = b'{"family_id": "analyst"}'

    await store.put(uri, payload, metadata={"generator_version": "v1"})
    assert await store.exists(uri) is True

    fetched = await store.fetch(uri)
    assert fetched == payload


async def test_s3_list(s3_bucket):
    store = ObjectStore(s3_client=s3_bucket)
    await store.put("s3://test-bucket/seeds/analyst/a.yaml", b"x")
    await store.put("s3://test-bucket/seeds/analyst/b.yaml", b"y")
    await store.put("s3://test-bucket/seeds/verifier/c.yaml", b"z")

    analyst_entries = [uri async for uri in store.list("s3://test-bucket/seeds/analyst/")]
    assert sorted(analyst_entries) == [
        "s3://test-bucket/seeds/analyst/a.yaml",
        "s3://test-bucket/seeds/analyst/b.yaml",
    ]


async def test_s3_fetch_missing_raises(s3_bucket):
    store = ObjectStore(s3_client=s3_bucket)
    with pytest.raises(ObjectStoreError):
        await store.fetch("s3://test-bucket/missing/key.json")


async def test_s3_exists_false_for_missing(s3_bucket):
    store = ObjectStore(s3_client=s3_bucket)
    assert await store.exists("s3://test-bucket/never-written.json") is False


async def test_default_s3_client_uses_r2_when_env_set(monkeypatch):
    """R2 env vars override the default AWS session and configure a
    sig-v4 boto3 client pointed at the R2 endpoint."""
    monkeypatch.setenv("EIREL_R2_ACCOUNT_ID", "acct-123")
    monkeypatch.setenv("EIREL_R2_ACCESS_KEY_ID", "AKIA-r2")
    monkeypatch.setenv("EIREL_R2_SECRET_ACCESS_KEY", "secret-r2")
    from shared.common.object_store import _build_default_s3_client
    client = _build_default_s3_client()
    # boto3 client carries the configured endpoint on .meta.endpoint_url
    assert client.meta.endpoint_url == "https://acct-123.r2.cloudflarestorage.com"


async def test_default_s3_client_falls_back_to_aws_when_no_r2(monkeypatch):
    for v in ("EIREL_R2_ACCOUNT_ID", "EIREL_R2_ACCESS_KEY_ID", "EIREL_R2_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
    from shared.common.object_store import _build_default_s3_client
    client = _build_default_s3_client()
    # Default AWS endpoint (no custom override).
    assert "r2.cloudflarestorage.com" not in (client.meta.endpoint_url or "")

