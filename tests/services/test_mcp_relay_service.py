"""Tests for tool_platforms.mcp_relay_service."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

import httpx

from shared.common.database import Database
from shared.common.models import (
    ConsumerMcpConnection,
    ConsumerUser,
    McpIntegration,
)
from shared.safety.token_encryption import TokenCipher, build_token_cipher
from tool_platforms.mcp_relay_service._capabilities import hash_capabilities
from tool_platforms.mcp_relay_service.app import create_app


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'mcp.db'}")
    db.create_all()
    return db


def _seed_user_integration_connection(
    session,
    *,
    cipher: TokenCipher,
    base_url: str = "https://notion.test/mcp",
    capabilities_hash: str = "h0",
    access_token: str = "secret-token",
    integration_status: str = "active",
    connection_status: str = "active",
) -> tuple[str, str, str]:
    user_id = str(uuid4())
    session.add(ConsumerUser(
        user_id=user_id, auth_subject=f"k:{user_id}", display_name="X",
    ))
    integration_id = str(uuid4())
    session.add(McpIntegration(
        id=integration_id, slug="notion", display_name="Notion",
        vendor="Notion", base_url=base_url,
        capabilities_json=[
            {"name": "search", "description": "search docs"},
        ],
        capabilities_hash=capabilities_hash,
        status=integration_status,
    ))
    connection_id = str(uuid4())
    session.add(ConsumerMcpConnection(
        id=connection_id, user_id=user_id, integration_id=integration_id,
        oauth_access_token_encrypted=cipher.encrypt(access_token.encode()),
        status=connection_status,
    ))
    session.commit()
    return user_id, integration_id, connection_id


def _build_app(
    db: Database,
    *,
    handler,
    cipher: TokenCipher | None = None,
):
    transport = httpx.MockTransport(handler)
    app = create_app(
        database=db,
        cipher=cipher or build_token_cipher(),
        transport=transport,
        allow_http=True,
    )
    return app


# -- list_tools ------------------------------------------------------------


async def test_list_tools_canonicalizes_and_hashes(tmp_path):
    db = _make_db(tmp_path)
    cipher = build_token_cipher()
    with db.sessionmaker() as session:
        _u, integration_id, _c = _seed_user_integration_connection(
            session, cipher=cipher,
        )

    declared = [
        {"name": "search", "description": "search"},
        {"name": "create_page", "description": "make a page"},
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/list_tools")
        return httpx.Response(200, json={"tools": declared})

    app = _build_app(db, handler=handler, cipher=cipher)
    async with app.router.lifespan_context(app):
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://relay",
        )
        async with client:
            resp = await client.post(
                f"/v1/relay/integrations/{integration_id}/list_tools"
            )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["capabilities_hash"] == hash_capabilities(declared)
    # Sorted by name.
    assert [t["name"] for t in body["tools"]] == ["create_page", "search"]


async def test_list_tools_404_when_integration_unknown(tmp_path):
    db = _make_db(tmp_path)

    def handler(req):
        return httpx.Response(500)

    app = _build_app(db, handler=handler)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://relay",
        ) as client:
            resp = await client.post(
                "/v1/relay/integrations/ghost/list_tools"
            )
    assert resp.status_code == 404


# -- call -----------------------------------------------------------------


async def test_call_invokes_upstream_with_decrypted_oauth_token(tmp_path):
    db = _make_db(tmp_path)
    cipher = build_token_cipher()
    with db.sessionmaker() as session:
        _u, _i, connection_id = _seed_user_integration_connection(
            session, cipher=cipher, access_token="t-secret-42",
        )

    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append({
            "url": str(req.url),
            "auth": req.headers.get("authorization"),
            "body": json.loads(req.content) if req.content else None,
        })
        return httpx.Response(200, json={"output": "ok", "rows": [1, 2]})

    app = _build_app(db, handler=handler, cipher=cipher)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://relay",
        ) as client:
            resp = await client.post(
                f"/v1/relay/connections/{connection_id}/call",
                json={"tool_name": "search", "args": {"q": "hello"}},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["result"]["output"] == "ok"
    assert body["latency_ms"] >= 0
    # Outbound call carried the decrypted bearer token.
    assert len(captured) == 1
    assert captured[0]["auth"] == "Bearer t-secret-42"
    assert captured[0]["url"].endswith("/tools/search")
    assert captured[0]["body"] == {"q": "hello"}


async def test_call_rejects_capabilities_hash_drift(tmp_path):
    db = _make_db(tmp_path)
    cipher = build_token_cipher()
    with db.sessionmaker() as session:
        _u, _i, connection_id = _seed_user_integration_connection(
            session, cipher=cipher, capabilities_hash="stored",
        )

    def handler(req):
        # Should never be called on hash mismatch.
        raise AssertionError("upstream contacted on hash mismatch")

    app = _build_app(db, handler=handler, cipher=cipher)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://relay",
        ) as client:
            resp = await client.post(
                f"/v1/relay/connections/{connection_id}/call",
                json={
                    "tool_name": "search",
                    "args": {},
                    "capabilities_hash": "drifted",
                },
            )
    assert resp.status_code == 409
    assert "capabilities_hash_mismatch" in resp.json()["detail"]


async def test_call_404_when_connection_unknown(tmp_path):
    db = _make_db(tmp_path)

    def handler(req):
        return httpx.Response(500)

    app = _build_app(db, handler=handler)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://relay",
        ) as client:
            resp = await client.post(
                "/v1/relay/connections/ghost/call",
                json={"tool_name": "x", "args": {}},
            )
    assert resp.status_code == 404


async def test_call_409_when_connection_inactive(tmp_path):
    db = _make_db(tmp_path)
    cipher = build_token_cipher()
    with db.sessionmaker() as session:
        _u, _i, connection_id = _seed_user_integration_connection(
            session, cipher=cipher, connection_status="revoked",
        )

    def handler(req):
        raise AssertionError("upstream contacted on inactive connection")

    app = _build_app(db, handler=handler, cipher=cipher)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://relay",
        ) as client:
            resp = await client.post(
                f"/v1/relay/connections/{connection_id}/call",
                json={"tool_name": "x", "args": {}},
            )
    assert resp.status_code == 409


async def test_call_409_when_integration_disabled(tmp_path):
    db = _make_db(tmp_path)
    cipher = build_token_cipher()
    with db.sessionmaker() as session:
        _u, _i, connection_id = _seed_user_integration_connection(
            session, cipher=cipher, integration_status="disabled",
        )

    def handler(req):
        raise AssertionError("upstream contacted on disabled integration")

    app = _build_app(db, handler=handler, cipher=cipher)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://relay",
        ) as client:
            resp = await client.post(
                f"/v1/relay/connections/{connection_id}/call",
                json={"tool_name": "x", "args": {}},
            )
    assert resp.status_code == 409
    assert "integration_disabled" in resp.json()["detail"]


async def test_call_returns_error_on_upstream_failure(tmp_path):
    db = _make_db(tmp_path)
    cipher = build_token_cipher()
    with db.sessionmaker() as session:
        _u, _i, connection_id = _seed_user_integration_connection(
            session, cipher=cipher,
        )

    def handler(req):
        return httpx.Response(500, text="nope")

    app = _build_app(db, handler=handler, cipher=cipher)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://relay",
        ) as client:
            resp = await client.post(
                f"/v1/relay/connections/{connection_id}/call",
                json={"tool_name": "x", "args": {}},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "upstream_error" in body["error"]


async def test_call_blocks_private_ip_base_url(tmp_path):
    db = _make_db(tmp_path)
    cipher = build_token_cipher()
    with db.sessionmaker() as session:
        _u, _i, connection_id = _seed_user_integration_connection(
            session, cipher=cipher,
            base_url="http://169.254.169.254/mcp",
        )

    def handler(req):
        raise AssertionError("upstream contacted despite SSRF policy")

    # require_https=False keeps this isolated to the IP-range check.
    app = _build_app(db, handler=handler, cipher=cipher)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://relay",
        ) as client:
            resp = await client.post(
                f"/v1/relay/connections/{connection_id}/call",
                json={"tool_name": "x", "args": {}},
            )
    assert resp.status_code == 400
    assert "ssrf_blocked" in resp.json()["detail"]


async def test_auth_required_when_token_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("EIREL_MCP_RELAY_TOKEN", "relay-secret")
    db = _make_db(tmp_path)
    cipher = build_token_cipher()
    with db.sessionmaker() as session:
        _u, _i, connection_id = _seed_user_integration_connection(
            session, cipher=cipher,
        )

    def handler(req):
        return httpx.Response(200, json={"ok": True})

    app = _build_app(db, handler=handler, cipher=cipher)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://relay",
        ) as client:
            unauth = await client.post(
                f"/v1/relay/connections/{connection_id}/call",
                json={"tool_name": "x", "args": {}},
            )
            authed = await client.post(
                f"/v1/relay/connections/{connection_id}/call",
                json={"tool_name": "x", "args": {}},
                headers={"Authorization": "Bearer relay-secret"},
            )
    assert unauth.status_code == 401
    assert authed.status_code == 200


async def test_call_updates_connection_last_used_at(tmp_path):
    db = _make_db(tmp_path)
    cipher = build_token_cipher()
    with db.sessionmaker() as session:
        _u, _i, connection_id = _seed_user_integration_connection(
            session, cipher=cipher,
        )

    def handler(req):
        return httpx.Response(200, json={"ok": True})

    app = _build_app(db, handler=handler, cipher=cipher)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://relay",
        ) as client:
            await client.post(
                f"/v1/relay/connections/{connection_id}/call",
                json={"tool_name": "x", "args": {}},
            )
    with db.sessionmaker() as session:
        conn = session.get(ConsumerMcpConnection, connection_id)
        assert conn.last_used_at is not None
