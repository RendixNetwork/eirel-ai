"""Tests for orchestration.consumer_api.mcp_routes."""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import httpx
import pytest
from fastapi import Depends, FastAPI, Header, HTTPException, status
from httpx import ASGITransport, AsyncClient

from shared.common.database import Database
from shared.common.models import (
    ConsumerMcpConnection,
    ConsumerUser,
    McpIntegration,
)
from shared.safety.token_encryption import build_token_cipher
from orchestration.consumer_api.mcp_routes import (
    build_admin_router,
    build_catalog_router,
    build_connections_router,
)


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'mcp_routes.db'}")
    db.create_all()
    return db


def _seed_user(session) -> str:
    user_id = str(uuid4())
    session.add(ConsumerUser(
        user_id=user_id, auth_subject=f"k:{user_id}", display_name="X",
    ))
    session.commit()
    return user_id


def _build_admin_app(db: Database) -> FastAPI:
    app = FastAPI()
    app.state.database = db
    app.state.mcp_token_cipher = build_token_cipher()

    async def _require_admin(authorization: str | None = Header(default=None)):
        if authorization != "Bearer admin-token":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return "admin@test"

    app.include_router(
        build_admin_router(
            require_admin=_require_admin, allow_http_base_urls=True,
        )
    )
    return app


def _build_consumer_app(
    db: Database,
    *,
    user_id: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.database = db
    app.state.mcp_token_cipher = build_token_cipher()

    async def _require_user(x_user_id: str | None = Header(default=None)) -> str:
        if x_user_id != user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return user_id

    app.include_router(build_catalog_router(require_user=_require_user))
    app.include_router(
        build_connections_router(
            require_user=_require_user, transport=transport,
        )
    )
    return app


# -- Admin router ----------------------------------------------------------


async def test_admin_create_lists_and_hashes(tmp_path):
    db = _make_db(tmp_path)
    app = _build_admin_app(db)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as client:
        # Unauth rejected.
        unauth = await client.post(
            "/v1/admin/mcp_integrations",
            json={"slug": "x", "display_name": "X", "base_url": "http://x.test/m"},
        )
        assert unauth.status_code == 401
        # Create.
        resp = await client.post(
            "/v1/admin/mcp_integrations",
            json={
                "slug": "notion", "display_name": "Notion",
                "base_url": "http://notion.test/mcp",
                "capabilities": [
                    {"name": "search", "description": "search"},
                    {"name": "create_page", "description": "create"},
                ],
            },
            headers={"Authorization": "Bearer admin-token"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["slug"] == "notion"
        assert len(body["capabilities_hash"]) == 64  # sha256 hex
        # Capabilities are canonicalized (sorted by name).
        names = [c["name"] for c in body["capabilities"]]
        assert names == ["create_page", "search"]
        # List.
        listed = await client.get(
            "/v1/admin/mcp_integrations",
            headers={"Authorization": "Bearer admin-token"},
        )
        assert listed.status_code == 200
        assert len(listed.json()) == 1


async def test_admin_create_blocks_private_ip_base_url(tmp_path):
    db = _make_db(tmp_path)
    app = _build_admin_app(db)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as client:
        resp = await client.post(
            "/v1/admin/mcp_integrations",
            json={
                "slug": "evil", "display_name": "Evil",
                "base_url": "http://169.254.169.254/m",
            },
            headers={"Authorization": "Bearer admin-token"},
        )
    assert resp.status_code == 400
    assert "ssrf_blocked" in resp.json()["detail"]


async def test_admin_create_rejects_duplicate_slug(tmp_path):
    db = _make_db(tmp_path)
    app = _build_admin_app(db)
    body = {
        "slug": "notion", "display_name": "Notion",
        "base_url": "http://notion.test/mcp",
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as client:
        h = {"Authorization": "Bearer admin-token"}
        first = await client.post("/v1/admin/mcp_integrations", json=body, headers=h)
        assert first.status_code == 201
        dup = await client.post("/v1/admin/mcp_integrations", json=body, headers=h)
    assert dup.status_code == 409


async def test_admin_patch_can_disable(tmp_path):
    db = _make_db(tmp_path)
    app = _build_admin_app(db)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as client:
        h = {"Authorization": "Bearer admin-token"}
        created = (await client.post(
            "/v1/admin/mcp_integrations",
            json={"slug": "x", "display_name": "X",
                  "base_url": "http://x.test/m"},
            headers=h,
        )).json()
        patch = await client.patch(
            f"/v1/admin/mcp_integrations/{created['id']}",
            json={"status": "disabled"},
            headers=h,
        )
    assert patch.status_code == 200
    assert patch.json()["status"] == "disabled"


async def test_admin_reprobe_replaces_capabilities_hash(tmp_path):
    db = _make_db(tmp_path)
    app = _build_admin_app(db)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as client:
        h = {"Authorization": "Bearer admin-token"}
        created = (await client.post(
            "/v1/admin/mcp_integrations",
            json={
                "slug": "x", "display_name": "X",
                "base_url": "http://x.test/m",
                "capabilities": [{"name": "old"}],
            },
            headers=h,
        )).json()
        old_hash = created["capabilities_hash"]
        reprobe = await client.post(
            f"/v1/admin/mcp_integrations/{created['id']}/reprobe",
            json=[
                {"name": "new1"},
                {"name": "new2"},
            ],
            headers=h,
        )
    body = reprobe.json()
    assert body["capabilities_hash"] != old_hash
    assert [c["name"] for c in body["capabilities"]] == ["new1", "new2"]


# -- Catalog router (consumer browse) -------------------------------------


async def test_catalog_lists_only_active(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
        session.add(McpIntegration(
            slug="active-slug", display_name="Active",
            base_url="https://a.test/m",
            capabilities_json=[{"name": "search"}],
            capabilities_hash="hh",
            status="active",
        ))
        session.add(McpIntegration(
            slug="disabled-slug", display_name="Disabled",
            base_url="https://b.test/m",
            status="disabled",
        ))
        session.commit()
    app = _build_consumer_app(db, user_id=user_id)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as client:
        resp = await client.get(
            "/v1/mcp/integrations",
            headers={"x-user-id": user_id},
        )
    assert resp.status_code == 200
    slugs = [r["slug"] for r in resp.json()]
    assert slugs == ["active-slug"]
    # Public projection — internal fields aren't exposed.
    item = resp.json()[0]
    assert "base_url" not in item
    assert "capabilities_hash" not in item


# -- Connections router (consumer manage) ---------------------------------


async def test_connection_post_creates_pending_row_and_returns_authorize_url(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
        integration = McpIntegration(
            slug="notion", display_name="Notion",
            base_url="https://notion.test/mcp",
            oauth_authorize_url="https://notion.test/oauth/authorize",
            oauth_token_url="https://notion.test/oauth/token",
            oauth_scopes_json=["read:docs"],
            status="active",
        )
        session.add(integration)
        session.commit()
        integration_id = integration.id
    app = _build_consumer_app(db, user_id=user_id)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as client:
        resp = await client.post(
            "/v1/users/me/mcp_connections",
            json={"integration_id": integration_id},
            headers={"x-user-id": user_id},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["authorize_url"].startswith("https://notion.test/oauth/authorize")
    assert "state=" in body["authorize_url"]
    # Pending row persisted.
    with db.sessionmaker() as session:
        conn = session.get(ConsumerMcpConnection, body["connection_id"])
        assert conn.status == "pending"
        assert conn.user_id == user_id


async def test_connection_post_rejects_disabled_integration(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
        integration = McpIntegration(
            slug="x", display_name="X",
            base_url="https://x.test/m", status="disabled",
        )
        session.add(integration)
        session.commit()
        integration_id = integration.id
    app = _build_consumer_app(db, user_id=user_id)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as client:
        resp = await client.post(
            "/v1/users/me/mcp_connections",
            json={"integration_id": integration_id},
            headers={"x-user-id": user_id},
        )
    assert resp.status_code == 404


async def test_connection_post_rejects_duplicate(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
        integration = McpIntegration(
            slug="notion", display_name="Notion",
            base_url="https://notion.test/m",
            oauth_token_url="https://notion.test/oauth/token",
            status="active",
        )
        session.add(integration)
        session.commit()
        integration_id = integration.id
    app = _build_consumer_app(db, user_id=user_id)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as client:
        h = {"x-user-id": user_id}
        first = await client.post(
            "/v1/users/me/mcp_connections",
            json={"integration_id": integration_id}, headers=h,
        )
        assert first.status_code == 202
        dup = await client.post(
            "/v1/users/me/mcp_connections",
            json={"integration_id": integration_id}, headers=h,
        )
    assert dup.status_code == 409


async def test_oauth_callback_exchanges_code_and_persists_encrypted_token(tmp_path):
    db = _make_db(tmp_path)
    cipher = build_token_cipher()
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
        integration = McpIntegration(
            slug="notion", display_name="Notion",
            base_url="https://notion.test/m",
            oauth_authorize_url="https://notion.test/oauth/authorize",
            oauth_token_url="https://notion.test/oauth/token",
            status="active",
        )
        session.add(integration)
        session.commit()
        integration_id = integration.id

    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(200, json={
            "access_token": "tok-abc",
            "refresh_token": "ref-xyz",
            "expires_in": 3600,
        })

    app = _build_consumer_app(
        db, user_id=user_id, transport=httpx.MockTransport(handler),
    )
    app.state.mcp_token_cipher = cipher  # ensure same cipher instance
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as client:
        h = {"x-user-id": user_id}
        # Begin connect to get state token.
        begin = await client.post(
            "/v1/users/me/mcp_connections",
            json={"integration_id": integration_id}, headers=h,
        )
        body = begin.json()
        state = body["state"]
        # OAuth callback.
        callback = await client.get(
            "/v1/users/me/mcp_connections/oauth/callback",
            params={"code": "AUTH_CODE", "state": state},
            headers=h,
        )
    assert callback.status_code == 200
    # Access token persisted, encrypted, status active.
    with db.sessionmaker() as session:
        conn = session.get(ConsumerMcpConnection, body["connection_id"])
        assert conn.status == "active"
        assert conn.oauth_access_token_encrypted is not None
        assert (
            cipher.decrypt(bytes(conn.oauth_access_token_encrypted)).decode()
            == "tok-abc"
        )
        assert conn.oauth_refresh_token_encrypted is not None
        assert conn.oauth_expires_at is not None


async def test_oauth_callback_rejects_unknown_state(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
    app = _build_consumer_app(db, user_id=user_id)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as client:
        resp = await client.get(
            "/v1/users/me/mcp_connections/oauth/callback",
            params={"code": "x", "state": "ghost"},
            headers={"x-user-id": user_id},
        )
    assert resp.status_code == 400
    assert "state_token_unknown" in resp.json()["detail"]


async def test_connection_delete_revokes(tmp_path):
    db = _make_db(tmp_path)
    cipher = build_token_cipher()
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
        integration = McpIntegration(
            slug="notion", display_name="Notion",
            base_url="https://notion.test/m", status="active",
        )
        session.add(integration)
        session.flush()
        connection = ConsumerMcpConnection(
            user_id=user_id, integration_id=integration.id,
            oauth_access_token_encrypted=cipher.encrypt(b"tok"),
            status="active",
        )
        session.add(connection)
        session.commit()
        cid = connection.id
    app = _build_consumer_app(db, user_id=user_id)
    app.state.mcp_token_cipher = cipher
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as client:
        resp = await client.delete(
            f"/v1/users/me/mcp_connections/{cid}",
            headers={"x-user-id": user_id},
        )
    assert resp.status_code == 204
    with db.sessionmaker() as session:
        assert session.get(ConsumerMcpConnection, cid) is None


async def test_connection_list_only_returns_users_own_rows(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        u1 = _seed_user(session)
        u2 = _seed_user(session)
        integration = McpIntegration(
            slug="notion", display_name="Notion",
            base_url="https://notion.test/m", status="active",
        )
        session.add(integration)
        session.flush()
        session.add(ConsumerMcpConnection(
            user_id=u1, integration_id=integration.id, status="active",
        ))
        session.add(ConsumerMcpConnection(
            user_id=u2, integration_id=integration.id, status="active",
        ))
        session.commit()
    app = _build_consumer_app(db, user_id=u1)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as client:
        resp = await client.get(
            "/v1/users/me/mcp_connections",
            headers={"x-user-id": u1},
        )
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["integration_slug"] == "notion"
