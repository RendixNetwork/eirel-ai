"""HTTP routes for operator-curated MCP integrations + consumer connections.

Three router groups bundled in this module:

  * ``admin_router`` — operator catalog CRUD. Add / list / patch / reprobe
    integrations. Guarded by an admin Bearer token.
  * ``catalog_router`` — consumer-facing browse. Lists active catalog
    integrations with a public projection (no internal fields).
  * ``connections_router`` — per-user OAuth connect / list / disconnect
    flow. Returns the integration's ``oauth_authorize_url`` on POST;
    the OAuth callback exchanges the code for an access token and
    persists it encrypted.

Routers are exported as bare :class:`APIRouter` instances. Wiring into
the consumer-chat-api / owner-api apps happens via ``app.include_router``;
tests assemble a minimal :class:`FastAPI` to exercise the contracts
without bringing up the full services.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from shared.common.database import Database
from shared.common.models import (
    ConsumerMcpConnection,
    ConsumerUser,
    McpIntegration,
)
from shared.safety.token_encryption import TokenCipher
from tool_platforms.mcp_relay_service._capabilities import (
    canonicalize_tools,
    hash_capabilities,
)
from tool_platforms.mcp_relay_service._ssrf import (
    MCPSSRFError,
    validate_base_url,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "ConnectRequest",
    "IntegrationCreate",
    "IntegrationUpdate",
    "build_admin_router",
    "build_catalog_router",
    "build_connections_router",
]


def _get_db(request: Request) -> Database:
    db = getattr(request.app.state, "database", None)
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="database not initialized",
        )
    return db


def _get_cipher(request: Request) -> TokenCipher:
    cipher = getattr(request.app.state, "mcp_token_cipher", None)
    if cipher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="mcp token cipher not initialized",
        )
    return cipher


# -- Admin router ----------------------------------------------------------


class IntegrationCreate(BaseModel):
    slug: str = Field(min_length=1, max_length=64)
    display_name: str = Field(min_length=1, max_length=128)
    vendor: str = Field(default="", max_length=128)
    description: str = Field(default="", max_length=8000)
    base_url: str = Field(min_length=1, max_length=1024)
    transport: str = Field(default="http", pattern="^(http|sse|stdio_via_relay)$")
    oauth_provider: str | None = Field(default=None, max_length=64)
    oauth_authorize_url: str | None = Field(default=None, max_length=1024)
    oauth_token_url: str | None = Field(default=None, max_length=1024)
    oauth_scopes: list[str] = Field(default_factory=list, max_length=32)
    capabilities: list[dict[str, Any]] = Field(default_factory=list, max_length=128)


class IntegrationUpdate(BaseModel):
    display_name: str | None = Field(default=None, max_length=128)
    vendor: str | None = Field(default=None, max_length=128)
    description: str | None = Field(default=None, max_length=8000)
    status: str | None = Field(default=None, pattern="^(active|disabled)$")


def build_admin_router(
    *,
    require_admin: Callable[..., Any],
    allow_http_base_urls: bool = False,
) -> APIRouter:
    router = APIRouter(
        prefix="/v1/admin/mcp_integrations",
        tags=["mcp-admin"],
    )

    @router.post("", status_code=status.HTTP_201_CREATED)
    async def create(
        payload: IntegrationCreate,
        request: Request,
        admin: Any = Depends(require_admin),
    ) -> dict[str, Any]:
        try:
            validate_base_url(payload.base_url, allow_http=allow_http_base_urls)
        except MCPSSRFError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"ssrf_blocked: {exc}",
            ) from exc
        canonical = canonicalize_tools(payload.capabilities)
        h = hash_capabilities(canonical)
        db = _get_db(request)
        with db.sessionmaker() as session:
            existing = session.scalar(
                select(McpIntegration).where(McpIntegration.slug == payload.slug)
            )
            if existing is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"integration with slug {payload.slug!r} already exists",
                )
            integration = McpIntegration(
                slug=payload.slug,
                display_name=payload.display_name,
                vendor=payload.vendor,
                description=payload.description,
                base_url=payload.base_url,
                transport=payload.transport,
                oauth_provider=payload.oauth_provider,
                oauth_authorize_url=payload.oauth_authorize_url,
                oauth_token_url=payload.oauth_token_url,
                oauth_scopes_json=list(payload.oauth_scopes),
                capabilities_json=canonical,
                capabilities_hash=h,
                status="active",
                created_by_admin_id=str(admin)[:128] if admin else None,
            )
            session.add(integration)
            session.commit()
            return _admin_view(integration)

    @router.get("")
    async def list_all(
        request: Request,
        admin: Any = Depends(require_admin),
    ) -> list[dict[str, Any]]:
        del admin
        db = _get_db(request)
        with db.sessionmaker() as session:
            rows = list(session.scalars(select(McpIntegration)))
            return [_admin_view(r) for r in rows]

    @router.patch("/{integration_id}")
    async def patch(
        integration_id: str,
        payload: IntegrationUpdate,
        request: Request,
        admin: Any = Depends(require_admin),
    ) -> dict[str, Any]:
        del admin
        db = _get_db(request)
        with db.sessionmaker() as session:
            integration = session.get(McpIntegration, integration_id)
            if integration is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="integration not found",
                )
            if payload.display_name is not None:
                integration.display_name = payload.display_name
            if payload.vendor is not None:
                integration.vendor = payload.vendor
            if payload.description is not None:
                integration.description = payload.description
            if payload.status is not None:
                integration.status = payload.status
            integration.updated_at = datetime.utcnow()
            session.commit()
            return _admin_view(integration)

    @router.post("/{integration_id}/reprobe")
    async def reprobe(
        integration_id: str,
        request: Request,
        admin: Any = Depends(require_admin),
        capabilities: list[dict[str, Any]] = Body(default_factory=list),
    ) -> dict[str, Any]:
        """Replace ``capabilities_json`` + ``capabilities_hash`` from
        operator-supplied input.

        In production the operator typically calls the relay service's
        ``list_tools`` endpoint, then posts the result here. Bundling
        the two steps in the operator workflow keeps the relay
        side-effect free; this endpoint is just persistence.
        """
        del admin
        db = _get_db(request)
        with db.sessionmaker() as session:
            integration = session.get(McpIntegration, integration_id)
            if integration is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="integration not found",
                )
            canonical = canonicalize_tools(capabilities)
            integration.capabilities_json = canonical
            integration.capabilities_hash = hash_capabilities(canonical)
            integration.updated_at = datetime.utcnow()
            session.commit()
            return _admin_view(integration)

    return router


def _admin_view(row: McpIntegration) -> dict[str, Any]:
    return {
        "id": row.id,
        "slug": row.slug,
        "display_name": row.display_name,
        "vendor": row.vendor,
        "description": row.description,
        "base_url": row.base_url,
        "transport": row.transport,
        "status": row.status,
        "oauth_provider": row.oauth_provider,
        "oauth_authorize_url": row.oauth_authorize_url,
        "oauth_token_url": row.oauth_token_url,
        "oauth_scopes": list(row.oauth_scopes_json or []),
        "capabilities": list(row.capabilities_json or []),
        "capabilities_hash": row.capabilities_hash,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _public_view(row: McpIntegration) -> dict[str, Any]:
    return {
        "id": row.id,
        "slug": row.slug,
        "display_name": row.display_name,
        "vendor": row.vendor,
        "description": row.description,
        "transport": row.transport,
        "oauth_provider": row.oauth_provider,
        "oauth_scopes": list(row.oauth_scopes_json or []),
        "capabilities": [
            {
                "name": c.get("name"),
                "description": c.get("description") or "",
            }
            for c in (row.capabilities_json or [])
        ],
    }


# -- Catalog router (consumer-facing browse) ------------------------------


def build_catalog_router(
    *,
    require_user: Callable[..., Any],
) -> APIRouter:
    router = APIRouter(prefix="/v1/mcp", tags=["mcp-catalog"])

    @router.get("/integrations")
    async def list_active(
        request: Request,
        user: Any = Depends(require_user),
    ) -> list[dict[str, Any]]:
        del user
        db = _get_db(request)
        with db.sessionmaker() as session:
            rows = list(session.scalars(
                select(McpIntegration).where(McpIntegration.status == "active")
            ))
            return [_public_view(r) for r in rows]

    return router


# -- Connections router (consumer-facing manage) --------------------------


class ConnectRequest(BaseModel):
    integration_id: str = Field(min_length=1, max_length=36)
    redirect_uri: str | None = Field(default=None, max_length=1024)


def build_connections_router(
    *,
    require_user: Callable[..., str],
    transport: httpx.AsyncBaseTransport | None = None,
) -> APIRouter:
    router = APIRouter(
        prefix="/v1/users/me/mcp_connections",
        tags=["mcp-connections"],
    )

    @router.get("")
    async def list_my(
        request: Request,
        user_id: str = Depends(require_user),
    ) -> list[dict[str, Any]]:
        db = _get_db(request)
        with db.sessionmaker() as session:
            stmt = (
                select(ConsumerMcpConnection, McpIntegration)
                .join(
                    McpIntegration,
                    McpIntegration.id == ConsumerMcpConnection.integration_id,
                )
                .where(ConsumerMcpConnection.user_id == user_id)
            )
            return [
                {
                    "id": conn.id,
                    "integration_id": integration.id,
                    "integration_slug": integration.slug,
                    "integration_display_name": integration.display_name,
                    "status": conn.status,
                    "last_used_at": (
                        conn.last_used_at.isoformat()
                        if conn.last_used_at else None
                    ),
                    "oauth_expires_at": (
                        conn.oauth_expires_at.isoformat()
                        if conn.oauth_expires_at else None
                    ),
                }
                for conn, integration in session.execute(stmt).all()
            ]

    @router.post("", status_code=status.HTTP_202_ACCEPTED)
    async def begin_connect(
        payload: ConnectRequest,
        request: Request,
        user_id: str = Depends(require_user),
    ) -> dict[str, Any]:
        db = _get_db(request)
        with db.sessionmaker() as session:
            integration = session.get(McpIntegration, payload.integration_id)
            if integration is None or integration.status != "active":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="integration not found or disabled",
                )
            existing = session.scalar(
                select(ConsumerMcpConnection).where(
                    ConsumerMcpConnection.user_id == user_id,
                    ConsumerMcpConnection.integration_id == integration.id,
                )
            )
            if existing is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="already_connected",
                )
            user = session.get(ConsumerUser, user_id)
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="user not found",
                )
            # Provisional connection row in pending state — populated
            # with the access token by the OAuth callback.
            state_token = uuid4().hex
            connection = ConsumerMcpConnection(
                user_id=user_id,
                integration_id=integration.id,
                status="pending",
                metadata_json={"oauth_state": state_token},
            )
            session.add(connection)
            session.commit()
            authorize = _build_authorize_url(
                integration=integration,
                state=state_token,
                redirect_uri=payload.redirect_uri,
            )
            return {
                "connection_id": connection.id,
                "authorize_url": authorize,
                "state": state_token,
            }

    @router.get("/oauth/callback")
    async def oauth_callback(
        request: Request,
        code: str,
        state: str,
        user_id: str = Depends(require_user),
    ) -> dict[str, Any]:
        db = _get_db(request)
        cipher = _get_cipher(request)
        with db.sessionmaker() as session:
            stmt = select(ConsumerMcpConnection).where(
                ConsumerMcpConnection.user_id == user_id,
            )
            connection = next(
                (
                    c for c in session.scalars(stmt)
                    if (c.metadata_json or {}).get("oauth_state") == state
                ),
                None,
            )
            if connection is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="state_token_unknown",
                )
            integration = session.get(McpIntegration, connection.integration_id)
            if integration is None or not integration.oauth_token_url:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="integration_missing_oauth_config",
                )
            token_url = integration.oauth_token_url
        kwargs: dict[str, Any] = {"timeout": 10.0}
        if transport is not None:
            kwargs["transport"] = transport
        try:
            async with httpx.AsyncClient(**kwargs) as client:
                token_resp = await client.post(
                    token_url,
                    data={
                        "grant_type": "authorization_code",
                        "code": code,
                        "state": state,
                    },
                )
                token_resp.raise_for_status()
                token_payload = token_resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"oauth_token_exchange_failed: {exc}",
            ) from exc
        access = token_payload.get("access_token")
        refresh = token_payload.get("refresh_token")
        expires_in = token_payload.get("expires_in")
        if not isinstance(access, str) or not access:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="oauth_token_missing_access_token",
            )
        with db.sessionmaker() as session:
            connection = session.get(ConsumerMcpConnection, connection.id)
            connection.oauth_access_token_encrypted = cipher.encrypt(
                access.encode("utf-8")
            )
            if isinstance(refresh, str) and refresh:
                connection.oauth_refresh_token_encrypted = cipher.encrypt(
                    refresh.encode("utf-8")
                )
            if isinstance(expires_in, (int, float)) and expires_in > 0:
                connection.oauth_expires_at = (
                    datetime.utcnow() + timedelta(seconds=int(expires_in))
                )
            connection.status = "active"
            connection.updated_at = datetime.utcnow()
            session.commit()
            return {"connection_id": connection.id, "status": "active"}

    @router.delete("/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def revoke(
        connection_id: str,
        request: Request,
        user_id: str = Depends(require_user),
    ) -> Response:  # type: ignore[name-defined]
        db = _get_db(request)
        with db.sessionmaker() as session:
            connection = session.get(ConsumerMcpConnection, connection_id)
            if connection is None or connection.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="connection not found",
                )
            session.delete(connection)
            session.commit()
        from fastapi import Response as _R
        return _R(status_code=status.HTTP_204_NO_CONTENT)

    return router


def _build_authorize_url(
    *,
    integration: McpIntegration,
    state: str,
    redirect_uri: str | None,
) -> str:
    base = integration.oauth_authorize_url or ""
    if not base:
        # Test/dev integrations without an explicit authorize URL fall
        # back to a synthetic stub so the consumer flow can still be
        # exercised end-to-end.
        return f"about:blank?state={state}"
    sep = "&" if "?" in base else "?"
    out = f"{base}{sep}state={state}"
    if redirect_uri:
        from urllib.parse import quote
        out += f"&redirect_uri={quote(redirect_uri, safe='')}"
    if integration.oauth_scopes_json:
        out += "&scope=" + "+".join(
            integration.oauth_scopes_json
        )
    return out


# Re-export Response for the delete handler annotation.
from fastapi import Response  # noqa: E402
