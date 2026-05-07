"""MCP relay service.

Sits between :class:`MCPToolDispatcher` (orchestrator) and the
operator-vetted MCP integration. Inbound auth is the internal-service
Bearer token — only the orchestrator should reach this. Outbound auth
is the per-connection encrypted OAuth access token loaded from the DB.

Endpoints:

  * ``POST /v1/relay/connections/{connection_id}/call`` — invoke one
    tool on behalf of one consumer connection. Verifies the
    integration's stored ``capabilities_hash`` matches what the caller
    has (passed in the body); refuses on drift.
  * ``POST /v1/relay/integrations/{integration_id}/list_tools`` —
    reprobe an integration's tool surface, return the canonical hash.
    Operator-only path; used by the admin reprobe route.

The relay does NOT write the audit row. The dispatcher does, because
it owns the conversation/message context. The relay returns latency +
cost so the dispatcher can stamp them into the audit.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status
from sqlalchemy import select

from shared.common.database import Database
from shared.common.models import (
    ConsumerMcpConnection,
    McpIntegration,
)
from shared.safety.token_encryption import TokenCipher, build_token_cipher
from tool_platforms.mcp_relay_service._capabilities import (
    canonicalize_tools,
    hash_capabilities,
)
from tool_platforms.mcp_relay_service._ssrf import (
    MCPSSRFError,
    validate_base_url,
)
from tool_platforms.mcp_relay_service.models import (
    RelayCallRequest,
    RelayCallResponse,
    RelayListToolsResponse,
)


_logger = logging.getLogger(__name__)

DEFAULT_RELAY_TIMEOUT_SECONDS: float = 10.0
DEFAULT_MAX_RESPONSE_BYTES: int = 256 * 1024
DEFAULT_PER_CALL_COST_USD: float = 0.0002


def create_app(
    *,
    database: Database,
    cipher: TokenCipher | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    allow_http: bool | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.auth_token = os.getenv("EIREL_MCP_RELAY_TOKEN", "")
        app.state.timeout_seconds = float(
            os.getenv(
                "EIREL_MCP_RELAY_TIMEOUT_SECONDS",
                str(DEFAULT_RELAY_TIMEOUT_SECONDS),
            )
        )
        app.state.max_response_bytes = int(
            os.getenv(
                "EIREL_MCP_RELAY_MAX_RESPONSE_BYTES",
                str(DEFAULT_MAX_RESPONSE_BYTES),
            )
        )
        app.state.per_call_cost_usd = float(
            os.getenv(
                "EIREL_MCP_RELAY_PER_CALL_COST_USD",
                str(DEFAULT_PER_CALL_COST_USD),
            )
        )
        # ``allow_http`` defaults to True in non-production so tests
        # against ``http://testserver`` work. Operators flip
        # ``EIREL_MCP_RELAY_REQUIRE_HTTPS=1`` in prod.
        if allow_http is None:
            require_https = os.getenv(
                "EIREL_MCP_RELAY_REQUIRE_HTTPS", "0"
            ).strip() in {"1", "true", "True"}
            app.state.allow_http = not require_https
        else:
            app.state.allow_http = bool(allow_http)
        app.state.database = database
        app.state.cipher = cipher or build_token_cipher()
        app.state.transport = transport
        yield

    app = FastAPI(
        title="mcp-relay-service",
        version="0.1.0",
        lifespan=lifespan,
    )

    async def require_auth(
        authorization: str | None = Header(default=None),
    ) -> None:
        configured: str = app.state.auth_token
        if not configured:
            return
        if authorization == f"Bearer {configured}":
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid mcp relay auth token",
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/v1/relay/integrations/{integration_id}/list_tools",
        response_model=RelayListToolsResponse,
    )
    async def list_tools(
        integration_id: str,
        _: None = Depends(require_auth),
    ) -> RelayListToolsResponse:
        db: Database = app.state.database
        with db.sessionmaker() as session:
            integration = session.get(McpIntegration, integration_id)
            if integration is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="integration not found",
                )
            base_url = integration.base_url
        try:
            validate_base_url(base_url, allow_http=app.state.allow_http)
        except MCPSSRFError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"ssrf_blocked: {exc}",
            ) from exc
        client_kwargs: dict[str, Any] = {"timeout": app.state.timeout_seconds}
        if app.state.transport is not None:
            client_kwargs["transport"] = app.state.transport
        async with httpx.AsyncClient(**client_kwargs) as client:
            try:
                resp = await client.post(
                    f"{base_url.rstrip('/')}/list_tools",
                    json={},
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"upstream_error: {exc}",
                ) from exc
            try:
                payload = resp.json()
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"upstream_invalid_json: {exc}",
                ) from exc
        tools_raw = payload.get("tools") if isinstance(payload, dict) else None
        if not isinstance(tools_raw, list):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="upstream_missing_tools",
            )
        canonical = canonicalize_tools(tools_raw)
        h = hash_capabilities(canonical)
        return RelayListToolsResponse(tools=canonical, capabilities_hash=h)

    @app.post(
        "/v1/relay/connections/{connection_id}/call",
        response_model=RelayCallResponse,
    )
    async def call(
        connection_id: str,
        payload: RelayCallRequest,
        _: None = Depends(require_auth),
    ) -> RelayCallResponse:
        db: Database = app.state.database
        cipher: TokenCipher = app.state.cipher
        with db.sessionmaker() as session:
            connection = session.get(ConsumerMcpConnection, connection_id)
            if connection is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="connection not found",
                )
            if connection.status != "active":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"connection_not_active: {connection.status}",
                )
            integration = session.get(McpIntegration, connection.integration_id)
            if integration is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="integration not found",
                )
            if integration.status != "active":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="integration_disabled",
                )
            if (
                payload.capabilities_hash
                and integration.capabilities_hash
                and payload.capabilities_hash != integration.capabilities_hash
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="capabilities_hash_mismatch",
                )
            base_url = integration.base_url
            access_token: str | None = None
            if connection.oauth_access_token_encrypted:
                try:
                    access_token = cipher.decrypt(
                        bytes(connection.oauth_access_token_encrypted)
                    ).decode("utf-8")
                except Exception as exc:  # noqa: BLE001 — defensive
                    _logger.warning(
                        "relay token decrypt failed connection=%s err=%s",
                        connection_id, exc,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="token_decrypt_failed",
                    ) from exc
            connection.last_used_at = datetime.utcnow()
            session.commit()

        try:
            validate_base_url(base_url, allow_http=app.state.allow_http)
        except MCPSSRFError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"ssrf_blocked: {exc}",
            ) from exc

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        client_kwargs: dict[str, Any] = {"timeout": app.state.timeout_seconds}
        if app.state.transport is not None:
            client_kwargs["transport"] = app.state.transport

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                resp = await asyncio.wait_for(
                    client.post(
                        f"{base_url.rstrip('/')}/tools/{payload.tool_name}",
                        json=payload.args,
                        headers=headers,
                    ),
                    timeout=app.state.timeout_seconds,
                )
                resp.raise_for_status()
                if (
                    resp.headers.get("content-length")
                    and int(resp.headers["content-length"])
                    > app.state.max_response_bytes
                ):
                    raise httpx.HTTPError(
                        f"response too large: {resp.headers['content-length']}"
                    )
                body = resp.content
                if len(body) > app.state.max_response_bytes:
                    raise httpx.HTTPError(
                        f"response too large: {len(body)} bytes"
                    )
                result = resp.json()
        except asyncio.TimeoutError:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return RelayCallResponse(
                ok=False,
                error="upstream_timeout",
                latency_ms=latency_ms,
                cost_usd=0.0,
            )
        except (httpx.HTTPError, ValueError) as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return RelayCallResponse(
                ok=False,
                error=f"upstream_error: {exc}",
                latency_ms=latency_ms,
                cost_usd=0.0,
            )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        if not isinstance(result, dict):
            return RelayCallResponse(
                ok=False,
                error="upstream_invalid_shape",
                latency_ms=latency_ms,
                cost_usd=0.0,
            )
        return RelayCallResponse(
            ok=True,
            result=result,
            latency_ms=latency_ms,
            cost_usd=float(app.state.per_call_cost_usd),
        )

    return app


def main() -> None:
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        raise RuntimeError("DATABASE_URL must be set for the mcp_relay_service")
    db = Database(db_url)
    app = create_app(database=db)
    port = int(os.getenv("EIREL_MCP_RELAY_PORT", "8093"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
