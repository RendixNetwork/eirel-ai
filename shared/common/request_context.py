"""Request context utilities for correlation ID propagation."""

from __future__ import annotations

import contextvars
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


class RequestIdMiddleware:
    """ASGI middleware that propagates X-Request-Id through the request lifecycle."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        req_id = (
            headers.get(b"x-request-id", b"").decode() or uuid.uuid4().hex[:12]
        )
        token = request_id_var.set(req_id)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = list(message.get("headers", []))
                response_headers.append([b"x-request-id", req_id.encode()])
                message["headers"] = response_headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)
