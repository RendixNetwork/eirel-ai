from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response
from pydantic import BaseModel, Field

from shared.common.config import get_settings
from shared.common.request_context import RequestIdMiddleware
from shared.common.database import Database
from shared.common.execution_store import ExecutionStore
from shared.common.http_control import (
    ApiKeyAuthorizer,
    SlidingWindowRateLimiter,
    parse_api_keys,
    rate_limit_principal,
)
from shared.common.models import ConsumerSessionState, TaskRequestRecord
from shared.common.tracing import init_tracing, get_tracer
from orchestration.consumer_api.chat import route_chat_request, stream_chat_request

init_tracing("consumer-chat-api")
_tracer = get_tracer(__name__)


class ChatRequest(BaseModel):
    prompt: str = Field(max_length=32000)
    user_id: str = Field(default="anonymous", max_length=64)
    session_id: str | None = Field(default=None, max_length=128)
    context_history: list[dict] = Field(default_factory=list, max_length=50)
    # Per-session toggles. Persisted on the session row in the
    # orchestrator (consumer-chat-api forwards them as-is).
    mode: str = Field(default="instant", pattern="^(instant|thinking)$")
    web_search: bool = Field(default=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    db = Database(settings.database_url)
    db.create_all()
    app.state.settings = settings
    app.state.execution_store = ExecutionStore(db)
    app.state.api_key_authorizer = ApiKeyAuthorizer(parse_api_keys(settings.consumer_api_keys))
    app.state.rate_limiter = SlidingWindowRateLimiter(
        max_requests=settings.consumer_rate_limit_requests,
        window_seconds=settings.consumer_rate_limit_window_seconds,
    )
    yield


app = FastAPI(title="consumer-chat-api", lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)


async def _request_guard(request: Request) -> str:
    identity = await request.app.state.api_key_authorizer(request)
    principal = rate_limit_principal(request, authenticated_identity=identity)
    request.app.state.rate_limiter.check(principal)
    return identity


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    store: ExecutionStore = app.state.execution_store
    with store.db.sessionmaker() as session:
        session_total = session.query(ConsumerSessionState).count()
        active_sessions = session.query(ConsumerSessionState).filter_by(status="active").count()
        task_total = session.query(TaskRequestRecord).count()
    return Response(
        content=(
            "# TYPE eirel_consumer_api_up gauge\n"
            "eirel_consumer_api_up 1\n"
            "# TYPE eirel_consumer_api_sessions_total gauge\n"
            f"eirel_consumer_api_sessions_total {session_total}\n"
            "# TYPE eirel_consumer_api_active_sessions gauge\n"
            f"eirel_consumer_api_active_sessions {active_sessions}\n"
            "# TYPE eirel_consumer_api_task_views_total gauge\n"
            f"eirel_consumer_api_task_views_total {task_total}\n"
        ),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.post("/v1/chat")
async def chat(payload: ChatRequest, caller_identity: str = Depends(_request_guard)):
    del caller_identity
    with _tracer.start_as_current_span("consumer_chat") as span:
        span.set_attribute("chat.user_id", payload.user_id)
        span.set_attribute("chat.prompt_length", len(payload.prompt))
        response = await route_chat_request(
            prompt=payload.prompt,
            user_id=payload.user_id,
            session_id=payload.session_id,
            context_history=payload.context_history,
        )
        if isinstance(response, tuple):
            status_code, response_payload = response
        else:
            status_code, response_payload = 200, response
        span.set_attribute("chat.status_code", status_code)
        return JSONResponse(status_code=status_code, content=response_payload)


@app.post("/v1/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    caller_identity: str = Depends(_request_guard),
):
    """Server-Sent Events streaming chat.

    Resolves the current serving miner for the chat family and proxies
    the miner's `/v1/agent/infer/stream` NDJSON back to the browser as
    SSE. Falls back to the unary endpoint for older miners (eirel SDK
    < 0.2.3) and emits the answer as a single delta+done so the client
    UX is identical regardless of miner SDK version.

    Event types (matches eirel.schemas.StreamChunk):
      - started     : task_id + family_id (stream metadata)
      - delta       : `{text: "next slice"}`
      - tool_call   : `{tool_call: {...}}`
      - citation    : `{citation: {url, title}}`
      - done        : terminal; carries final output + citations + status
      - error       : terminal failure with `{message}`
    """
    del caller_identity
    return StreamingResponse(
        stream_chat_request(
            prompt=payload.prompt,
            user_id=payload.user_id,
            session_id=payload.session_id,
            context_history=payload.context_history,
            mode=payload.mode,
            web_search=payload.web_search,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
        },
    )


@app.get("/v1/tasks/{task_id}")
async def get_task(task_id: str, caller_identity: str = Depends(_request_guard)):
    del caller_identity
    store: ExecutionStore = app.state.execution_store
    record = store.get_task(task_id=task_id)
    if record is None:
        return JSONResponse(status_code=404, content={"detail": "task not found"})
    return store.task_payload(record)


@app.get("/v1/sessions/{session_id}")
async def get_session(session_id: str, caller_identity: str = Depends(_request_guard)):
    del caller_identity
    store: ExecutionStore = app.state.execution_store
    record = store.get_session(session_id=session_id)
    if record is None:
        return JSONResponse(status_code=404, content={"detail": "session not found"})
    return store.session_payload(record)


def main() -> None:
    uvicorn.run("orchestration.consumer_api.main:app", host="0.0.0.0", port=8080, reload=False)
