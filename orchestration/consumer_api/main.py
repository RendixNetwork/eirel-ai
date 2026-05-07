from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
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
from shared.common.models import (
    ConsumerAttachment,
    ConsumerSessionState,
    ConsumerUser,
    TaskRequestRecord,
)
from shared.common.tracing import init_tracing, get_tracer
from orchestration.orchestrator.document_extractor import (
    MAX_RAW_BYTES,
    extract_text,
)
from orchestration.orchestrator.product_orchestrator import (
    ProductOrchestrator,
    ProductOrchestratorError,
)
from orchestration.orchestrator.serving_picker import ServingPicker

init_tracing("consumer-chat-api")
_tracer = get_tracer(__name__)


class GraphChatRequest(BaseModel):
    """Product-mode chat request — routes through ProductOrchestrator.

    User-facing state (history, preferences, project memory) is loaded
    from the product DB by user_id; the request body only carries
    transient per-turn knobs (prompt, mode, optional conversation /
    project ids).
    """

    prompt: str = Field(max_length=32000)
    user_id: str = Field(min_length=1, max_length=64)
    conversation_id: str | None = Field(default=None, max_length=64)
    project_id: str | None = Field(default=None, max_length=64)
    attachment_ids: list[str] = Field(default_factory=list, max_length=20)
    mode: str = Field(default="instant", pattern="^(instant|thinking)$")
    web_search: bool = Field(default=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    db = Database(settings.database_url)
    db.create_all()
    app.state.settings = settings
    app.state.execution_store = ExecutionStore(db)
    app.state.database = db
    app.state.api_key_authorizer = ApiKeyAuthorizer(parse_api_keys(settings.consumer_api_keys))
    app.state.rate_limiter = SlidingWindowRateLimiter(
        max_requests=settings.consumer_rate_limit_requests,
        window_seconds=settings.consumer_rate_limit_window_seconds,
    )
    # Product-mode orchestrator wired against the same DB. Reads
    # ServingDeployment rows for routing; never talks to ManagedDeployment
    # (eval-only) directly.
    app.state.product_orchestrator = ProductOrchestrator(
        database=db,
        serving_picker=ServingPicker(database=db),
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


# ---------------------------------------------------------------------------
# Product-mode endpoints (graph-runtime path)
# ---------------------------------------------------------------------------


@app.post("/v1/graph/chat")
async def graph_chat(
    payload: GraphChatRequest,
    caller_identity: str = Depends(_request_guard),
):
    """Unary product chat. Routes through ProductOrchestrator.

    Returns ``{conversation_id, message_id, response}``. Echo the
    ``conversation_id`` on follow-up turns to keep the same thread.
    """
    del caller_identity
    orchestrator: ProductOrchestrator = app.state.product_orchestrator
    try:
        result = await orchestrator.invoke(
            user_id=payload.user_id,
            prompt=payload.prompt,
            conversation_id=payload.conversation_id,
            project_id=payload.project_id,
            attachment_ids=list(payload.attachment_ids or []),
            mode=payload.mode,
            web_search=payload.web_search,
        )
    except ProductOrchestratorError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    return JSONResponse(content=result)


@app.post("/v1/graph/chat/stream")
async def graph_chat_stream(
    payload: GraphChatRequest,
    caller_identity: str = Depends(_request_guard),
):
    """NDJSON streaming product chat.

    First event is a ``conversation`` event carrying the
    ``conversation_id`` (so a brand-new conversation surfaces its id
    immediately). Then ``delta`` / ``tool_call`` / ``tool_result`` /
    ``citation`` / ``checkpoint`` passthrough from the serving
    deployment. Terminal ``done`` event includes the orchestrator
    audit block.
    """
    del caller_identity
    orchestrator: ProductOrchestrator = app.state.product_orchestrator

    import json as _json

    async def _ndjson():
        async for event in orchestrator.astream(
            user_id=payload.user_id,
            prompt=payload.prompt,
            conversation_id=payload.conversation_id,
            project_id=payload.project_id,
            attachment_ids=list(payload.attachment_ids or []),
            mode=payload.mode,
            web_search=payload.web_search,
        ):
            yield (_json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")

    return StreamingResponse(
        _ndjson(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/graph/attachments")
async def graph_attachment_upload(
    file: UploadFile = File(...),
    user_id: str = Form(min_length=1, max_length=64),
    conversation_id: str | None = Form(default=None),
    caller_identity: str = Depends(_request_guard),
):
    """Upload a single file (PDF/DOCX/CSV/TXT/JSON/markdown).

    Pattern matches ChatGPT/Claude: orchestrator preprocesses the
    upload into LLM-ready text immediately. The agent never has to
    open the file at chat time — extracted text rides on
    ``metadata.attached_files`` of the next chat turn.

    Pass ``attachment_id`` from the response on the next ``/v1/graph/chat``
    call (in the ``attachment_ids`` array) to attach this file to that
    turn. Attachments expire / are GC'd by the product layer's
    retention policy (out of scope here).
    """
    del caller_identity
    db: Database = app.state.database

    raw = await file.read()
    if len(raw) == 0:
        return JSONResponse(
            status_code=400,
            content={"detail": "empty file upload"},
        )
    if len(raw) > MAX_RAW_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "detail": (
                    f"file exceeds {MAX_RAW_BYTES} bytes; consider chunking "
                    "or upload to external storage and pass a URL"
                ),
                "size_bytes": len(raw),
            },
        )

    extracted = extract_text(
        raw,
        filename=file.filename or "",
        content_type=file.content_type or "",
    )

    with db.sessionmaker() as session:
        user = session.get(ConsumerUser, user_id)
        if user is None:
            return JSONResponse(
                status_code=404, content={"detail": f"unknown user_id {user_id!r}"},
            )
        # Best-effort conversation binding — if the caller passes a
        # conversation_id that doesn't belong to this user, skip the
        # link rather than failing the upload.
        bound_conversation_id: str | None = None
        if conversation_id:
            from shared.common.models import ConsumerConversation

            convo = session.get(ConsumerConversation, conversation_id)
            if convo is not None and convo.user_id == user_id:
                bound_conversation_id = conversation_id

        attachment = ConsumerAttachment(
            user_id=user_id,
            conversation_id=bound_conversation_id,
            filename=file.filename or "uploaded",
            content_type=file.content_type or "application/octet-stream",
            size_bytes=len(raw),
            blob_ref=None,  # production: write to S3 here and store the key
            extracted_text=extracted.text,
            extraction_metadata_json=dict(extracted.metadata or {}),
            extraction_status=extracted.status,
        )
        session.add(attachment)
        session.flush()
        attachment_id = attachment.id
        session.commit()

    return JSONResponse(content={
        "attachment_id": attachment_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(raw),
        "extracted_chars": len(extracted.text),
        "extraction_status": extracted.status,
        "extraction_metadata": dict(extracted.metadata or {}),
    })


def main() -> None:
    uvicorn.run("orchestration.consumer_api.main:app", host="0.0.0.0", port=8080, reload=False)
