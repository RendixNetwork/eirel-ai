from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SandboxAttachment(BaseModel):
    """Pre-existing file mounted into the sandbox working directory.

    ``content_b64`` is base64-encoded raw bytes. ``max_length`` here is a
    safety bound on the request body — typical attachments come from
    ``ConsumerAttachment`` rows that the orchestrator has already
    size-bounded at upload.
    """

    filename: str = Field(min_length=1, max_length=255)
    content_b64: str = Field(default="", max_length=14_000_000)


class SandboxFileWrite(BaseModel):
    """One file that the sandbox produced during execution.

    ``content_b64`` is ``None`` when the file exceeds the per-file or
    total cap; the path and size still come back so the agent can decide
    whether to retry with a smaller output or split the work.
    """

    path: str
    size: int
    content_b64: str | None = None


class SandboxExecuteRequest(BaseModel):
    code: str = Field(min_length=1, max_length=65_536)
    timeout_seconds: float | None = Field(default=None, gt=0.0, le=30.0)
    memory_mb: int | None = Field(default=None, gt=0, le=1024)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    attachments: list[SandboxAttachment] | None = Field(default=None, max_length=16)


class SandboxExecuteResponse(BaseModel):
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: int = 0
    truncated: bool = False
    retrieved_at: str | None = None
    retrieval_ledger_id: str | None = None
    session_id: str | None = None
    files: list[SandboxFileWrite] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
