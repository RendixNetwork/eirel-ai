from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SandboxExecuteRequest(BaseModel):
    code: str = Field(min_length=1, max_length=65_536)
    timeout_seconds: float | None = Field(default=None, gt=0.0, le=30.0)
    memory_mb: int | None = Field(default=None, gt=0, le=1024)


class SandboxExecuteResponse(BaseModel):
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: int = 0
    truncated: bool = False
    retrieved_at: str | None = None
    retrieval_ledger_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
