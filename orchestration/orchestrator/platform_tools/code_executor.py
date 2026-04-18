"""Platform tool: code_exec

Executes user code in an isolated sandbox. In production this will use
Firecracker microVMs with <200ms cold start. For now, delegates to the
existing sandbox-service over HTTP.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from orchestration.orchestrator.platform_tools.base import PlatformTool, ToolResult

_logger = logging.getLogger(__name__)

SANDBOX_URL = os.getenv("SANDBOX_SERVICE_URL", "http://sandbox-service:8090")
SANDBOX_TIMEOUT = float(os.getenv("CODE_EXEC_TIMEOUT_SECONDS", "30"))
MAX_OUTPUT_BYTES = int(os.getenv("CODE_EXEC_MAX_OUTPUT_BYTES", str(64 * 1024)))


class CodeExecutorTool(PlatformTool):
    @property
    def name(self) -> str:
        return "code_exec"

    @property
    def description(self) -> str:
        return (
            "Execute Python, JavaScript, or shell code in a sandboxed environment. "
            "No outbound network access. 2 vCPU, 512MB RAM, 10 min max."
        )

    async def execute(self, *, params: dict[str, Any]) -> ToolResult:
        code = params.get("code", "")
        language = params.get("language", "python")
        if not code.strip():
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="no code provided",
            )

        payload = {
            "code": code,
            "language": language,
            "timeout_seconds": min(params.get("timeout", 30), 600),
        }
        try:
            async with httpx.AsyncClient(timeout=SANDBOX_TIMEOUT) as client:
                resp = await client.post(
                    f"{SANDBOX_URL}/v1/execute",
                    json=payload,
                )
                resp.raise_for_status()
                result_data = resp.json()
                stdout = str(result_data.get("stdout", ""))[:MAX_OUTPUT_BYTES]
                stderr = str(result_data.get("stderr", ""))[:MAX_OUTPUT_BYTES]
                exit_code = result_data.get("exit_code", -1)
                return ToolResult(
                    tool_name=self.name,
                    success=exit_code == 0,
                    output={
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": exit_code,
                        "language": language,
                    },
                    error=stderr if exit_code != 0 else None,
                    metadata={"language": language},
                )
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"sandbox returned {exc.response.status_code}",
            )
        except httpx.ConnectError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="sandbox service unavailable",
            )
