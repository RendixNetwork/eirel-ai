"""Platform tool: file_manager

Manages temporary files within a conversation session. Allows the
orchestrator to pass file artifacts between tool invocations and
specialist calls.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from orchestration.orchestrator.platform_tools.base import PlatformTool, ToolResult

_logger = logging.getLogger(__name__)

FILE_STORAGE_ROOT = os.getenv(
    "FILE_MANAGER_STORAGE_ROOT",
    os.path.join(tempfile.gettempdir(), "eirel-file-manager"),
)
MAX_FILE_SIZE = int(os.getenv("FILE_MANAGER_MAX_SIZE_BYTES", str(10 * 1024 * 1024)))


class FileManagerTool(PlatformTool):
    @property
    def name(self) -> str:
        return "file_manager"

    @property
    def description(self) -> str:
        return (
            "Read, write, and list files in a session-scoped temporary directory. "
            "10MB max file size."
        )

    async def execute(self, *, params: dict[str, Any]) -> ToolResult:
        action = params.get("action", "list")
        session_id = params.get("session_id", "default")
        session_dir = Path(FILE_STORAGE_ROOT) / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        if action == "write":
            return self._write(session_dir, params)
        elif action == "read":
            return self._read(session_dir, params)
        elif action == "list":
            return self._list(session_dir)
        else:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"unknown action: {action}",
            )

    def _write(self, session_dir: Path, params: dict[str, Any]) -> ToolResult:
        filename = params.get("filename", "")
        content = params.get("content", "")
        if not filename:
            return ToolResult(tool_name=self.name, success=False, error="no filename")
        safe_name = Path(filename).name
        if len(content.encode()) > MAX_FILE_SIZE:
            return ToolResult(tool_name=self.name, success=False, error="file too large")
        filepath = session_dir / safe_name
        filepath.write_text(content, encoding="utf-8")
        return ToolResult(
            tool_name=self.name,
            success=True,
            output={"path": str(filepath), "filename": safe_name, "size": len(content)},
        )

    def _read(self, session_dir: Path, params: dict[str, Any]) -> ToolResult:
        filename = params.get("filename", "")
        if not filename:
            return ToolResult(tool_name=self.name, success=False, error="no filename")
        safe_name = Path(filename).name
        filepath = session_dir / safe_name
        if not filepath.exists():
            return ToolResult(tool_name=self.name, success=False, error="file not found")
        content = filepath.read_text(encoding="utf-8")
        return ToolResult(
            tool_name=self.name,
            success=True,
            output={"filename": safe_name, "content": content, "size": len(content)},
        )

    def _list(self, session_dir: Path) -> ToolResult:
        files = [
            {"name": f.name, "size": f.stat().st_size}
            for f in session_dir.iterdir()
            if f.is_file()
        ]
        return ToolResult(
            tool_name=self.name,
            success=True,
            output={"files": files, "count": len(files)},
        )
