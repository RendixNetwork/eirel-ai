from __future__ import annotations

import asyncio
import logging
import resource
import sys
import time
from dataclasses import dataclass
from typing import Protocol


_logger = logging.getLogger(__name__)


# Preamble injected before user code. Blocks dangerous imports at the
# Python level and strips the nastiest os attributes. This is DEFENSE IN
# DEPTH only — real security comes from running the whole service inside
# a hardened container (seccomp, no network, read-only fs, unprivileged
# user) or a gVisor / Firecracker micro-VM. See SubprocessBackend
# docstring for the production hardening checklist.
_PREAMBLE = '''
import sys as _sys

_BLOCKED_ROOTS = frozenset({
    "socket", "urllib", "http", "httpx", "requests",
    "subprocess", "ctypes", "fcntl", "termios",
    "pty", "tty", "multiprocessing", "_socket", "ssl",
    "asyncio", "select", "selectors", "signal",
    "_ssl", "_thread",
})


class _SandboxImportFinder:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in _BLOCKED_ROOTS:
            raise ImportError(f"sandbox: import of {name!r} is blocked")
        return None


_sys.meta_path.insert(0, _SandboxImportFinder())

try:
    import os as _os
    for _attr in (
        "system", "popen", "spawnl", "spawnv", "spawnle", "spawnve",
        "execl", "execle", "execlp", "execlpe", "execv", "execve",
        "execvp", "execvpe", "fork", "forkpty", "kill", "killpg",
        "chmod", "chown", "lchown", "fchown", "rename", "replace",
    ):
        try:
            delattr(_os, _attr)
        except AttributeError:
            pass
    del _os
except ImportError:
    pass

del _sys
'''


@dataclass(slots=True, frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    truncated: bool = False
    timed_out: bool = False


class SandboxBackend(Protocol):
    async def execute(
        self,
        *,
        code: str,
        timeout_seconds: float,
        memory_mb: int,
    ) -> ExecutionResult: ...


@dataclass(slots=True)
class SubprocessBackend:
    """Execute Python code in an isolated subprocess with resource limits.

    Security model: this backend is defense-in-depth suitable for
    deployment **inside a hardened container**. The subprocess itself
    enforces:
      - RLIMIT_AS (memory) — hard cap per process
      - RLIMIT_CPU (CPU time) — hard wall, backs up wall-clock timeout
      - RLIMIT_NOFILE (file descriptors) — 64 max
      - RLIMIT_NPROC (processes) — 8 max, blocks fork bombs
      - RLIMIT_FSIZE (file size) — 1 MB max per file
      - RLIMIT_CORE — core dumps disabled
      - Python ``-I -S`` flags (isolated mode, no site-packages)
      - Import preamble blocking network / subprocess / ctypes / ssl

    This does NOT provide kernel-level isolation. For production,
    run this service in a Docker container hardened with::

        docker run \\
            --user nobody \\
            --cap-drop ALL \\
            --security-opt no-new-privileges \\
            --security-opt seccomp=<seccomp.json> \\
            --network none \\
            --read-only \\
            --tmpfs /tmp:size=64m,mode=1777 \\
            --memory 512m --memory-swap 512m \\
            --pids-limit 100 \\
            sandbox-tool-service

    For stronger isolation (adversarial miners, shared infrastructure),
    implement a ``GVisorBackend`` or ``FirecrackerBackend`` satisfying
    the ``SandboxBackend`` Protocol. They plug in transparently — no
    app.py change required.
    """

    python_path: str = sys.executable
    default_timeout: float = 5.0
    default_memory_mb: int = 128
    stdout_max_bytes: int = 65_536
    stderr_max_bytes: int = 16_384

    async def execute(
        self,
        *,
        code: str,
        timeout_seconds: float | None = None,
        memory_mb: int | None = None,
    ) -> ExecutionResult:
        wall_clock = timeout_seconds or self.default_timeout
        mem_mb = memory_mb or self.default_memory_mb
        mem_bytes = mem_mb * 1024 * 1024
        cpu_limit = max(1, int(wall_clock) + 2)
        full_code = _PREAMBLE + "\n" + code

        def _set_limits() -> None:
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (OSError, ValueError):
                pass
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
            except (OSError, ValueError):
                pass
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            except (OSError, ValueError):
                pass
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (8, 8))
            except (OSError, ValueError, AttributeError):
                pass
            try:
                resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
            except (OSError, ValueError):
                pass
            try:
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            except (OSError, ValueError):
                pass

        t0 = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                self.python_path,
                "-I",  # isolated mode — ignore PYTHON* env vars, no user site
                "-S",  # do not import site module
                "-c",
                full_code,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=_set_limits,
            )
        except OSError as exc:
            _logger.warning("sandbox spawn failed: %s", exc)
            return ExecutionResult(
                stdout="",
                stderr=f"sandbox_spawn_error: {exc}",
                exit_code=-1,
                duration_ms=0,
                truncated=False,
            )

        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=wall_clock,
            )
            exit_code = proc.returncode if proc.returncode is not None else 0
        except asyncio.TimeoutError:
            timed_out = True
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                stdout_bytes, stderr_bytes = await proc.communicate()
            except Exception:  # pragma: no cover - communicate after kill
                stdout_bytes, stderr_bytes = b"", b""
            exit_code = -9

        duration_ms = int((time.perf_counter() - t0) * 1000)

        truncated = False
        if len(stdout_bytes) > self.stdout_max_bytes:
            stdout_bytes = stdout_bytes[: self.stdout_max_bytes]
            truncated = True
        if len(stderr_bytes) > self.stderr_max_bytes:
            stderr_bytes = stderr_bytes[: self.stderr_max_bytes]
            truncated = True

        return ExecutionResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            exit_code=exit_code,
            duration_ms=duration_ms,
            truncated=truncated,
            timed_out=timed_out,
        )
