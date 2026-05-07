from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import resource
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


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


# Worker loop for session-persistent kernels. Reads newline-delimited
# JSON commands from stdin and writes length-prefixed JSON responses
# directly to fd 1, bypassing the per-call sys.stdout swap. User code
# in exec() sees its own StringIO for stdout/stderr, so our framing
# does not collide with user prints.
_WORKER_LOOP = '''
import io
import json
import sys
import traceback

_globals = {"__name__": "__main__"}
_real_buf = sys.__stdout__.buffer


def _send(obj):
    data = json.dumps(obj).encode("utf-8")
    _real_buf.write(f"{len(data)}\\n".encode("ascii"))
    _real_buf.write(data)
    _real_buf.flush()


while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        cmd = json.loads(line)
    except Exception:
        continue
    code = cmd.get("code", "")
    out = io.StringIO()
    err = io.StringIO()
    prev_out, prev_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    exit_code = 0
    try:
        exec(compile(code, "<sandbox>", "exec"), _globals)
    except SystemExit as e:
        exit_code = int(e.code) if isinstance(e.code, int) else 1
    except BaseException:
        traceback.print_exc(file=err)
        exit_code = 1
    finally:
        sys.stdout, sys.stderr = prev_out, prev_err
    _send({
        "stdout": out.getvalue(),
        "stderr": err.getvalue(),
        "exit_code": exit_code,
    })
'''


@dataclass(slots=True, frozen=True)
class SandboxFile:
    """One file produced by the sandbox during execution."""

    path: str
    size: int
    content_b64: str | None = None  # ``None`` when the file exceeds the per-file cap.


@dataclass(slots=True, frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    truncated: bool = False
    timed_out: bool = False
    files: tuple[SandboxFile, ...] = ()


class SandboxBackend(Protocol):
    async def execute(
        self,
        *,
        code: str,
        timeout_seconds: float | None = None,
        memory_mb: int | None = None,
        session_id: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> ExecutionResult: ...


@dataclass
class _SessionKernel:
    """Long-lived worker subprocess for one session."""

    proc: asyncio.subprocess.Process
    workdir: str
    last_access: float
    lock: asyncio.Lock


@dataclass(slots=True)
class SubprocessBackend:
    """Execute Python code in an isolated subprocess with resource limits.

    Two execution modes:

    - **One-shot** (no ``session_id``): spawn a fresh subprocess per call,
      isolated mode (``-I -S``), apply RLIMIT_*, run user code, capture
      stdout/stderr, return. ``attachments`` (if any) are mounted into a
      throwaway temp directory; files written during execution come back
      in ``ExecutionResult.files``.
    - **Session** (``session_id`` set): reuse a long-lived worker
      subprocess keyed by ``session_id``. The worker keeps a globals dict
      across exec calls so variables, imports, and the working directory
      persist between turns. Idle for more than ``session_idle_ttl``
      seconds → kernel evicted on next access. A timeout or worker crash
      also evicts the kernel; the next call starts fresh.

    Security model: this backend is defense-in-depth suitable for
    deployment **inside a hardened container**. The subprocess itself
    enforces:
      - RLIMIT_AS (memory) — hard cap per process
      - RLIMIT_CPU (CPU time) — hard wall (cumulative across a session)
      - RLIMIT_NOFILE (file descriptors) — 64 max
      - RLIMIT_NPROC (processes) — 8 max, blocks fork bombs
      - RLIMIT_FSIZE (file size) — 1 MB max per file
      - RLIMIT_CORE — core dumps disabled
      - Python ``-I -S`` flags (isolated mode, no site-packages)
      - Import preamble blocking network / subprocess / ctypes / ssl

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
    session_idle_ttl: float = 300.0
    file_size_cap: int = 256 * 1024
    files_total_cap: int = 1024 * 1024
    _sessions: dict[str, _SessionKernel] = field(default_factory=dict)
    _sessions_lock: asyncio.Lock | None = None

    async def execute(
        self,
        *,
        code: str,
        timeout_seconds: float | None = None,
        memory_mb: int | None = None,
        session_id: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> ExecutionResult:
        timeout = timeout_seconds or self.default_timeout
        mem_mb = memory_mb or self.default_memory_mb
        if session_id:
            return await self._execute_session(
                session_id=session_id,
                code=code,
                timeout=timeout,
                memory_mb=mem_mb,
                attachments=attachments,
            )
        return await self._execute_oneshot(
            code=code,
            timeout=timeout,
            memory_mb=mem_mb,
            attachments=attachments,
        )

    async def close(self) -> None:
        for sid in list(self._sessions.keys()):
            await self._evict(sid)

    # -- one-shot path -----------------------------------------------------

    async def _execute_oneshot(
        self,
        *,
        code: str,
        timeout: float,
        memory_mb: int,
        attachments: list[dict[str, Any]] | None,
    ) -> ExecutionResult:
        mem_bytes = memory_mb * 1024 * 1024
        cpu_limit = max(1, int(timeout) + 2)
        full_code = _PREAMBLE + "\n" + code

        workdir: str | None = None
        snap_before: dict[str, int] = {}
        if attachments:
            workdir = tempfile.mkdtemp(prefix="sandbox-oneshot-")
            self._mount_attachments(workdir, attachments)
            snap_before = self._snapshot(workdir)

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
                cwd=workdir,
                preexec_fn=_make_preexec(mem_bytes, cpu_limit),
            )
        except OSError as exc:
            _logger.warning("sandbox spawn failed: %s", exc)
            if workdir is not None:
                shutil.rmtree(workdir, ignore_errors=True)
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
                timeout=timeout,
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

        files: tuple[SandboxFile, ...] = ()
        if workdir is not None:
            files = self._collect_files(workdir, snap_before)
            shutil.rmtree(workdir, ignore_errors=True)

        return ExecutionResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            exit_code=exit_code,
            duration_ms=duration_ms,
            truncated=truncated,
            timed_out=timed_out,
            files=files,
        )

    # -- session path ------------------------------------------------------

    async def _execute_session(
        self,
        *,
        session_id: str,
        code: str,
        timeout: float,
        memory_mb: int,
        attachments: list[dict[str, Any]] | None,
    ) -> ExecutionResult:
        if self._sessions_lock is None:
            self._sessions_lock = asyncio.Lock()

        async with self._sessions_lock:
            kernel = self._sessions.get(session_id)
            now = time.time()
            if kernel is not None and (
                kernel.proc.returncode is not None
                or now - kernel.last_access > self.session_idle_ttl
            ):
                await self._evict(session_id)
                kernel = None
            if kernel is None:
                try:
                    kernel = await self._spawn_kernel(memory_mb)
                except Exception as exc:
                    _logger.warning("sandbox kernel spawn failed: %s", exc)
                    return ExecutionResult(
                        stdout="",
                        stderr=f"sandbox_spawn_error: {exc}",
                        exit_code=-1,
                        duration_ms=0,
                    )
                self._sessions[session_id] = kernel

        async with kernel.lock:
            if attachments:
                self._mount_attachments(kernel.workdir, attachments)
            snap_before = self._snapshot(kernel.workdir)

            assert kernel.proc.stdin is not None
            assert kernel.proc.stdout is not None
            t0 = time.perf_counter()
            try:
                msg = (json.dumps({"code": code}) + "\n").encode("utf-8")
                kernel.proc.stdin.write(msg)
                await kernel.proc.stdin.drain()

                length_line = await asyncio.wait_for(
                    kernel.proc.stdout.readline(), timeout=timeout
                )
                if not length_line:
                    raise RuntimeError("worker terminated unexpectedly")
                try:
                    length = int(length_line.strip())
                except ValueError as exc:
                    raise RuntimeError(
                        f"sandbox framing error: {length_line!r}"
                    ) from exc
                data = await asyncio.wait_for(
                    kernel.proc.stdout.readexactly(length), timeout=timeout
                )
                response = json.loads(data)
            except asyncio.TimeoutError:
                duration_ms = int((time.perf_counter() - t0) * 1000)
                await self._evict(session_id)
                return ExecutionResult(
                    stdout="",
                    stderr="sandbox session timeout",
                    exit_code=-9,
                    duration_ms=duration_ms,
                    timed_out=True,
                )
            except Exception as exc:  # noqa: BLE001 — surface to caller
                _logger.warning(
                    "sandbox session %s exec error: %s", session_id, exc
                )
                duration_ms = int((time.perf_counter() - t0) * 1000)
                await self._evict(session_id)
                return ExecutionResult(
                    stdout="",
                    stderr=f"sandbox_session_error: {exc}",
                    exit_code=-1,
                    duration_ms=duration_ms,
                )

            duration_ms = int((time.perf_counter() - t0) * 1000)
            kernel.last_access = time.time()

            stdout = str(response.get("stdout", ""))
            stderr = str(response.get("stderr", ""))
            exit_code = int(response.get("exit_code", 0))
            stdout_bytes = stdout.encode("utf-8")
            stderr_bytes = stderr.encode("utf-8")
            truncated = False
            if len(stdout_bytes) > self.stdout_max_bytes:
                stdout = stdout_bytes[: self.stdout_max_bytes].decode(
                    "utf-8", errors="replace"
                )
                truncated = True
            if len(stderr_bytes) > self.stderr_max_bytes:
                stderr = stderr_bytes[: self.stderr_max_bytes].decode(
                    "utf-8", errors="replace"
                )
                truncated = True

            files = self._collect_files(kernel.workdir, snap_before)

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                duration_ms=duration_ms,
                truncated=truncated,
                timed_out=False,
                files=files,
            )

    async def _spawn_kernel(self, memory_mb: int) -> _SessionKernel:
        mem_bytes = memory_mb * 1024 * 1024
        # Cumulative CPU cap for the lifetime of the kernel; idle sessions
        # don't consume CPU, so this caps abuse without throttling normal
        # multi-turn use.
        cpu_limit = max(1, int(self.session_idle_ttl))
        full_code = _PREAMBLE + "\n" + _WORKER_LOOP
        workdir = tempfile.mkdtemp(prefix="sandbox-session-")
        try:
            proc = await asyncio.create_subprocess_exec(
                self.python_path,
                "-I",
                "-S",
                "-c",
                full_code,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                preexec_fn=_make_preexec(mem_bytes, cpu_limit),
            )
        except OSError:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        return _SessionKernel(
            proc=proc,
            workdir=workdir,
            last_access=time.time(),
            lock=asyncio.Lock(),
        )

    async def _evict(self, session_id: str) -> None:
        kernel = self._sessions.pop(session_id, None)
        if kernel is None:
            return
        try:
            kernel.proc.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(kernel.proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:  # pragma: no cover - kill should win
            pass
        except Exception:  # pragma: no cover
            pass
        shutil.rmtree(kernel.workdir, ignore_errors=True)

    # -- file helpers ------------------------------------------------------

    def _mount_attachments(
        self,
        workdir: str,
        attachments: list[dict[str, Any]],
    ) -> None:
        if not attachments:
            return
        real_workdir = os.path.realpath(workdir)
        for att in attachments:
            if isinstance(att, dict):
                filename = att.get("filename") or ""
                content_b64 = att.get("content_b64") or ""
            else:  # pragma: no cover — defensive
                filename = getattr(att, "filename", "") or ""
                content_b64 = getattr(att, "content_b64", "") or ""
            if not isinstance(filename, str) or not isinstance(content_b64, str):
                continue
            safe = os.path.basename(filename)
            if not safe or safe in (".", ".."):
                continue
            target = os.path.join(workdir, safe)
            real_target = os.path.realpath(target)
            try:
                if os.path.commonpath([real_target, real_workdir]) != real_workdir:
                    continue
            except ValueError:
                continue
            try:
                data = base64.b64decode(content_b64, validate=False)
            except Exception:
                continue
            try:
                with open(target, "wb") as fh:
                    fh.write(data)
            except OSError:
                continue

    def _snapshot(self, workdir: str) -> dict[str, int]:
        snap: dict[str, int] = {}
        for root, _, files in os.walk(workdir):
            for fname in files:
                full = os.path.join(root, fname)
                try:
                    snap[os.path.relpath(full, workdir)] = os.stat(full).st_mtime_ns
                except OSError:
                    pass
        return snap

    def _collect_files(
        self,
        workdir: str,
        snap_before: dict[str, int],
    ) -> tuple[SandboxFile, ...]:
        out: list[SandboxFile] = []
        total = 0
        for root, _, files in os.walk(workdir):
            for fname in files:
                full = os.path.join(root, fname)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                rel = os.path.relpath(full, workdir)
                prev = snap_before.get(rel)
                if prev is not None and prev == st.st_mtime_ns:
                    continue
                size = st.st_size
                content_b64: str | None = None
                if size <= self.file_size_cap and total + size <= self.files_total_cap:
                    try:
                        with open(full, "rb") as fh:
                            content_b64 = base64.b64encode(fh.read()).decode("ascii")
                        total += size
                    except OSError:
                        pass
                out.append(SandboxFile(path=rel, size=size, content_b64=content_b64))
        return tuple(out)


def _make_preexec(mem_bytes: int, cpu_limit: int):
    """Return a preexec_fn that applies RLIMIT_* before exec."""

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

    return _set_limits
