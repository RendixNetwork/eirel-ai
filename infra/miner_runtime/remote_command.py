from __future__ import annotations

"""Async SSH command execution for baremetal miner node management.

All remote operations (Docker commands, file transfers) go through these
helpers.  SSH multiplexing is used to avoid repeated handshakes.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from control_plane.owner_api.baremetal_inventory import BaremetalNode

logger = logging.getLogger(__name__)

_SSH_COMMON_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=/tmp/eirel-ssh-%r@%h:%p",
    "-o", "ControlPersist=60",
]


class RemoteCommandError(RuntimeError):
    """Raised when a remote SSH command fails."""

    def __init__(self, host: str, command: str, returncode: int, stderr: str) -> None:
        self.host = host
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"remote command failed on {host} (exit {returncode}): {stderr[:200]}")


async def run_remote_command(
    *,
    ssh_host: str,
    ssh_user: str,
    ssh_key_path: str,
    ssh_port: int = 22,
    command: list[str],
    check: bool = True,
    timeout_seconds: float = 120.0,
) -> str:
    """Execute a command on a remote host via SSH."""
    ssh_cmd = [
        "ssh",
        *_SSH_COMMON_OPTS,
        "-i", ssh_key_path,
        "-p", str(ssh_port),
        f"{ssh_user}@{ssh_host}",
        "--",
        *command,
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        process.kill()
        raise RemoteCommandError(
            ssh_host, " ".join(command), -1, "command timed out"
        )
    if check and process.returncode != 0:
        raise RemoteCommandError(
            ssh_host,
            " ".join(command),
            process.returncode or -1,
            stderr.decode().strip(),
        )
    return stdout.decode().strip()


async def run_remote_docker(
    *,
    node: BaremetalNode,
    docker_args: list[str],
    check: bool = True,
    timeout_seconds: float = 120.0,
) -> str:
    """Run a Docker command on a remote baremetal node via SSH."""
    return await run_remote_command(
        ssh_host=node.ssh_host,
        ssh_user=node.ssh_user,
        ssh_key_path=node.ssh_key_path,
        ssh_port=node.ssh_port,
        command=["docker", *docker_args],
        check=check,
        timeout_seconds=timeout_seconds,
    )


async def scp_to_remote(
    *,
    node: BaremetalNode,
    local_path: str,
    remote_path: str,
    timeout_seconds: float = 300.0,
) -> None:
    """Transfer a file to a remote node via scp."""
    scp_cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-i", node.ssh_key_path,
        "-P", str(node.ssh_port),
        local_path,
        f"{node.ssh_user}@{node.ssh_host}:{remote_path}",
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *scp_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        process.kill()
        raise RemoteCommandError(
            node.ssh_host, f"scp {local_path}", -1, "transfer timed out"
        )
    if process.returncode != 0:
        raise RemoteCommandError(
            node.ssh_host,
            f"scp {local_path}",
            process.returncode or -1,
            stderr.decode().strip(),
        )


async def rsync_to_remote(
    *,
    node: BaremetalNode,
    local_dir: str,
    remote_dir: str,
    timeout_seconds: float = 300.0,
) -> None:
    """Sync a local directory to a remote node via rsync over SSH."""
    ssh_opts = (
        f"ssh -i {node.ssh_key_path} -p {node.ssh_port}"
        " -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=10"
    )
    rsync_cmd = [
        "rsync", "-az", "--delete",
        "-e", ssh_opts,
        f"{local_dir}/",
        f"{node.ssh_user}@{node.ssh_host}:{remote_dir}/",
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *rsync_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        process.kill()
        raise RemoteCommandError(
            node.ssh_host, f"rsync {local_dir}", -1, "transfer timed out"
        )
    if process.returncode != 0:
        raise RemoteCommandError(
            node.ssh_host,
            f"rsync {local_dir}",
            process.returncode or -1,
            stderr.decode().strip(),
        )
