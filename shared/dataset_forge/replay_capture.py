"""Stub for replay-research web snapshot capture.

The full implementation needs live ``provider_proxy`` access to fetch and
record web pages, and writes the results to
``s3://eirel-owner-private/datasets/replay_captures/{run_id}/``. That has
moving parts (browser sessions, rate limits, retry policy) that warrant a
dedicated workstream — see Phase 2.5 in the plan.

For now this module exposes the *interface* the rest of the forge consumes,
so the orchestration code does not need to special-case "replay capture
implemented yet?". The default ``StaticReplayCaptureClient`` returns a fixed
snapshot id (the same one analyst.json points at today) and is sufficient for
end-to-end forge runs while the live capture pipeline is being built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ReplaySnapshotRef:
    snapshot_id: str
    base_url: str
    mode: str = "replay_web"

    def to_retrieval_environment(self) -> dict[str, str]:
        return {
            "mode": self.mode,
            "base_url": self.base_url,
            "snapshot_id": self.snapshot_id,
        }


class ReplayCaptureClient(Protocol):
    async def snapshot_for(self, *, run_id: str, topic: str) -> ReplaySnapshotRef:
        ...


class StaticReplayCaptureClient:
    """Returns a single fixed snapshot ref. Suitable for tests and bring-up.

    Once ``replay_capture.py`` grows the real capture path, replace this with
    ``LiveReplayCaptureClient`` (or similar) wired to provider_proxy.
    """

    def __init__(
        self,
        *,
        snapshot_id: str = "retrieval-default-v1",
        base_url: str = "http://retrieval-service:8080",
    ) -> None:
        self.snapshot_id = snapshot_id
        self.base_url = base_url

    async def snapshot_for(self, *, run_id: str, topic: str) -> ReplaySnapshotRef:
        del run_id, topic
        return ReplaySnapshotRef(snapshot_id=self.snapshot_id, base_url=self.base_url)
