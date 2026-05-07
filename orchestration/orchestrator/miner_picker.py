"""Miner picker for the graph-runtime orchestrator.

Picks one healthy ``ManagedDeployment`` per (family, request) so the
execution coordinator can stream against a single miner pod. The
policy is simple by design — round-robin among the top-K by
``latency_ms_p50`` with a healthy/active gate. Sophistication
(quality-weighted ranking, caching, sticky sessions across regions)
lives in later milestones; flag risks #12 in the plan.

Thread-id continuity
--------------------

For graph-runtime miners, multi-turn flows must land on the same
deployment so the SDK's checkpointer hits its own thread state. When
``thread_id`` resolves to an existing :class:`ConversationThread`,
this module returns the pinned deployment regardless of latency rank.
A fresh ``thread_id`` (no row yet) takes the round-robin path.
"""
from __future__ import annotations

import itertools
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from shared.common.database import Database
from shared.common.models import (
    ConversationThread,
    ManagedDeployment,
    ManagedMinerSubmission,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "MinerCandidate",
    "MinerPicker",
    "NoEligibleMinerError",
]


class NoEligibleMinerError(RuntimeError):
    """Raised when no healthy deployment is available for the family."""


@dataclass(frozen=True, slots=True)
class MinerCandidate:
    """One miner the orchestrator can stream against."""

    deployment_id: str
    miner_hotkey: str
    family_id: str
    endpoint: str
    latency_ms_p50: int
    runtime_kind: str = "base_agent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "miner_hotkey": self.miner_hotkey,
            "family_id": self.family_id,
            "endpoint": self.endpoint,
            "latency_ms_p50": self.latency_ms_p50,
            "runtime_kind": self.runtime_kind,
        }


class MinerPicker:
    """Picks a deployment for the family + thread.

    Backed by a :class:`Database` handle so the picker reads ground
    truth from the same tables eirel-ai's submission/deployment loop
    writes to. Cursor state for round-robin is in-process (single
    orchestrator replica today); when the orchestrator scales out,
    this becomes a Redis HINCRBY.
    """

    def __init__(self, *, database: Database, top_k: int = 5):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._db = database
        self._top_k = top_k
        # Per-family round-robin counters. Lock-protected because
        # multiple incoming requests may pick concurrently.
        self._cursor: dict[str, itertools.count] = {}
        self._cursor_lock = threading.Lock()

    def _next_cursor(self, family_id: str) -> int:
        with self._cursor_lock:
            counter = self._cursor.setdefault(family_id, itertools.count())
            return next(counter)

    def _runtime_kind_for(self, session, deployment: ManagedDeployment) -> str:
        submission = session.get(ManagedMinerSubmission, deployment.submission_id)
        if submission is None:
            return "base_agent"
        manifest = submission.manifest_json or {}
        runtime = manifest.get("runtime") if isinstance(manifest, dict) else None
        if isinstance(runtime, dict):
            kind = runtime.get("kind")
            if isinstance(kind, str) and kind:
                return kind
        return "base_agent"

    def pick(
        self,
        *,
        family_id: str,
        thread_id: str | None = None,
        excluded_deployment_ids: Sequence[str] = (),
    ) -> MinerCandidate:
        """Return one healthy miner for the family.

        Order of resolution:
          1. If ``thread_id`` is bound to a deployment in
             ``conversation_threads``, return that deployment (sticky
             affinity for graph checkpoint resume).
          2. Otherwise rank healthy/active deployments for the family
             by ``latency_ms_p50`` ascending and round-robin among the
             top K.

        Raises :class:`NoEligibleMinerError` when the pinned
        deployment is gone (e.g. retired since the thread was created)
        AND no fallback deployments exist for the family. When the
        pinned deployment is gone but fallbacks exist, the picker
        falls forward to the round-robin path and logs a warning —
        thread continuity is best-effort, never a hard fail.
        """
        excluded = set(excluded_deployment_ids)
        with self._db.sessionmaker() as session:
            if thread_id:
                thread = session.get(ConversationThread, thread_id)
                if thread is not None and thread.deployment_id not in excluded:
                    pinned = session.get(ManagedDeployment, thread.deployment_id)
                    if pinned is not None and self._is_eligible(pinned, family_id):
                        return MinerCandidate(
                            deployment_id=pinned.id,
                            miner_hotkey=pinned.miner_hotkey,
                            family_id=pinned.family_id,
                            endpoint=pinned.endpoint,
                            latency_ms_p50=int(pinned.latency_ms_p50 or 0),
                            runtime_kind=self._runtime_kind_for(session, pinned),
                        )
                    if pinned is not None:
                        _logger.warning(
                            "thread %s pinned to deployment %s but it is no "
                            "longer eligible; falling forward to round-robin",
                            thread_id, thread.deployment_id,
                        )

            stmt = (
                select(ManagedDeployment)
                .where(
                    ManagedDeployment.family_id == family_id,
                    ManagedDeployment.status == "active",
                    ManagedDeployment.health_status == "healthy",
                )
                .order_by(ManagedDeployment.latency_ms_p50.asc())
                .limit(self._top_k)
            )
            top = [d for d in session.scalars(stmt) if d.id not in excluded]
            if not top:
                raise NoEligibleMinerError(
                    f"no healthy deployment available for family {family_id!r}"
                )
            idx = self._next_cursor(family_id) % len(top)
            chosen = top[idx]
            return MinerCandidate(
                deployment_id=chosen.id,
                miner_hotkey=chosen.miner_hotkey,
                family_id=chosen.family_id,
                endpoint=chosen.endpoint,
                latency_ms_p50=int(chosen.latency_ms_p50 or 0),
                runtime_kind=self._runtime_kind_for(session, chosen),
            )

    @staticmethod
    def _is_eligible(deployment: ManagedDeployment, family_id: str) -> bool:
        return (
            deployment.family_id == family_id
            and deployment.status == "active"
            and deployment.health_status == "healthy"
        )
