"""Picker for the product-mode serving deployments.

Mirrors :class:`~orchestration.orchestrator.miner_picker.MinerPicker` but
reads ``ServingDeployment`` rows. Big difference: **no thread-id pinning**.

Why no pinning: in product mode the deployment can change between turns
(when a new winner gets promoted), and user state lives in the product
DB — not in the deployment's checkpointer. Pinning a conversation to a
specific deployment would either freeze the user on a deprecated agent
or surface confusing "this conversation can't be resumed" errors. The
agent is stateless from the user's perspective; the orchestrator hands
it the full history on every turn.

Today's policy is single-replica per family (one ``ServingDeployment``
per ``family_id``). When we scale to multi-replica, this widens into a
top-K-by-latency_p50 round-robin similar to ``MinerPicker``; the
interface stays the same.
"""
from __future__ import annotations

import itertools
import logging
import os
import random
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from shared.common.database import Database
from shared.common.models import (
    ManagedDeployment,
    ManagedMinerSubmission,
    ServingDeployment,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "ServingCandidate",
    "ServingPicker",
    "NoEligibleServingDeploymentError",
]


# Default floor for inverse-latency weighting. A replica reporting <200ms
# would otherwise blow up the weight relative to a 1s replica; clamping
# avoids that pathological case.
_DEFAULT_LATENCY_FLOOR_MS: int = int(
    os.getenv("EIREL_SERVING_PICKER_LATENCY_FLOOR_MS", "200")
)


class NoEligibleServingDeploymentError(RuntimeError):
    """Raised when no healthy serving deployment is available for the family."""


@dataclass(frozen=True, slots=True)
class ServingCandidate:
    """One serving deployment the orchestrator can stream against."""

    deployment_id: str
    serving_release_id: str
    miner_hotkey: str
    family_id: str
    endpoint: str
    runtime_kind: str = "base_agent"
    latency_ms_p50: int = 0
    picker_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "serving_release_id": self.serving_release_id,
            "miner_hotkey": self.miner_hotkey,
            "family_id": self.family_id,
            "endpoint": self.endpoint,
            "runtime_kind": self.runtime_kind,
            "latency_ms_p50": self.latency_ms_p50,
            "picker_weight": self.picker_weight,
        }


class ServingPicker:
    """Picks one healthy serving deployment for a given family.

    Among the top-K candidates (by published_at desc), select via
    inverse-latency weighted random sampling so the fastest replica
    gets more traffic without monopolizing it. The ``latency_floor_ms``
    parameter clamps the denominator so a replica with no telemetry
    yet (or pathologically low latency) doesn't dominate. Replicas
    with ``latency_ms_p50 == 0`` (typically newcomers) get the *median*
    weight of the candidate pool — they have to earn traffic share
    rather than getting starved or overwhelmed.

    Set ``weighted=False`` to fall back to round-robin — useful for
    tests and for environments where the latency telemetry is
    known-bad.
    """

    def __init__(
        self,
        *,
        database: Database,
        top_k: int = 5,
        weighted: bool = True,
        latency_floor_ms: int = _DEFAULT_LATENCY_FLOOR_MS,
        rng: random.Random | None = None,
    ):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if latency_floor_ms < 1:
            raise ValueError("latency_floor_ms must be at least 1")
        self._db = database
        self._top_k = top_k
        self._weighted = weighted
        self._latency_floor_ms = latency_floor_ms
        self._cursor: dict[str, itertools.count] = {}
        self._cursor_lock = threading.Lock()
        self._rng = rng or random.Random()

    def _next_cursor(self, family_id: str) -> int:
        with self._cursor_lock:
            counter = self._cursor.setdefault(family_id, itertools.count())
            return next(counter)

    def _compute_weights(self, latencies_ms: Sequence[int]) -> list[float]:
        """Inverse-latency weight per candidate, with newcomer fallback.

        Replicas with ``latency_ms_p50 == 0`` (no telemetry yet) get the
        median weight of replicas that DO have telemetry. With no
        telemetry anywhere, every weight is 1.0 (uniform random).
        """
        floor = self._latency_floor_ms
        weights: list[float | None] = []
        known: list[float] = []
        for latency in latencies_ms:
            if latency <= 0:
                weights.append(None)  # placeholder for newcomer
                continue
            w = 1.0 / max(int(latency), floor)
            weights.append(w)
            known.append(w)
        if known:
            sorted_known = sorted(known)
            median = sorted_known[len(sorted_known) // 2]
        else:
            median = 1.0
        return [(median if w is None else w) for w in weights]

    def _runtime_kind_for(self, session, deployment: ServingDeployment) -> str:
        submission = session.get(ManagedMinerSubmission, deployment.source_submission_id)
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
        excluded_deployment_ids: Sequence[str] = (),
    ) -> ServingCandidate:
        """Return one healthy serving deployment for the family.

        Raises :class:`NoEligibleServingDeploymentError` when no healthy
        deployment is available — caller should surface this as a 503
        and either fall back to the eval orchestrator or return a stale
        cached response, depending on product policy.
        """
        excluded = set(excluded_deployment_ids)
        with self._db.sessionmaker() as session:
            # Pull latency_ms_p50 via the eval-side ManagedDeployment row
            # — ServingDeployment doesn't carry the metric directly today.
            stmt = (
                select(ServingDeployment, ManagedDeployment.latency_ms_p50)
                .join(
                    ManagedDeployment,
                    ManagedDeployment.id == ServingDeployment.source_deployment_id,
                    isouter=True,
                )
                .where(
                    ServingDeployment.family_id == family_id,
                    ServingDeployment.status == "healthy",
                    ServingDeployment.health_status == "healthy",
                )
                .order_by(ServingDeployment.published_at.desc())
                .limit(self._top_k)
            )
            rows = [
                (d, int(latency or 0))
                for d, latency in session.execute(stmt).all()
                if d.id not in excluded
            ]
            if not rows:
                raise NoEligibleServingDeploymentError(
                    f"no healthy serving deployment for family {family_id!r}"
                )
            if self._weighted and len(rows) > 1:
                weights = self._compute_weights([r[1] for r in rows])
                chosen, latency = self._rng.choices(
                    rows, weights=weights, k=1,
                )[0]
                weight = weights[rows.index((chosen, latency))]
            else:
                idx = self._next_cursor(family_id) % len(rows)
                chosen, latency = rows[idx]
                weight = 1.0
            return ServingCandidate(
                deployment_id=chosen.id,
                serving_release_id=chosen.release_id,
                miner_hotkey=chosen.miner_hotkey,
                family_id=chosen.family_id,
                endpoint=chosen.endpoint,
                runtime_kind=self._runtime_kind_for(session, chosen),
                latency_ms_p50=latency,
                picker_weight=float(weight),
            )
