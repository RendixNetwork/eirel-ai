from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import select

from shared.common.models import EvaluationRun, ManagedDeployment, ManagedMinerSubmission

if TYPE_CHECKING:
    from control_plane.owner_api.managed import ManagedOwnerServices

_logger = logging.getLogger(__name__)


async def _reap_pending_runtime_stops(
    services: ManagedOwnerServices,
) -> None:
    with services.db.sessionmaker() as session:
        ids = list(
            session.execute(
                select(ManagedDeployment.id).where(
                    ManagedDeployment.pending_runtime_stop.is_(True)
                )
            ).scalars()
        )
    for deployment_id in ids:
        try:
            await services.stop_deployment_runtime(
                deployment_id=deployment_id, reason="reaper", retire=True,
            )
        except (RuntimeError, OSError, TimeoutError, ValueError) as exc:
            _logger.warning("reaper stop failed for %s: %s", deployment_id, exc)


async def _retry_pending_capacity(
    services: ManagedOwnerServices,
) -> None:
    """Re-drive deployments stranded in build/placement limbo.

    Two placement states need a timed retry:

    * ``pending_capacity`` — the cluster ran out of schedulable
      capacity at build time. ``reconcile_family_deployments`` picks
      these up via its ``bootstrapping_ids`` set.
    * ``queued`` — a queued deployment whose *target* run has already
      opened but ``start_queued_deployments`` hasn't drained it yet
      (e.g. owner-api restarted between submit and run-open, or the
      submission arrived before run-1's first tick).

    **Critical gate**: we only promote a queued deployment if its
    ``introduced_run_id`` matches the currently-open run. Submissions
    arriving during an open run target run-N+1 via
    ``submission_target_run``; they must remain pooled until run-N+1
    opens. Activating them early would let validators evaluate a miner
    whose code was never supposed to be scored in the current run.
    """
    with services.db.sessionmaker() as session:
        current_run = session.execute(
            select(EvaluationRun)
            .where(EvaluationRun.status == "open")
            .order_by(EvaluationRun.sequence.desc())
            .limit(1)
        ).scalar_one_or_none()
        current_run_id = current_run.id if current_run is not None else None

        stuck = list(
            session.execute(
                select(
                    ManagedDeployment.id,
                    ManagedDeployment.family_id,
                    ManagedDeployment.placement_status,
                    ManagedMinerSubmission.introduced_run_id,
                )
                .join(
                    ManagedMinerSubmission,
                    ManagedDeployment.submission_id == ManagedMinerSubmission.id,
                )
                .where(
                    ManagedDeployment.placement_status.in_(
                        ("pending_capacity", "queued")
                    )
                )
                .where(ManagedDeployment.status.notin_(("build_failed", "retired")))
            )
        )
    queued_ids = [
        row.id
        for row in stuck
        if row.placement_status == "queued"
        and current_run_id is not None
        and row.introduced_run_id == current_run_id
    ]
    pending_families = {
        row.family_id for row in stuck if row.placement_status == "pending_capacity"
    }

    if queued_ids:
        from control_plane.owner_api._helpers import utcnow
        with services.db.sessionmaker() as session:
            rows = list(
                session.execute(
                    select(ManagedDeployment).where(ManagedDeployment.id.in_(queued_ids))
                ).scalars()
            )
            now = utcnow()
            for dep in rows:
                if dep.placement_status != "queued":
                    continue
                if dep.status not in ("queued", "standby_cold"):
                    continue
                dep.status = "received"
                dep.health_status = "starting"
                dep.placement_status = "pending"
                dep.health_details_json = {
                    **(dep.health_details_json or {}),
                    "build": "pending",
                    "deploy": "pending",
                }
                dep.updated_at = now
            session.commit()
        try:
            await services.deployments.schedule_queued_deployments(queued_ids)
        except (RuntimeError, OSError) as exc:
            _logger.warning(
                "queued-placement retry failed for %d deployments: %s",
                len(queued_ids), exc,
            )

    for family_id in pending_families:
        try:
            await services.reconcile_family_deployments(family_id=family_id)
        except (RuntimeError, OSError) as exc:
            _logger.warning(
                "pending_capacity retry failed for family %s: %s", family_id, exc,
            )
