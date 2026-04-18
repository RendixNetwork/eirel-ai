from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import select

from shared.common.models import ManagedDeployment

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
      capacity at build time.  ``reconcile_family_deployments`` picks
      these up via its ``bootstrapping_ids`` set.
    * ``queued`` — a miner submitted *after* the current run opened.
      ``start_queued_deployments`` only fires on run-open events
      (``run_manager.py:162``), so without this sweep a queued
      deployment sits in ``standby_cold / queued`` until the next
      rollover — days in production.  Closes the gap operators hit
      during multi-miner rehearsals (we had to rollover twice per
      submit to unstick them).  ``reconcile_family_deployments``'s
      bootstrapping filter does NOT include ``queued`` placements,
      so we route them through ``schedule_queued_deployments``
      directly — the same entry point the run-open path uses.
    """
    with services.db.sessionmaker() as session:
        stuck = list(
            session.execute(
                select(
                    ManagedDeployment.id,
                    ManagedDeployment.family_id,
                    ManagedDeployment.placement_status,
                )
                .where(
                    ManagedDeployment.placement_status.in_(
                        ("pending_capacity", "queued")
                    )
                )
                .where(ManagedDeployment.status.notin_(("build_failed", "retired")))
            )
        )
    queued_ids = [row.id for row in stuck if row.placement_status == "queued"]
    pending_families = {
        row.family_id for row in stuck if row.placement_status == "pending_capacity"
    }

    if queued_ids:
        # Queued → building is a gated transition — the state machine
        # only permits ``queued -> received``, and the run-open path
        # (``run_manager.py:start_queued_deployments``) does that
        # transition only for submissions introduced in the *current*
        # open run.  Submissions arriving during an open run are
        # assigned to the *next* run by ``submission_target_run``, so
        # they'd normally wait until that run opens to start building.
        # That's fine for activation (they can't be evaluated until
        # next run anyway), but it means the build artifact isn't
        # ready at activation time — the 1–2 minute build window
        # becomes observable downtime.  Promote them to ``received``
        # here so they build eagerly; activation still gates on
        # run-open.
        from control_plane.owner_api._helpers import utcnow
        from shared.common.models import ManagedDeployment as _MD
        with services.db.sessionmaker() as session:
            rows = list(
                session.execute(
                    select(_MD).where(_MD.id.in_(queued_ids))
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
