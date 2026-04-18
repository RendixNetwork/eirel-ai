from __future__ import annotations

"""Serving deployments, releases, and lease operations.

Extracted from ``ManagedOwnerServices`` (Item 15) to reduce the size of
the god-object.  Each public method here has a thin delegation wrapper
in ``ManagedOwnerServices`` for backward compatibility.
"""

from typing import Any, TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.common.models import (
    ServingDeployment,
    ServingRelease,
)
from shared.contracts.models import MinerRegistryEntry

if TYPE_CHECKING:
    from control_plane.owner_api.managed import ManagedOwnerServices


class LeaseManager:
    """Handles serving deployments, releases, and the serving registry."""

    def __init__(self, owner: ManagedOwnerServices) -> None:
        self._owner = owner

    @property
    def db(self):
        return self._owner.db

    @property
    def settings(self):
        return self._owner.settings

    def latest_published_release(self, session: Session) -> ServingRelease | None:
        return session.execute(
            select(ServingRelease)
            .where(ServingRelease.status == "published")
            .order_by(ServingRelease.created_at.desc())
        ).scalar_one_or_none()

    def serving_release_by_id(self, session: Session, *, release_id: str) -> ServingRelease | None:
        return session.get(ServingRelease, release_id)

    def list_release_deployments(
        self, session: Session, *, release_id: str
    ) -> list[ServingDeployment]:
        return list(
            session.execute(
                select(ServingDeployment)
                .where(ServingDeployment.release_id == release_id)
                .order_by(ServingDeployment.family_id.asc())
            ).scalars()
        )

    def current_serving_fleet(self, session: Session) -> list[ServingDeployment]:
        release = self.latest_published_release(session)
        if release is None:
            return []
        return self.list_release_deployments(session, release_id=release.id)

    def serving_release_payload(self, release: ServingRelease | None) -> dict[str, Any] | None:
        if release is None:
            return None
        return {
            "id": release.id,
            "trigger_type": release.trigger_type,
            "status": release.status,
            "scheduled_for": release.scheduled_for.isoformat() if release.scheduled_for else None,
            "published_at": release.published_at.isoformat() if release.published_at else None,
            "cancelled_at": release.cancelled_at.isoformat() if release.cancelled_at else None,
            "metadata": release.metadata_json,
        }
