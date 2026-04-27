from __future__ import annotations

"""Weight aggregation, chain publication, and scorecard management.

Extracted from ``ManagedOwnerServices`` (Item 15) to reduce the size of
the god-object.  Each public method here has a thin delegation wrapper
in ``ManagedOwnerServices`` for backward compatibility.
"""

from typing import Any, TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.common.models import (
    AggregateFamilyScoreSnapshot,
)
if TYPE_CHECKING:
    from control_plane.owner_api.managed import ManagedOwnerServices


class WeightManager:
    """Handles weight aggregation, score snapshots, and chain publication."""

    def __init__(self, owner: ManagedOwnerServices) -> None:
        self._owner = owner

    @property
    def db(self):
        return self._owner.db

    @property
    def settings(self):
        return self._owner.settings

    def aggregate_snapshot_for_family(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
    ) -> AggregateFamilyScoreSnapshot | None:
        return session.execute(
            select(AggregateFamilyScoreSnapshot).where(
                AggregateFamilyScoreSnapshot.run_id == run_id,
                AggregateFamilyScoreSnapshot.family_id == family_id,
            )
        ).scalar_one_or_none()

