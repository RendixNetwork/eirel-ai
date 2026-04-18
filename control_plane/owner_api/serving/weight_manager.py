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

    def chain_publication_state_payload(self, session: Session) -> dict[str, Any]:
        from control_plane.owner_api._constants import CHAIN_PUBLICATION_STATE_KEY
        record = self._owner._state_record(session, state_key=CHAIN_PUBLICATION_STATE_KEY)
        payload = dict(record.value_json or {}) if record is not None else {}
        return {
            "latest_run_id": payload.get("latest_run_id"),
            "latest_publication_batch_id": payload.get("latest_publication_batch_id"),
            "latest_build_inputs": payload.get("latest_build_inputs"),
            "latest_emitted_family_allocations": payload.get("latest_emitted_family_allocations"),
            "latest_publication_status": payload.get("latest_publication_status"),
            "latest_publication_error": payload.get("latest_publication_error"),
            "latest_submission_mode": payload.get("latest_submission_mode"),
            "latest_weight_setter_results": payload.get("latest_weight_setter_results"),
            "latest_published_at": payload.get("latest_published_at"),
            "publication_history": payload.get("publication_history") or [],
            "updated_at": payload.get("updated_at"),
        }

    def update_chain_publication_state(
        self,
        session: Session,
        *,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        from control_plane.owner_api._constants import CHAIN_PUBLICATION_STATE_KEY
        from control_plane.owner_api._helpers import utcnow
        record = self._owner._state_record(session, state_key=CHAIN_PUBLICATION_STATE_KEY, create=True)
        assert record is not None
        payload = {
            **dict(record.value_json or {}),
            **values,
            "updated_at": utcnow().isoformat(),
        }
        record.value_json = payload
        record.updated_at = utcnow()
        session.flush()
        return self.chain_publication_state_payload(session)
