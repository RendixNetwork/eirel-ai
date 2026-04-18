from __future__ import annotations

"""Miner submissions, registrations, and candidate selection.

Extracted from ``ManagedOwnerServices`` (Item 15) to reduce the size of
the god-object.  Each public method here has a thin delegation wrapper
in ``ManagedOwnerServices`` for backward compatibility.
"""

from typing import Any, TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.common.models import (
    ManagedDeployment,
    ManagedMinerSubmission,
)
from shared.contracts.models import MinerRegistryEntry

if TYPE_CHECKING:
    from control_plane.owner_api.managed import ManagedOwnerServices


class MinerRegistry:
    """Handles miner submissions, deployments, and the candidate registry."""

    def __init__(self, owner: ManagedOwnerServices) -> None:
        self._owner = owner

    @property
    def db(self):
        return self._owner.db

    @property
    def settings(self):
        return self._owner.settings

    def latest_submission_for_hotkey(
        self, session: Session, *, miner_hotkey: str
    ) -> ManagedMinerSubmission | None:
        return session.execute(
            select(ManagedMinerSubmission)
            .where(ManagedMinerSubmission.miner_hotkey == miner_hotkey)
            .order_by(ManagedMinerSubmission.submission_seq.desc())
            .limit(1)
        ).scalar_one_or_none()

    def get_deployment_for_submission(
        self, session: Session, submission_id: str
    ) -> ManagedDeployment | None:
        return session.execute(
            select(ManagedDeployment)
            .where(ManagedDeployment.submission_id == submission_id)
            .order_by(ManagedDeployment.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()

    def candidate_runtime_endpoint(self, *, deployment_id: str) -> str:
        return f"{self.settings.owner_api_internal_url.rstrip('/')}/runtime/{deployment_id}"
