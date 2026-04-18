from __future__ import annotations

"""Minimal CalibrationManager stub.

The legacy analyst calibration pipeline has been retired. This class
remains only so ``ManagedOwnerServices`` can wire a ``services.calibration``
attribute; all methods are no-ops and return benign defaults. When a new
family (``deep_research``, ``coding``) needs calibration, reintroduce the
real implementation here.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from control_plane.owner_api.managed import ManagedOwnerServices


class CalibrationManager:
    def __init__(self, owner: "ManagedOwnerServices") -> None:
        self._owner = owner

    @property
    def settings(self) -> Any:
        return self._owner.settings

    @property
    def db(self) -> Any:
        return self._owner.db

    def recalculate_family_calibration_report(
        self,
        session: Any,
        *,
        run_id: str,
        family_id: str,
        deployment_id: str | None = None,
        submission_id: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        del session, deployment_id, submission_id
        return (
            {
                "run_id": run_id,
                "family_id": family_id,
                "status": "not_applicable",
                "reason": "calibration retired under general_chat 4D scoring",
            },
            "",
        )
