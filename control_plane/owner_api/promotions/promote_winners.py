"""Pick the eval-winner per family at run close and record the promotion.

Two-step pipeline:

  1. :func:`select_winners_for_run` — pure read: per (family, run),
     return the highest-scoring eligible :class:`DeploymentScoreRecord`.
  2. :func:`promote_winners_for_run` — write: for each winner, create
     (or fetch idempotently) a :class:`ServingPromotion` row that
     anchors the run-end winner to the most recent
     :class:`ServingRelease`. Returns a structured
     :class:`PromotionResult` for the caller (typically a scheduled
     job in the owner-api operations loop).

The actual ``ServingDeployment`` rollout still happens via the
existing ``serving.serving_manager`` flow — this module is the audit
+ dispatch trigger, not the rollout engine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from shared.common.database import Database
from shared.common.models import (
    DeploymentScoreRecord,
    ManagedDeployment,
    ServingPromotion,
    ServingRelease,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "PromotionDecision",
    "PromotionResult",
    "select_winners_for_run",
    "promote_winners_for_run",
]


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """One per (family, run): the chosen winning deployment."""

    family_id: str
    run_id: str
    source_deployment_id: str
    submission_id: str
    miner_hotkey: str
    raw_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "run_id": self.run_id,
            "source_deployment_id": self.source_deployment_id,
            "submission_id": self.submission_id,
            "miner_hotkey": self.miner_hotkey,
            "raw_score": self.raw_score,
        }


@dataclass(slots=True)
class PromotionResult:
    """Outcome of one ``promote_winners_for_run`` call."""

    run_id: str
    promoted: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "promoted": list(self.promoted),
            "skipped": list(self.skipped),
        }


def select_winners_for_run(
    *,
    database: Database,
    run_id: str,
    families: list[str] | None = None,
) -> list[PromotionDecision]:
    """Return one :class:`PromotionDecision` per family for a given run.

    Eligible records are those with ``is_eligible=True`` and a
    ``health_status="healthy"`` deployment at the time of the read.
    The winner is the highest ``raw_score``; ties broken by
    ``deployment_id`` (deterministic).
    """
    decisions: list[PromotionDecision] = []
    with database.sessionmaker() as session:
        stmt = (
            select(DeploymentScoreRecord, ManagedDeployment)
            .join(
                ManagedDeployment,
                ManagedDeployment.id == DeploymentScoreRecord.deployment_id,
            )
            .where(DeploymentScoreRecord.run_id == run_id)
            .where(DeploymentScoreRecord.is_eligible.is_(True))
            .where(ManagedDeployment.health_status == "healthy")
        )
        if families:
            stmt = stmt.where(DeploymentScoreRecord.family_id.in_(families))
        rows = list(session.execute(stmt))

    by_family: dict[str, tuple[DeploymentScoreRecord, ManagedDeployment]] = {}
    for record, deployment in rows:
        prev = by_family.get(record.family_id)
        candidate_key = (-record.raw_score, record.deployment_id)
        if prev is None:
            by_family[record.family_id] = (record, deployment)
            continue
        prev_record = prev[0]
        prev_key = (-prev_record.raw_score, prev_record.deployment_id)
        if candidate_key < prev_key:
            by_family[record.family_id] = (record, deployment)

    for family_id in sorted(by_family):
        record, _deployment = by_family[family_id]
        decisions.append(
            PromotionDecision(
                family_id=family_id,
                run_id=run_id,
                source_deployment_id=record.deployment_id,
                submission_id=record.submission_id,
                miner_hotkey=record.miner_hotkey,
                raw_score=record.raw_score,
            )
        )
    return decisions


def _latest_published_release(session) -> ServingRelease | None:
    """Most recently published ``ServingRelease``, if any.

    The serving manager owns ``ServingRelease`` creation. The promotion
    job links its audit row to whatever the serving manager has most
    recently published — that's the release whose ``ServingDeployment``
    is currently answering product traffic.
    """
    stmt = (
        select(ServingRelease)
        .where(ServingRelease.status == "published")
        .order_by(ServingRelease.published_at.desc())
        .limit(1)
    )
    return session.scalar(stmt)


def promote_winners_for_run(
    *,
    database: Database,
    run_id: str,
    families: list[str] | None = None,
) -> PromotionResult:
    """Record one :class:`ServingPromotion` per family winner.

    Idempotent on ``(family_id, run_id)`` — the unique constraint on
    :class:`ServingPromotion` ensures re-running is safe. Skipped
    families (no eligible winner, no published release) are reported
    in :attr:`PromotionResult.skipped` for the caller to log.
    """
    decisions = select_winners_for_run(
        database=database, run_id=run_id, families=families
    )
    result = PromotionResult(run_id=run_id)

    with database.sessionmaker() as session:
        release = _latest_published_release(session)
        if release is None:
            for decision in decisions:
                result.skipped.append({
                    **decision.to_dict(),
                    "reason": "no_published_serving_release",
                })
            return result

        for decision in decisions:
            existing = session.scalar(
                select(ServingPromotion).where(
                    ServingPromotion.family_id == decision.family_id,
                    ServingPromotion.run_id == decision.run_id,
                )
            )
            if existing is not None:
                result.skipped.append({
                    **decision.to_dict(),
                    "reason": "already_promoted",
                    "promotion_id": existing.id,
                })
                continue
            row = ServingPromotion(
                family_id=decision.family_id,
                run_id=decision.run_id,
                source_deployment_id=decision.source_deployment_id,
                serving_release_id=release.id,
                metadata_json={
                    "submission_id": decision.submission_id,
                    "miner_hotkey": decision.miner_hotkey,
                    "raw_score": decision.raw_score,
                },
            )
            session.add(row)
            session.flush()
            result.promoted.append({
                **decision.to_dict(),
                "promotion_id": row.id,
                "serving_release_id": release.id,
            })
        session.commit()

    if result.promoted:
        _logger.info(
            "promoted %d eval-winners for run %s",
            len(result.promoted),
            run_id,
        )
    return result
