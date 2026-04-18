from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select

from shared.common.database import Database
from shared.common.metagraph import (
    BittensorMetagraphClient,
    FileMetagraphClient,
    MetagraphClient,
    MetagraphNeuron,
)
from shared.common.models import MetagraphSyncSnapshot, RegisteredNeuron, ValidatorRecord
from sqlalchemy import func


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class MetagraphSyncService:
    """Syncs registered neurons (uid:hotkey pairs) from the metagraph.

    Validator authorization is NOT handled here — the `validators` table is
    an operator-managed whitelist (see ValidatorRecord). This service only
    tracks which hotkeys are registered on the subnet so the owner API can
    gate submissions.
    """

    def __init__(
        self,
        *,
        db: Database,
        network: str,
        netuid: int,
        snapshot_path: str,
    ) -> None:
        self.db = db
        self.network = network
        self.netuid = netuid
        self.snapshot_path = snapshot_path

    def client(self) -> MetagraphClient:
        if self.snapshot_path:
            return FileMetagraphClient(self.snapshot_path)
        return BittensorMetagraphClient(network=self.network, netuid=self.netuid)

    async def run_sync(
        self, *, neurons: list[MetagraphNeuron] | None = None
    ) -> dict[str, Any]:
        try:
            if neurons is None:
                client = self.client()
                neurons = await asyncio.wait_for(
                    client.fetch_all_neurons(), timeout=60.0,
                )
            self._persist_neurons(neurons=neurons)
            return self.status_payload()
        except Exception as exc:
            with self.db.sessionmaker() as session:
                session.add(
                    MetagraphSyncSnapshot(
                        netuid=self.netuid,
                        network=self.network,
                        validator_count=0,
                        miner_count=0,
                        status="failed",
                        error_text=str(exc),
                        payload_json={},
                    )
                )
                session.commit()
            raise

    def _persist_neurons(self, *, neurons: list[MetagraphNeuron]) -> None:
        """Upsert synced hotkeys and DELETE deregistered ones.

        Presence in registered_neurons = registered on chain = allowed to
        submit. A hotkey that drops off the metagraph (deregistered by the
        subnet) is removed from the table entirely.
        """
        with self.db.sessionmaker() as session:
            synced_hotkeys = {n.hotkey for n in neurons}
            for neuron in neurons:
                record = session.get(RegisteredNeuron, neuron.hotkey)
                if record is None:
                    session.add(RegisteredNeuron(hotkey=neuron.hotkey, uid=neuron.uid))
                else:
                    record.uid = neuron.uid
                    record.last_synced_at = utcnow()
            if synced_hotkeys:
                session.execute(
                    delete(RegisteredNeuron).where(
                        RegisteredNeuron.hotkey.notin_(synced_hotkeys)
                    )
                )
            session.add(
                MetagraphSyncSnapshot(
                    netuid=self.netuid,
                    network=self.network,
                    validator_count=0,
                    miner_count=len(neurons),
                    status="success",
                    payload_json={
                        "neurons": [
                            {"hotkey": n.hotkey, "uid": n.uid} for n in neurons
                        ],
                    },
                )
            )
            session.commit()

    def status_payload(self) -> dict[str, Any]:
        with self.db.sessionmaker() as session:
            latest = session.execute(
                select(MetagraphSyncSnapshot)
                .order_by(MetagraphSyncSnapshot.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            validator_count = int(
                session.execute(
                    select(func.count())
                    .select_from(ValidatorRecord)
                    .where(ValidatorRecord.is_active.is_(True))
                ).scalar_one()
            )
            if latest is None:
                return {
                    "status": "never_synced",
                    "network": self.network,
                    "netuid": self.netuid,
                    "neuron_count": 0,
                    "validator_count": validator_count,
                }
            return {
                "status": latest.status,
                "network": latest.network,
                "netuid": latest.netuid,
                "neuron_count": latest.miner_count,
                "validator_count": validator_count,
                "error": latest.error_text,
                "created_at": latest.created_at.isoformat(),
            }
