from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import bittensor as bt


@dataclass(slots=True)
class MetagraphNeuron:
    """A hotkey registered on the metagraph. We don't track axon activity —
    Bittensor's `active` flag refers to serving an axon/validator, which we
    don't use. The uid:hotkey pair is what matters."""

    hotkey: str
    uid: int


class MetagraphClient:
    async def fetch_all_neurons(self) -> list[MetagraphNeuron]:
        raise NotImplementedError


class FileMetagraphClient(MetagraphClient):
    def __init__(self, snapshot_path: str):
        self.snapshot_path = snapshot_path

    async def fetch_all_neurons(self) -> list[MetagraphNeuron]:
        if not self.snapshot_path:
            return []
        payload = json.loads(Path(self.snapshot_path).read_text())
        return [MetagraphNeuron(hotkey=item["hotkey"], uid=item["uid"]) for item in payload]


class BittensorMetagraphClient(MetagraphClient):
    def __init__(self, *, network: str, netuid: int):
        self.network = network
        self.netuid = netuid

    async def fetch_all_neurons(self) -> list[MetagraphNeuron]:
        subtensor = bt.Subtensor(network=self.network)
        metagraph = subtensor.metagraph(netuid=self.netuid, lite=True)
        return [
            MetagraphNeuron(hotkey=hotkey, uid=uid)
            for uid, hotkey in enumerate(metagraph.hotkeys)
        ]
