from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import bittensor as bt


def build_signature_message(method: str, path: str, body_hash: str, timestamp: str) -> str:
    return f"{method.upper()}{path}{body_hash}{timestamp}"


@dataclass(slots=True)
class LoadedSigner:
    keypair: bt.Keypair

    @property
    def hotkey(self) -> str:
        return self.keypair.ss58_address

    def sign(self, method: str, path: str, body_hash: str, timestamp: str) -> str:
        message = build_signature_message(method, path, body_hash, timestamp)
        return f"0x{self.keypair.sign(message).hex()}"

    def signed_headers(self, method: str, path: str, body_hash: str) -> dict[str, str]:
        timestamp = datetime.now(UTC).isoformat()
        return {
            "X-Hotkey": self.hotkey,
            "X-Signature": self.sign(method, path, body_hash, timestamp),
            "X-Timestamp": timestamp,
            "X-Request-Id": str(uuid4()),
        }


def load_signer(
    *,
    wallet_name: str | None = None,
    hotkey_name: str | None = None,
    wallet_path: str | None = None,
    mnemonic: str | None = None,
) -> LoadedSigner:
    if mnemonic:
        return LoadedSigner(bt.Keypair.create_from_mnemonic(mnemonic))
    if not wallet_name or not hotkey_name:
        raise ValueError("wallet_name and hotkey_name are required when mnemonic is not provided")
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_path)
    return LoadedSigner(wallet.hotkey)
