from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import bittensor as bt
from fastapi import HTTPException, Request, status
from redis import asyncio as redis_asyncio

from shared.common.bittensor_signing import build_signature_message

_security_logger = logging.getLogger('eirel.security')


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(slots=True)
class SignatureHeaders:
    hotkey: str
    signature: str
    timestamp: str


class ReplayProtector(Protocol):
    async def claim(
        self,
        *,
        hotkey: str,
        request_id: str,
        ttl_seconds: int,
    ) -> bool:
        ...

    async def close(self) -> None:
        ...


class InMemoryReplayProtector:
    MAX_ENTRIES = 100_000

    def __init__(self) -> None:
        self._entries: dict[str, float] = {}

    async def claim(
        self,
        *,
        hotkey: str,
        request_id: str,
        ttl_seconds: int,
    ) -> bool:
        key = f"{hotkey}:{request_id}"
        now = time.monotonic()
        # Remove expired entries
        expired = [item for item, deadline in self._entries.items() if deadline <= now]
        for item in expired:
            self._entries.pop(item, None)
        # If still over limit after TTL cleanup, evict oldest 25%
        if len(self._entries) > self.MAX_ENTRIES:
            sorted_keys = sorted(self._entries, key=self._entries.get)  # type: ignore[arg-type]
            for k in sorted_keys[: len(self._entries) // 4]:
                del self._entries[k]
        if key in self._entries:
            return False
        self._entries[key] = now + ttl_seconds
        return True

    async def close(self) -> None:
        return None


class RedisReplayProtector:
    def __init__(self, *, redis_url: str) -> None:
        self.client = redis_asyncio.from_url(redis_url, decode_responses=True)

    async def claim(
        self,
        *,
        hotkey: str,
        request_id: str,
        ttl_seconds: int,
    ) -> bool:
        key = f"eirel:request-replay:{hotkey}:{request_id}"
        stored = await self.client.set(key, "1", ex=ttl_seconds, nx=True)
        return bool(stored)

    async def close(self) -> None:
        await self.client.aclose()


def create_replay_protector(redis_url: str) -> ReplayProtector:
    if redis_url.strip():
        return RedisReplayProtector(redis_url=redis_url)
    return InMemoryReplayProtector()


class SignatureVerifier:
    def verify(
        self,
        *,
        hotkey: str,
        signature: str,
        timestamp: str,
        method: str,
        path: str,
        body: bytes,
        ttl_seconds: int,
    ) -> None:
        try:
            issued_at = datetime.fromisoformat(timestamp)
        except ValueError as exc:
            _security_logger.warning(
                'auth_failure reason=%s hotkey=%s method=%s path=%s',
                'invalid timestamp', hotkey, method, path,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid timestamp"
            ) from exc
        now = datetime.now(UTC)
        skew = abs((now - issued_at.astimezone(UTC)).total_seconds())
        if skew > ttl_seconds:
            _security_logger.warning(
                'auth_failure reason=%s hotkey=%s method=%s path=%s',
                'stale timestamp', hotkey, method, path,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="stale timestamp"
            )
        body_hash = sha256_hex(body)
        message = build_signature_message(method, path, body_hash, timestamp)
        try:
            keypair = bt.Keypair(ss58_address=hotkey)
        except Exception as exc:
            _security_logger.warning(
                'auth_failure reason=%s hotkey=%s method=%s path=%s',
                'invalid hotkey', hotkey, method, path,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid hotkey"
            ) from exc
        if not keypair.verify(message, signature):
            _security_logger.warning(
                'auth_failure reason=%s hotkey=%s method=%s path=%s',
                'invalid signature', hotkey, method, path,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid signature"
            )


async def authenticate_request(
    request: Request,
    *,
    verifier: SignatureVerifier,
    ttl_seconds: int,
    replay_protector: ReplayProtector | None = None,
) -> str:
    hotkey = request.headers.get("X-Hotkey")
    signature = request.headers.get("X-Signature")
    timestamp = request.headers.get("X-Timestamp")
    if not hotkey or not signature or not timestamp:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing signature headers",
        )
    body = request.scope.get("cached_body")
    if body is None:
        body = getattr(request.state, "cached_body", None)
    if body is None:
        body = await request.body()
    verifier.verify(
        hotkey=hotkey,
        signature=signature,
        timestamp=timestamp,
        method=request.method,
        path=request.url.path,
        body=body,
        ttl_seconds=ttl_seconds,
    )
    if replay_protector is not None and request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
        request_id = request.headers.get("X-Request-Id")
        if not request_id:
            request_id = sha256_hex(
                f"{hotkey}:{signature}:{timestamp}:{request.method.upper()}:{request.url.path}".encode()
            )
        accepted = await replay_protector.claim(
            hotkey=hotkey,
            request_id=request_id,
            ttl_seconds=ttl_seconds,
        )
        if not accepted:
            _security_logger.warning(
                'replay_rejected hotkey=%s request_id=%s method=%s path=%s',
                hotkey, request_id, request.method, request.url.path,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="replayed request",
            )
    _security_logger.info(
        'auth_success hotkey=%s method=%s path=%s',
        hotkey, request.method, request.url.path,
    )
    return hotkey
