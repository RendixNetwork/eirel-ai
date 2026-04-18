from __future__ import annotations

from collections import defaultdict, deque
from time import monotonic

from fastapi import HTTPException, Request, status


def parse_api_keys(raw_value: str) -> set[str]:
    return {item.strip() for item in raw_value.split(",") if item.strip()}


class ApiKeyAuthorizer:
    def __init__(self, allowed_keys: set[str]) -> None:
        self.allowed_keys = allowed_keys

    async def __call__(self, request: Request) -> str:
        if not self.allowed_keys:
            return "anonymous"
        api_key = request.headers.get("X-API-Key", "").strip()
        if not api_key or api_key not in self.allowed_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid api key",
            )
        return api_key


class SlidingWindowRateLimiter:
    def __init__(self, *, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, principal: str) -> None:
        if self.max_requests <= 0 or self.window_seconds <= 0:
            return
        now = monotonic()
        window_floor = now - self.window_seconds
        hits = self._hits[principal]
        while hits and hits[0] <= window_floor:
            hits.popleft()
        if len(hits) >= self.max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="rate limit exceeded",
            )
        hits.append(now)


def rate_limit_principal(request: Request, *, authenticated_identity: str | None) -> str:
    if authenticated_identity and authenticated_identity != "anonymous":
        return authenticated_identity
    client = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    return forwarded or client
