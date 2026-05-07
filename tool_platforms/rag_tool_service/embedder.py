"""OpenAI embeddings client for the RAG indexer.

Uses ``text-embedding-3-small`` by default (~$0.00002 per 1K tokens —
i.e. roughly $0.02 per million tokens, well below judging cost).
Async + bounded retry on transient errors. Batches requests up to the
provider's documented batch ceiling (~2048 inputs per call) to keep
indexing latency low for medium-sized corpora.

Environment knobs:
  * ``EIREL_RAG_EMBEDDING_BASE_URL`` (default: OpenAI public)
  * ``EIREL_RAG_EMBEDDING_API_KEY`` (defaults to ``OPENAI_API_KEY``)
  * ``EIREL_RAG_EMBEDDING_MODEL`` (default: ``text-embedding-3-small``)
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Sequence

import httpx

_logger = logging.getLogger(__name__)

__all__ = ["EmbeddingClient", "EmbeddingError"]


_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_BATCH_SIZE = 256
_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_MAX_RETRIES = 3
_RETRY_STATUSES: frozenset[int] = frozenset({429, 502, 503, 504})


class EmbeddingError(RuntimeError):
    """Generic embedding-call failure — surfaces as 502 to the caller."""


class EmbeddingClient:
    """Thin async wrapper around ``POST {base_url}/embeddings``."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("EIREL_RAG_EMBEDDING_BASE_URL")
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.getenv("EIREL_RAG_EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        self.model = (
            model
            or os.getenv("EIREL_RAG_EMBEDDING_MODEL")
            or _DEFAULT_MODEL
        )
        self.batch_size = max(1, int(batch_size))
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(0, int(max_retries))
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model and self.base_url)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(transport=self._transport)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def embed(self, inputs: Sequence[str]) -> list[list[float]]:
        """Return one embedding per input string, in order."""
        if not self.configured:
            raise EmbeddingError(
                "embedding client requires base_url + api_key + model"
            )
        if not inputs:
            return []
        out: list[list[float]] = []
        for i in range(0, len(inputs), self.batch_size):
            batch = list(inputs[i : i + self.batch_size])
            out.extend(await self._embed_batch(batch))
        return out

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        client = await self._get_client()
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": batch}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await client.post(
                    url, json=payload, headers=headers,
                    timeout=self.timeout_seconds,
                )
            except httpx.TimeoutException as exc:
                if attempt > self.max_retries:
                    raise EmbeddingError(
                        f"embedding timeout after {attempt} attempts: {exc}"
                    ) from exc
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
                continue
            except httpx.HTTPError as exc:
                if attempt > self.max_retries:
                    raise EmbeddingError(
                        f"embedding network error after {attempt} attempts: {exc}"
                    ) from exc
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
                continue
            if resp.status_code in _RETRY_STATUSES and attempt <= self.max_retries:
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
                continue
            if resp.status_code != 200:
                raise EmbeddingError(
                    f"embedding HTTP {resp.status_code}: "
                    f"{(resp.text or '')[:512]}"
                )
            data = resp.json()
            try:
                vectors = [item["embedding"] for item in data["data"]]
            except (KeyError, TypeError) as exc:
                raise EmbeddingError(
                    f"embedding response missing 'data[].embedding': {exc}"
                ) from exc
            if len(vectors) != len(batch):
                raise EmbeddingError(
                    f"embedding count mismatch: got {len(vectors)} for "
                    f"{len(batch)} inputs"
                )
            return vectors
