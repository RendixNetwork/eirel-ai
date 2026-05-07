"""Embedding client for project-memory recall.

Pluggable via :class:`EmbeddingClient` so tests can substitute a
deterministic stub. Production deployments wire :class:`ProxyEmbeddingClient`
against an OpenAI-compatible ``/embeddings`` endpoint (the subnet
provider-proxy or an external service).

Why pluggable: the production embedding choice (model, provider,
dimension) is a deployment-time decision driven by
``EIREL_EMBEDDING_BASE_URL`` / ``EIREL_EMBEDDING_MODEL`` env vars, but
tests need determinism — a stub that hashes text to a fixed vector lets
us verify ranking without spinning up a real model.

The client returns plain ``list[list[float]]`` — no numpy dependency
here. Vector size is the caller's contract; the writer pickles via
:mod:`struct` so storage is dependency-free.
"""
from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import httpx

_logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DIMENSION",
    "EmbeddingClient",
    "ProxyEmbeddingClient",
    "StubEmbeddingClient",
    "build_default_embedding_client",
]


DEFAULT_DIMENSION: int = 1536  # OpenAI text-embedding-3-small default
_STUB_DIMENSION: int = 64  # smaller for fast tests


class EmbeddingClient(ABC):
    """Async embedding endpoint."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in order."""

    async def aclose(self) -> None:
        return None


# -- Stub: deterministic, no network ----------------------------------------


class StubEmbeddingClient(EmbeddingClient):
    """Hash-based pseudo-embedding for tests.

    Produces a deterministic ``_STUB_DIMENSION``-element float vector
    derived from the SHA-256 of the input text. Two strings with
    overlapping tokens yield similar vectors (we add per-token unit
    vectors so cosine similarity is meaningful) — enough to validate
    ranking logic without touching a real model.
    """

    @property
    def dimension(self) -> int:
        return _STUB_DIMENSION

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    @staticmethod
    def _embed_one(text: str) -> list[float]:
        vec = [0.0] * _STUB_DIMENSION
        # Tokenize on whitespace; each token contributes a unit-vector
        # bump based on its hash. Repeated tokens stack, so longer
        # documents containing the same word as the query have higher
        # cosine similarity to it — exactly what real embeddings do
        # qualitatively.
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(_STUB_DIMENSION):
                vec[i] += (digest[i % len(digest)] / 255.0) - 0.5
        # Normalize to unit length so cosine == dot product downstream.
        norm = sum(v * v for v in vec) ** 0.5
        if norm == 0:
            return vec
        return [v / norm for v in vec]


# -- Proxy: OpenAI-compatible ------------------------------------------------


class ProxyEmbeddingClient(EmbeddingClient):
    """Calls an OpenAI-compatible ``/embeddings`` endpoint.

    Configure via env / constructor:
      * ``EIREL_EMBEDDING_BASE_URL`` — e.g. ``http://provider-proxy:8092/v1``
        or ``https://api.openai.com/v1``
      * ``EIREL_EMBEDDING_API_KEY`` — bearer token for the upstream
      * ``EIREL_EMBEDDING_MODEL`` — model name (default ``text-embedding-3-small``)
      * ``EIREL_EMBEDDING_DIMENSION`` — expected output dimension
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        dimension: int | None = None,
        timeout_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._base_url = (
            base_url or os.getenv("EIREL_EMBEDDING_BASE_URL", "")
        ).rstrip("/")
        self._api_key = api_key if api_key is not None else os.getenv(
            "EIREL_EMBEDDING_API_KEY", "",
        )
        self._model = model or os.getenv(
            "EIREL_EMBEDDING_MODEL", "text-embedding-3-small",
        )
        env_dim = os.getenv("EIREL_EMBEDDING_DIMENSION")
        self._dimension = int(dimension or env_dim or DEFAULT_DIMENSION)
        self._timeout = timeout_seconds
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            kwargs: dict[str, Any] = {"timeout": self._timeout}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self._base_url:
            raise RuntimeError(
                "ProxyEmbeddingClient requires EIREL_EMBEDDING_BASE_URL "
                "(or base_url= argument) to be set",
            )
        if not texts:
            return []
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload: dict[str, Any] = {
            "model": self._model,
            "input": list(texts),
        }
        client = await self._get_client()
        resp = await client.post(
            f"{self._base_url}/embeddings", json=payload, headers=headers,
        )
        resp.raise_for_status()
        body = resp.json()
        # OpenAI-compatible shape: {data: [{embedding: [...], index: i}, ...]}
        items = body.get("data") or []
        # Don't trust upstream order — sort by index.
        items_sorted = sorted(items, key=lambda r: int(r.get("index", 0)))
        out: list[list[float]] = []
        for item in items_sorted:
            embedding = item.get("embedding") or []
            if not isinstance(embedding, list):
                raise RuntimeError(
                    f"embedding endpoint returned non-list for index "
                    f"{item.get('index')}: {type(embedding).__name__}"
                )
            out.append([float(x) for x in embedding])
        return out


def build_default_embedding_client() -> EmbeddingClient:
    """Pick an embedding client based on env config.

    Falls back to :class:`StubEmbeddingClient` when no
    ``EIREL_EMBEDDING_BASE_URL`` is configured — useful for local dev
    and tests without spinning up the provider-proxy / OpenAI.
    """
    if os.getenv("EIREL_EMBEDDING_BASE_URL", "").strip():
        return ProxyEmbeddingClient()
    _logger.info(
        "EIREL_EMBEDDING_BASE_URL not set; using StubEmbeddingClient. "
        "Recall quality will be hash-based until a real embedding "
        "endpoint is configured.",
    )
    return StubEmbeddingClient()
