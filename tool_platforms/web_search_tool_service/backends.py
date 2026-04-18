from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse, urlunparse

import httpx

from tool_platforms.web_search_tool_service.models import SearchDocument

_logger = logging.getLogger(__name__)

DEFAULT_BRAVE_SEARCH_BASE_URL = "https://api.search.brave.com/res/v1"
DEFAULT_FETCH_USER_AGENT = "EIREL-Web-Search-Tool/0.1 (+https://github.com/eirel)"

DATE_PATTERNS = (
    re.compile(r'article:published_time["\']?\s*content=["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'article:modified_time["\']?\s*content=["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'"datePublished"\s*:\s*"([^"]+)"', re.IGNORECASE),
    re.compile(r'"dateModified"\s*:\s*"([^"]+)"', re.IGNORECASE),
    re.compile(r'<time[^>]+datetime=["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b"),
)


# -- Exceptions --------

class RetryableBackendError(RuntimeError):
    pass


class HardBackendError(RuntimeError):
    pass


class AllBackendsFailedError(RuntimeError):
    def __init__(self, failures: list[tuple[str, str]]) -> None:
        self.failures = failures
        super().__init__(f"all backends failed: {failures}")


# -- Result container --------

@dataclass(slots=True, frozen=True)
class BackendResult:
    documents: list[SearchDocument]
    backend_name: str
    latency_ms: int
    attempted: list[str]


# -- Protocol --------

class SearchBackend(Protocol):
    name: str

    async def search(
        self,
        *,
        query: str,
        count: int,
        freshness: str | None = None,
        site: str | None = None,
    ) -> list[SearchDocument]: ...


# -- Shared data models --------

@dataclass(slots=True)
class ResearchDocumentRecord:
    document_id: str
    title: str
    url: str
    snippet: str
    content: str
    links: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResearchCatalogStore:
    documents: dict[str, ResearchDocumentRecord]


# -- Catalog backend --------

class CatalogBackend:
    name = "catalog"

    def __init__(self, catalog_store: ResearchCatalogStore) -> None:
        self._catalog_store = catalog_store

    async def search(
        self,
        *,
        query: str,
        count: int,
        freshness: str | None = None,
        site: str | None = None,
    ) -> list[SearchDocument]:
        query_terms = _normalize_terms(query)
        ranked: list[tuple[float, ResearchDocumentRecord]] = []
        for document in self._catalog_store.documents.values():
            haystack = " ".join(
                [
                    document.title,
                    document.url,
                    document.snippet,
                    document.content,
                    json.dumps(document.metadata, sort_keys=True),
                ]
            ).lower()
            score = sum(1.0 for term in query_terms if term in haystack)
            if score > 0:
                ranked.append((score, document))
        ranked.sort(key=lambda item: (-item[0], item[1].document_id))
        documents: list[SearchDocument] = []
        for score, document in ranked[:count]:
            documents.append(
                SearchDocument(
                    document_id=document.document_id,
                    title=document.title,
                    url=document.url,
                    snippet=document.snippet,
                    score=float(score),
                    metadata={
                        **document.metadata,
                        "canonical_url": _canonical_url(document.url),
                        "published_at": _published_at_from_text(
                            document.metadata.get("published_at"),
                            "\n".join([document.title, document.snippet, document.content]),
                        ),
                        "domain": _canonical_domain(document.url),
                        "search_provider": "catalog",
                    },
                )
            )
        return documents


# -- Brave backend --------

class BraveBackend:
    name = "brave"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BRAVE_SEARCH_BASE_URL,
        timeout_seconds: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def search(
        self,
        *,
        query: str,
        count: int,
        freshness: str | None = None,
        site: str | None = None,
    ) -> list[SearchDocument]:
        if not self._api_key:
            raise HardBackendError("brave search API key is not configured")
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.get(
                    f"{self._base_url}/web/search",
                    params={"q": query, "count": max(1, min(count, 10))},
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": self._api_key,
                        "User-Agent": DEFAULT_FETCH_USER_AGENT,
                    },
                )
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
            raise RetryableBackendError(str(exc)) from exc
        if response.status_code >= 500:
            raise RetryableBackendError(
                "brave returned %d: %s" % (response.status_code, response.text[:200])
            )
        if response.status_code >= 400:
            raise HardBackendError(
                "brave returned %d: %s" % (response.status_code, response.text[:200])
            )
        payload = response.json()
        results = payload.get("web", {}).get("results", [])
        if not isinstance(results, list):
            results = []
        documents: list[SearchDocument] = []
        for index, item in enumerate(results[:count]):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            canonical = _canonical_url(url)
            domain = _canonical_domain(canonical)
            published_at = _published_at_from_text(
                item.get("page_age") or item.get("age") or item.get("published_at"),
                json.dumps(item, sort_keys=True),
            )
            base_score = max(0.0, 1.0 - (index * 0.08))
            documents.append(
                SearchDocument(
                    document_id=_document_id_for_url(canonical),
                    title=str(item.get("title") or canonical),
                    url=url,
                    snippet=str(item.get("description") or item.get("meta_description") or ""),
                    score=round(base_score, 4),
                    metadata={
                        "canonical_url": canonical,
                        "search_provider": "brave",
                        "domain": domain,
                        "published_at": published_at,
                    },
                )
            )
        documents.sort(key=lambda item: (-item.score, item.document_id))
        return documents[:count]


# -- Serper backend --------

_FRESHNESS_TO_TBS: dict[str, str] = {
    "day": "qdr:d",
    "week": "qdr:w",
    "month": "qdr:m",
    "year": "qdr:y",
}

DEFAULT_SERPER_BASE_URL = "https://google.serper.dev"


class SerperBackend:
    name = "serper"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_SERPER_BASE_URL,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise HardBackendError("serper API key is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def search(
        self,
        *,
        query: str,
        count: int,
        freshness: str | None = None,
        site: str | None = None,
    ) -> list[SearchDocument]:
        q = f"{query} site:{site}" if site else query
        body: dict[str, Any] = {"q": q, "num": count}
        if freshness and freshness in _FRESHNESS_TO_TBS:
            body["tbs"] = _FRESHNESS_TO_TBS[freshness]

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    f"{self._base_url}/search",
                    headers={
                        "X-API-KEY": self._api_key,
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
            raise RetryableBackendError(str(exc)) from exc

        if response.status_code >= 500:
            raise RetryableBackendError(
                "serper returned %d" % response.status_code
            )
        if response.status_code == 429:
            raise RetryableBackendError("serper rate-limited (429)")
        if response.status_code >= 400:
            raise HardBackendError(
                "serper returned %d" % response.status_code
            )

        try:
            payload = response.json()
        except Exception as exc:
            raise RetryableBackendError("serper returned unparseable JSON") from exc

        organic = payload.get("organic")
        if not isinstance(organic, list):
            _logger.warning("serper response missing 'organic' key")
            raise RetryableBackendError("serper response missing 'organic' key")

        documents: list[SearchDocument] = []
        for entry in organic[:count]:
            if not isinstance(entry, dict):
                continue
            position = entry.get("position", 0)
            documents.append(
                SearchDocument(
                    document_id=f"serper-{position}",
                    title=str(entry.get("title", "")),
                    url=str(entry.get("link", "")),
                    snippet=str(entry.get("snippet", "")),
                    score=0.0,
                    metadata={
                        "source": "serper",
                        "position": position,
                        "date": entry.get("date", ""),
                    },
                )
            )
        return documents


# -- Tavily backend --------

_FRESHNESS_TO_TIME_RANGE: dict[str, str] = {
    "day": "day",
    "week": "week",
    "month": "month",
    "year": "year",
}

DEFAULT_TAVILY_BASE_URL = "https://api.tavily.com"


class TavilyBackend:
    name = "tavily"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_TAVILY_BASE_URL,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
        search_depth: str = "basic",
    ) -> None:
        if not api_key:
            raise HardBackendError("tavily API key is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport
        self._search_depth = search_depth

    async def search(
        self,
        *,
        query: str,
        count: int,
        freshness: str | None = None,
        site: str | None = None,
    ) -> list[SearchDocument]:
        body: dict[str, Any] = {
            "api_key": self._api_key,
            "query": query,
            "max_results": count,
            "search_depth": self._search_depth,
            "include_answer": True,
            "include_raw_content": False,
        }
        if site:
            body["include_domains"] = [site]
        if freshness and freshness in _FRESHNESS_TO_TIME_RANGE:
            body["time_range"] = _FRESHNESS_TO_TIME_RANGE[freshness]

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    f"{self._base_url}/search",
                    json=body,
                )
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
            raise RetryableBackendError(str(exc)) from exc

        if response.status_code >= 500:
            raise RetryableBackendError(
                "tavily returned %d" % response.status_code
            )
        if response.status_code == 429:
            raise RetryableBackendError("tavily rate-limited (429)")
        if response.status_code >= 400:
            raise HardBackendError(
                "tavily returned %d" % response.status_code
            )

        try:
            payload = response.json()
        except Exception as exc:
            raise RetryableBackendError("tavily returned unparseable JSON") from exc

        results = payload.get("results")
        if not isinstance(results, list):
            _logger.warning("tavily response missing 'results' key")
            raise RetryableBackendError("tavily response missing 'results' key")

        answer = payload.get("answer")
        documents: list[SearchDocument] = []
        for i, entry in enumerate(results[:count]):
            if not isinstance(entry, dict):
                continue
            content = str(entry.get("content", ""))
            meta: dict[str, Any] = {
                "source": "tavily",
                "score": entry.get("score", 0.0),
            }
            if i == 0 and answer:
                meta["answer"] = answer
            documents.append(
                SearchDocument(
                    document_id=f"tavily-{i}",
                    title=str(entry.get("title", "")),
                    url=str(entry.get("url", "")),
                    snippet=content[:400],
                    score=float(entry.get("score", 0.0)),
                    metadata=meta,
                )
            )
        return documents


# -- Fallback chain --------

class FallbackSearchBackend:
    def __init__(
        self,
        backends: list[SearchBackend],
        *,
        per_backend_timeout: float = 10.0,
    ) -> None:
        self._backends = backends
        self._per_backend_timeout = per_backend_timeout

    @property
    def name(self) -> str:
        names = ", ".join(b.name for b in self._backends)
        return f"fallback({names})"

    async def search(
        self,
        *,
        query: str,
        count: int,
        freshness: str | None = None,
        site: str | None = None,
    ) -> BackendResult:
        attempted: list[str] = []
        failures: list[tuple[str, str]] = []
        for backend in self._backends:
            attempted.append(backend.name)
            t0 = asyncio.get_running_loop().time()
            try:
                documents = await asyncio.wait_for(
                    backend.search(
                        query=query,
                        count=count,
                        freshness=freshness,
                        site=site,
                    ),
                    timeout=self._per_backend_timeout,
                )
                latency_ms = int(
                    (asyncio.get_running_loop().time() - t0) * 1000
                )
                return BackendResult(
                    documents=documents,
                    backend_name=backend.name,
                    latency_ms=latency_ms,
                    attempted=list(attempted),
                )
            except HardBackendError:
                raise
            except (RetryableBackendError, asyncio.TimeoutError) as exc:
                failures.append((backend.name, str(exc)))
                continue
        raise AllBackendsFailedError(failures)


# -- Shared utilities --------

def _normalize_terms(query: str) -> list[str]:
    return [term for term in query.lower().replace("-", " ").split() if term]


def _canonical_url(url: str) -> str:
    parsed = urlparse(url.strip())
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme.lower() or "https", netloc, path, "", "", ""))


def _canonical_domain(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _document_id_for_url(url: str) -> str:
    return f"web-{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}"


def _extract_published_at(html: str) -> tuple[str | None, float]:
    for index, pattern in enumerate(DATE_PATTERNS):
        match = pattern.search(html)
        if match:
            return match.group(1).strip(), max(0.35, 1.0 - (index * 0.15))
    return None, 0.0


def _published_at_from_text(value: object, fallback_text: str) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    published_at, _ = _extract_published_at(fallback_text)
    return published_at
