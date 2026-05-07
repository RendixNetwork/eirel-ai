"""HTML → readable-text extraction.

Bare-bones BeautifulSoup pipeline: drop ``<script>``/``<style>``/``<nav>``/
``<footer>``/``<header>`` boilerplate, prefer ``<main>``/``<article>`` if
present, otherwise the whole ``<body>``. Collapse whitespace, return
text + extracted ``<title>`` and outbound link list.

Not as polished as Trafilatura — but Trafilatura isn't a dependency
today, and the trade-off is acceptable for v1: most pages produce
clean-enough text for an LLM consumer. Producers that need fancier
extraction (sidebar removal, ad detection) can swap the extractor
backend without changing the wire shape.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urljoin

from bs4 import BeautifulSoup

__all__ = ["ExtractedDocument", "extract_text"]


_BOILERPLATE_TAGS = (
    "script", "style", "noscript", "nav", "footer", "header", "aside",
    "form", "svg", "iframe",
)
_PREFERRED_ROOTS = ("main", "article")
_LINK_LIMIT = 50
_TITLE_LIMIT = 200
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    title: str
    content: str
    links: list[dict[str, str]] = field(default_factory=list)


def _strip_boilerplate(soup: BeautifulSoup) -> None:
    for tag_name in _BOILERPLATE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()


def _pick_root(soup: BeautifulSoup):
    for name in _PREFERRED_ROOTS:
        node = soup.find(name)
        if node is not None:
            return node
    return soup.body or soup


def _collect_links(soup: BeautifulSoup, base_url: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        absolute = urljoin(base_url, href)
        text = anchor.get_text(" ", strip=True)
        out.append({"href": absolute, "text": text[:200]})
        if len(out) >= _LINK_LIMIT:
            break
    return out


def extract_text(
    html: str,
    *,
    base_url: str = "",
    max_chars: int = 50_000,
    include_links: bool = True,
) -> ExtractedDocument:
    """Pull readable text + links + title out of an HTML document."""
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = (title_tag.get_text(" ", strip=True) if title_tag else "")[:_TITLE_LIMIT]

    links = _collect_links(soup, base_url) if include_links else []

    _strip_boilerplate(soup)
    root = _pick_root(soup)
    raw_text = root.get_text("\n", strip=True) if root is not None else ""
    cleaned = _WS_RE.sub(" ", raw_text).strip()
    if max_chars and len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars].rstrip() + "..."

    return ExtractedDocument(title=title, content=cleaned, links=links)
