from __future__ import annotations

"""Honeytoken canary URLs for trace integrity enforcement.

A honeytoken is a fictional URL that the tool proxy injects into a small
fraction of tool responses. If a miner response cites a honeytoken, we
know the miner hallucinated — honeytokens do not exist in the real world,
so a citation proves the miner is fabricating plausible-looking URLs
instead of actually reading what the tool returned.

Key properties:
  * Per-run rotation — a fresh set is generated on run open, so miners
    cannot build a blocklist across runs.
  * Deterministic generation from ``(run_id, seed)`` — recovery-safe,
    easy to audit.
  * Plausible shape — the URLs look like real archive / news links so a
    non-reading miner that scrapes URL patterns from the response body
    cannot distinguish them from legitimate search results.
  * Injection rate is low (~2%) so honest miners rarely encounter them
    and honeytokens remain scarce enough to serve as a detection signal.
"""

import hashlib
import random
from typing import Any


# Canonical marker embedded in every honeytoken path. Allows fast rejection
# checks and helps operators grep logs / DB rows.
HONEYTOKEN_MARKER = "eirel-canary"

# Plausible-looking hostnames that will never resolve to real content.
# ``.example`` is an IANA reserved TLD — guaranteed not to resolve.
_HONEYTOKEN_HOSTS: tuple[str, ...] = (
    "archive-news.example",
    "wire-research.example",
    "press-releases.example",
    "docs-library.example",
    "open-registry.example",
)


def generate_honeytoken_set(
    run_id: str,
    *,
    count: int = 8,
    seed: int | None = None,
) -> list[str]:
    """Return a deterministic list of honeytoken URLs for a run.

    Same ``(run_id, seed, count)`` always produces the same list —
    stored alongside the run and queried by the scoring pipeline at
    evaluation time. ``run_id`` alone is sufficient in production;
    ``seed`` exists for tests.
    """
    if count <= 0:
        return []
    base = f"{run_id}:{seed if seed is not None else 0}".encode("utf-8")
    digest = hashlib.sha256(base).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    out: list[str] = []
    for idx in range(count):
        host = rng.choice(_HONEYTOKEN_HOSTS)
        suffix = hashlib.sha256(f"{digest}:{idx}".encode("utf-8")).hexdigest()[:10]
        out.append(f"https://{host}/{HONEYTOKEN_MARKER}/{idx:02d}-{suffix}")
    return out


def is_honeytoken(url: str, active_set: list[str] | set[str] | None) -> bool:
    """Return True if ``url`` is in the active honeytoken set.

    Membership test is case-insensitive and tolerates trailing
    punctuation / whitespace. Also accepts substring match against the
    ``HONEYTOKEN_MARKER`` as a safety net — any URL that looks like a
    honeytoken (path contains the marker) is treated as one even if the
    active set was lost (fail-closed against fabricated URLs that share
    the marker shape).
    """
    if not url:
        return False
    lowered = url.strip().rstrip(".,;:)").lower()
    if HONEYTOKEN_MARKER in lowered:
        return True
    if not active_set:
        return False
    lower_set = {item.strip().lower() for item in active_set if item}
    return lowered in lower_set


def detect_honeytoken_citation(
    claims: list[str],
    active_set: list[str] | set[str] | None,
) -> bool:
    """Return True iff any claim names an active honeytoken URL.

    Operates on the tagged-claim list produced by
    ``extract_attribution_claims``: strips the ``url:`` prefix before
    testing. Also catches bare-URL legacy claims. Ignores
    ``search:*`` and ``authority:*`` claims — those are not URL-based.
    """
    if not claims:
        return False
    for claim in claims:
        if claim.startswith("search:") or claim.startswith("authority:"):
            continue
        candidate = claim[len("url:"):] if claim.startswith("url:") else claim
        if is_honeytoken(candidate, active_set):
            return True
    return False


def inject_honeytokens_into_search_payload(
    payload: Any,
    *,
    active_set: list[str],
    conversation_id: str,
    call_index: int,
    rate: float = 0.02,
) -> Any:
    """Inject a honeytoken result into a search-style tool payload.

    Only runs when the deterministic dice roll lands — ``rate`` of calls
    receive an injection, keyed by ``(conversation_id, call_index)`` so
    two independent runs against the same honeytoken set can differ.

    Safe no-op when:
      * ``active_set`` is empty
      * the payload is not a dict or has no ``results``/``items``/``data`` list
      * the dice roll doesn't hit

    Returns the (possibly-mutated) payload. The original object is
    modified in place to avoid copying large responses.
    """
    if not active_set or not isinstance(payload, dict):
        return payload
    key = f"{conversation_id}:{call_index}".encode("utf-8")
    roll = int(hashlib.sha256(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    if roll > rate:
        return payload
    # Find the first list-of-results field in common search response shapes.
    list_key: str | None = None
    for candidate in ("results", "items", "data", "hits", "documents"):
        value = payload.get(candidate)
        if isinstance(value, list):
            list_key = candidate
            break
    if list_key is None:
        return payload
    # Deterministic pick of which honeytoken to inject.
    pick_idx = int(hashlib.sha256(key).hexdigest()[8:16], 16) % len(active_set)
    token_url = active_set[pick_idx]
    injected = {
        "title": f"Reference archive — entry {pick_idx:02d}",
        "url": token_url,
        "snippet": (
            "Historical record of relevant proceedings. "
            "See full document for details."
        ),
        "source": "archive",
    }
    payload[list_key].append(injected)
    return payload
