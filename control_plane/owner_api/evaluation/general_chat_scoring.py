from __future__ import annotations

import logging
import re
from statistics import mean
from typing import Any, Protocol
from urllib.parse import urlparse

from shared.core.evaluation_models import (
    ConversationScore,
    ConversationTrace,
    ConversationTurn,
    MinerGeneralChatScore,
    ModeBudget,
)
from shared.core.honeytokens import detect_honeytoken_citation


_logger = logging.getLogger(__name__)


# -- Weights --------------------------------------------------------------

QUALITY_WEIGHT = 0.80
LATENCY_WEIGHT = 0.20

INSTANT_MODE_WEIGHT = 0.60
THINKING_MODE_WEIGHT = 0.40

COST_MODIFIER_FLOOR = 0.80
COST_MODIFIER_CEILING = 1.00

# Tier 1 body-overlap gate: minimum fraction of content words from the
# claim's surrounding sentence that must appear in the cited URL's stored
# body excerpt. Tuned to be forgiving of rephrasing — a miner that summarizes
# faithfully will share common content words; a miner fabricating a summary
# will not. Applied only when an excerpt is available (skipped gracefully
# for older trace entries that predate the field).
_BODY_OVERLAP_MIN_RATIO = 0.25
_BODY_OVERLAP_MIN_WORDS = 3

# English stopwords excluded from content-word overlap counting.
_OVERLAP_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "but", "for", "from",
    "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "of",
    "on", "or", "our", "she", "that", "the", "their", "them", "they", "this",
    "to", "was", "were", "will", "with", "you", "your", "we", "us", "my",
    "me", "so", "if", "then", "than", "not", "no", "yes", "do", "does",
    "did", "can", "could", "would", "should", "may", "might", "about",
    "into", "over", "under", "some", "any", "all", "each", "every", "one",
    "two", "three", "per", "via", "through", "according",
})


# -- Attribution extraction ----------------------------------------------

# URLs (http/https)
_URL_RE = re.compile(r"https?://[^\s\])}>,;\"']+", re.IGNORECASE)
# Bracketed numeric citations like [1] with a following URL: "[1] https://..."
_CITATION_RE = re.compile(r"\[\d+\][^\n]*?(https?://[^\s\])}>,;\"']+)", re.IGNORECASE)

# Broad search-intent keyword set (Tier 0 #2). If ANY of these appear in the
# response and the trace has zero search-style tool calls, the gate hard-fails.
# Kept as lowercase tokens for plain substring matching against the lowercased
# response text.
_SEARCH_KEYWORDS: tuple[str, ...] = (
    "searched",
    "search result",
    "search results",
    "looked up",
    "lookup",
    "looked it up",
    "queried",
    "a query",
    "ran a query",
    "fetched",
    "retrieved",
    "consulted",
    "per my research",
    "based on my research",
    "sources indicate",
    "according to sources",
    "a quick search",
    "i found online",
    "found online",
    "i investigated",
    "i checked",
)

# Search-style tools that satisfy a search-intent claim.
_SEARCH_TOOLS: frozenset[str] = frozenset({
    "web_search", "semantic_scholar", "x_api", "sec_edgar", "news_search",
})

# Uncited authority claims — prose name-drops that imply external sourcing
# without a URL. "According to <Capitalized Entity>", "per a <adj> report
# by <Entity>", "<Entity> recently announced". The trigger phrases are
# matched case-insensitively via (?i:...); the entity capture group stays
# case-sensitive so we only accept proper-noun entities.
_AUTHORITY_CLAIM_RE = re.compile(
    r"(?i:\baccording\s+to|"
    r"\bper\s+a(?:n)?\s+\w+\s+(?:report|statement|post|study|article)\s+by|"
    r"\bas\s+reported\s+by|\bas\s+stated\s+by|\bcited\s+by)"
    r"\s+"
    r"([A-Z][A-Za-z0-9][A-Za-z0-9\-&.]{1,40}(?:\s+[A-Z][A-Za-z0-9\-&.]{1,40}){0,3})",
)
# "<Capitalized Entity> recently announced/reported/stated/revealed"
_ANNOUNCEMENT_CLAIM_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9][A-Za-z0-9\-&.]{1,40}(?:\s+[A-Z][A-Za-z0-9\-&.]{1,40}){0,3})\s+"
    r"(?i:(?:recently\s+)?(?:announced|reported|stated|revealed|confirmed|disclosed|published))\b",
)

# Stopword entities we do NOT treat as authority claims — too generic and
# trigger false positives on casual language.
_ENTITY_STOPWORDS: frozenset[str] = frozenset({
    "The", "This", "That", "These", "Those", "It", "He", "She", "They", "We",
    "I", "You", "My", "Our", "Your", "Their", "His", "Her", "Its",
    "Some", "Many", "Most", "Few", "All", "Any", "Each", "Every",
})


def _normalize_url(raw: str) -> str | None:
    """Canonicalize a URL to ``scheme://host/path`` (lowercased).

    Drops query, fragment, port, userinfo, trailing slash, and any leading/
    trailing punctuation that slipped through the extraction regex. Returns
    ``None`` for strings that don't parse as http(s) URLs.
    """
    if not raw:
        return None
    raw = raw.strip().rstrip(".,;:)")
    try:
        parsed = urlparse(raw)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    host = parsed.hostname.lower().rstrip(".")
    path = (parsed.path or "").rstrip("/")
    return f"{parsed.scheme}://{host}{path}"


def _registered_domain(raw: str) -> str | None:
    """Return the lowercased host for a URL — used for domain-level checks.

    No public-suffix aware logic: ``news.bbc.co.uk`` stays as-is. Exact-host
    matching is strict on purpose — miners cannot cite bbc.co.uk if they
    fetched example.com.
    """
    if not raw:
        return None
    try:
        parsed = urlparse(raw.strip())
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return parsed.hostname.lower().rstrip(".")


def extract_attribution_claims(response_text: str) -> list[str]:
    """Return every URL, bracket citation, search-intent, and authority claim.

    Pure string extraction — used by ``verify_trace_integrity`` to build the
    set of claims that must be backed by owner-api trace entries.

    Returns a list of tagged claims:
      - ``url:<canonical_url>``     — cited URL (normalized)
      - ``search:present``          — response uses search-intent keywords
      - ``authority:<entity>``      — prose grounding ("according to <Entity>")
    Legacy bare-URL strings are still included for backward compatibility with
    existing tests that assert ``"https://..." in claims``.
    """
    if not response_text:
        return []
    claims: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        if value and value not in seen:
            seen.add(value)
            claims.append(value)

    raw_urls: list[str] = []
    for url in _URL_RE.findall(response_text):
        raw_urls.append(url)
    for url in _CITATION_RE.findall(response_text):
        raw_urls.append(url)

    for raw_url in raw_urls:
        canonical = _normalize_url(raw_url)
        if canonical is None:
            continue
        # Legacy bare form kept for back-compat with existing tests.
        _add(canonical)
        _add(f"url:{canonical}")

    lowered = response_text.lower()
    if any(keyword in lowered for keyword in _SEARCH_KEYWORDS):
        _add("search:present")

    for pattern in (_AUTHORITY_CLAIM_RE, _ANNOUNCEMENT_CLAIM_RE):
        for match in pattern.findall(response_text):
            entity = match.strip()
            head = entity.split()[0] if entity else ""
            if head in _ENTITY_STOPWORDS:
                continue
            _add(f"authority:{entity.lower()}")
    return claims


_WORD_RE = re.compile(r"[a-zA-Z0-9]{2,}")


def _content_words(text: str) -> set[str]:
    """Return lowercased content words (>=2 chars, minus stopwords)."""
    return {
        word.lower()
        for word in _WORD_RE.findall(text)
        if word.lower() not in _OVERLAP_STOPWORDS
    }


def _surrounding_sentence(response_text: str, needle: str) -> str:
    """Return the sentence in ``response_text`` that contains ``needle``.

    Splits on ``. ?!`` boundaries and returns the first sentence containing
    the needle substring. Falls back to a +/- 200-char window if sentence
    splitting fails.
    """
    if not needle or not response_text:
        return ""
    idx = response_text.lower().find(needle.lower())
    if idx < 0:
        return ""
    # Walk backwards to the prior sentence boundary.
    start = idx
    for char in range(idx - 1, -1, -1):
        if response_text[char] in ".!?\n":
            start = char + 1
            break
        start = char
    # Walk forward to the next sentence boundary.
    end = idx + len(needle)
    for char in range(end, len(response_text)):
        end = char
        if response_text[char] in ".!?\n":
            break
    else:
        end = len(response_text)
    return response_text[start:end].strip()


def _entry_for_url(trace: ConversationTrace, canonical_url: str):
    """Return the first ``TraceEntry`` whose args/metadata/digest mention the URL."""
    canonical_lower = canonical_url.lower()
    for entry in trace.entries:
        haystack_parts = [entry.result_digest.lower()]
        for value in entry.args.values():
            haystack_parts.append(str(value).lower())
        for value in (entry.metadata or {}).values():
            haystack_parts.append(str(value).lower())
        haystack = " ".join(haystack_parts)
        if canonical_lower in haystack:
            return entry
    return None


def _body_overlap_ok(
    *,
    response_text: str,
    canonical_url: str,
    trace: ConversationTrace,
) -> bool:
    """Check that the claim's sentence shares content words with the body.

    Skipped (returns True) when either the trace entry has no
    ``result_body_excerpt`` (older entries or tool services that don't
    populate it) or the response text is empty.
    """
    if not response_text:
        return True
    entry = _entry_for_url(trace, canonical_url)
    if entry is None or not entry.result_body_excerpt:
        return True
    # Use the raw URL substring (not canonical) to find the citation in
    # the response — miners may cite URL with a trailing slash.
    sentence = _surrounding_sentence(response_text, canonical_url)
    if not sentence:
        return True
    sentence_words = _content_words(sentence)
    if len(sentence_words) < _BODY_OVERLAP_MIN_WORDS:
        # Too few content words to judge — don't fail, let the judge decide.
        return True
    body_words = _content_words(entry.result_body_excerpt)
    if not body_words:
        return True
    matched = sentence_words & body_words
    if len(matched) < _BODY_OVERLAP_MIN_WORDS:
        return False
    ratio = len(matched) / len(sentence_words)
    return ratio >= _BODY_OVERLAP_MIN_RATIO


def verify_trace_integrity(
    claims: list[str],
    trace: ConversationTrace,
    *,
    response_text: str = "",
) -> bool:
    """Return True iff every claim is backed by a trace entry.

    Decision rules (Tier 0 hardened):
      - ``url:<canonical>`` / bare URL → canonical form must appear in the
        trace's normalized URL set AND the registered domain must appear in
        the trace's domain set. Trailing query/fragment variations are
        normalized away in ``_normalize_url``.
      - ``search:present`` → response uses search-intent language; trace
        must contain at least one tool call in ``_SEARCH_TOOLS``.
      - ``authority:<entity>`` → entity name must appear somewhere in the
        trace's flattened text (tool_name / result_digest / args / metadata).
        Covers "according to Bloomberg" only when a tool call actually
        touched Bloomberg.
      - legacy ``search:<phrase>`` → same rule as ``search:present``.
      - empty claims → pass by default.
    """
    if not claims:
        return True
    trace_text = _trace_searchable_text(trace)
    trace_tools = {entry.tool_name.lower() for entry in trace.entries}
    trace_urls = _trace_canonical_urls(trace)
    trace_domains = _trace_domains(trace)

    has_search_tool = bool(_SEARCH_TOOLS.intersection(trace_tools))

    for claim in claims:
        if claim.startswith("search:"):
            if not has_search_tool:
                return False
            continue
        if claim.startswith("authority:"):
            entity = claim[len("authority:"):].strip()
            if not entity or entity not in trace_text:
                return False
            continue
        if claim.startswith("url:"):
            canonical = claim[len("url:"):].strip()
        else:
            # Bare-URL legacy form — normalize on the fly.
            normalized = _normalize_url(claim)
            if normalized is None:
                # Not a URL and not a tagged claim — fall back to substring.
                if claim.lower() not in trace_text:
                    return False
                continue
            canonical = normalized

        if canonical not in trace_urls:
            return False
        domain = _registered_domain(canonical)
        if domain is not None and domain not in trace_domains:
            return False
        if not _body_overlap_ok(
            response_text=response_text,
            canonical_url=canonical,
            trace=trace,
        ):
            return False
    return True


def _trace_searchable_text(trace: ConversationTrace) -> str:
    parts: list[str] = []
    for entry in trace.entries:
        parts.append(entry.tool_name.lower())
        parts.append(entry.result_digest.lower())
        for value in entry.args.values():
            parts.append(str(value).lower())
        for value in (entry.metadata or {}).values():
            parts.append(str(value).lower())
    return " ".join(parts)


def _trace_canonical_urls(trace: ConversationTrace) -> set[str]:
    """Collect canonicalized URLs from every trace entry.

    Looks in ``args`` (common keys: ``url``, ``href``, ``link``) and
    ``metadata`` for http(s) values, normalizes each, and returns the set.
    Also scans the ``result_digest`` field for substring URLs in case the
    tool stored the fetched URL inline rather than hashed.
    """
    out: set[str] = set()

    def _collect(value: Any) -> None:
        if isinstance(value, str):
            for match in _URL_RE.findall(value):
                normalized = _normalize_url(match)
                if normalized is not None:
                    out.add(normalized)

    for entry in trace.entries:
        for value in entry.args.values():
            _collect(value)
        for value in (entry.metadata or {}).values():
            _collect(value)
        _collect(entry.result_digest)
    return out


def _trace_domains(trace: ConversationTrace) -> set[str]:
    """Collect registered domains from every trace entry URL."""
    out: set[str] = set()
    for canonical in _trace_canonical_urls(trace):
        domain = _registered_domain(canonical)
        if domain is not None:
            out.add(domain)
    return out


# -- Per-dimension scoring -----------------------------------------------


def compute_latency_score(latency_ms: int, budget_ms: int) -> float:
    if budget_ms <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (float(latency_ms) / float(budget_ms))))


# -- ModeBudget derivation -------------------------------------------------


def budget_for_mode(mode: str | None, *, web_search: bool = True) -> ModeBudget:
    """Return the canonical ``ModeBudget`` for a task mode.

    ``"instant"`` — tight latency (20s), short output (1024 tokens), no
    reasoning. Used for crisp chat responses.

    ``"thinking"`` — relaxed latency (60s), large output (4096 tokens),
    reasoning tokens permitted (16384). Used for deep-research tasks.

    Unknown or ``None`` mode falls back to ``instant`` defaults — the
    scoring pipeline always gets a valid budget even when the task
    definition omits the field.
    """
    normalized = (mode or "").strip().lower()
    if normalized == "thinking":
        return ModeBudget(
            mode="thinking",
            web_search=web_search,
            latency_seconds=60.0,
            output_tokens=4096,
            reasoning_tokens=16384,
            cost_usd=0.50,
        )
    return ModeBudget(
        mode="instant",
        web_search=web_search,
        latency_seconds=20.0,
        output_tokens=1024,
        reasoning_tokens=0,
        cost_usd=0.10,
    )


# -- Judge protocol (lightweight so callers can inject fakes) ------------


class QualityJudge(Protocol):
    async def score_quality(
        self,
        *,
        conversation_history: list[ConversationTurn],
        response: str,
        mode: str,
    ) -> float:
        ...


class _StaticJudge:
    """Deterministic fallback used when no judge client is configured."""

    def __init__(self, default: float = 0.5) -> None:
        self.default = default

    async def score_quality(
        self,
        *,
        conversation_history: list[ConversationTurn],
        response: str,
        mode: str,
    ) -> float:
        del conversation_history, response, mode
        return self.default


class _PreComputedQualityJudge:
    """Adapter that returns a quality score computed by a prior judge call.

    Used when the production judge sidecar has already scored the
    conversation and we want to layer the trace integrity gate, latency,
    and penalty on top without re-invoking the judge. Prevents double
    judge costs and ensures scoring stays deterministic for a given
    ``(judge_result, trace)`` pair.
    """

    def __init__(self, quality_score: float) -> None:
        self._quality = max(0.0, min(1.0, float(quality_score)))

    async def score_quality(
        self,
        *,
        conversation_history: list[ConversationTurn],
        response: str,
        mode: str,
    ) -> float:
        del conversation_history, response, mode
        return self._quality


# -- End-to-end conversation scorer --------------------------------------


async def score_general_chat_conversation(
    *,
    conversation_history: list[ConversationTurn],
    response: str,
    trace: ConversationTrace,
    budget: ModeBudget,
    judge_client: QualityJudge | None = None,
    metadata: dict[str, Any] | None = None,
    trace_gate_penalty_usd: float = 0.0,
    active_honeytokens: list[str] | None = None,
) -> ConversationScore:
    """Compute a per-conversation score.

    Formula: ``trace_gate × (0.80 × quality + 0.20 × latency)``

    Cost is now handled as a multiplicative modifier at the miner-aggregate
    level (see ``aggregate_miner_score``), not per-conversation.

    ``trace_gate_penalty_usd`` is stamped on the returned score whenever
    the trace integrity gate fails. The scoring manager is responsible for
    actually debiting the run budget via the provider proxy — this function
    is pure and does not itself perform IO.

    ``active_honeytokens`` is the set of canary URLs seeded into this
    run's tool responses. If any claim cites one, the conversation is
    flagged in metadata and the gate fails immediately — honeytoken
    citation is a fabrication proof, independent of trace integrity.
    """
    claims = extract_attribution_claims(response)
    honeytoken_hit = detect_honeytoken_citation(claims, active_honeytokens)
    if honeytoken_hit:
        trace_gate_pass = False
    else:
        trace_gate_pass = verify_trace_integrity(
            claims, trace, response_text=response
        )
    trace_gate = 1.0 if trace_gate_pass else 0.0

    judge = judge_client or _StaticJudge()
    try:
        quality_raw = await judge.score_quality(
            conversation_history=list(conversation_history),
            response=response,
            mode=budget.mode,
        )
    except Exception as exc:
        _logger.warning("judge call failed, falling back to 0.0: %s", exc)
        quality_raw = 0.0
    quality_score = max(0.0, min(1.0, float(quality_raw)))

    latency_score = compute_latency_score(
        trace.total_latency_ms(), budget.latency_ms
    )

    weighted = (
        QUALITY_WEIGHT * quality_score
        + LATENCY_WEIGHT * latency_score
    )
    total = round(trace_gate * weighted, 6)

    per_dimension = {
        "quality": round(quality_score, 6),
        "latency": round(latency_score, 6),
        "trace_gate": round(trace_gate, 6),
    }

    penalty = (
        float(trace_gate_penalty_usd)
        if (not trace_gate_pass and trace_gate_penalty_usd > 0.0)
        else 0.0
    )

    return ConversationScore(
        quality=quality_score,
        latency=latency_score,
        cost=0.0,
        trace_gate=trace_gate,
        total=total,
        per_dimension=per_dimension,
        mode=budget.mode,
        web_search=budget.web_search,
        trace_gate_penalty_usd=penalty,
        metadata={
            "claims_extracted": len(claims),
            "trace_entries": len(trace.entries),
            "trace_gate_passed": trace_gate_pass,
            "honeytoken_cited": honeytoken_hit,
            **(metadata or {}),
        },
    )


# -- Miner-level aggregation ---------------------------------------------


def aggregate_miner_score(
    *,
    miner_hotkey: str,
    conversation_scores: list[ConversationScore],
    run_budget_usd: float = 30.0,
    run_cost_usd_used: float = 0.0,
) -> MinerGeneralChatScore:
    """Blend per-mode means into a miner score with cost-efficiency modifier.

    ``blended_quality = 0.6 × instant_mean + 0.4 × thinking_mean``
    ``cost_efficiency = max(0, 1 - run_cost_usd_used / run_budget_usd)``
    ``miner_score = blended_quality × (0.80 + 0.20 × cost_efficiency)``
    """
    instant_scores = [cs for cs in conversation_scores if cs.mode == "instant"]
    thinking_scores = [cs for cs in conversation_scores if cs.mode == "thinking"]

    instant_mean = (
        round(mean(cs.total for cs in instant_scores), 6) if instant_scores else 0.0
    )
    thinking_mean = (
        round(mean(cs.total for cs in thinking_scores), 6)
        if thinking_scores
        else 0.0
    )
    blended_quality = round(
        INSTANT_MODE_WEIGHT * instant_mean + THINKING_MODE_WEIGHT * thinking_mean, 6
    )

    if run_budget_usd > 0:
        cost_efficiency = max(0.0, min(1.0, 1.0 - run_cost_usd_used / run_budget_usd))
    else:
        cost_efficiency = 0.0
    cost_modifier = COST_MODIFIER_FLOOR + (COST_MODIFIER_CEILING - COST_MODIFIER_FLOOR) * cost_efficiency
    blended = round(blended_quality * cost_modifier, 6)

    # Honeytoken short-circuit: if any conversation in the run cited an
    # active honeytoken URL, the miner is a confirmed bad actor.
    # Blended score collapses to zero regardless of quality or cost.
    honeytoken_cited = any(
        bool(cs.metadata.get("honeytoken_cited", False))
        for cs in conversation_scores
    )
    if honeytoken_cited:
        blended = 0.0

    return MinerGeneralChatScore(
        miner_hotkey=miner_hotkey,
        instant_mean=instant_mean,
        thinking_mean=thinking_mean,
        blended=blended,
        cost_efficiency=round(cost_efficiency, 6),
        run_cost_usd_used=run_cost_usd_used,
        run_budget_usd=run_budget_usd,
        honeytoken_cited=honeytoken_cited,
        conversation_scores=list(conversation_scores),
        metadata={
            "instant_count": len(instant_scores),
            "thinking_count": len(thinking_scores),
            "cost_modifier": round(cost_modifier, 6),
            "honeytoken_cited": honeytoken_cited,
        },
    )
