"""Regex PII redaction for inbound prompts and outbound content.

Matches four high-frequency leak vectors:

  * **email** — RFC-5322-ish; intentionally tighter than the spec to
    avoid grabbing arbitrary at-signs in code blocks.
  * **phone** — North American formats (``(123) 456-7890``,
    ``123-456-7890``, ``+1 555 123 4567``) plus generic 10–15 digit
    runs with country-code prefixes. Not bulletproof internationally;
    the goal is to mask plausible matches, not all possible numbers.
  * **SSN** — US format ``DDD-DD-DDDD``, no other variants.
  * **credit card** — 13–19 contiguous digits with optional spaces or
    dashes; validated with the Luhn checksum so plain digit strings
    that aren't payment cards (order numbers, IDs) don't get masked.

False positives are cheap (one masked token in a chat reply); false
negatives are expensive (PII leaks downstream). The guard tilts toward
catching too much, not too little.

The regexes deliberately do not chase obscure formats — production
deployments wanting Microsoft Presidio or AWS Comprehend integration
implement another :class:`OrchestratorGuard` subclass and stack it via
the :class:`SafetyPipeline`.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from shared.safety.guard import GuardVerdict, OrchestratorGuard, Redaction

__all__ = ["PIIRedactionGuard"]


# -- Patterns ----------------------------------------------------------------
#
# Order matters: more-specific patterns first so they win on overlap.

_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
)

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Credit card: 13–19 digits with optional space/dash separators between
# 3–4 digit groups. Validated with Luhn below, so non-card digit runs
# don't trip.
_CC_RE = re.compile(
    r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)",
)

# Phone numbers: a permissive shape that accepts NA-style formatting.
# Avoids matching small numbers (under 10 digits) by requiring at
# least one separator and a 3-digit prefix or area code.
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?"  # optional country code
    r"(?:\(\d{3}\)\s?|\d{3}[\s.-])"  # area code with separator
    r"\d{3}[\s.-]\d{4}\b",
)


def _luhn(digits: str) -> bool:
    """Return ``True`` when ``digits`` (no separators) passes Luhn."""
    if not digits.isdigit() or len(digits) < 13 or len(digits) > 19:
        return False
    total = 0
    odd = True  # we walk right-to-left
    for ch in reversed(digits):
        d = ord(ch) - 48
        if odd:
            total += d
        else:
            doubled = d * 2
            total += doubled if doubled < 10 else doubled - 9
        odd = not odd
    return total % 10 == 0


def _scan(text: str) -> list[tuple[str, int, int, str]]:
    """Return ``[(kind, start, end, replacement), ...]`` non-overlapping.

    Kinds in priority order: email, ssn, cc, phone. Once a span is
    masked, later patterns won't match inside it.
    """
    if not text:
        return []
    matches: list[tuple[str, int, int, str]] = []
    masked: list[bool] = [False] * len(text)

    def _claim(s: int, e: int) -> bool:
        if any(masked[s:e]):
            return False
        for i in range(s, e):
            masked[i] = True
        return True

    for m in _EMAIL_RE.finditer(text):
        if _claim(m.start(), m.end()):
            matches.append(("email", m.start(), m.end(), "[REDACTED-EMAIL]"))
    for m in _SSN_RE.finditer(text):
        if _claim(m.start(), m.end()):
            matches.append(("ssn", m.start(), m.end(), "[REDACTED-SSN]"))
    for m in _CC_RE.finditer(text):
        digits = re.sub(r"[^\d]", "", m.group(0))
        if not _luhn(digits):
            continue
        if _claim(m.start(), m.end()):
            matches.append(("credit_card", m.start(), m.end(), "[REDACTED-CC]"))
    for m in _PHONE_RE.finditer(text):
        if _claim(m.start(), m.end()):
            matches.append(("phone", m.start(), m.end(), "[REDACTED-PHONE]"))

    matches.sort(key=lambda x: x[1])
    return matches


def _apply(text: str, matches: list[tuple[str, int, int, str]]) -> str:
    """Splice the replacements into ``text`` left-to-right."""
    if not matches:
        return text
    out: list[str] = []
    cursor = 0
    for _kind, start, end, replacement in matches:
        out.append(text[cursor:start])
        out.append(replacement)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


class PIIRedactionGuard(OrchestratorGuard):
    """Mask SSN / CC / email / phone in both inbound and outbound text.

    Always allows the turn — the deny path is reserved for actively
    hostile inputs (prompt injection). PII redaction is non-blocking
    by design; the user might *legitimately* be asking the model to
    parse a contact card. Mask, don't reject.

    Set ``redact_input=False`` or ``redact_output=False`` to keep the
    guard one-sided; useful for staging environments where the operator
    wants to see raw inbound text in logs but still mask outbound.
    """

    __slots__ = ("_redact_input", "_redact_output")

    def __init__(
        self,
        *,
        redact_input: bool = True,
        redact_output: bool = True,
    ) -> None:
        self._redact_input = redact_input
        self._redact_output = redact_output

    async def pre_input(
        self, text: str, context: Mapping[str, Any]
    ) -> GuardVerdict:
        if not self._redact_input:
            return GuardVerdict.ok()
        return self._verdict(text)

    async def post_output(
        self, text: str, context: Mapping[str, Any]
    ) -> GuardVerdict:
        if not self._redact_output:
            return GuardVerdict.ok()
        return self._verdict(text)

    def _verdict(self, text: str) -> GuardVerdict:
        matches = _scan(text)
        if not matches:
            return GuardVerdict.ok()
        redacted = _apply(text, matches)
        redactions = tuple(
            Redaction(kind=kind, replacement=replacement, span=(start, end))
            for kind, start, end, replacement in matches
        )
        return GuardVerdict(
            allow=True,
            redacted_text=redacted,
            redactions=redactions,
            metadata={"pii_kinds": sorted({r.kind for r in redactions})},
        )
