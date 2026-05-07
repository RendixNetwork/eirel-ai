"""Prompt-injection guard for the orchestrator boundary.

Two-layer detection:

  1. **Regex denylist** (always on). Catches the well-trodden patterns —
     "ignore previous instructions", "you are now DAN", system-impersonation
     headers, etc. False-positive rate matters; the denylist tilts toward
     specificity over recall, since a misfire blocks a real chat turn.

  2. **LLM classifier** (optional). When the regex layer abstains and a
     classifier is wired in, the guard escalates the prompt to the
     classifier and respects its verdict. Skipped entirely without a
     classifier — keeps the cheap path zero-cost.

A deny verdict short-circuits the turn at the orchestrator. The miner
pod never sees the prompt, no envelope is built, and the consumer gets
a canned refusal with the matched-rule name in metadata for audit.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any, Protocol

from shared.safety.guard import GuardVerdict, OrchestratorGuard

_logger = logging.getLogger(__name__)

__all__ = ["PromptInjectionClassifier", "PromptInjectionGuard"]


# -- Denylist regexes -------------------------------------------------------
#
# Each entry is (rule_name, compiled_pattern). Order is informational;
# the first match wins for the metadata.matched_rule field.

_DENYLIST: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_previous_instructions",
        re.compile(
            r"\b(?:please\s+)?(?:ignore|disregard|forget|bypass|override)"
            r"\s+(?:all|any|the)?\s*(?:prior|previous|preceding|above|earlier|"
            r"prior\s+system|system|safety)\s+"
            r"(?:instruction|instructions|prompt|prompts|rule|rules|"
            r"guideline|guidelines|directive|directives)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "system_impersonation",
        re.compile(
            r"^\s*(?:###\s*)?(?:\[?\s*)?system\s*[:>\]]"
            r"|"
            r"<\|im_start\|>\s*system",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "role_hijack_dan",
        re.compile(
            r"\b(?:you\s+are\s+now\s+)?(?:DAN|do\s+anything\s+now|"
            r"developer\s+mode|jailbreak\s+mode|unrestricted\s+mode)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "reveal_system_prompt",
        re.compile(
            r"\b(?:reveal|show|print|leak|repeat|recite|output)\s+"
            r"(?:your|the|all)?\s*"
            r"(?:system|hidden|original|initial|raw)\s+"
            r"(?:prompt|instructions|message|context)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "instruction_override",
        re.compile(
            r"\bnew\s+instructions?:\s*",
            re.IGNORECASE,
        ),
    ),
)


class PromptInjectionClassifier(Protocol):
    """Optional escalation path for prompts the regex layer didn't catch.

    Implementations call out to a model (or any heuristic). They MUST
    return a (deny, reason) pair. Failures should raise; the guard
    catches and logs them as ``allow`` so the chat turn isn't broken
    by a flaky classifier.
    """

    async def classify(self, text: str) -> tuple[bool, str | None]: ...


class PromptInjectionGuard(OrchestratorGuard):
    """Two-layer guard: regex denylist + optional classifier escalation.

    Parameters
    ----------
    classifier
        Optional :class:`PromptInjectionClassifier`. When ``None``, the
        guard runs the regex layer only.
    enable_classifier
        Master switch for the classifier (independent of whether one is
        passed). Lets operators ship the classifier off by default and
        flip it on per-deployment via env without changing wiring.
    apply_to_output
        When ``True``, the guard also runs at :meth:`post_output`.
        Defaults to ``False`` — assistant content tripping the denylist
        usually means the model is parroting the user's prompt back, not
        an injection attempt; the input-side block already handled it.
    """

    __slots__ = ("_classifier", "_enable_classifier", "_apply_to_output")

    def __init__(
        self,
        *,
        classifier: PromptInjectionClassifier | None = None,
        enable_classifier: bool = True,
        apply_to_output: bool = False,
    ) -> None:
        self._classifier = classifier
        self._enable_classifier = enable_classifier
        self._apply_to_output = apply_to_output

    async def pre_input(
        self, text: str, context: Mapping[str, Any]
    ) -> GuardVerdict:
        return await self._evaluate(text, stage="pre_input")

    async def post_output(
        self, text: str, context: Mapping[str, Any]
    ) -> GuardVerdict:
        if not self._apply_to_output:
            return GuardVerdict.ok()
        return await self._evaluate(text, stage="post_output")

    async def _evaluate(self, text: str, *, stage: str) -> GuardVerdict:
        if not text:
            return GuardVerdict.ok()
        rule = self._regex_match(text)
        if rule is not None:
            return GuardVerdict(
                allow=False,
                reason=f"prompt_injection: {rule}",
                metadata={
                    "layer": "regex",
                    "matched_rule": rule,
                    "stage": stage,
                },
            )
        if self._classifier is not None and self._enable_classifier:
            try:
                deny, reason = await self._classifier.classify(text)
            except Exception as exc:  # noqa: BLE001 — best-effort
                _logger.warning(
                    "prompt-injection classifier failed: %s", exc,
                )
                return GuardVerdict(
                    allow=True,
                    metadata={"layer": "classifier_error"},
                )
            if deny:
                return GuardVerdict(
                    allow=False,
                    reason=f"prompt_injection: {reason or 'classifier'}",
                    metadata={
                        "layer": "classifier",
                        "stage": stage,
                    },
                )
        return GuardVerdict(
            allow=True,
            metadata={"layer": "regex", "stage": stage},
        )

    def _regex_match(self, text: str) -> str | None:
        for name, pattern in _DENYLIST:
            if pattern.search(text):
                return name
        return None
