"""Sequential guard chain wired at the :class:`ProductOrchestrator` boundary.

Two passes run per turn:

  * **pre_input** — before the orchestrator builds the envelope. The
    raw user prompt walks through every guard's :meth:`pre_input`. A
    deny short-circuits the turn (the orchestrator returns a canned
    refusal); an allow may rewrite the prompt via ``redacted_text`` so
    the next guard (and ultimately the miner) sees the cleaned
    version.
  * **post_output** — after the miner replies, before the assistant
    turn is persisted. Same chain logic, but operating on the
    assistant content. Deny here replaces the persisted content with
    the canned refusal — the user already saw the streamed response,
    so deny on output is rare; redaction is the common path.

The pipeline reports a "safety_verdict" block via the metadata it
returns: which guard fired, what was redacted, and any layer
information the guard surfaced. Consumers attach this to
``ConsumerMessage.metadata_json`` for audit.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from shared.safety.guard import GuardVerdict, OrchestratorGuard

_logger = logging.getLogger(__name__)

__all__ = [
    "SafetyOutcome",
    "SafetyPipeline",
    "SafetyVerdict",
]


# Fixed refusal copy. Stays terse on purpose — the metadata block is
# where audit detail lives.
_DEFAULT_INPUT_REFUSAL = (
    "I can't help with that — the request looks like an attempt to "
    "override my instructions or contains content I won't process."
)
_DEFAULT_OUTPUT_REFUSAL = (
    "I'm withholding that response — it didn't pass safety checks."
)


@dataclass(frozen=True, slots=True)
class SafetyVerdict:
    """One guard's contribution to the pipeline outcome."""

    guard: str
    allow: bool
    reason: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SafetyOutcome:
    """Pipeline outcome handed back to :class:`ProductOrchestrator`.

    ``allow`` is ``False`` when any guard denied. ``text`` is the
    final text the caller should use — either the (possibly
    redacted) input/output or the canned refusal on deny. ``verdicts``
    lists every guard's contribution in run order.
    """

    allow: bool
    text: str
    verdicts: tuple[SafetyVerdict, ...] = ()

    @property
    def metadata(self) -> dict[str, Any]:
        """Audit block to stamp into ``ConsumerMessage.metadata_json``."""
        return {
            "allow": self.allow,
            "verdicts": [
                {
                    "guard": v.guard,
                    "allow": v.allow,
                    "reason": v.reason,
                    "metadata": dict(v.metadata),
                }
                for v in self.verdicts
            ],
        }


class SafetyPipeline:
    """Sequential :class:`OrchestratorGuard` chain.

    Construct with the guard list in run order. Pass the pipeline
    to :class:`ProductOrchestrator` via the ``safety_pipeline``
    constructor kwarg. When ``guards`` is empty the pipeline is a
    pass-through (returns the original text unchanged) — this is the
    "safety off" mode for tests and dev environments.
    """

    def __init__(
        self,
        guards: Sequence[OrchestratorGuard],
        *,
        input_refusal: str = _DEFAULT_INPUT_REFUSAL,
        output_refusal: str = _DEFAULT_OUTPUT_REFUSAL,
    ) -> None:
        self._guards: tuple[OrchestratorGuard, ...] = tuple(guards)
        self._input_refusal = input_refusal
        self._output_refusal = output_refusal

    @property
    def empty(self) -> bool:
        return not self._guards

    async def pre_input(
        self,
        text: str,
        context: Mapping[str, Any],
    ) -> SafetyOutcome:
        return await self._run(
            text, context, stage="pre_input", refusal=self._input_refusal,
        )

    async def post_output(
        self,
        text: str,
        context: Mapping[str, Any],
    ) -> SafetyOutcome:
        return await self._run(
            text, context, stage="post_output", refusal=self._output_refusal,
        )

    async def _run(
        self,
        text: str,
        context: Mapping[str, Any],
        *,
        stage: str,
        refusal: str,
    ) -> SafetyOutcome:
        if not self._guards:
            return SafetyOutcome(allow=True, text=text)
        current = text
        verdicts: list[SafetyVerdict] = []
        for guard in self._guards:
            try:
                if stage == "pre_input":
                    verdict = await guard.pre_input(current, context)
                else:
                    verdict = await guard.post_output(current, context)
            except Exception as exc:  # noqa: BLE001 — best-effort
                _logger.warning(
                    "safety guard %s raised at %s: %s",
                    type(guard).__name__, stage, exc,
                )
                verdicts.append(SafetyVerdict(
                    guard=type(guard).__name__,
                    allow=True,
                    reason=None,
                    metadata={"error": str(exc)},
                ))
                continue
            verdicts.append(SafetyVerdict(
                guard=type(guard).__name__,
                allow=verdict.allow,
                reason=verdict.reason,
                metadata=dict(verdict.metadata or {}),
            ))
            if not verdict.allow:
                return SafetyOutcome(
                    allow=False, text=refusal,
                    verdicts=tuple(verdicts),
                )
            if verdict.redacted_text is not None:
                current = verdict.redacted_text
        return SafetyOutcome(allow=True, text=current, verdicts=tuple(verdicts))
