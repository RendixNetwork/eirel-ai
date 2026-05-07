"""Orchestrator-side guard ABC.

Distinct from :class:`eirel.safety.Guard` (which scans graph state
inside a miner pod). This guard runs at the
:class:`ProductOrchestrator` boundary on plain strings and is shaped
for two product-layer concerns: rejecting prompt-injection attempts
before the miner sees them, and redacting PII from both inbound user
prompts and outbound assistant content.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["GuardVerdict", "OrchestratorGuard", "Redaction"]


@dataclass(frozen=True, slots=True)
class Redaction:
    """One PII match the guard wants the orchestrator to mask.

    ``replacement`` is the substitution token (e.g. ``"[REDACTED-EMAIL]"``).
    ``span`` is the original ``(start, end)`` byte offsets — useful for
    audit logs where the operator needs to know what was masked
    without storing the raw value.
    """

    kind: str
    replacement: str
    span: tuple[int, int]


@dataclass(frozen=True, slots=True)
class GuardVerdict:
    """Outcome of one guard call at the orchestrator boundary.

    ``allow=False`` short-circuits the turn — the orchestrator returns
    a canned refusal message and never calls the miner. ``redacted_text``
    (when non-None) replaces the original prompt or content; the miner
    sees the redacted version, not the raw one.
    """

    allow: bool
    reason: str | None = None
    redacted_text: str | None = None
    redactions: tuple[Redaction, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, **metadata: Any) -> "GuardVerdict":
        return cls(allow=True, metadata=dict(metadata) if metadata else {})

    @classmethod
    def deny(cls, reason: str, **metadata: Any) -> "GuardVerdict":
        return cls(
            allow=False, reason=reason,
            metadata=dict(metadata) if metadata else {},
        )


class OrchestratorGuard(ABC):
    """ABC for guards wired into :class:`SafetyPipeline`.

    Two-call contract:

      * :meth:`pre_input` runs before the orchestrator builds the
        envelope. Receives the raw user prompt + a context dict
        (``user_id``, ``conversation_id``, ``project_id`` when
        present). May return a redacted prompt; an ``allow=False``
        verdict aborts the turn.
      * :meth:`post_output` runs after the miner replies, before the
        assistant turn is persisted. Receives the assistant content +
        the same context shape. Same redaction semantics.

    Implementations MUST be pure-async and side-effect free except for
    their own scanner calls. The pipeline applies the redacted text
    sequentially through the chain so multiple guards can stack.
    """

    @abstractmethod
    async def pre_input(
        self,
        text: str,
        context: Mapping[str, Any],
    ) -> GuardVerdict: ...

    @abstractmethod
    async def post_output(
        self,
        text: str,
        context: Mapping[str, Any],
    ) -> GuardVerdict: ...

    async def aclose(self) -> None:
        return None
