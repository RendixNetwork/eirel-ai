"""Oracle client ABC + shared types.

Each oracle vendor (OpenAI, Gemini, Grok) implements ``OracleClient``
producing an ``OracleGrounding`` from a task's full context. The
fanout layer calls all three in parallel; the reconciler consumes the
list and synthesizes a consensus claim set.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


OracleStatus = Literal["ok", "error", "blocked"]


@dataclass(frozen=True)
class OracleContext:
    """Full input the miner saw, passed to oracle calls verbatim.

    No summarization. Each oracle vendor renders this into its own
    request shape (system + user prompt). ``conversation_recent``
    carries multi-turn fixtures verbatim; ``attached_document``
    carries any document the miner had access to.
    """

    task_id: str
    prompt: str
    conversation_recent: list[dict[str, str]] = field(default_factory=list)
    attached_document: str | None = None
    category: str | None = None


@dataclass(frozen=True)
class OracleGrounding:
    """One vendor's answer + telemetry.

    ``raw_text`` is the model's free-form response (the reconciler
    parses it). On failure (``status != "ok"``) raw_text is empty and
    ``error_msg`` carries diagnostic context for operator review;
    scoring continues with the remaining successful groundings.
    """

    vendor: str
    status: OracleStatus
    raw_text: str = ""
    latency_ms: int = 0
    cost_usd: float | None = None
    error_msg: str | None = None
    finish_reason: str | None = None
    # URLs the vendor cited as sources during the web-search call.
    # Empty when the vendor didn't search, didn't surface citations
    # (e.g. error/blocked), or web search was disabled. The reconciler
    # uses this to weight grounded answers higher than purely
    # parametric ones.
    citations: tuple[str, ...] = ()


class OracleClient(ABC):
    """Abstract base for the 3 oracle vendor clients."""

    @property
    @abstractmethod
    def vendor(self) -> str:
        """Stable vendor tag — ``"openai"`` / ``"gemini"`` / ``"grok"``.

        Used as the dict key in per-vendor agreement telemetry; do not
        change between releases.
        """

    @abstractmethod
    async def produce_grounding(
        self, context: OracleContext,
    ) -> OracleGrounding:
        """Produce one oracle grounding for the task.

        Implementations MUST NOT raise; convert provider exceptions
        into ``OracleGrounding(status="error", ...)`` so the fanout
        layer can keep going on partial vendor success.
        """

    @abstractmethod
    async def aclose(self) -> None:
        """Close the underlying provider client."""
