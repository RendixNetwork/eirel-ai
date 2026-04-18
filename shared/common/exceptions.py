"""Shared exception types for the EIREL control plane."""

from __future__ import annotations


class WorkflowEpisodeLeaseFencedError(RuntimeError):
    """Raised when a workflow episode lease is fenced by a newer holder."""
    pass


class WorkflowEpisodeCancelledError(RuntimeError):
    """Raised when a workflow episode has been cancelled."""
    pass


class WorkflowEpisodeAbortedError(RuntimeError):
    """Raised when a workflow episode execution is aborted."""
    pass
