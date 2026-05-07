"""In-process per-task cache of ``ReconciledOracle`` values.

Populated at the validator's task-claim phase: for each task in the
batch, run oracle fanout + reconciler ONCE and cache the result. The
per-(task, miner) judge phase reads from this cache so the oracle
work amortizes across all N miners that judge each task.

Lifetime: one batch (between ``populate(...)`` and ``clear()`` /
process exit). The cache is intentionally a plain dict — no
expiration policy, no LRU. Each batch's cache is small (~100 tasks);
clearing between batches keeps memory bounded without coordination.

Not thread-safe. The validator engine runs the claim phase before the
judge phase (per-task fan-out across miners) within a single
asyncio loop, so synchronous dict operations are fine.
"""

from __future__ import annotations

from collections.abc import Iterator

from validation.validator.reconciler import ReconciledOracle


class TaskOracleCache:
    """Plain in-process map ``task_id → ReconciledOracle``."""

    def __init__(self) -> None:
        self._store: dict[str, ReconciledOracle] = {}

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def set(self, task_id: str, reconciled: ReconciledOracle) -> None:
        self._store[task_id] = reconciled

    def get(self, task_id: str) -> ReconciledOracle | None:
        return self._store.get(task_id)

    def require(self, task_id: str) -> ReconciledOracle:
        """Fetch the cached value or raise — useful in ``_judge_miner``
        where a missing cache entry indicates an enrichment-phase bug."""
        cached = self._store.get(task_id)
        if cached is None:
            raise KeyError(
                f"no oracle cache entry for task_id={task_id!r}; "
                f"did the claim-phase enrichment run?"
            )
        return cached

    def clear(self) -> None:
        """Drop all cached entries — call between claim batches."""
        self._store.clear()


__all__ = ["TaskOracleCache"]
