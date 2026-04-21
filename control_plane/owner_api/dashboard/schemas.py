from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


WindowLiteral = Literal["latest", "7d", "30d", "all"]
TrendLiteral = Literal["up", "down", "stable", "new"]


class PillarScores(BaseModel):
    capability: float | None = None
    robustness: float | None = None
    anti_gaming: float | None = None


class PillarWeights(BaseModel):
    capability: float
    robustness: float
    anti_gaming: float


class RobustnessBreakdown(BaseModel):
    cross_task_consistency: float | None = None
    latency_consistency: float | None = None


class OverviewResponse(BaseModel):
    total_miners: int
    active_validators: int
    total_families: int
    netuid: int
    network: str
    current_run_id: str | None
    current_run_sequence: int | None


class FamilySummary(BaseModel):
    id: str
    label: str
    weight: float
    active: bool


class FamiliesResponse(BaseModel):
    families: list[FamilySummary]


class LeaderboardEntry(BaseModel):
    rank: int
    hotkey: str
    hotkey_short: str
    raw_score: float
    normalized_score: float | None
    is_serving_winner: bool
    is_eligible: bool
    is_running: bool = False
    trend: TrendLiteral
    previous_rank: int | None
    score_delta: float
    pillar_summary: PillarScores
    epochs_participated: int
    win_count: int


class LeaderboardResponse(BaseModel):
    run_id: str | None
    run_sequence: int | None
    window: WindowLiteral
    family_id: str
    total: int
    entries: list[LeaderboardEntry] = Field(default_factory=list)


class MinerProfileResponse(BaseModel):
    hotkey: str
    hotkey_short: str
    uid: int | None
    family_id: str
    current_rank: int | None
    current_score: float | None
    current_weight: float | None
    lifetime_wins: int
    epochs_participated: int
    first_seen_at: str | None
    pillar_scores: PillarScores


class MinerRunSummary(BaseModel):
    run_id: str
    epoch_sequence: int
    status: str
    started_at: str
    closed_at: str | None
    rank: int | None
    raw_score: float | None
    normalized_score: float | None
    was_winner: bool


class MinerRunsResponse(BaseModel):
    miner_hotkey: str
    family_id: str
    runs: list[MinerRunSummary] = Field(default_factory=list)


class TaskEvaluation(BaseModel):
    task_id: str
    task_index: int
    track: str | None
    category: str | None
    difficulty: str | None
    validator_hotkey: str | None
    task_score: float | None
    task_status: str | None
    evaluated_at: str | None
    # Phase 1 populated (from bundle lookup + existing columns):
    prompt: dict[str, Any] | str | None
    miner_response: dict[str, Any] | None
    # Deferred to Phase 2 (always null in Phase 1):
    expected_output: dict[str, Any] | None = None
    citations: list[dict[str, Any]] | None = None
    execution_trace: dict[str, Any] | None = None
    scoring_breakdown: dict[str, Any] | None = None
    judge_output: dict[str, Any] | None = None


class RunDetailResponse(BaseModel):
    run_id: str
    epoch_sequence: int
    status: str
    official_score: float | None
    pillar_scores: PillarScores
    pillar_weights: PillarWeights
    robustness_breakdown: RobustnessBreakdown
    anti_gaming_flags: list[str] = Field(default_factory=list)
    protocol_gate_passed: bool | None
    tasks: list[TaskEvaluation] = Field(default_factory=list)
