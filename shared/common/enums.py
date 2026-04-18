from enum import StrEnum


class SubmissionStatus(StrEnum):
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    SUPERSEDED = "superseded"
    FAILED = "failed"


class AssignmentStatus(StrEnum):
    ASSIGNED = "assigned"
    CLAIMED = "claimed"
    RUNNING = "running"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED = "completed"
    EXPIRED = "expired"


class EvaluationStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class CanonicalResultStatus(StrEnum):
    PENDING = "pending"
    FINALIZED = "finalized"
    FAILED = "failed"


class SubmissionTaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Capability(StrEnum):
    CHAT = "chat"
    TOOL_USE = "tool_use"
    IMAGE = "image"
    VIDEO = "video"
    CODE = "code"
