"""Fact card schema and content-hash verification.

A fact card is a structured, verifiable claim extracted from a primary source
during the harvest-distill pipeline. Each card carries a ``content_sha256``
digest of the source material at fetch time so the scorer can later verify
that miner citations point at real bytes, not hallucinated content.

Fact cards serve as the answer key for generated evaluation tasks.
"""

from __future__ import annotations

import hashlib
from datetime import date as date_type
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Domain = Literal[
    "financial",
    "tech_platform",
    "regulatory_legal",
    "scientific",
    "current_events",
]


class FactCard(BaseModel):
    card_id: str
    domain: Domain
    claim: str = Field(max_length=2000)
    primary_source_url: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    date: str
    numeric_facts: dict[str, str] = Field(default_factory=dict)
    related_cards: list[str] = Field(default_factory=list)
    source_content: str = Field(max_length=50000)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        date_type.fromisoformat(v)
        return v

    @field_validator("content_sha256")
    @classmethod
    def validate_sha256_matches_content(cls, v: str) -> str:
        if len(v) != 64:
            raise ValueError("content_sha256 must be 64 hex characters")
        return v


class ContentHashMismatch(ValueError):
    pass


def compute_content_sha256(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def verify_content_sha256(card: FactCard, content: str | bytes) -> bool:
    actual = compute_content_sha256(content)
    if actual != card.content_sha256:
        raise ContentHashMismatch(
            f"card {card.card_id}: expected {card.content_sha256}, got {actual}"
        )
    return True


def make_card_id(domain: str, source_url: str, claim: str) -> str:
    payload = f"{domain}:{source_url}:{claim}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
