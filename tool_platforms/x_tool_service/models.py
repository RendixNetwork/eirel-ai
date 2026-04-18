from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class XSearchRequest(BaseModel):
    query: str
    since: str | None = None
    until: str | None = None
    verified_only: bool = False
    max_results: int = Field(default=10, ge=1, le=100)


class XTweet(BaseModel):
    tweet_id: str
    author_id: str
    author_username: str
    text: str
    created_at: str
    verified: bool = False
    retweet_count: int = 0
    like_count: int = 0
    reply_count: int = 0
    url: str = ""
    content_sha256: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class XSearchResponse(BaseModel):
    query: str
    tweets: list[XTweet] = Field(default_factory=list)
    result_count: int = 0
    retrieved_at: str | None = None
    retrieval_ledger_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
