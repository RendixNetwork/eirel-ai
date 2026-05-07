"""Tests for UserMemoryWriter / UserMemoryReader."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from shared.common.database import Database
from shared.common.models import ConsumerUser, ConsumerUserMemory
from orchestration.orchestrator.embedding_client import StubEmbeddingClient
from orchestration.orchestrator.user_memory import (
    UserFactCandidate,
    UserMemoryReader,
    UserMemoryWriter,
    default_pre_filter,
)


class _StubLLM:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        if not self._replies:
            raise AssertionError("StubLLM exhausted")
        return {"choices": [{"message": {"content": self._replies.pop(0)}}]}


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'um.db'}")
    db.create_all()
    return db


def _seed_user(session, *, user_id: str | None = None) -> str:
    user_id = user_id or str(uuid4())
    session.add(ConsumerUser(
        user_id=user_id, auth_subject=f"api-key:{user_id}", display_name="X",
    ))
    session.commit()
    return user_id


# -- Pre-filter -------------------------------------------------------------


def test_pre_filter_matches_self_descriptive_phrases():
    assert default_pre_filter("I am a Python developer") is True
    assert default_pre_filter("I work at a startup") is True
    assert default_pre_filter("I prefer concise answers") is True
    assert default_pre_filter("I'm new to this") is True
    assert default_pre_filter("My name is Alice") is True
    assert default_pre_filter("My job involves data engineering") is True
    assert default_pre_filter("Call me Bob") is True


def test_pre_filter_skips_non_self_descriptive():
    assert default_pre_filter("What's the weather today?") is False
    assert default_pre_filter("Explain Python decorators.") is False
    assert default_pre_filter("Can you help with this code?") is False
    assert default_pre_filter("") is False


# -- Extraction parsing ----------------------------------------------------


def test_parse_candidates_handles_clean_json():
    db_dummy = None  # not used by the parse-only path
    writer = UserMemoryWriter.__new__(UserMemoryWriter)
    writer._instruction = ""
    writer._llm = None  # type: ignore[assignment]
    writer._pre_filter = default_pre_filter
    writer._max_tokens = 0
    writer._db = db_dummy
    writer._embed = None  # type: ignore[assignment]
    out = writer._parse_candidates(
        '[{"kind": "skill", "text": "Python", "confidence": 0.95},'
        ' {"kind": "preference", "text": "concise replies", "confidence": 0.8}]'
    )
    assert len(out) == 2
    assert out[0].kind == "skill"
    assert out[0].text == "Python"
    assert pytest.approx(out[0].confidence) == 0.95


def test_parse_candidates_strips_fences_and_prose():
    writer = UserMemoryWriter.__new__(UserMemoryWriter)
    writer._instruction = ""
    out = writer._parse_candidates(
        'Sure! ```json\n[{"kind": "fact", "text": "lives in Tokyo"}]\n``` done.'
    )
    assert len(out) == 1
    assert out[0].text == "lives in Tokyo"


def test_parse_candidates_empty_array_returns_empty():
    writer = UserMemoryWriter.__new__(UserMemoryWriter)
    out = writer._parse_candidates("[]")
    assert out == []


def test_parse_candidates_invalid_kind_falls_back_to_fact():
    writer = UserMemoryWriter.__new__(UserMemoryWriter)
    out = writer._parse_candidates('[{"kind": "wat", "text": "..."}]')
    assert len(out) == 1
    assert out[0].kind == "fact"


def test_parse_candidates_clamps_confidence():
    writer = UserMemoryWriter.__new__(UserMemoryWriter)
    out = writer._parse_candidates('[{"text": "x", "confidence": 5.0}]')
    assert out[0].confidence == 1.0
    out = writer._parse_candidates('[{"text": "x", "confidence": -1.0}]')
    assert out[0].confidence == 0.0


# -- Writer end-to-end -----------------------------------------------------


async def test_writer_pre_filter_miss_skips_extractor(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
    llm = _StubLLM([])  # raises if called
    writer = UserMemoryWriter(
        database=db,
        embedding_client=StubEmbeddingClient(),
        extractor_llm=llm,
    )
    n = await writer.extract_and_write(
        user_id=user_id, user_message="What is the weather?",
    )
    assert n == 0
    assert llm.calls == []


async def test_writer_extracts_and_persists_facts(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
    llm = _StubLLM([
        '[{"kind": "skill", "text": "Python", "confidence": 0.95},'
        ' {"kind": "preference", "text": "concise replies", "confidence": 0.8}]'
    ])
    writer = UserMemoryWriter(
        database=db,
        embedding_client=StubEmbeddingClient(),
        extractor_llm=llm,
    )
    n = await writer.extract_and_write(
        user_id=user_id,
        user_message="I'm a Python developer and I prefer concise replies.",
    )
    assert n == 2
    with db.sessionmaker() as session:
        rows = session.query(ConsumerUserMemory).filter_by(user_id=user_id).all()
        assert len(rows) == 2
        kinds = {r.kind for r in rows}
        assert kinds == {"skill", "preference"}


async def test_writer_dedupes_existing_facts(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
    llm = _StubLLM([
        '[{"kind": "skill", "text": "Python", "confidence": 0.95}]',
        # Second turn returns the same fact slightly differently capitalized.
        '[{"kind": "skill", "text": "python", "confidence": 0.9}]',
    ])
    writer = UserMemoryWriter(
        database=db,
        embedding_client=StubEmbeddingClient(),
        extractor_llm=llm,
    )
    n1 = await writer.extract_and_write(
        user_id=user_id, user_message="I work in Python.",
    )
    n2 = await writer.extract_and_write(
        user_id=user_id, user_message="I prefer Python.",
    )
    assert n1 == 1
    # Dedup by normalized text → second write skips.
    assert n2 == 0
    with db.sessionmaker() as session:
        rows = session.query(ConsumerUserMemory).filter_by(user_id=user_id).all()
        assert len(rows) == 1


async def test_writer_swallows_extractor_failure(tmp_path):
    class _BoomLLM:
        async def chat_completions(self, payload):
            raise RuntimeError("provider down")

    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
    writer = UserMemoryWriter(
        database=db,
        embedding_client=StubEmbeddingClient(),
        extractor_llm=_BoomLLM(),
    )
    n = await writer.extract_and_write(
        user_id=user_id, user_message="I prefer concise answers.",
    )
    assert n == 0


async def test_writer_handles_empty_extractor_response(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
    llm = _StubLLM(["[]"])
    writer = UserMemoryWriter(
        database=db,
        embedding_client=StubEmbeddingClient(),
        extractor_llm=llm,
    )
    n = await writer.extract_and_write(
        user_id=user_id, user_message="I am someone.",
    )
    # Pre-filter hit → 1 LLM call; extractor returned [] → 0 rows written.
    assert n == 0
    assert len(llm.calls) == 1


# -- Reader -----------------------------------------------------------------


async def test_reader_returns_top_k_by_cosine(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
    llm = _StubLLM([
        '[{"text": "I prefer concise replies"},'
        ' {"text": "I work in Python"},'
        ' {"text": "I live in Tokyo"}]'
    ])
    embed = StubEmbeddingClient()
    writer = UserMemoryWriter(
        database=db, embedding_client=embed, extractor_llm=llm,
    )
    await writer.extract_and_write(
        user_id=user_id,
        user_message="I prefer concise replies. I work in Python. I live in Tokyo.",
    )
    reader = UserMemoryReader(database=db, embedding_client=embed)
    hits = await reader.recall(user_id=user_id, query="Python developer", k=2)
    assert len(hits) == 2
    # All hits belong to this user; scores are sorted descending.
    assert hits[0].score >= hits[1].score


async def test_reader_isolates_users(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        u1 = _seed_user(session)
        u2 = _seed_user(session)
    llm = _StubLLM([
        '[{"text": "I love Rust"}]',
        '[{"text": "I love Python"}]',
    ])
    embed = StubEmbeddingClient()
    writer = UserMemoryWriter(
        database=db, embedding_client=embed, extractor_llm=llm,
    )
    await writer.extract_and_write(user_id=u1, user_message="I love Rust.")
    await writer.extract_and_write(user_id=u2, user_message="I love Python.")
    reader = UserMemoryReader(database=db, embedding_client=embed)
    hits_u1 = await reader.recall(user_id=u1, query="favorite language", k=5)
    hits_u2 = await reader.recall(user_id=u2, query="favorite language", k=5)
    assert {h.text for h in hits_u1} == {"I love Rust"}
    assert {h.text for h in hits_u2} == {"I love Python"}


async def test_reader_returns_empty_for_unknown_user(tmp_path):
    db = _make_db(tmp_path)
    embed = StubEmbeddingClient()
    reader = UserMemoryReader(database=db, embedding_client=embed)
    hits = await reader.recall(user_id="nope", query="anything", k=3)
    assert hits == []


async def test_reader_swallows_embed_failure(tmp_path):
    class _BoomEmbed:
        async def aembed(self, texts):
            raise RuntimeError("embed down")

    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
    reader = UserMemoryReader(database=db, embedding_client=_BoomEmbed())
    hits = await reader.recall(user_id=user_id, query="x", k=3)
    assert hits == []
