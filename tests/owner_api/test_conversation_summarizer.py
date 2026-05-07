"""Tests for ConversationSummarizer."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from shared.common.database import Database
from shared.common.models import (
    ConsumerConversation,
    ConsumerMessage,
    ConsumerUser,
)
from orchestration.orchestrator.conversation_summarizer import (
    ConversationSummarizer,
    DEFAULT_KEEP_RECENT,
    DEFAULT_STALE_THRESHOLD,
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
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'summary.db'}")
    db.create_all()
    return db


def _seed_user_and_conversation(session) -> tuple[str, str]:
    user_id = str(uuid4())
    convo_id = str(uuid4())
    session.add(ConsumerUser(
        user_id=user_id, auth_subject=f"api-key:{user_id}", display_name="X",
    ))
    session.add(ConsumerConversation(
        conversation_id=convo_id, user_id=user_id, family_id="general_chat",
    ))
    session.commit()
    return user_id, convo_id


def _add_messages(session, conversation_id: str, n: int, *, start_idx: int = 0) -> list[str]:
    """Add ``n`` alternating user/assistant messages, return their ids."""
    ids: list[str] = []
    for i in range(n):
        role = "user" if (start_idx + i) % 2 == 0 else "assistant"
        msg = ConsumerMessage(
            conversation_id=conversation_id,
            turn_idx=start_idx + i,
            role=role,
            content=f"{role} message {start_idx + i}",
        )
        session.add(msg)
        session.flush()
        ids.append(msg.id)
    session.commit()
    return ids


# -- needs_summary ----------------------------------------------------------


def test_needs_summary_false_for_short_conversation(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _user, convo = _seed_user_and_conversation(session)
        _add_messages(session, convo, 4)
    summarizer = ConversationSummarizer(
        database=db,
        llm=_StubLLM([]),
        stale_threshold_messages=DEFAULT_STALE_THRESHOLD,
        keep_recent_messages=DEFAULT_KEEP_RECENT,
    )
    assert summarizer.needs_summary(conversation_id=convo) is False


def test_needs_summary_true_when_tail_exceeds_threshold(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _user, convo = _seed_user_and_conversation(session)
        _add_messages(session, convo, 30)  # well past threshold
    summarizer = ConversationSummarizer(
        database=db,
        llm=_StubLLM([]),
        stale_threshold_messages=10,
        keep_recent_messages=4,
    )
    assert summarizer.needs_summary(conversation_id=convo) is True


def test_needs_summary_false_when_tail_under_threshold_after_summary(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _user, convo_id = _seed_user_and_conversation(session)
        ids = _add_messages(session, convo_id, 30)
        convo = session.get(ConsumerConversation, convo_id)
        # Mark the 25th message as the summary boundary; tail is now 5.
        convo.rolling_summary = "summary so far"
        convo.last_summarized_message_id = ids[24]
        session.commit()
    summarizer = ConversationSummarizer(
        database=db,
        llm=_StubLLM([]),
        stale_threshold_messages=10,
        keep_recent_messages=4,
    )
    assert summarizer.needs_summary(conversation_id=convo_id) is False


# -- maybe_summarize --------------------------------------------------------


async def test_maybe_summarize_writes_first_summary(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _user, convo_id = _seed_user_and_conversation(session)
        ids = _add_messages(session, convo_id, 30)
    llm = _StubLLM(["the user discussed Python projects with the assistant"])
    summarizer = ConversationSummarizer(
        database=db,
        llm=llm,
        stale_threshold_messages=10,
        keep_recent_messages=8,
    )
    wrote = await summarizer.maybe_summarize(conversation_id=convo_id)
    assert wrote is True
    with db.sessionmaker() as session:
        convo = session.get(ConsumerConversation, convo_id)
        assert convo.rolling_summary == "the user discussed Python projects with the assistant"
        # Boundary should be the last of the head rows = 30 - 8 = 22nd
        # message (index 21 in our ids list, since keep_recent=8).
        assert convo.last_summarized_message_id == ids[21]
    # After writing, tail = 8 < threshold, so needs_summary should now be False.
    assert summarizer.needs_summary(conversation_id=convo_id) is False
    assert len(llm.calls) == 1


async def test_maybe_summarize_skips_when_under_threshold(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _user, convo = _seed_user_and_conversation(session)
        _add_messages(session, convo, 5)
    llm = _StubLLM([])  # would raise if called
    summarizer = ConversationSummarizer(
        database=db,
        llm=llm,
        stale_threshold_messages=10,
        keep_recent_messages=4,
    )
    wrote = await summarizer.maybe_summarize(conversation_id=convo)
    assert wrote is False
    assert llm.calls == []


async def test_maybe_summarize_extends_existing_summary(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _user, convo_id = _seed_user_and_conversation(session)
        _add_messages(session, convo_id, 30)
        convo = session.get(ConsumerConversation, convo_id)
        convo.rolling_summary = "Earlier: weather discussion."
        # Pretend nothing has been summarized yet so the tail re-triggers.
        session.commit()
    llm = _StubLLM(["Extended summary including weather."])
    summarizer = ConversationSummarizer(
        database=db,
        llm=llm,
        stale_threshold_messages=10,
        keep_recent_messages=4,
    )
    await summarizer.maybe_summarize(conversation_id=convo_id)
    # The user prompt should reference the previous summary.
    user_msg = llm.calls[0]["messages"][1]["content"]
    assert "Previous summary" in user_msg
    assert "weather discussion" in user_msg


async def test_maybe_summarize_swallows_llm_failure(tmp_path):
    class _BoomLLM:
        async def chat_completions(self, payload):
            raise RuntimeError("provider exploded")

    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _user, convo_id = _seed_user_and_conversation(session)
        _add_messages(session, convo_id, 30)
    summarizer = ConversationSummarizer(
        database=db,
        llm=_BoomLLM(),
        stale_threshold_messages=10,
        keep_recent_messages=4,
    )
    wrote = await summarizer.maybe_summarize(conversation_id=convo_id)
    assert wrote is False
    with db.sessionmaker() as session:
        convo = session.get(ConsumerConversation, convo_id)
        # Prior summary preserved (was None here, but stays None — no overwrite).
        assert convo.rolling_summary is None


async def test_maybe_summarize_swallows_empty_response(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _user, convo_id = _seed_user_and_conversation(session)
        _add_messages(session, convo_id, 30)
    llm = _StubLLM(["   "])  # whitespace-only
    summarizer = ConversationSummarizer(
        database=db,
        llm=llm,
        stale_threshold_messages=10,
        keep_recent_messages=4,
    )
    wrote = await summarizer.maybe_summarize(conversation_id=convo_id)
    assert wrote is False


def test_constructor_validates_keep_recent_smaller_than_threshold(tmp_path):
    db = _make_db(tmp_path)
    try:
        ConversationSummarizer(
            database=db,
            llm=_StubLLM([]),
            stale_threshold_messages=5,
            keep_recent_messages=10,
        )
    except ValueError as exc:
        assert "smaller than" in str(exc)
    else:
        raise AssertionError("expected ValueError")
