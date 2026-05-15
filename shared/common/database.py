from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from shared.common.migrations import run_migrations
from shared.common.models import Base


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


class Database:
    def __init__(self, url: str):
        normalized_url = url.replace("sqlite+aiosqlite://", "sqlite://")
        engine_kwargs: dict[str, Any] = {"future": True}
        if not normalized_url.startswith("sqlite"):
            engine_kwargs.update({
                "pool_size": _int_env("EIREL_DB_POOL_SIZE", 20),
                "max_overflow": _int_env("EIREL_DB_MAX_OVERFLOW", 30),
                "pool_pre_ping": True,
                "pool_recycle": 1800,
                "connect_args": {"connect_timeout": 10},
            })
        self.engine = create_engine(normalized_url, **engine_kwargs)
        self.sessionmaker = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        run_migrations(self.engine)
        Base.metadata.create_all(self.engine)

    def drop_all(self) -> None:
        Base.metadata.drop_all(self.engine)

    def session(self) -> Iterator[Session]:
        with self.sessionmaker() as session:
            yield session
