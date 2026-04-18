from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from shared.common.migrations import run_migrations
from shared.common.models import Base


class Database:
    def __init__(self, url: str):
        normalized_url = url.replace("sqlite+aiosqlite://", "sqlite://")
        engine_kwargs: dict[str, Any] = {"future": True}
        if not normalized_url.startswith("sqlite"):
            engine_kwargs.update({
                "pool_size": 5,
                "max_overflow": 10,
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
