from __future__ import annotations

from shared.common.config import get_settings
from shared.common.database import Database
from shared.common.migrations import run_migrations
from shared.common.models import Base


def main() -> None:
    settings = get_settings()
    db = Database(settings.database_url)
    applied = run_migrations(db.engine)
    Base.metadata.create_all(db.engine)
    if applied:
        print("applied migrations:")
        for version in applied:
            print(version)
        return
    print("no migrations pending")


if __name__ == "__main__":
    main()
