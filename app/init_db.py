"""Create all database tables. Run once with:  python -m app.init_db"""
from __future__ import annotations

from app.database import Base, engine
from app import models  # noqa: F401  (importing registers the History model)


def init_database() -> None:
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Done. Tables are ready.")


if __name__ == "__main__":
    init_database()