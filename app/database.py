"""Database engine and session setup for SQLAlchemy."""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import DATABASE_URL

# The engine is the core connection pool to PostgreSQL.
# pool_pre_ping checks a connection is alive before using it (avoids
# "stale connection" errors after the DB restarts).
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# A session factory: calling SessionLocal() gives us a new session.
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Base class that our ORM models inherit from. SQLAlchemy uses it to
# keep track of all tables we define.
Base = declarative_base()


def get_db():
    """Yield a database session and always close it afterwards.

    FastAPI will call this per-request so every request gets a fresh,
    self-contained session that is cleanly closed even if an error occurs.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()