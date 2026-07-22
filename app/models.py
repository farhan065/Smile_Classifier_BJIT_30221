"""ORM models: Python classes that map to database tables."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String

from app.database import Base


class History(Base):
    """One row per classification result shown on the History page."""

    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=False)          # where the image is saved
    predicted_class = Column(String, nullable=False)      # "Smiling" / "Not Smiling"
    created_at = Column(                                  # the Date-Time column
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<History {self.id} {self.predicted_class} {self.created_at}>"