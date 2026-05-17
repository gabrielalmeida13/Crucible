"""SQLite read/write via SQLAlchemy. All side effects are isolated here."""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import Engine

logger = logging.getLogger(__name__)


def save_scan(engine: Engine, df: pd.DataFrame, universe_id: str) -> None:
    """Persist a monthly scan result to the database."""
    raise NotImplementedError("save_scan not yet implemented")
