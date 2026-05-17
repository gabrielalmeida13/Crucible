"""Data extraction from yfinance (dev) and FMP (Phase 2+)."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_universe(universe_id: str) -> pd.DataFrame:
    """Return raw fundamentals DataFrame for all tickers in the given universe."""
    raise NotImplementedError(f"fetch_universe not yet implemented for {universe_id}")
