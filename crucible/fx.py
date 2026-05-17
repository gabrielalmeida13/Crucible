"""Currency conversion cost identification and score penalty."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def apply_fx_penalty(df: pd.DataFrame, account_currency: str, penalty: float = -0.5) -> pd.DataFrame:
    """Apply score penalty to tickers denominated in a currency other than account_currency."""
    raise NotImplementedError("apply_fx_penalty not yet implemented")
