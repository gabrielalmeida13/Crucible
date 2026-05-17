"""Layer 1 fundamental filters — hard rules applied before scoring."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all Layer 1 filters; return DataFrame of companies that pass all rules."""
    raise NotImplementedError("apply_filters not yet implemented")
