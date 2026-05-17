"""Layer 2 composite quality + valuation scorer with sector/region normalization."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute composite score; comparisons are within GICS sector + accounting region."""
    raise NotImplementedError("score not yet implemented")
