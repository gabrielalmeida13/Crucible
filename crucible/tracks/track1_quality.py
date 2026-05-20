"""Track 1 — Quality Compounders.

Thin wrapper over crucible.filters and crucible.scorer.  No logic lives here;
the value is a consistent interface so scripts can import any track identically.
"""
from __future__ import annotations

import pandas as pd

from crucible.config import CrucibleConfig, FilterThresholds
from crucible.filters import apply_filters as _apply_filters
from crucible.scorer import score as _score


def apply_filters(df: pd.DataFrame, thresholds: FilterThresholds) -> pd.DataFrame:
    """Apply Track 1 Layer 1 filters (delegates to crucible.filters)."""
    return _apply_filters(df, thresholds)


def score(df: pd.DataFrame, config: CrucibleConfig) -> pd.DataFrame:
    """Apply Track 1 Layer 2 scorer (delegates to crucible.scorer)."""
    return _score(df, config)


def run(df: pd.DataFrame, config: CrucibleConfig) -> pd.DataFrame:
    """Filter then score; return top candidates sorted by composite_score."""
    filtered = apply_filters(df, config.filters)
    return score(filtered, config)
