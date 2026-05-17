"""Cleaning, normalization, and missing value detection."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw fundamentals DataFrame; output must pass Pandera schema."""
    raise NotImplementedError("clean not yet implemented")
