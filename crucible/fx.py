"""Currency conversion cost identification and score penalty.

XTB charges a 0.5% FX conversion cost on stocks denominated in a currency
different from the account currency.  This module translates that real
transaction cost into a configurable score penalty applied before ranking.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def apply_fx_penalty(
    df: pd.DataFrame,
    account_currency: str,
    penalty: float = -0.5,
) -> pd.DataFrame:
    """Add fx_penalty column: 0.0 for same currency, penalty for different.

    Tickers with unknown currency (NaN) are treated conservatively as requiring
    conversion and receive the full penalty.
    """
    df = df.copy()
    needs_conversion = df["currency"].fillna("") != account_currency
    df["fx_penalty"] = needs_conversion.map({True: penalty, False: 0.0})

    n_penalised = int(needs_conversion.sum())
    if n_penalised:
        logger.info(
            "FX penalty %.2f applied to %d/%d tickers (account currency: %s)",
            penalty,
            n_penalised,
            len(df),
            account_currency,
        )
    return df
