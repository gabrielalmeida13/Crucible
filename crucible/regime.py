"""Market regime detection — GROWTH / DEFENSIVE / HIGH_VOL.

Rules (all free inputs via yfinance)
--------------------------------------
  HIGH_VOL   — VIX close > 25
  DEFENSIVE  — yield curve inverted (10y-2y spread < 0) AND SP500 12m momentum < 0
  GROWTH     — everything else

Inputs
------
  VIX            ^VIX  daily close, most recent
  10y yield      ^TNX  daily close (percent), most recent
  2y  yield      ^IRX  daily close (percent × 10, so divide by 10), most recent
  SP500          ^GSPC daily close; 12-month momentum = (latest / 12m-ago) - 1

Notes
-----
  ^IRX is quoted as the annualised 13-week T-bill rate × 10 in some feeds —
  we divide by 10 so it is comparable to ^TNX in percentage points.
  The spread is not used as a precise economic estimate; it is a directional
  signal only, so the normalisation mismatch is acceptable.
"""

from __future__ import annotations

import logging
from enum import Enum

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

VIX_HIGH_VOL_THRESHOLD = 25.0
_LOOKBACK_DAYS         = 370  # ~12 months + buffer


class Regime(Enum):
    GROWTH    = "GROWTH"
    DEFENSIVE = "DEFENSIVE"
    HIGH_VOL  = "HIGH_VOL"


def _fetch_regime_inputs() -> tuple[float | None, float | None, float | None]:
    """Return (vix, yield_spread_10y2y, sp500_12m_momentum).

    Returns None for any input that cannot be fetched.
    """
    tickers = ["^VIX", "^TNX", "^IRX", "^GSPC"]
    try:
        raw = yf.download(
            tickers,
            period="14mo",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        close = raw["Close"] if "Close" in raw else raw
        if isinstance(close, pd.Series):
            close = close.to_frame()
    except Exception:
        log.warning("yfinance download failed for regime inputs", exc_info=True)
        return None, None, None

    def _last(col: str) -> float | None:
        if col not in close.columns:
            return None
        s = close[col].dropna()
        return float(s.iloc[-1]) if not s.empty else None

    vix = _last("^VIX")

    tnx = _last("^TNX")
    irx_raw = _last("^IRX")
    if tnx is not None and irx_raw is not None:
        irx = irx_raw / 10.0
        spread = tnx - irx
    else:
        spread = None

    if "^GSPC" in close.columns:
        gspc = close["^GSPC"].dropna()
        if len(gspc) >= 2:
            latest = float(gspc.iloc[-1])
            # find the price closest to 12 months ago
            cutoff = gspc.index[-1] - pd.DateOffset(months=12)
            past   = gspc[gspc.index <= cutoff]
            if not past.empty:
                year_ago  = float(past.iloc[-1])
                sp500_mom = latest / year_ago - 1.0
            else:
                sp500_mom = None
        else:
            sp500_mom = None
    else:
        sp500_mom = None

    return vix, spread, sp500_mom


def detect_regime() -> Regime:
    """Detect the current market regime.

    Fetches live inputs via yfinance and applies the three-state rule set.
    Falls back to GROWTH when inputs are unavailable (avoids false DEFENSIVE calls).
    """
    vix, spread, sp500_mom = _fetch_regime_inputs()

    log.info(
        "Regime inputs — VIX: %s  10y-2y spread: %s  SP500 12m: %s",
        f"{vix:.1f}" if vix is not None else "N/A",
        f"{spread:.2f}pp" if spread is not None else "N/A",
        f"{sp500_mom:+.2%}" if sp500_mom is not None else "N/A",
    )

    if vix is not None and vix > VIX_HIGH_VOL_THRESHOLD:
        regime = Regime.HIGH_VOL
    elif spread is not None and sp500_mom is not None and spread < 0 and sp500_mom < 0:
        regime = Regime.DEFENSIVE
    else:
        regime = Regime.GROWTH

    log.info("Detected regime: %s", regime.value)
    return regime


def regime_allocation_hint(regime: Regime) -> str:
    """Return a one-line allocation hint for the monthly screener output."""
    hints = {
        Regime.GROWTH:    "Regime GROWTH — favour Track 2 (Growth Inflection).",
        Regime.DEFENSIVE: (
            "Regime DEFENSIVE — favour Tracks 1 (Quality) and 3 (Value); "
            "reduce Track 2 exposure."
        ),
        Regime.HIGH_VOL:  (
            "Regime HIGH VOLATILITY — consider reducing all position sizes; "
            "VIX > 25 indicates elevated risk."
        ),
    }
    return hints[regime]
