"""Tests for crucible/ml/features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crucible.ml.features import (
    FEATURE_COLS,
    add_roic_direction,
    build_feature_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_snapshot(
    tickers: list[str],
    roic: float = 0.20,
) -> pd.DataFrame:
    n = len(tickers)
    return pd.DataFrame(
        {
            "roic_proxy_avg": [roic] * n,
            "fcf_positive_years": [4] * n,
            "gross_margin_avg": [0.40] * n,
            "net_debt_ebitda": [1.0] * n,
            "revenue_growth_positive_years": [3] * n,
            "p_e": [15.0] * n,
            "p_fcf": [12.0] * n,
            "ev_ebitda": [10.0] * n,
            "momentum_raw": [0.05] * n,
            "interest_coverage": [8.0] * n,
            "cfo_to_ni": [1.1] * n,
            "capex_intensity": [0.05] * n,
            "operating_margin_trend": [0.01] * n,
            "revenue_acceleration": [0.02] * n,
            "share_buyback_signal": [0.02] * n,
            "insider_buy_ratio": [np.nan] * n,
        },
        index=pd.Index(tickers, name="ticker"),
    )


def _make_prices(
    tickers: list[str],
    dates: list[pd.Timestamp],
    base: float = 100.0,
    growth: float = 0.01,
) -> pd.DataFrame:
    """Deterministic price series: price[i] = base * (1 + growth)^i."""
    data = {
        t: [base * (1 + growth) ** i for i in range(len(dates))]
        for t in tickers
    }
    return pd.DataFrame(data, index=pd.DatetimeIndex(dates))


# ---------------------------------------------------------------------------
# Tests: add_roic_direction
# ---------------------------------------------------------------------------


def test_roic_direction_improves() -> None:
    d0 = pd.Timestamp("2010-01-31")
    d1 = pd.Timestamp("2011-01-31")
    fund = {
        d0: _make_snapshot(["AAPL"], roic=0.15),
        d1: _make_snapshot(["AAPL"], roic=0.20),
    }
    add_roic_direction(fund)
    assert fund[d1].at["AAPL", "roic_direction"] == 1.0


def test_roic_direction_declines() -> None:
    d0 = pd.Timestamp("2010-01-31")
    d1 = pd.Timestamp("2011-01-31")
    fund = {
        d0: _make_snapshot(["AAPL"], roic=0.25),
        d1: _make_snapshot(["AAPL"], roic=0.20),
    }
    add_roic_direction(fund)
    assert fund[d1].at["AAPL", "roic_direction"] == 0.0


def test_roic_direction_first_snapshot_is_nan() -> None:
    d0 = pd.Timestamp("2010-01-31")
    fund = {d0: _make_snapshot(["AAPL"])}
    add_roic_direction(fund)
    assert np.isnan(fund[d0].at["AAPL", "roic_direction"])


def test_roic_direction_nan_when_prior_missing_ticker() -> None:
    d0 = pd.Timestamp("2010-01-31")
    d1 = pd.Timestamp("2011-01-31")
    snap0 = _make_snapshot(["MSFT"])  # AAPL not in prior
    snap1 = _make_snapshot(["AAPL"])
    fund = {d0: snap0, d1: snap1}
    add_roic_direction(fund)
    assert np.isnan(fund[d1].at["AAPL", "roic_direction"])


# ---------------------------------------------------------------------------
# Tests: build_feature_matrix
# ---------------------------------------------------------------------------


def test_feature_cols_count() -> None:
    assert len(FEATURE_COLS) == 17


def test_build_feature_matrix_shape() -> None:
    tickers = ["AAPL", "MSFT", "SP500"]
    dates = pd.date_range("2010-01-31", periods=15, freq="ME")
    prices = _make_prices(tickers, list(dates), growth=0.01)

    d0 = dates[0]
    d1 = dates[1]
    fund_by_date = {d: _make_snapshot(["AAPL", "MSFT"]) for d in dates}

    X, y = build_feature_matrix(fund_by_date, prices, start_date=d0, end_date=d1)

    assert set(X.columns) == set(FEATURE_COLS)
    assert len(X) == len(y)
    assert X.index.names == ["snapshot_date", "ticker"]


def test_build_feature_matrix_binary_label() -> None:
    tickers = ["AAPL", "MSFT", "SP500"]
    dates = pd.date_range("2010-01-31", periods=15, freq="ME")
    prices = _make_prices(tickers, list(dates), growth=0.01)
    fund_by_date = {d: _make_snapshot(["AAPL", "MSFT"]) for d in dates}

    X, y = build_feature_matrix(fund_by_date, prices, start_date=dates[0], end_date=dates[2])
    assert set(y.unique()).issubset({0, 1})


def test_build_feature_matrix_empty_when_no_dates_in_window() -> None:
    tickers = ["AAPL", "SP500"]
    dates = pd.date_range("2010-01-31", periods=15, freq="ME")
    prices = _make_prices(tickers, list(dates))
    fund_by_date = {d: _make_snapshot(["AAPL"]) for d in dates}

    future = pd.Timestamp("2030-01-31")
    X, y = build_feature_matrix(fund_by_date, prices, start_date=future, end_date=future)
    assert len(X) == 0
    assert len(y) == 0


def test_build_feature_matrix_tz_aware_fund_dates() -> None:
    """fund_by_date with UTC keys and tz-naive start/end bounds must not raise."""
    tickers = ["AAPL", "MSFT", "SP500"]
    dates = pd.date_range("2010-01-31", periods=15, freq="ME", tz="UTC")
    prices = _make_prices(tickers, list(dates), growth=0.01)
    fund_by_date = {d: _make_snapshot(["AAPL", "MSFT"]) for d in dates}

    # tz-naive bounds — must compare cleanly with UTC keys
    start = pd.Timestamp("2010-01-31")
    end = pd.Timestamp("2010-03-31")
    X, y = build_feature_matrix(fund_by_date, prices, start_date=start, end_date=end)
    assert len(X) >= 0  # no exception raised


def test_build_feature_matrix_drops_rows_without_forward_price() -> None:
    """Tickers with no price data produce no rows."""
    tickers = ["SP500"]  # AAPL is in fund but not in prices
    dates = pd.date_range("2010-01-31", periods=15, freq="ME")
    prices = _make_prices(tickers, list(dates))
    fund_by_date = {d: _make_snapshot(["AAPL"]) for d in dates}

    X, y = build_feature_matrix(fund_by_date, prices, start_date=dates[0], end_date=dates[2])
    assert len(X) == 0
