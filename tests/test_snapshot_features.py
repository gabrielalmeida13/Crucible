# tests/test_snapshot_features.py
"""Tests for new Track 2 columns in crucible/snapshot.py."""
from __future__ import annotations

import pandas as pd
import pytest


def _dt(year: int) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-12-31", tz="UTC")


def _series(*values: float, start_year: int = 2015) -> pd.Series:
    dates = pd.DatetimeIndex([_dt(start_year + i) for i in range(len(values))])
    return pd.Series(list(values), index=dates, dtype=float)


def _pivoted(revenues, gross_profits=None, ocf=None, capex=None) -> dict:
    p: dict = {"Revenues": _series(*revenues)}
    if gross_profits:
        p["GrossProfit"] = _series(*gross_profits)
    if ocf:
        p["NetCashProvidedByUsedInOperatingActivities"] = _series(*ocf)
    if capex:
        p["PaymentsToAcquirePropertyPlantAndEquipment"] = _series(*[abs(c) for c in capex])
    return p


# ---------------------------------------------------------------------------
# revenue_growth_yr1 and revenue_growth_yr2
# ---------------------------------------------------------------------------

def test_revenue_growth_yr1():
    from crucible.snapshot import compute_snapshot_row
    row = compute_snapshot_row("T1", _pivoted([1000, 1100, 1200, 1320]))
    assert row["revenue_growth_yr1"] == pytest.approx(1320 / 1200 - 1.0, abs=1e-9)

def test_revenue_growth_yr2():
    from crucible.snapshot import compute_snapshot_row
    row = compute_snapshot_row("T1", _pivoted([1000, 1100, 1200, 1320]))
    assert row["revenue_growth_yr2"] == pytest.approx(1200 / 1100 - 1.0, abs=1e-9)

def test_revenue_growth_none_when_single_point():
    from crucible.snapshot import compute_snapshot_row
    row = compute_snapshot_row("T1", _pivoted([1000]))
    assert row["revenue_growth_yr1"] is None
    assert row["revenue_growth_yr2"] is None

def test_revenue_growth_yr2_none_when_only_two_points():
    from crucible.snapshot import compute_snapshot_row
    row = compute_snapshot_row("T1", _pivoted([1000, 1100]))
    assert row["revenue_growth_yr1"] == pytest.approx(0.10, abs=1e-9)
    assert row["revenue_growth_yr2"] is None


# ---------------------------------------------------------------------------
# fcf_positive_last2yr
# ---------------------------------------------------------------------------

def test_fcf_positive_last2yr_both_positive():
    from crucible.snapshot import compute_snapshot_row
    # FCF = OCF - capex: last 2 = [300, 250] both positive
    row = compute_snapshot_row("T1", _pivoted(
        [1000, 1000, 1000, 1000],
        ocf=[500, 200, 400, 350],
        capex=[100, 300, 100, 100],
    ))
    assert row["fcf_positive_last2yr"] == 2

def test_fcf_positive_last2yr_one_positive():
    from crucible.snapshot import compute_snapshot_row
    # last 2 FCF: [300, -50] → 1 positive
    row = compute_snapshot_row("T1", _pivoted(
        [1000, 1000, 1000, 1000],
        ocf=[500, 200, 400, 50],
        capex=[100, 300, 100, 100],
    ))
    assert row["fcf_positive_last2yr"] == 1

def test_fcf_positive_last2yr_none_when_no_fcf_data():
    from crucible.snapshot import compute_snapshot_row
    row = compute_snapshot_row("T1", _pivoted([1000, 1100]))
    assert row["fcf_positive_last2yr"] is None


# ---------------------------------------------------------------------------
# gross_margin_yr1_change
# ---------------------------------------------------------------------------

def test_gross_margin_yr1_change_positive():
    from crucible.snapshot import compute_snapshot_row
    # margins: 0.40, 0.42, 0.45 → change = 0.03
    row = compute_snapshot_row("T1", _pivoted(
        [1000, 1000, 1000],
        gross_profits=[400, 420, 450],
    ))
    assert row["gross_margin_yr1_change"] == pytest.approx(0.03, abs=1e-9)

def test_gross_margin_yr1_change_none_single_year():
    from crucible.snapshot import compute_snapshot_row
    row = compute_snapshot_row("T1", _pivoted([1000], gross_profits=[400]))
    assert row["gross_margin_yr1_change"] is None


# ---------------------------------------------------------------------------
# fcf_trajectory
# ---------------------------------------------------------------------------

def test_fcf_trajectory_positive_slope():
    from crucible.snapshot import compute_snapshot_row
    # FCF = 100, 200, 300 → positive slope
    row = compute_snapshot_row("T1", _pivoted(
        [1000, 1000, 1000],
        ocf=[200, 300, 400],
        capex=[100, 100, 100],
    ))
    assert row["fcf_trajectory"] is not None
    assert row["fcf_trajectory"] > 0

def test_fcf_trajectory_none_when_no_fcf():
    from crucible.snapshot import compute_snapshot_row
    row = compute_snapshot_row("T1", _pivoted([1000]))
    assert row["fcf_trajectory"] is None


# ---------------------------------------------------------------------------
# p_s
# ---------------------------------------------------------------------------

def test_p_s_computed_from_price_and_shares():
    from crucible.snapshot import compute_snapshot_row
    # market cap = 100 * 50_000 = 5_000_000; revenue = 1_000_000 → p_s = 5.0
    row = compute_snapshot_row("T1", _pivoted([1_000_000]), price=100.0, shares=50_000)
    assert row["p_s"] == pytest.approx(5.0, abs=1e-9)

def test_p_s_none_without_price():
    from crucible.snapshot import compute_snapshot_row
    row = compute_snapshot_row("T1", _pivoted([1_000_000]))
    assert row["p_s"] is None


# ---------------------------------------------------------------------------
# attach_momentum — momentum_3m
# ---------------------------------------------------------------------------

def test_attach_momentum_adds_momentum_3m():
    from crucible.snapshot import attach_momentum
    tickers = ["AAPL", "MSFT"]
    dates = pd.date_range("2020-01-31", periods=16, freq="ME", tz="UTC")
    # Prices: 100, 101, ..., 115
    prices = pd.DataFrame(
        {t: [100.0 + i for i in range(16)] for t in tickers},
        index=dates,
    )
    snap_date = dates[14]  # pos=14, pos_1m=13, pos_3m_lag=10
    df = pd.DataFrame(
        {"sector": ["Tech", "Tech"]},
        index=pd.Index(tickers, name="ticker"),
    )
    fund_by_date = {snap_date: df}
    attach_momentum(fund_by_date, prices)

    assert "momentum_raw" in df.columns
    assert "momentum_3m" in df.columns
    # momentum_raw = prices[13] / prices[2] - 1 = 113/102 - 1
    assert df.loc["AAPL", "momentum_raw"] == pytest.approx(113 / 102 - 1.0, abs=1e-9)
    # momentum_3m = prices[13] / prices[10] - 1 = 113/110 - 1
    assert df.loc["AAPL", "momentum_3m"] == pytest.approx(113 / 110 - 1.0, abs=1e-9)

def test_attach_momentum_nan_when_insufficient_history():
    from crucible.snapshot import attach_momentum
    dates = pd.date_range("2020-01-31", periods=3, freq="ME", tz="UTC")
    prices = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0]}, index=dates)
    snap_date = dates[2]
    df = pd.DataFrame({"sector": ["Tech"]}, index=pd.Index(["AAPL"], name="ticker"))
    fund_by_date = {snap_date: df}
    attach_momentum(fund_by_date, prices)
    assert pd.isna(df.loc["AAPL", "momentum_raw"])  # pos_12m < 0
    assert pd.isna(df.loc["AAPL", "momentum_3m"])   # pos_3m_lag < 0
