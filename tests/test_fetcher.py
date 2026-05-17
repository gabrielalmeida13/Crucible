"""Unit tests for fetcher.py — synthetic data only, no real API calls."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from crucible.fetcher import fetch_sp500_tickers, fetch_universe, save_raw


@pytest.fixture()
def tickers() -> list[str]:
    return ["AAPL", "MSFT", "GOOGL"]


@pytest.fixture()
def info_df(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sector": ["Technology", "Technology", "Technology"],
            "sub_industry": ["Consumer Electronics", "Systems Software", "Internet Services"],
            "currency": ["USD", "USD", "USD"],
            "p_e": [25.0, 30.0, 22.0],
            "p_fcf": [20.0, 25.0, 18.0],
            "ev_ebitda": [15.0, 18.0, 12.0],
        },
        index=pd.Index(tickers, name="ticker"),
    )


@pytest.fixture()
def panel_df(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        for year in ["2021-12-31", "2022-12-31", "2023-12-31"]:
            for stmt, metric, value in [
                ("income", "Total Revenue", 1e10),
                ("income", "Gross Profit", 4e9),
                ("income", "Net Income", 2e9),
                ("income", "EBITDA", 3e9),
                ("balance", "Total Assets", 2e11),
                ("balance", "Current Liabilities", 5e10),
                ("balance", "Total Debt", 3e10),
                ("balance", "Cash And Cash Equivalents", 1e10),
                ("cashflow", "Operating Cash Flow", 2.5e9),
                ("cashflow", "Capital Expenditure", -5e8),
            ]:
                rows.append(
                    {
                        "ticker": ticker,
                        "fiscal_year": pd.Timestamp(year, tz="UTC"),
                        "statement": stmt,
                        "metric": metric,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def test_save_raw_creates_three_files(
    tmp_path: Path, tickers: list[str], info_df: pd.DataFrame, panel_df: pd.DataFrame
) -> None:
    """save_raw must create all three Parquet files."""
    paths = save_raw(tickers, info_df, panel_df, tmp_path, "20240101T000000Z")
    assert len(paths) == 3
    for p in paths:
        assert Path(p).exists(), f"Missing: {p}"


def test_save_raw_parquet_round_trip(
    tmp_path: Path, tickers: list[str], info_df: pd.DataFrame, panel_df: pd.DataFrame
) -> None:
    """Data written by save_raw must survive a Parquet round-trip."""
    run_ts = "20240101T000000Z"
    tickers_path, info_path, panel_path = save_raw(
        tickers, info_df, panel_df, tmp_path, run_ts
    )
    assert list(pd.read_parquet(tickers_path)["ticker"]) == tickers
    assert list(pd.read_parquet(info_path).index) == tickers
    assert len(pd.read_parquet(panel_path)) == len(panel_df)


def test_save_raw_empty_panel(
    tmp_path: Path, tickers: list[str], info_df: pd.DataFrame
) -> None:
    """save_raw must handle an empty panel without error."""
    empty = pd.DataFrame(
        columns=["ticker", "fiscal_year", "statement", "metric", "value"]
    )
    _, _, panel_path = save_raw(tickers, info_df, empty, tmp_path, "20240101T000000Z")
    assert pd.read_parquet(panel_path).empty


def test_save_raw_creates_subdir(
    tmp_path: Path, tickers: list[str], info_df: pd.DataFrame, panel_df: pd.DataFrame
) -> None:
    """save_raw must create raw_dir if it does not exist."""
    nested = tmp_path / "a" / "b" / "raw"
    save_raw(tickers, info_df, panel_df, nested, "20240101T000000Z")
    assert nested.exists()


def test_fetch_sp500_tickers_normalises_dots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dots in ticker symbols must be replaced with dashes (e.g. BRK.B → BRK-B)."""
    mock_table = pd.DataFrame({"Symbol": ["AAPL", "BRK.B", "BF.A"]})
    monkeypatch.setattr("pandas.read_html", lambda *a, **kw: [mock_table])
    result = fetch_sp500_tickers()
    assert "BRK-B" in result
    assert "BF-A" in result
    assert "BRK.B" not in result


def test_fetch_sp500_tickers_returns_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch_sp500_tickers must return a list of non-empty strings."""
    mock_table = pd.DataFrame({"Symbol": ["AAPL", "MSFT", "AMZN"]})
    monkeypatch.setattr("pandas.read_html", lambda *a, **kw: [mock_table])
    result = fetch_sp500_tickers()
    assert isinstance(result, list)
    assert all(isinstance(t, str) and len(t) > 0 for t in result)


def test_fetch_universe_unsupported_raises(tmp_path: Path) -> None:
    """fetch_universe must raise NotImplementedError for non-SP500 universes."""
    with pytest.raises(NotImplementedError, match="EUROPE_LARGE"):
        fetch_universe("EUROPE_LARGE", tmp_path)
