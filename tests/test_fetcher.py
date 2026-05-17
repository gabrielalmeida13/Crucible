"""Unit tests for fetcher.py — no real HTTP calls, no real API key required."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

from crucible.fetcher import (
    _FMPCache,
    _raw_to_info_row,
    _raw_to_panel_rows,
    fetch_financials,
    fetch_info,
    fetch_sp500_tickers,
    fetch_universe,
    save_raw,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def cache(tmp_path: Path) -> _FMPCache:
    """File-backed cache with a generous limit so tests never hit it."""
    return _FMPCache(tmp_path / "test_cache.db", daily_limit=10_000)


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
                ("cashflow", "Free Cash Flow", 2.0e9),
            ]:
                rows.append({
                    "ticker": ticker,
                    "fiscal_year": pd.Timestamp(year, tz="UTC"),
                    "statement": stmt,
                    "metric": metric,
                    "value": float(value),
                })
    return pd.DataFrame(rows)


def _fmp_profile(ticker: str = "AAPL") -> list[dict]:
    return [{
        "symbol": ticker,
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "currency": "USD",
        "pe": 28.5,
        "mktCap": 3_000_000_000_000,
    }]


def _fmp_income(ticker: str = "AAPL") -> list[dict]:
    return [
        {
            "date": "2024-09-28",
            "symbol": ticker,
            "revenue": 391_035_000_000,
            "grossProfit": 180_683_000_000,
            "netIncome": 93_736_000_000,
            "ebitda": 134_661_000_000,
        },
        {
            "date": "2023-09-30",
            "symbol": ticker,
            "revenue": 383_285_000_000,
            "grossProfit": 169_148_000_000,
            "netIncome": 96_995_000_000,
            "ebitda": 130_000_000_000,
        },
    ]


def _fmp_balance(ticker: str = "AAPL") -> list[dict]:
    return [{
        "date": "2024-09-28",
        "symbol": ticker,
        "totalAssets": 364_980_000_000,
        "totalCurrentLiabilities": 176_392_000_000,
        "totalDebt": 101_304_000_000,
        "cashAndCashEquivalents": 29_943_000_000,
    }]


def _fmp_cashflow(ticker: str = "AAPL") -> list[dict]:
    return [{
        "date": "2024-09-28",
        "symbol": ticker,
        "operatingCashFlow": 118_254_000_000,
        "capitalExpenditure": -9_447_000_000,
        "freeCashFlow": 108_807_000_000,
    }]


def _make_session(ticker: str = "AAPL") -> MagicMock:
    """Mock requests.Session that returns minimal FMP-shaped responses."""
    session = MagicMock(spec=requests.Session)

    def mock_get(url, params=None, timeout=None):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        if "/sp500_constituent" in url:
            mock_resp.json.return_value = [
                {"symbol": "AAPL"},
                {"symbol": "BRK.B"},
                {"symbol": "BF.A"},
            ]
        elif "/profile/" in url:
            t = url.split("/profile/")[-1]
            mock_resp.json.return_value = _fmp_profile(t)
        elif "/income-statement/" in url:
            t = url.split("/income-statement/")[-1]
            mock_resp.json.return_value = _fmp_income(t)
        elif "/balance-sheet-statement/" in url:
            t = url.split("/balance-sheet-statement/")[-1]
            mock_resp.json.return_value = _fmp_balance(t)
        elif "/cash-flow-statement/" in url:
            t = url.split("/cash-flow-statement/")[-1]
            mock_resp.json.return_value = _fmp_cashflow(t)
        else:
            mock_resp.json.return_value = []
        return mock_resp

    session.get.side_effect = mock_get
    return session


# ---------------------------------------------------------------------------
# _FMPCache
# ---------------------------------------------------------------------------


def test_cache_miss_returns_none(cache: _FMPCache) -> None:
    """A key not yet in the cache must return None."""
    assert cache.get("missing_key", ttl=3600) is None


def test_cache_set_and_get_roundtrip(cache: _FMPCache) -> None:
    """Data written with set must be returned by get within TTL."""
    data = [{"symbol": "AAPL"}]
    cache.set("test_key", data)
    assert cache.get("test_key", ttl=3600) == data


def test_cache_expired_returns_none(cache: _FMPCache) -> None:
    """Data older than TTL must not be returned."""
    cache.set("old_key", [{"x": 1}])
    assert cache.get("old_key", ttl=0) is None


def test_cache_overwrite_updates_value(cache: _FMPCache) -> None:
    """set on an existing key must overwrite the previous value."""
    cache.set("k", [1])
    cache.set("k", [2])
    assert cache.get("k", ttl=3600) == [2]


def test_rate_limit_allows_up_to_limit(tmp_path: Path) -> None:
    """check_and_increment must return True up to the daily limit."""
    c = _FMPCache(tmp_path / "c.db", daily_limit=3)
    assert c.check_and_increment() is True
    assert c.check_and_increment() is True
    assert c.check_and_increment() is True


def test_rate_limit_blocks_at_limit(tmp_path: Path) -> None:
    """check_and_increment must return False once the limit is reached."""
    c = _FMPCache(tmp_path / "c.db", daily_limit=2)
    c.check_and_increment()
    c.check_and_increment()
    assert c.check_and_increment() is False


def test_requests_today_reflects_increments(tmp_path: Path) -> None:
    """requests_today must match the number of successful increments."""
    c = _FMPCache(tmp_path / "c.db", daily_limit=5)
    c.check_and_increment()
    c.check_and_increment()
    assert c.requests_today() == 2


def test_cache_hit_does_not_count_as_request(cache: _FMPCache) -> None:
    """Serving from cache must not increment the request counter."""
    cache.set("sp500_constituents", [{"symbol": "AAPL"}])
    before = cache.requests_today()
    cache.get("sp500_constituents", ttl=86400)
    assert cache.requests_today() == before


# ---------------------------------------------------------------------------
# fetch_sp500_tickers
# ---------------------------------------------------------------------------


def test_fetch_sp500_tickers_normalises_dots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dots in ticker symbols must be replaced with dashes (e.g. BRK.B → BRK-B)."""
    c = _FMPCache(tmp_path / "c.db")
    sess = _make_session()
    monkeypatch.setenv("CRUCIBLE_FMP_API_KEY", "test_key")
    result = fetch_sp500_tickers("test_key", _cache=c, _session=sess)
    assert "BRK-B" in result
    assert "BF-A" in result
    assert "BRK.B" not in result


def test_fetch_sp500_tickers_returns_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fetch_sp500_tickers must return a non-empty list of strings."""
    c = _FMPCache(tmp_path / "c.db")
    sess = _make_session()
    monkeypatch.setenv("CRUCIBLE_FMP_API_KEY", "test_key")
    result = fetch_sp500_tickers("test_key", _cache=c, _session=sess)
    assert isinstance(result, list)
    assert all(isinstance(t, str) and len(t) > 0 for t in result)


def test_fetch_sp500_tickers_uses_cache_on_second_call(tmp_path: Path) -> None:
    """Second call must be served from cache without making another HTTP request."""
    c = _FMPCache(tmp_path / "c.db")
    sess = _make_session()
    fetch_sp500_tickers("test_key", _cache=c, _session=sess)
    first_call_count = sess.get.call_count
    fetch_sp500_tickers("test_key", _cache=c, _session=sess)
    assert sess.get.call_count == first_call_count  # no additional HTTP calls


def test_fetch_sp500_tickers_raises_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fetch_sp500_tickers must raise ValueError when no API key is available."""
    monkeypatch.delenv("CRUCIBLE_FMP_API_KEY", raising=False)
    c = _FMPCache(tmp_path / "c.db")
    with pytest.raises(ValueError, match="CRUCIBLE_FMP_API_KEY"):
        fetch_sp500_tickers(api_key="", _cache=c)


# ---------------------------------------------------------------------------
# _raw_to_info_row
# ---------------------------------------------------------------------------


def test_raw_to_info_row_sector_and_currency() -> None:
    """Profile fields must map to sector, sub_industry, currency."""
    raw = {"profile": _fmp_profile(), "income": _fmp_income(),
           "balance": _fmp_balance(), "cashflow": _fmp_cashflow()}
    row = _raw_to_info_row("AAPL", raw)
    assert row["sector"] == "Technology"
    assert row["sub_industry"] == "Consumer Electronics"
    assert row["currency"] == "USD"


def test_raw_to_info_row_p_e_from_profile() -> None:
    """p_e must come directly from the FMP profile pe field."""
    raw = {"profile": _fmp_profile(), "income": _fmp_income(),
           "balance": _fmp_balance(), "cashflow": _fmp_cashflow()}
    row = _raw_to_info_row("AAPL", raw)
    assert abs(row["p_e"] - 28.5) < 1e-6


def test_raw_to_info_row_p_fcf_computed() -> None:
    """p_fcf = mktCap / freeCashFlow must be within a reasonable range."""
    raw = {"profile": _fmp_profile(), "income": _fmp_income(),
           "balance": _fmp_balance(), "cashflow": _fmp_cashflow()}
    row = _raw_to_info_row("AAPL", raw)
    expected = 3_000_000_000_000 / 108_807_000_000
    assert abs(row["p_fcf"] - expected) < 1e-3


def test_raw_to_info_row_ev_ebitda_computed() -> None:
    """ev_ebitda = (mktCap + debt - cash) / ebitda."""
    raw = {"profile": _fmp_profile(), "income": _fmp_income(),
           "balance": _fmp_balance(), "cashflow": _fmp_cashflow()}
    row = _raw_to_info_row("AAPL", raw)
    mkt_cap = 3_000_000_000_000
    debt = 101_304_000_000
    cash = 29_943_000_000
    ebitda = 134_661_000_000
    expected = (mkt_cap + debt - cash) / ebitda
    assert abs(row["ev_ebitda"] - expected) < 1e-3


def test_raw_to_info_row_p_fcf_none_when_negative_fcf() -> None:
    """p_fcf must be None when FCF is negative (avoid negative multiples)."""
    cf = [{"date": "2024-09-28", "freeCashFlow": -1_000_000_000,
           "operatingCashFlow": 0, "capitalExpenditure": 0}]
    raw = {"profile": _fmp_profile(), "income": _fmp_income(),
           "balance": _fmp_balance(), "cashflow": cf}
    row = _raw_to_info_row("AAPL", raw)
    assert row["p_fcf"] is None


def test_raw_to_info_row_empty_data_returns_nulls() -> None:
    """All metric fields must be None when no FMP data is available."""
    raw = {"profile": [], "income": [], "balance": [], "cashflow": []}
    row = _raw_to_info_row("EMPTY", raw)
    for field in ("sector", "currency", "p_e", "p_fcf", "ev_ebitda"):
        assert row[field] is None


# ---------------------------------------------------------------------------
# _raw_to_panel_rows
# ---------------------------------------------------------------------------


def test_raw_to_panel_rows_metric_names_match_cleaner_expectations() -> None:
    """Metric names in panel rows must match the names cleaner.py looks up."""
    raw = {"profile": [], "income": _fmp_income(), "balance": _fmp_balance(),
           "cashflow": _fmp_cashflow()}
    rows = _raw_to_panel_rows("AAPL", raw)
    metric_names = {r["metric"] for r in rows}
    assert "Total Revenue" in metric_names
    assert "Gross Profit" in metric_names
    assert "Net Income" in metric_names
    assert "EBITDA" in metric_names
    assert "Total Assets" in metric_names
    assert "Current Liabilities" in metric_names
    assert "Total Debt" in metric_names
    assert "Cash And Cash Equivalents" in metric_names
    assert "Operating Cash Flow" in metric_names
    assert "Capital Expenditure" in metric_names
    assert "Free Cash Flow" in metric_names


def test_raw_to_panel_rows_fiscal_year_is_utc_timestamp() -> None:
    """fiscal_year values must be timezone-aware UTC Timestamps."""
    raw = {"profile": [], "income": _fmp_income(), "balance": [],  "cashflow": []}
    rows = _raw_to_panel_rows("AAPL", raw)
    for row in rows:
        ts = row["fiscal_year"]
        assert isinstance(ts, pd.Timestamp)
        assert ts.tzinfo is not None


def test_raw_to_panel_rows_statement_field_correct() -> None:
    """Each row must have the correct statement label (income/balance/cashflow)."""
    raw = {"profile": [], "income": _fmp_income(), "balance": _fmp_balance(),
           "cashflow": _fmp_cashflow()}
    rows = _raw_to_panel_rows("AAPL", raw)
    stmt_by_metric = {r["metric"]: r["statement"] for r in rows}
    assert stmt_by_metric["Total Revenue"] == "income"
    assert stmt_by_metric["Total Assets"] == "balance"
    assert stmt_by_metric["Operating Cash Flow"] == "cashflow"


def test_raw_to_panel_rows_multiple_years() -> None:
    """Multiple annual filings must produce distinct fiscal_year entries."""
    raw = {"profile": [], "income": _fmp_income(), "balance": [], "cashflow": []}
    rows = _raw_to_panel_rows("AAPL", raw)
    revenue_rows = [r for r in rows if r["metric"] == "Total Revenue"]
    assert len(revenue_rows) == 2  # two years in _fmp_income fixture


def test_raw_to_panel_rows_empty_returns_empty_list() -> None:
    """Empty FMP data must produce no panel rows."""
    raw = {"profile": [], "income": [], "balance": [], "cashflow": []}
    assert _raw_to_panel_rows("EMPTY", raw) == []


# ---------------------------------------------------------------------------
# fetch_info
# ---------------------------------------------------------------------------


def test_fetch_info_returns_dataframe_indexed_by_ticker(
    tmp_path: Path,
) -> None:
    """fetch_info must return a DataFrame with ticker as index."""
    c = _FMPCache(tmp_path / "c.db")
    sess = _make_session()
    result = fetch_info(["AAPL"], "test_key", c, sess)
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == "ticker"
    assert "AAPL" in result.index


def test_fetch_info_has_expected_columns(tmp_path: Path) -> None:
    """fetch_info must produce sector, currency, p_e, p_fcf, ev_ebitda columns."""
    c = _FMPCache(tmp_path / "c.db")
    sess = _make_session()
    result = fetch_info(["AAPL"], "test_key", c, sess)
    for col in ("sector", "sub_industry", "currency", "p_e", "p_fcf", "ev_ebitda"):
        assert col in result.columns, f"Missing column: {col}"


def test_fetch_info_cache_prevents_duplicate_requests(tmp_path: Path) -> None:
    """Calling fetch_info twice for the same tickers must not double-hit the API."""
    c = _FMPCache(tmp_path / "c.db")
    sess = _make_session()
    fetch_info(["AAPL"], "test_key", c, sess)
    first_count = sess.get.call_count
    fetch_info(["AAPL"], "test_key", c, sess)
    assert sess.get.call_count == first_count


# ---------------------------------------------------------------------------
# fetch_financials
# ---------------------------------------------------------------------------


def test_fetch_financials_returns_long_panel(tmp_path: Path) -> None:
    """fetch_financials must return a DataFrame with the five panel columns."""
    c = _FMPCache(tmp_path / "c.db")
    sess = _make_session()
    result = fetch_financials(["AAPL"], "test_key", c, sess)
    assert isinstance(result, pd.DataFrame)
    for col in ("ticker", "fiscal_year", "statement", "metric", "value"):
        assert col in result.columns, f"Missing column: {col}"


def test_fetch_financials_fiscal_year_is_utc(tmp_path: Path) -> None:
    """fiscal_year in the panel must be UTC-aware Timestamps."""
    c = _FMPCache(tmp_path / "c.db")
    sess = _make_session()
    result = fetch_financials(["AAPL"], "test_key", c, sess)
    assert not result.empty
    assert result["fiscal_year"].dt.tz is not None


def test_fetch_financials_uses_cache_from_fetch_info(tmp_path: Path) -> None:
    """fetch_financials after fetch_info must make zero additional HTTP requests."""
    c = _FMPCache(tmp_path / "c.db")
    sess = _make_session()
    fetch_info(["AAPL"], "test_key", c, sess)
    count_after_info = sess.get.call_count
    fetch_financials(["AAPL"], "test_key", c, sess)
    assert sess.get.call_count == count_after_info


# ---------------------------------------------------------------------------
# save_raw (unchanged contract)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# fetch_universe
# ---------------------------------------------------------------------------


def test_fetch_universe_unsupported_raises(tmp_path: Path) -> None:
    """fetch_universe must raise NotImplementedError for non-SP500 universes."""
    with pytest.raises(NotImplementedError, match="EUROPE_LARGE"):
        fetch_universe("EUROPE_LARGE", tmp_path)


def test_fetch_universe_raises_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fetch_universe must raise ValueError when CRUCIBLE_FMP_API_KEY is not set."""
    monkeypatch.delenv("CRUCIBLE_FMP_API_KEY", raising=False)
    with pytest.raises(ValueError, match="CRUCIBLE_FMP_API_KEY"):
        fetch_universe("SP500", tmp_path)
