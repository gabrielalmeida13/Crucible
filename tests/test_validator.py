"""Unit tests for validator.py and cleaner derivation logic — synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from crucible.validator import PROCESSED_COLUMNS, ProcessedFundamentalsSchema


def _valid_df(n: int = 3) -> pd.DataFrame:
    """Build a minimal valid processed DataFrame with n tickers."""
    tickers = [f"TICK{i}" for i in range(n)]
    return pd.DataFrame(
        {
            "sector": ["Technology"] * n,
            "sub_industry": ["Software"] * n,
            "currency": ["USD"] * n,
            "p_e": [20.0] * n,
            "p_fcf": [15.0] * n,
            "ev_ebitda": [10.0] * n,
            "data_years": [4] * n,
            "insufficient_data": [False] * n,
            "roic_proxy_avg": [0.20] * n,
            "fcf_latest": [1e9] * n,
            "fcf_positive_years": [4.0] * n,
            "net_debt_ebitda": [1.5] * n,
            "revenue_growth_positive_years": [3.0] * n,
            "gross_margin_latest": [0.45] * n,
            "gross_margin_avg": [0.43] * n,
            "gross_margin_trend_slope": [0.005] * n,
        },
        index=pd.Index(tickers, name="ticker"),
    )


def test_valid_dataframe_passes() -> None:
    """A correctly structured DataFrame must pass the schema without error."""
    ProcessedFundamentalsSchema.validate(_valid_df())


def test_nullable_columns_accept_nan() -> None:
    """All nullable columns must accept NaN without raising."""
    df = _valid_df()
    df.loc["TICK0", "roic_proxy_avg"] = np.nan
    df.loc["TICK0", "net_debt_ebitda"] = np.nan
    df.loc["TICK0", "gross_margin_latest"] = np.nan
    df.loc["TICK0", "sector"] = None
    ProcessedFundamentalsSchema.validate(df)


def test_roic_above_max_fails() -> None:
    """ROIC above 100x (schema upper bound) must fail validation."""
    df = _valid_df()
    df.loc["TICK0", "roic_proxy_avg"] = 101.0
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        ProcessedFundamentalsSchema.validate(df)


def test_gross_margin_above_one_fails() -> None:
    """Gross margin above 1.0 (impossible) must fail validation."""
    df = _valid_df()
    df.loc["TICK0", "gross_margin_latest"] = 1.5
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        ProcessedFundamentalsSchema.validate(df)


def test_negative_data_years_fails() -> None:
    """Negative data_years is impossible and must fail validation."""
    df = _valid_df()
    df["data_years"] = df["data_years"].astype(object)
    df.loc["TICK0", "data_years"] = -1
    df["data_years"] = df["data_years"].astype(int)
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        ProcessedFundamentalsSchema.validate(df)


def test_missing_required_column_fails() -> None:
    """Dropping a required column must fail schema validation."""
    df = _valid_df().drop(columns=["insufficient_data"])
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        ProcessedFundamentalsSchema.validate(df)


def test_extra_column_fails_strict() -> None:
    """strict=True means extra columns not in the schema must fail validation."""
    df = _valid_df()
    df["unexpected_column"] = 0.0
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        ProcessedFundamentalsSchema.validate(df)


def test_processed_columns_matches_schema() -> None:
    """PROCESSED_COLUMNS must exactly cover all non-index schema fields."""
    schema_fields = set(ProcessedFundamentalsSchema.to_schema().columns.keys())
    assert set(PROCESSED_COLUMNS) == schema_fields


def test_wrong_index_name_fails() -> None:
    """Index named something other than 'ticker' must fail validation."""
    df = _valid_df().rename_axis("symbol")
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        ProcessedFundamentalsSchema.validate(df)


# ---------------------------------------------------------------------------
# Cleaner derivation logic (via synthetic raw data)
# ---------------------------------------------------------------------------


def test_cleaner_derive_roic_proxy(tmp_path: Path) -> None:
    """ROIC proxy = Net Income / (Total Assets - Current Liabilities)."""
    from pathlib import Path

    from crucible.cleaner import clean
    from crucible.fetcher import save_raw

    tickers = ["FAKE"]
    info = pd.DataFrame(
        {"sector": ["Tech"], "sub_industry": ["SW"], "currency": ["USD"],
         "p_e": [20.0], "p_fcf": [15.0], "ev_ebitda": [10.0]},
        index=pd.Index(tickers, name="ticker"),
    )
    # Net Income=2e9, Assets=20e9, CL=5e9 → invested_capital=15e9 → ROIC=2/15 ≈ 0.1333
    rows = []
    for year in ["2021-12-31", "2022-12-31", "2023-12-31"]:
        ts = pd.Timestamp(year, tz="UTC")
        for stmt, metric, value in [
            ("income", "Total Revenue", 10e9),
            ("income", "Gross Profit", 4e9),
            ("income", "Net Income", 2e9),
            ("income", "EBITDA", 3e9),
            ("balance", "Total Assets", 20e9),
            ("balance", "Current Liabilities", 5e9),
            ("balance", "Total Debt", 3e9),
            ("balance", "Cash And Cash Equivalents", 1e9),
            ("cashflow", "Operating Cash Flow", 2.5e9),
            ("cashflow", "Capital Expenditure", -5e8),
        ]:
            rows.append({"ticker": "FAKE", "fiscal_year": ts,
                         "statement": stmt, "metric": metric, "value": float(value)})

    panel = pd.DataFrame(rows)
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    run_ts = "20240101T000000Z"
    save_raw(tickers, info, panel, raw_dir, run_ts)

    result = clean(raw_dir, run_ts, processed_dir)

    assert "FAKE" in result.index
    expected_roic = 2e9 / (20e9 - 5e9)
    assert abs(result.loc["FAKE", "roic_proxy_avg"] - expected_roic) < 1e-6
    assert result.loc["FAKE", "data_years"] == 3
    assert result.loc["FAKE", "insufficient_data"] is np.bool_(False)


def test_cleaner_insufficient_data_flag(tmp_path: Path) -> None:
    """Tickers with fewer than 3 years of data must be flagged as insufficient."""
    from pathlib import Path

    from crucible.cleaner import clean
    from crucible.fetcher import save_raw

    tickers = ["FEW"]
    info = pd.DataFrame(
        {"sector": ["Tech"], "sub_industry": [None], "currency": ["USD"],
         "p_e": [None], "p_fcf": [None], "ev_ebitda": [None]},
        index=pd.Index(tickers, name="ticker"),
    )
    # Only 1 year of data
    rows = [
        {"ticker": "FEW", "fiscal_year": pd.Timestamp("2023-12-31", tz="UTC"),
         "statement": "income", "metric": "Total Revenue", "value": 1e9},
    ]
    panel = pd.DataFrame(rows)
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    run_ts = "20240201T000000Z"
    save_raw(tickers, info, panel, raw_dir, run_ts)

    result = clean(raw_dir, run_ts, processed_dir)
    assert result.loc["FEW", "insufficient_data"] is np.bool_(True)


def test_cleaner_empty_panel_produces_valid_output(tmp_path: Path) -> None:
    """An empty panel must still produce a valid schema-passing DataFrame."""
    from pathlib import Path

    from crucible.cleaner import clean
    from crucible.fetcher import save_raw

    tickers = ["EMPTY"]
    info = pd.DataFrame(
        {"sector": ["Tech"], "sub_industry": [None], "currency": ["USD"],
         "p_e": [None], "p_fcf": [None], "ev_ebitda": [None]},
        index=pd.Index(tickers, name="ticker"),
    )
    empty_panel = pd.DataFrame(
        columns=["ticker", "fiscal_year", "statement", "metric", "value"]
    )
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    run_ts = "20240301T000000Z"
    save_raw(tickers, info, empty_panel, raw_dir, run_ts)

    result = clean(raw_dir, run_ts, processed_dir)
    assert result.loc["EMPTY", "data_years"] == 0
    assert result.loc["EMPTY", "insufficient_data"] is np.bool_(True)
    ProcessedFundamentalsSchema.validate(result)
