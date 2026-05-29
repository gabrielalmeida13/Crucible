"""Unit tests for fetcher.py — no real HTTP calls, no EDGAR bulk data required."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from crucible.fetcher import (
    _edgar_to_panel_rows,
    _load_cik_mapping,
    _parse_edgar_json,
    fetch_financials,
    fetch_universe,
    save_raw,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_cik_json(cik: int, concepts: dict[str, list[dict[str, Any]]]) -> dict:
    """Build a minimal EDGAR CIK JSON structure for testing."""
    return {
        "cik": cik,
        "entityName": "Test Corp",
        "facts": {
            "us-gaap": {
                name: {"units": {"USD": records}}
                for name, records in concepts.items()
            }
        },
    }


def _rec(
    end: str,
    val: float,
    filed: str,
    fy: int,
    form: str = "10-K",
    fp: str = "FY",
) -> dict:
    """Build one EDGAR fact record."""
    return {"end": end, "val": val, "filed": filed, "fy": fy, "form": form, "fp": fp}


def _write_cik_json(
    tmp_path: Path, cik: int, concepts: dict[str, list[dict[str, Any]]]
) -> Path:
    padded = str(cik).zfill(10)
    p = tmp_path / f"CIK{padded}.json"
    p.write_text(json.dumps(_make_cik_json(cik, concepts)))
    return p


def _write_cik_mapping(tmp_path: Path, entries: list[dict]) -> Path:
    """Write a SEC-format company_tickers.json to tmp_path."""
    mapping = {str(i): e for i, e in enumerate(entries)}
    p = tmp_path / "cik_mapping.json"
    p.write_text(json.dumps(mapping))
    return p


# ---------------------------------------------------------------------------
# Fixtures for save_raw (interface unchanged)
# ---------------------------------------------------------------------------


@pytest.fixture()
def tickers() -> list[str]:
    return ["AAPL", "MSFT", "GOOGL"]


@pytest.fixture()
def info_df(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sector": ["Technology"] * 3,
            "sub_industry": ["Consumer Electronics", "Systems Software", "Internet"],
            "currency": ["USD"] * 3,
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
            for metric, value in [
                ("Total Revenue", 1e10),
                ("Gross Profit", 4e9),
                ("Net Income", 2e9),
                ("EBITDA", 3e9),
                ("Total Assets", 2e11),
                ("Current Liabilities", 5e10),
                ("Total Debt", 3e10),
                ("Cash And Cash Equivalents", 1e10),
                ("Operating Cash Flow", 2.5e9),
                ("Capital Expenditure", 5e8),
                ("Free Cash Flow", 2.0e9),
            ]:
                rows.append({
                    "ticker": ticker,
                    "fiscal_year": pd.Timestamp(year, tz="UTC"),
                    "metric": metric,
                    "value": float(value),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _load_cik_mapping
# ---------------------------------------------------------------------------


def test_load_cik_mapping_returns_ticker_to_padded_cik(tmp_path: Path) -> None:
    """CIK integers must be zero-padded to 10 digits."""
    p = _write_cik_mapping(
        tmp_path, [{"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}]
    )
    result = _load_cik_mapping(p)
    assert result["AAPL"] == "0000320193"


def test_load_cik_mapping_uppercases_ticker(tmp_path: Path) -> None:
    """Tickers stored as lowercase in the SEC file must be normalised to uppercase."""
    p = _write_cik_mapping(
        tmp_path, [{"cik_str": 789019, "ticker": "msft", "title": "Microsoft"}]
    )
    result = _load_cik_mapping(p)
    assert "MSFT" in result
    assert result["MSFT"] == "0000789019"


def test_load_cik_mapping_multiple_tickers(tmp_path: Path) -> None:
    """All entries in the mapping file must be loaded."""
    p = _write_cik_mapping(
        tmp_path,
        [
            {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"},
            {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft"},
        ],
    )
    result = _load_cik_mapping(p)
    assert len(result) == 2
    assert "AAPL" in result and "MSFT" in result


def test_load_cik_mapping_missing_file_raises(tmp_path: Path) -> None:
    """FileNotFoundError must be raised when the mapping file does not exist."""
    with pytest.raises(FileNotFoundError, match="CIK mapping not found"):
        _load_cik_mapping(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# _parse_edgar_json — taxonomy fallback
# ---------------------------------------------------------------------------


def test_parse_edgar_json_finds_revenue_from_primary_tag(tmp_path: Path) -> None:
    """Total Revenue should be found under RevenueFromContractWithCustomerExcludingAssessedTax."""
    _write_cik_json(tmp_path, 320193, {
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            _rec("2022-09-24", 394328e6, "2022-10-28", 2022)
        ]
    })
    facts = _parse_edgar_json("0000320193", pd.Timestamp("2023-01-01", tz="UTC"), tmp_path)
    assert "Total Revenue" in facts
    assert facts["Total Revenue"][0]["val"] == 394328e6


def test_parse_edgar_json_falls_back_to_revenues_tag(tmp_path: Path) -> None:
    """If the primary tag is absent, the secondary tag (Revenues) must be used."""
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [_rec("2022-09-24", 394328e6, "2022-10-28", 2022)]
    })
    facts = _parse_edgar_json("0000320193", pd.Timestamp("2023-01-01", tz="UTC"), tmp_path)
    assert "Total Revenue" in facts


def test_parse_edgar_json_primary_tag_wins_over_secondary(tmp_path: Path) -> None:
    """When both primary and secondary tags exist, the primary value is used."""
    _write_cik_json(tmp_path, 320193, {
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            _rec("2022-09-24", 100e9, "2022-10-28", 2022)
        ],
        "Revenues": [
            _rec("2022-09-24", 999e9, "2022-10-28", 2022)
        ],
    })
    facts = _parse_edgar_json("0000320193", pd.Timestamp("2023-01-01", tz="UTC"), tmp_path)
    assert facts["Total Revenue"][0]["val"] == pytest.approx(100e9)


# ---------------------------------------------------------------------------
# _parse_edgar_json — point-in-time integrity
# ---------------------------------------------------------------------------


def test_parse_edgar_json_excludes_filings_after_as_of(tmp_path: Path) -> None:
    """Facts with filed > as_of_date must be excluded."""
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [
            _rec("2022-09-24", 394328e6, "2022-10-28", 2022),  # filed before
            _rec("2023-09-30", 383285e6, "2023-11-02", 2023),  # filed after
        ]
    })
    facts = _parse_edgar_json(
        "0000320193", pd.Timestamp("2023-01-01", tz="UTC"), tmp_path
    )
    assert len(facts.get("Total Revenue", [])) == 1
    assert facts["Total Revenue"][0]["fy"] == 2022


def test_parse_edgar_json_includes_filing_on_exact_as_of_date(tmp_path: Path) -> None:
    """A filing with filed == as_of_date must be included (boundary is inclusive)."""
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [_rec("2022-09-24", 394328e6, "2022-10-28", 2022)]
    })
    facts = _parse_edgar_json(
        "0000320193", pd.Timestamp("2022-10-28", tz="UTC"), tmp_path
    )
    assert "Total Revenue" in facts
    assert len(facts["Total Revenue"]) == 1


def test_parse_edgar_json_all_future_filings_returns_no_metric(tmp_path: Path) -> None:
    """If all filings are in the future, the metric must not appear in results."""
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [_rec("2023-09-30", 383285e6, "2023-11-02", 2023)]
    })
    facts = _parse_edgar_json(
        "0000320193", pd.Timestamp("2022-01-01", tz="UTC"), tmp_path
    )
    assert "Total Revenue" not in facts


# ---------------------------------------------------------------------------
# _parse_edgar_json — deduplication
# ---------------------------------------------------------------------------


def test_parse_edgar_json_dedup_keeps_latest_amendment(tmp_path: Path) -> None:
    """When 10-K and 10-K/A exist for the same fiscal year, keep the latest filed."""
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [
            {"end": "2022-09-24", "val": 394328e6, "filed": "2022-10-28",
             "fy": 2022, "form": "10-K", "fp": "FY"},
            {"end": "2022-09-24", "val": 394500e6, "filed": "2022-12-01",
             "fy": 2022, "form": "10-K/A", "fp": "FY"},
        ]
    })
    facts = _parse_edgar_json(
        "0000320193", pd.Timestamp("2023-01-01", tz="UTC"), tmp_path
    )
    assert len(facts["Total Revenue"]) == 1
    assert facts["Total Revenue"][0]["val"] == pytest.approx(394500e6)


def test_parse_edgar_json_dedup_ignores_amendment_after_as_of(tmp_path: Path) -> None:
    """If the 10-K/A was filed after as_of_date, only the original 10-K is used."""
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [
            {"end": "2022-09-24", "val": 394328e6, "filed": "2022-10-28",
             "fy": 2022, "form": "10-K", "fp": "FY"},
            {"end": "2022-09-24", "val": 394500e6, "filed": "2023-03-15",
             "fy": 2022, "form": "10-K/A", "fp": "FY"},  # after as_of
        ]
    })
    facts = _parse_edgar_json(
        "0000320193", pd.Timestamp("2023-01-01", tz="UTC"), tmp_path
    )
    assert len(facts["Total Revenue"]) == 1
    assert facts["Total Revenue"][0]["val"] == pytest.approx(394328e6)  # original


# ---------------------------------------------------------------------------
# _parse_edgar_json — other edge cases
# ---------------------------------------------------------------------------


def test_parse_edgar_json_excludes_quarterly_forms(tmp_path: Path) -> None:
    """10-Q records must be excluded even when filed before as_of_date."""
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [
            {"end": "2023-06-30", "val": 94760e6, "filed": "2023-08-01",
             "fy": 2023, "form": "10-Q", "fp": "Q3"},
            {"end": "2022-09-24", "val": 394328e6, "filed": "2022-10-28",
             "fy": 2022, "form": "10-K", "fp": "FY"},
        ]
    })
    facts = _parse_edgar_json(
        "0000320193", pd.Timestamp("2023-09-01", tz="UTC"), tmp_path
    )
    assert len(facts["Total Revenue"]) == 1
    assert facts["Total Revenue"][0]["fy"] == 2022  # annual 10-K, not the quarterly


def test_parse_edgar_json_missing_cik_file_returns_empty_dict(tmp_path: Path) -> None:
    """Missing CIK JSON must return an empty dict without raising."""
    facts = _parse_edgar_json(
        "9999999999", pd.Timestamp("2023-01-01", tz="UTC"), tmp_path
    )
    assert facts == {}


def test_parse_edgar_json_results_sorted_newest_first(tmp_path: Path) -> None:
    """Multiple fiscal years must be sorted newest end date first."""
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [
            _rec("2021-09-25", 365817e6, "2021-10-29", 2021),
            _rec("2022-09-24", 394328e6, "2022-10-28", 2022),
            _rec("2023-09-30", 383285e6, "2023-11-02", 2023),
        ]
    })
    facts = _parse_edgar_json(
        "0000320193", pd.Timestamp("2024-01-01", tz="UTC"), tmp_path
    )
    ends = [r["end"] for r in facts["Total Revenue"]]
    assert ends == sorted(ends, reverse=True)


# ---------------------------------------------------------------------------
# _edgar_to_panel_rows
# ---------------------------------------------------------------------------


def test_edgar_to_panel_rows_raw_metrics_present() -> None:
    """Every metric in the facts dict must appear as panel rows."""
    facts = {
        "Total Revenue": [
            {"end": "2022-09-24", "val": 394328e6, "filed": "2022-10-28", "fy": 2022}
        ],
        "Net Income": [
            {"end": "2022-09-24", "val": 99803e6, "filed": "2022-10-28", "fy": 2022}
        ],
    }
    rows = _edgar_to_panel_rows("AAPL", facts)
    metrics = {r["metric"] for r in rows}
    assert "Total Revenue" in metrics
    assert "Net Income" in metrics


def test_edgar_to_panel_rows_ebitda_computed_correctly() -> None:
    """EBITDA = Operating Income + Depreciation Amortization."""
    facts = {
        "Operating Income": [
            {"end": "2022-12-31", "val": 100e9, "filed": "2023-02-01", "fy": 2022}
        ],
        "Depreciation Amortization": [
            {"end": "2022-12-31", "val": 10e9, "filed": "2023-02-01", "fy": 2022}
        ],
    }
    rows = _edgar_to_panel_rows("TEST", facts)
    ebitda_rows = [r for r in rows if r["metric"] == "EBITDA"]
    assert len(ebitda_rows) == 1
    assert ebitda_rows[0]["value"] == pytest.approx(110e9)


def test_edgar_to_panel_rows_fcf_equals_ocf_minus_capex() -> None:
    """Free Cash Flow = Operating Cash Flow − |Capital Expenditure|."""
    facts = {
        "Operating Cash Flow": [
            {"end": "2022-12-31", "val": 90e9, "filed": "2023-02-01", "fy": 2022}
        ],
        "Capital Expenditure": [
            {"end": "2022-12-31", "val": 15e9, "filed": "2023-02-01", "fy": 2022}
        ],
    }
    rows = _edgar_to_panel_rows("TEST", facts)
    fcf_rows = [r for r in rows if r["metric"] == "Free Cash Flow"]
    assert len(fcf_rows) == 1
    assert fcf_rows[0]["value"] == pytest.approx(75e9)


def test_edgar_to_panel_rows_total_debt_sums_ltd_and_std() -> None:
    """Total Debt = Long Term Debt + Short Term Debt."""
    facts = {
        "Long Term Debt": [
            {"end": "2022-12-31", "val": 50e9, "filed": "2023-02-01", "fy": 2022}
        ],
        "Short Term Debt": [
            {"end": "2022-12-31", "val": 5e9, "filed": "2023-02-01", "fy": 2022}
        ],
    }
    rows = _edgar_to_panel_rows("TEST", facts)
    td_rows = [r for r in rows if r["metric"] == "Total Debt"]
    assert len(td_rows) == 1
    assert td_rows[0]["value"] == pytest.approx(55e9)


def test_edgar_to_panel_rows_total_debt_with_only_ltd() -> None:
    """Total Debt must still be computed when only Long Term Debt is present."""
    facts = {
        "Long Term Debt": [
            {"end": "2022-12-31", "val": 50e9, "filed": "2023-02-01", "fy": 2022}
        ],
    }
    rows = _edgar_to_panel_rows("TEST", facts)
    td_rows = [r for r in rows if r["metric"] == "Total Debt"]
    assert len(td_rows) == 1
    assert td_rows[0]["value"] == pytest.approx(50e9)


def test_edgar_to_panel_rows_no_ebitda_when_da_missing() -> None:
    """EBITDA must NOT appear if Depreciation Amortization is absent."""
    facts = {
        "Operating Income": [
            {"end": "2022-12-31", "val": 100e9, "filed": "2023-02-01", "fy": 2022}
        ],
    }
    rows = _edgar_to_panel_rows("TEST", facts)
    assert "EBITDA" not in {r["metric"] for r in rows}


def test_edgar_to_panel_rows_fiscal_year_is_utc_timestamp() -> None:
    """fiscal_year column must be UTC-aware Timestamps."""
    facts = {
        "Total Revenue": [
            {"end": "2022-09-24", "val": 394328e6, "filed": "2022-10-28", "fy": 2022}
        ],
    }
    rows = _edgar_to_panel_rows("AAPL", facts)
    for row in rows:
        ts = row["fiscal_year"]
        assert isinstance(ts, pd.Timestamp)
        assert ts.tzinfo is not None


def test_edgar_to_panel_rows_empty_facts_returns_empty_list() -> None:
    """Empty facts dict must produce an empty list."""
    assert _edgar_to_panel_rows("EMPTY", {}) == []


# ---------------------------------------------------------------------------
# fetch_financials
# ---------------------------------------------------------------------------


def test_fetch_financials_returns_dataframe_with_correct_columns(tmp_path: Path) -> None:
    """fetch_financials must return a DataFrame with the four panel columns."""
    mapping_path = _write_cik_mapping(
        tmp_path, [{"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}]
    )
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [_rec("2022-09-24", 394328e6, "2022-10-28", 2022)]
    })
    cik_map = _load_cik_mapping(mapping_path)
    result = fetch_financials(
        ["AAPL"], pd.Timestamp("2023-01-01", tz="UTC"), tmp_path, cik_map
    )
    assert isinstance(result, pd.DataFrame)
    for col in ("ticker", "fiscal_year", "metric", "value"):
        assert col in result.columns, f"Missing column: {col}"


def test_fetch_financials_point_in_time_excludes_future_filings(tmp_path: Path) -> None:
    """Calling with an early as_of_date must exclude later filings."""
    mapping_path = _write_cik_mapping(
        tmp_path, [{"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}]
    )
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [
            _rec("2022-09-24", 394328e6, "2022-10-28", 2022),
            _rec("2023-09-30", 383285e6, "2023-11-02", 2023),
        ]
    })
    cik_map = _load_cik_mapping(mapping_path)

    early = fetch_financials(
        ["AAPL"], pd.Timestamp("2023-01-01", tz="UTC"), tmp_path, cik_map
    )
    late = fetch_financials(
        ["AAPL"], pd.Timestamp("2024-01-01", tz="UTC"), tmp_path, cik_map
    )

    early_rev = early[early["metric"] == "Total Revenue"]
    late_rev  = late[late["metric"] == "Total Revenue"]
    assert len(early_rev) == 1
    assert len(late_rev)  == 2


def test_fetch_financials_skips_ticker_with_no_cik(tmp_path: Path) -> None:
    """Tickers absent from the CIK mapping must produce no rows."""
    result = fetch_financials(
        ["UNKNOWN"], pd.Timestamp("2023-01-01", tz="UTC"), tmp_path, {}
    )
    assert result.empty


def test_fetch_financials_handles_multiple_tickers(tmp_path: Path) -> None:
    """Results for all tickers must be combined into one DataFrame."""
    mapping_path = _write_cik_mapping(
        tmp_path,
        [
            {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"},
            {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft"},
        ],
    )
    for cik in (320193, 789019):
        _write_cik_json(tmp_path, cik, {
            "Revenues": [_rec("2022-12-31", 100e9, "2023-02-01", 2022)]
        })
    cik_map = _load_cik_mapping(mapping_path)
    result = fetch_financials(
        ["AAPL", "MSFT"], pd.Timestamp("2023-06-01", tz="UTC"), tmp_path, cik_map
    )
    assert set(result["ticker"].unique()) == {"AAPL", "MSFT"}


def test_fetch_financials_fiscal_year_is_utc(tmp_path: Path) -> None:
    """fiscal_year values in the panel must be UTC-aware Timestamps."""
    mapping_path = _write_cik_mapping(
        tmp_path, [{"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}]
    )
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [_rec("2022-09-24", 394328e6, "2022-10-28", 2022)]
    })
    cik_map = _load_cik_mapping(mapping_path)
    result = fetch_financials(
        ["AAPL"], pd.Timestamp("2023-01-01", tz="UTC"), tmp_path, cik_map
    )
    assert not result.empty
    assert result["fiscal_year"].dt.tz is not None


# ---------------------------------------------------------------------------
# save_raw — interface unchanged
# ---------------------------------------------------------------------------


def test_save_raw_creates_three_parquet_files(
    tmp_path: Path,
    tickers: list[str],
    info_df: pd.DataFrame,
    panel_df: pd.DataFrame,
) -> None:
    """save_raw must create all three Parquet files."""
    paths = save_raw(tickers, info_df, panel_df, tmp_path, "20240101T000000Z")
    assert len(paths) == 3
    for p in paths:
        assert Path(p).exists(), f"Missing: {p}"


def test_save_raw_parquet_round_trip(
    tmp_path: Path,
    tickers: list[str],
    info_df: pd.DataFrame,
    panel_df: pd.DataFrame,
) -> None:
    """Data written by save_raw must survive a Parquet round-trip."""
    run_ts = "20240101T000000Z"
    tickers_path, info_path, panel_path = save_raw(
        tickers, info_df, panel_df, tmp_path, run_ts
    )
    assert list(pd.read_parquet(tickers_path)["ticker"]) == tickers
    assert list(pd.read_parquet(info_path).index) == tickers
    assert len(pd.read_parquet(panel_path)) == len(panel_df)


def test_save_raw_empty_panel_writes_schema_only(
    tmp_path: Path, tickers: list[str], info_df: pd.DataFrame
) -> None:
    """save_raw must handle an empty panel without error."""
    empty = pd.DataFrame(columns=["ticker", "fiscal_year", "metric", "value"])
    _, _, panel_path = save_raw(tickers, info_df, empty, tmp_path, "20240101T000000Z")
    assert pd.read_parquet(panel_path).empty


def test_save_raw_creates_directory_if_missing(
    tmp_path: Path,
    tickers: list[str],
    info_df: pd.DataFrame,
    panel_df: pd.DataFrame,
) -> None:
    """save_raw must create raw_dir if it does not exist."""
    nested = tmp_path / "a" / "b" / "raw"
    save_raw(tickers, info_df, panel_df, nested, "20240101T000000Z")
    assert nested.exists()


# ---------------------------------------------------------------------------
# fetch_universe — error-path tests (no bulk EDGAR data needed)
# ---------------------------------------------------------------------------


def test_fetch_universe_unsupported_universe_raises() -> None:
    """fetch_universe must raise NotImplementedError for non-US universes."""
    with pytest.raises(NotImplementedError, match="EUROPE_LARGE"):
        fetch_universe("EUROPE_LARGE", Path("/tmp"))


def test_fetch_universe_russell3000_raises_file_not_found_not_not_implemented() -> None:
    """RUSSELL3000 is a supported universe — it must raise FileNotFoundError
    (CIK mapping absent), not NotImplementedError."""
    with pytest.raises(FileNotFoundError, match="CIK mapping not found"):
        fetch_universe(
            "RUSSELL3000",
            Path("/tmp"),
            tickers=["AAPL"],
            as_of_date=pd.Timestamp("2023-01-01", tz="UTC"),
        )


# ---------------------------------------------------------------------------
# _load_cik_annual_facts — LRU cache split
# ---------------------------------------------------------------------------


def test_load_cik_annual_facts_returns_empty_for_missing_file(tmp_path: Path) -> None:
    from crucible.fetcher import _load_cik_annual_facts, _load_cik_annual_facts as f
    result = _load_cik_annual_facts("0000000001", str(tmp_path))
    assert result == {}


def test_load_cik_annual_facts_extracts_correct_metrics(tmp_path: Path) -> None:
    from crucible.fetcher import _load_cik_annual_facts
    _write_cik_json(tmp_path, 1, {
        "Revenues": [_rec("2022-12-31", 1_000_000, "2023-01-20", 2022, "10-K", "FY")],
        "NetIncomeLoss": [_rec("2022-12-31", 100_000, "2023-01-20", 2022, "10-K", "FY")],
    })
    # Use a unique edgar_dir_str so lru_cache doesn't return a previous test's result
    result = _load_cik_annual_facts("0000000001", str(tmp_path / "unique_a"))
    # tmp_path/unique_a doesn't contain the file — but let's write to a subdir
    facts_dir = tmp_path / "cfacts_a"
    facts_dir.mkdir()
    (_make_cik_json(1, {
        "Revenues": [_rec("2022-12-31", 1_000_000, "2023-01-20", 2022, "10-K", "FY")],
    }))
    (facts_dir / "CIK0000000001.json").write_text(
        json.dumps(_make_cik_json(1, {
            "Revenues": [_rec("2022-12-31", 1_000_000, "2023-01-20", 2022, "10-K", "FY")],
        }))
    )
    result2 = _load_cik_annual_facts("0000000001", str(facts_dir))
    assert "Total Revenue" in result2
    assert len(result2["Total Revenue"]) == 1


def test_load_cik_annual_facts_excludes_quarterly_forms(tmp_path: Path) -> None:
    from crucible.fetcher import _load_cik_annual_facts
    facts_dir = tmp_path / "cfacts_b"
    facts_dir.mkdir()
    (facts_dir / "CIK0000000002.json").write_text(
        json.dumps(_make_cik_json(2, {
            "Revenues": [
                _rec("2022-12-31", 1_000_000, "2023-01-20", 2022, "10-K", "FY"),
                _rec("2022-09-30", 250_000, "2022-10-15", 2022, "10-Q", "Q3"),
            ],
        }))
    )
    result = _load_cik_annual_facts("0000000002", str(facts_dir))
    assert "Total Revenue" in result
    # Only the 10-K annual record should be present
    assert all(r["filed"] == "2023-01-20" for r in result["Total Revenue"])


def test_parse_edgar_json_still_applies_date_filter_on_cached_facts(tmp_path: Path) -> None:
    """_parse_edgar_json must apply as_of_date filter even when _load_cik_annual_facts
    is cached — verifies the two-stage separation is correct."""
    facts_dir = tmp_path / "cfacts_c"
    facts_dir.mkdir()
    (facts_dir / "CIK0000000003.json").write_text(
        json.dumps(_make_cik_json(3, {
            "Revenues": [
                _rec("2020-12-31", 800_000, "2021-02-01", 2020, "10-K", "FY"),
                _rec("2021-12-31", 900_000, "2022-02-01", 2021, "10-K", "FY"),
            ],
        }))
    )
    early = _parse_edgar_json("3", pd.Timestamp("2021-06-01", tz="UTC"), facts_dir)
    late  = _parse_edgar_json("3", pd.Timestamp("2022-06-01", tz="UTC"), facts_dir)
    # Early cutoff: only 2020 filing (filed 2021-02-01) is visible
    assert len(early.get("Total Revenue", [])) == 1
    assert early["Total Revenue"][0]["fy"] == 2020
    # Late cutoff: both filings visible
    assert len(late.get("Total Revenue", [])) == 2


# ---------------------------------------------------------------------------
# fetch_russell1000_tickers — iShares IWB CSV approach
# ---------------------------------------------------------------------------


def _iwb_csv(tickers: list[str], include_non_equity: bool = False) -> bytes:
    """Build a minimal iShares IWB-style CSV with the given equity tickers."""
    header_meta = "iShares Russell 1000 ETF,IWB\nAs of Date,2024-01-01\n\n"
    col_header = "Ticker,Name,Asset Class,Weight (%)\n"
    rows = "\n".join(f"{t},Company {t},Equity,0.1" for t in tickers)
    if include_non_equity:
        rows += "\nXXX,Cash Component,Cash,1.0\n-,Derivative,Other,0.0"
    return (header_meta + col_header + rows + "\n").encode()


def test_parse_iwb_csv_extracts_equity_tickers() -> None:
    """_parse_iwb_csv must return only equity tickers."""
    from crucible.fetcher import _parse_iwb_csv

    result = _parse_iwb_csv(_iwb_csv(["AAPL", "MSFT", "GOOGL"]))
    assert set(result) == {"AAPL", "MSFT", "GOOGL"}


def test_parse_iwb_csv_filters_out_non_equity() -> None:
    """Cash and derivative rows must not appear in the output."""
    from crucible.fetcher import _parse_iwb_csv

    result = _parse_iwb_csv(_iwb_csv(["AAPL"], include_non_equity=True))
    assert "AAPL" in result
    assert "XXX" not in result  # cash row
    assert "-" not in result    # derivative placeholder


def test_parse_iwb_csv_strips_dash_tickers() -> None:
    """Tickers containing '-' (cash/derivatives) must be excluded."""
    from crucible.fetcher import _parse_iwb_csv

    csv_bytes = (
        "Ticker,Name,Asset Class\n"
        "AAPL,Apple,Equity\n"
        "-,Cash,Cash\n"
        "BRK-B,Berkshire,Equity\n"
    ).encode()
    result = _parse_iwb_csv(csv_bytes)
    assert "AAPL" in result
    # BRK-B contains '-' and must be excluded
    assert "BRK-B" not in result
    # bare '-' must be excluded
    assert "-" not in result


def test_parse_iwb_csv_deduplicates() -> None:
    """Duplicate tickers must be deduplicated."""
    from crucible.fetcher import _parse_iwb_csv

    result = _parse_iwb_csv(_iwb_csv(["AAPL", "AAPL", "MSFT"]))
    assert result.count("AAPL") == 1


def test_fetch_russell1000_tickers_uses_csv_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch_russell1000_tickers must try the iShares CSV before the file fallback."""
    import crucible.fetcher as fetcher_mod

    fake_tickers = [f"T{i:04d}" for i in range(600)]

    class FakeResponse:
        status_code = 200
        content = fetcher_mod._iwb_csv.__func__(None, fake_tickers) if False else (
            "Ticker,Name,Asset Class\n" +
            "\n".join(f"T{i:04d},Co,Equity" for i in range(600)) + "\n"
        ).encode()

        def raise_for_status(self) -> None:
            pass

    monkeypatch.setattr(fetcher_mod.requests, "get", lambda *a, **kw: FakeResponse())
    result = fetcher_mod.fetch_russell1000_tickers()
    assert len(result) == 600


def test_fetch_russell1000_tickers_falls_back_to_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When the HTTP request fails, fetch_russell1000_tickers reads the local file."""
    import crucible.fetcher as fetcher_mod

    fallback_tickers = [f"T{i:04d}" for i in range(700)]
    fallback_file = tmp_path / "russell1000_tickers.txt"
    fallback_file.write_text("\n".join(fallback_tickers))

    monkeypatch.setattr(fetcher_mod, "_RUSSELL1000_FALLBACK_PATH", fallback_file)
    monkeypatch.setattr(
        fetcher_mod.requests,
        "get",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("network error")),
    )
    result = fetcher_mod.fetch_russell1000_tickers()
    assert result == fallback_tickers


def test_fetch_russell1000_tickers_raises_when_all_sources_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """RuntimeError must be raised when HTTP fails and local file is absent."""
    import crucible.fetcher as fetcher_mod

    monkeypatch.setattr(
        fetcher_mod, "_RUSSELL1000_FALLBACK_PATH", tmp_path / "nonexistent.txt"
    )
    monkeypatch.setattr(
        fetcher_mod.requests,
        "get",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("network error")),
    )
    with pytest.raises(RuntimeError):
        fetcher_mod.fetch_russell1000_tickers()


# ---------------------------------------------------------------------------
# generate_picks_csv
# ---------------------------------------------------------------------------


def test_generate_picks_csv_creates_file(tmp_path: Path) -> None:
    from crucible.backtest import (
        BacktestConfig, BacktestResult, MonthlyResult, generate_picks_csv
    )
    dates = pd.date_range("2020-01-31", periods=4, freq="ME", tz="UTC")
    prices = pd.DataFrame(
        {"AAPL": [100.0, 110.0, 105.0, 115.0], "SP500": [300.0, 310.0, 305.0, 320.0]},
        index=dates,
    )
    monthly = [
        MonthlyResult(
            date=dates[0], portfolio_return=0.10, benchmark_return=0.03,
            n_picks=1, tickers=["AAPL"], ticker_returns={"AAPL": 0.10},
        )
    ]
    result = BacktestResult(
        monthly_results=monthly, hit_rate_returns=[0.10],
        bt_config=BacktestConfig(holding_months=1),
    )
    out = tmp_path / "picks.csv"
    generate_picks_csv(result, prices, out)
    assert out.exists()
    df = pd.read_csv(out)
    assert list(df.columns) == ["date", "ticker", "entry_price", "exit_price", "return_pct"]
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"
    assert abs(df.iloc[0]["entry_price"] - 100.0) < 0.01
    assert abs(df.iloc[0]["exit_price"] - 110.0) < 0.01
    assert abs(df.iloc[0]["return_pct"] - 10.0) < 0.01


def test_generate_picks_csv_empty_result(tmp_path: Path) -> None:
    from crucible.backtest import BacktestConfig, BacktestResult, generate_picks_csv
    prices = pd.DataFrame({"SP500": [300.0]},
                          index=pd.date_range("2020-01-31", periods=1, freq="ME", tz="UTC"))
    result = BacktestResult(monthly_results=[], hit_rate_returns=[], bt_config=BacktestConfig())
    out = tmp_path / "picks.csv"
    generate_picks_csv(result, prices, out)
    assert out.exists()
    assert pd.read_csv(out).empty
