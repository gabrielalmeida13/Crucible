"""Unit tests for store.py — all using in-memory SQLite, no files created."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crucible.store import (
    create_tables,
    get_engine,
    list_scans,
    load_all_for_scan,
    load_shortlist,
    save_scan,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine():
    """Fresh in-memory SQLite engine with tables created."""
    eng = get_engine(":memory:")
    create_tables(eng)
    return eng


def _processed_row(**kwargs) -> dict:
    defaults = dict(
        sector="Technology",
        sub_industry="Software",
        currency="USD",
        p_e=20.0,
        p_fcf=15.0,
        ev_ebitda=10.0,
        data_years=5,
        insufficient_data=False,
        roic_proxy_avg=0.20,
        fcf_latest=1e9,
        fcf_positive_years=4.0,
        net_debt_ebitda=1.0,
        revenue_growth_positive_years=4.0,
        gross_margin_latest=0.45,
        gross_margin_avg=0.44,
        gross_margin_trend_slope=0.01,
    )
    defaults.update(kwargs)
    return defaults


def _scored_row(**kwargs) -> dict:
    defaults = dict(
        **_processed_row(),
        quality_score=0.8,
        valuation_score=0.7,
        fx_penalty=0.0,
        composite_score=0.76,
    )
    defaults.update(kwargs)
    return defaults


def _make_processed(*rows: dict, tickers: list[str] | None = None) -> pd.DataFrame:
    if tickers is None:
        tickers = [f"T{i}" for i in range(len(rows))]
    return pd.DataFrame(list(rows), index=pd.Index(tickers, name="ticker"))


def _make_shortlist(*rows: dict, tickers: list[str] | None = None) -> pd.DataFrame:
    if tickers is None:
        tickers = [f"T{i}" for i in range(len(rows))]
    return pd.DataFrame(list(rows), index=pd.Index(tickers, name="ticker"))


# ---------------------------------------------------------------------------
# save_scan
# ---------------------------------------------------------------------------


def test_save_scan_returns_integer_scan_id(engine) -> None:
    """save_scan must return an integer primary key."""
    processed = _make_processed(_processed_row(), tickers=["AAPL"])
    shortlist = _make_shortlist(_scored_row(), tickers=["AAPL"])
    scan_id = save_scan(engine, processed, shortlist, "SP500", "20240101_120000")
    assert isinstance(scan_id, int)
    assert scan_id >= 1


def test_save_scan_increments_scan_id(engine) -> None:
    """Each call to save_scan must produce a unique, incrementing scan_id."""
    processed = _make_processed(_processed_row(), tickers=["AAPL"])
    shortlist = _make_shortlist(_scored_row(), tickers=["AAPL"])
    id1 = save_scan(engine, processed, shortlist, "SP500", "20240101_120000")
    id2 = save_scan(engine, processed, shortlist, "SP500", "20240201_120000")
    assert id2 > id1


def test_save_scan_stores_correct_counts(engine) -> None:
    """n_processed and n_passed_filters must match the DataFrames passed in."""
    processed = _make_processed(
        _processed_row(), _processed_row(), _processed_row(),
        tickers=["A", "B", "C"],
    )
    shortlist = _make_shortlist(_scored_row(), tickers=["A"])
    save_scan(engine, processed, shortlist, "SP500", "20240101_120000")
    scans = list_scans(engine)
    assert scans.iloc[0]["n_processed"] == 3
    assert scans.iloc[0]["n_passed_filters"] == 1


def test_save_scan_all_tickers_stored_in_companies(engine) -> None:
    """Every ticker from processed_df must appear in the companies table."""
    processed = _make_processed(
        _processed_row(), _processed_row(),
        tickers=["AAPL", "MSFT"],
    )
    shortlist = _make_shortlist(_scored_row(), tickers=["AAPL"])
    save_scan(engine, processed, shortlist, "SP500", "20240101_120000")
    result = load_all_for_scan(engine)
    assert set(result.index) == {"AAPL", "MSFT"}


def test_save_scan_only_shortlist_has_passed_filters_true(engine) -> None:
    """passed_filters=True only for tickers in shortlist_df."""
    processed = _make_processed(
        _processed_row(), _processed_row(),
        tickers=["PASS", "FAIL"],
    )
    shortlist = _make_shortlist(_scored_row(), tickers=["PASS"])
    save_scan(engine, processed, shortlist, "SP500", "20240101_120000")
    all_df = load_all_for_scan(engine)
    assert all_df.loc["PASS", "passed_filters"] == True  # noqa: E712
    assert all_df.loc["FAIL", "passed_filters"] == False  # noqa: E712


def test_save_scan_excluded_ticker_has_null_scores(engine) -> None:
    """Excluded tickers must have NULL (NaN) for all score columns."""
    processed = _make_processed(
        _processed_row(), _processed_row(),
        tickers=["PASS", "FAIL"],
    )
    shortlist = _make_shortlist(_scored_row(), tickers=["PASS"])
    save_scan(engine, processed, shortlist, "SP500", "20240101_120000")
    all_df = load_all_for_scan(engine)
    for col in ("quality_score", "valuation_score", "fx_penalty", "composite_score"):
        assert pd.isna(all_df.loc["FAIL", col]), f"{col} should be NaN for excluded ticker"


def test_save_scan_shortlisted_ticker_has_scores(engine) -> None:
    """Shortlisted tickers must have their score values persisted correctly."""
    processed = _make_processed(_processed_row(), tickers=["AAPL"])
    shortlist = _make_shortlist(
        _scored_row(quality_score=0.9, valuation_score=0.8, fx_penalty=0.0, composite_score=0.86),
        tickers=["AAPL"],
    )
    save_scan(engine, processed, shortlist, "SP500", "20240101_120000")
    result = load_shortlist(engine)
    assert abs(result.loc["AAPL", "quality_score"] - 0.9) < 1e-6
    assert abs(result.loc["AAPL", "composite_score"] - 0.86) < 1e-6


def test_save_scan_handles_nan_fundamentals(engine) -> None:
    """NaN values in fundamentals must be stored as NULL without raising."""
    processed = _make_processed(
        _processed_row(roic_proxy_avg=np.nan, p_fcf=np.nan),
        tickers=["AAPL"],
    )
    shortlist = _make_shortlist(_scored_row(), tickers=["AAPL"])
    save_scan(engine, processed, shortlist, "SP500", "20240101_120000")
    result = load_all_for_scan(engine)
    assert pd.isna(result.loc["AAPL", "roic_proxy_avg"])
    assert pd.isna(result.loc["AAPL", "p_fcf"])


def test_save_scan_handles_empty_shortlist(engine) -> None:
    """save_scan must succeed when no tickers pass filters."""
    processed = _make_processed(_processed_row(), tickers=["AAPL"])
    shortlist = _make_shortlist()  # empty
    scan_id = save_scan(engine, processed, shortlist, "SP500", "20240101_120000")
    assert isinstance(scan_id, int)
    all_df = load_all_for_scan(engine)
    assert not all_df.loc["AAPL", "passed_filters"]


# ---------------------------------------------------------------------------
# load_shortlist
# ---------------------------------------------------------------------------


def test_load_shortlist_returns_only_passed(engine) -> None:
    """load_shortlist must return only tickers with passed_filters=True."""
    processed = _make_processed(
        _processed_row(), _processed_row(),
        tickers=["PASS", "FAIL"],
    )
    shortlist = _make_shortlist(_scored_row(), tickers=["PASS"])
    save_scan(engine, processed, shortlist, "SP500", "20240101_120000")
    result = load_shortlist(engine)
    assert "PASS" in result.index
    assert "FAIL" not in result.index


def test_load_shortlist_sorted_by_composite_score(engine) -> None:
    """load_shortlist must be sorted by composite_score descending."""
    processed = _make_processed(
        _processed_row(), _processed_row(), _processed_row(),
        tickers=["LOW", "MID", "HIGH"],
    )
    shortlist = _make_shortlist(
        _scored_row(composite_score=0.50),
        _scored_row(composite_score=0.70),
        _scored_row(composite_score=0.90),
        tickers=["LOW", "MID", "HIGH"],
    )
    save_scan(engine, processed, shortlist, "SP500", "20240101_120000")
    result = load_shortlist(engine)
    scores = result["composite_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_load_shortlist_defaults_to_latest_scan(engine) -> None:
    """When scan_id is None, load_shortlist must return data from the latest scan."""
    p1 = _make_processed(_processed_row(), tickers=["OLD"])
    s1 = _make_shortlist(_scored_row(composite_score=0.5), tickers=["OLD"])
    save_scan(engine, p1, s1, "SP500", "20240101_120000")

    p2 = _make_processed(_processed_row(), tickers=["NEW"])
    s2 = _make_shortlist(_scored_row(composite_score=0.8), tickers=["NEW"])
    save_scan(engine, p2, s2, "SP500", "20240201_120000")

    result = load_shortlist(engine)
    assert "NEW" in result.index
    assert "OLD" not in result.index


def test_load_shortlist_specific_scan_id(engine) -> None:
    """Passing an explicit scan_id must load that scan, not the latest."""
    p1 = _make_processed(_processed_row(), tickers=["OLD"])
    s1 = _make_shortlist(_scored_row(), tickers=["OLD"])
    first_id = save_scan(engine, p1, s1, "SP500", "20240101_120000")

    p2 = _make_processed(_processed_row(), tickers=["NEW"])
    s2 = _make_shortlist(_scored_row(), tickers=["NEW"])
    save_scan(engine, p2, s2, "SP500", "20240201_120000")

    result = load_shortlist(engine, scan_id=first_id)
    assert "OLD" in result.index
    assert "NEW" not in result.index


def test_load_shortlist_empty_when_no_scans(engine) -> None:
    """load_shortlist must return an empty DataFrame when the DB has no scans."""
    result = load_shortlist(engine)
    assert result.empty


def test_load_shortlist_index_is_ticker(engine) -> None:
    """The returned DataFrame must have ticker as index."""
    processed = _make_processed(_processed_row(), tickers=["AAPL"])
    shortlist = _make_shortlist(_scored_row(), tickers=["AAPL"])
    save_scan(engine, processed, shortlist, "SP500", "20240101_120000")
    result = load_shortlist(engine)
    assert result.index.name == "ticker"
    assert "AAPL" in result.index


# ---------------------------------------------------------------------------
# load_all_for_scan
# ---------------------------------------------------------------------------


def test_load_all_for_scan_includes_excluded_tickers(engine) -> None:
    """load_all_for_scan must include tickers that did not pass filters."""
    processed = _make_processed(
        _processed_row(), _processed_row(),
        tickers=["PASS", "FAIL"],
    )
    shortlist = _make_shortlist(_scored_row(), tickers=["PASS"])
    save_scan(engine, processed, shortlist, "SP500", "20240101_120000")
    result = load_all_for_scan(engine)
    assert "PASS" in result.index
    assert "FAIL" in result.index


def test_load_all_for_scan_has_passed_filters_column(engine) -> None:
    """load_all_for_scan must include the passed_filters boolean column."""
    processed = _make_processed(_processed_row(), tickers=["AAPL"])
    shortlist = _make_shortlist(_scored_row(), tickers=["AAPL"])
    save_scan(engine, processed, shortlist, "SP500", "20240101_120000")
    result = load_all_for_scan(engine)
    assert "passed_filters" in result.columns


def test_load_all_for_scan_empty_when_no_scans(engine) -> None:
    """load_all_for_scan must return an empty DataFrame when no scans exist."""
    result = load_all_for_scan(engine)
    assert result.empty


# ---------------------------------------------------------------------------
# list_scans
# ---------------------------------------------------------------------------


def test_list_scans_returns_newest_first(engine) -> None:
    """list_scans must return rows ordered newest (highest id) first."""
    processed = _make_processed(_processed_row(), tickers=["AAPL"])
    shortlist = _make_shortlist(_scored_row(), tickers=["AAPL"])
    save_scan(engine, processed, shortlist, "SP500", "20240101_120000")
    save_scan(engine, processed, shortlist, "SP500", "20240201_120000")
    scans = list_scans(engine)
    ids = scans["id"].tolist()
    assert ids == sorted(ids, reverse=True)


def test_list_scans_contains_universe_id(engine) -> None:
    """list_scans must store and return the universe_id correctly."""
    processed = _make_processed(_processed_row(), tickers=["AAPL"])
    shortlist = _make_shortlist(_scored_row(), tickers=["AAPL"])
    save_scan(engine, processed, shortlist, "EUROPE_LARGE", "20240101_120000")
    scans = list_scans(engine)
    assert scans.iloc[0]["universe_id"] == "EUROPE_LARGE"


def test_list_scans_empty_db(engine) -> None:
    """list_scans must return a DataFrame (possibly empty) when no scans exist."""
    scans = list_scans(engine)
    assert isinstance(scans, pd.DataFrame)
