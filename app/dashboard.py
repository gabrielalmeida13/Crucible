"""Streamlit dashboard — browse monthly scan results stored in the SQLite database."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from crucible.store import get_engine, list_scans, load_all_for_scan, load_shortlist  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "crucible.db"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Crucible — Stock Screener",
    page_icon="⚗️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Engine / data loading
# ---------------------------------------------------------------------------


@st.cache_resource
def _engine():
    return get_engine(DB_PATH)


@st.cache_data(ttl=60)
def _scans() -> pd.DataFrame:
    return list_scans(_engine())


@st.cache_data(ttl=60)
def _shortlist(scan_id: int) -> pd.DataFrame:
    return load_shortlist(_engine(), scan_id=scan_id)


@st.cache_data(ttl=60)
def _all_tickers(scan_id: int) -> pd.DataFrame:
    return load_all_for_scan(_engine(), scan_id=scan_id)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

_SCORE_COLS = ["quality_score", "valuation_score", "fx_penalty", "composite_score"]
_FUNDAMENTAL_COLS = [
    "sector", "sub_industry", "currency",
    "roic_proxy_avg", "fcf_positive_years", "gross_margin_avg",
    "net_debt_ebitda", "revenue_growth_positive_years",
    "p_e", "p_fcf", "ev_ebitda",
]
_DISPLAY_COLS = _FUNDAMENTAL_COLS + _SCORE_COLS + ["passed_filters"]

_COL_LABELS = {
    "sector": "Sector",
    "sub_industry": "Sub-industry",
    "currency": "Currency",
    "roic_proxy_avg": "ROIC Avg",
    "fcf_positive_years": "FCF+ Yrs",
    "gross_margin_avg": "GM Avg",
    "net_debt_ebitda": "Net Debt/EBITDA",
    "revenue_growth_positive_years": "Rev Growth+ Yrs",
    "p_e": "P/E",
    "p_fcf": "P/FCF",
    "ev_ebitda": "EV/EBITDA",
    "quality_score": "Quality",
    "valuation_score": "Valuation",
    "fx_penalty": "FX Penalty",
    "composite_score": "Composite",
    "passed_filters": "Passed Filters",
}


def _pct(val: float) -> str:
    if pd.isna(val):
        return "—"
    return f"{val:.1%}"


def _fmt(val: float, decimals: int = 2) -> str:
    if pd.isna(val):
        return "—"
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("⚗️ Crucible — Monthly Stock Screener")

    scans = _scans()

    if scans.empty:
        st.warning(
            "No scan results found. Run `python scripts/run_scan.py` to generate your first scan."
        )
        return

    # ------------------------------------------------------------------
    # Sidebar — scan selector + filters
    # ------------------------------------------------------------------

    with st.sidebar:
        st.header("Scan")
        scan_options = {
            f"#{row['id']}  {row['run_ts']}  ({row['universe_id']})": int(row["id"])
            for _, row in scans.iterrows()
        }
        selected_label = st.selectbox("Select scan", list(scan_options.keys()))
        selected_scan_id = scan_options[selected_label]

        # Optional: compare with previous scan
        compare_scan_id: int | None = None
        if len(scans) > 1:
            compare_options = {"None": None} | {
                f"#{row['id']}  {row['run_ts']}": int(row["id"])
                for _, row in scans.iterrows()
                if int(row["id"]) != selected_scan_id
            }
            compare_label = st.selectbox("Compare with", list(compare_options.keys()))
            compare_scan_id = compare_options[compare_label]

        st.divider()
        st.header("Filters")

        show_all = st.checkbox("Show all tickers (including excluded)", value=False)

        min_composite = st.slider(
            "Min composite score", min_value=0.0, max_value=1.0, value=0.0, step=0.05
        )

        fx_filter = st.selectbox(
            "FX exposure",
            options=["All", "No FX penalty only", "FX penalty only"],
        )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    if show_all:
        df = _all_tickers(selected_scan_id)
    else:
        df = _shortlist(selected_scan_id)

    if df.empty:
        st.info("No tickers to display for this scan with current filters.")
        return

    # ------------------------------------------------------------------
    # Dynamic sector multiselect (must come after data is loaded)
    # ------------------------------------------------------------------

    all_sectors = sorted(df["sector"].dropna().unique().tolist())
    with st.sidebar:
        sectors = st.multiselect("Sectors", options=all_sectors, placeholder="All sectors")

    # ------------------------------------------------------------------
    # Apply sidebar filters
    # ------------------------------------------------------------------

    filtered = df.copy()

    if sectors:
        filtered = filtered[filtered["sector"].isin(sectors)]

    if not show_all and "composite_score" in filtered.columns:
        filtered = filtered[
            filtered["composite_score"].fillna(-999) >= min_composite
        ]

    if fx_filter == "No FX penalty only":
        filtered = filtered[filtered["fx_penalty"].fillna(-1) == 0.0]
    elif fx_filter == "FX penalty only":
        filtered = filtered[filtered["fx_penalty"].fillna(0) < 0.0]

    # ------------------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------------------

    selected_scan_row = scans[scans["id"] == selected_scan_id].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Universe", selected_scan_row["universe_id"])
    col2.metric("Processed", int(selected_scan_row["n_processed"]))
    col3.metric("Passed filters", int(selected_scan_row["n_passed_filters"]))
    col4.metric("Shown", len(filtered))

    st.caption(
        f"Scan run: {selected_scan_row['run_ts']}  |  Saved: {selected_scan_row['created_at']}"
    )

    st.divider()

    # ------------------------------------------------------------------
    # Main table
    # ------------------------------------------------------------------

    display_cols = [c for c in _DISPLAY_COLS if c in filtered.columns]
    display_df = filtered[display_cols].copy()

    for col in ("roic_proxy_avg", "gross_margin_avg"):
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(_pct)
    for col in ("quality_score", "valuation_score", "composite_score"):
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda v: _fmt(v, 3))
    if "fx_penalty" in display_df.columns:
        display_df["fx_penalty"] = display_df["fx_penalty"].apply(lambda v: _fmt(v, 2))

    display_df = display_df.rename(columns=_COL_LABELS)
    display_df.index.name = "Ticker"

    st.dataframe(display_df, use_container_width=True, height=520)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    csv_bytes = filtered.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"crucible_scan_{selected_scan_id}.csv",
        mime="text/csv",
    )

    # ------------------------------------------------------------------
    # Scan comparison (new vs previous)
    # ------------------------------------------------------------------

    if compare_scan_id is not None:
        st.divider()
        st.subheader("Scan comparison")

        prev_df = _shortlist(compare_scan_id) if not show_all else _all_tickers(compare_scan_id)
        prev_scan_row = scans[scans["id"] == compare_scan_id].iloc[0]

        current_tickers = set(df.index)
        previous_tickers = set(prev_df.index) if not prev_df.empty else set()

        new_entries = sorted(current_tickers - previous_tickers)
        dropped = sorted(previous_tickers - current_tickers)

        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.markdown(
                f"**New entries** vs scan #{compare_scan_id} ({prev_scan_row['run_ts']})"
            )
            if new_entries:
                st.dataframe(pd.DataFrame({"Ticker": new_entries}), use_container_width=True)
            else:
                st.write("None")
        with comp_col2:
            st.markdown(f"**Dropped** vs scan #{compare_scan_id}")
            if dropped:
                st.dataframe(pd.DataFrame({"Ticker": dropped}), use_container_width=True)
            else:
                st.write("None")

    # ------------------------------------------------------------------
    # Score distribution (shortlist only)
    # ------------------------------------------------------------------

    if not show_all and "composite_score" in filtered.columns and not filtered.empty:
        st.divider()
        st.subheader("Score distribution")
        score_data = filtered[["quality_score", "valuation_score", "composite_score"]].apply(
            pd.to_numeric, errors="coerce"
        )
        st.bar_chart(score_data.dropna().sort_values("composite_score", ascending=False))


if __name__ == "__main__":
    main()
