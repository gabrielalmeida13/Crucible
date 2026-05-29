"""Crucible dashboard — Monthly Picks · Portfolio · Manual Import · History · Performance"""
from __future__ import annotations

import io
import json
import re
import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))   # ensure crucible package importable

MONTHLY_DIR   = PROJECT_ROOT / "data" / "monthly"
PORTFOLIO_CSV = PROJECT_ROOT / "data" / "portfolio.csv"
PICKS_DB      = PROJECT_ROOT / "data" / "crucible_picks.db"

# Crucible package imports (available after sys.path insert above)
from crucible.portfolio import allocation_advice
from crucible.regime import (
    Regime,
    VIX_HIGH_VOL_THRESHOLD,
    _fetch_regime_inputs,
)

st.set_page_config(page_title="Crucible ⚗️", page_icon="⚗️", layout="wide")


# ===========================================================================
# Markdown parser
# ===========================================================================

def _parse_md_table(md: str, heading: str) -> pd.DataFrame:
    """Extract the first markdown table after ## {heading}."""
    m = re.search(rf"## {re.escape(heading)}\n([\s\S]*?)(?=\n## |\Z)", md)
    if not m:
        return pd.DataFrame()
    block = m.group(1)
    lines = [
        l for l in block.splitlines()
        if l.strip().startswith("|") and "---" not in l
    ]
    if len(lines) < 2:
        return pd.DataFrame()

    def _cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    headers = _cells(lines[0])
    n       = len(headers)
    rows    = [_cells(l) for l in lines[1:]]
    rows    = [r[:n] + [""] * max(0, n - len(r)) for r in rows]
    df      = pd.DataFrame(rows, columns=headers)
    for col in df.columns:
        if col != "ticker":
            numeric = pd.to_numeric(df[col], errors="coerce")
            if not numeric.isna().all():
                df[col] = numeric
    if "ticker" in df.columns:
        df = df.set_index("ticker")
    return df


# ===========================================================================
# Cached data loaders
# ===========================================================================

def _available_months() -> list[str]:
    if not MONTHLY_DIR.exists():
        return []
    return sorted(
        [d.name for d in MONTHLY_DIR.iterdir()
         if d.is_dir() and re.match(r"^\d{4}-\d{2}$", d.name)],
        reverse=True,
    )


@st.cache_data(ttl=300)
def _load_track_picks(month: str, track: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = MONTHLY_DIR / month / f"track{track}_picks.md"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    text      = path.read_text()
    shortlist = _parse_md_table(text, "Shortlist")
    full_dump = _parse_md_table(
        text, "Full metric dump (all candidates post-filter, for AI reasoning)"
    )
    return shortlist, full_dump


@st.cache_data(ttl=300)
def _load_manifest(month: str) -> dict:
    path = MONTHLY_DIR / month / "run_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


@st.cache_data(ttl=60)
def _load_portfolio_csv() -> pd.DataFrame:
    _empty = pd.DataFrame(columns=[
        "ticker", "entry_price", "entry_date", "track", "shares",
        "entry_pfcf", "entry_ps",
    ])
    if not PORTFOLIO_CSV.exists():
        return _empty
    try:
        df = pd.read_csv(PORTFOLIO_CSV, parse_dates=["entry_date"])
        df["ticker"] = df["ticker"].str.strip().str.upper()
        df = df.dropna(subset=["ticker", "entry_price", "track"])
        for col in ("entry_pfcf", "entry_ps"):
            if col not in df.columns:
                df[col] = float("nan")
        return df.reset_index(drop=True)
    except Exception:
        return _empty


@st.cache_data(ttl=120)
def _fetch_current_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    if not tickers:
        return {}
    if len(tickers) == 1:
        raw   = yf.download(tickers[0], period="5d", progress=False, auto_adjust=True)
        close = raw["Close"].to_frame(name=tickers[0]) if "Close" in raw.columns else pd.DataFrame()
    else:
        raw = yf.download(list(tickers), period="5d", progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else pd.DataFrame()
        else:
            close = pd.DataFrame()
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    prices: dict[str, float] = {}
    for tkr in tickers:
        if tkr in close.columns:
            s = close[tkr].dropna()
            if not s.empty:
                prices[tkr] = float(s.iloc[-1])
    return prices


@st.cache_data(ttl=60)
def _load_history_db() -> pd.DataFrame:
    if not PICKS_DB.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(PICKS_DB) as conn:
            return pd.read_sql(
                "SELECT * FROM monthly_picks ORDER BY run_date DESC, track, ticker",
                conn,
            )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def _regime_widget_data() -> dict:
    """Fetch current regime + raw inputs for the regime badge. Cached 1 hour."""
    try:
        vix, spread, sp500_mom = _fetch_regime_inputs()
        if vix is not None and vix > VIX_HIGH_VOL_THRESHOLD:
            regime = "HIGH_VOL"
        elif (
            spread is not None and sp500_mom is not None
            and spread < 0 and sp500_mom < 0
        ):
            regime = "DEFENSIVE"
        else:
            regime = "GROWTH"
        return {"regime": regime, "vix": vix, "spread": spread, "sp500_mom": sp500_mom}
    except Exception:
        return {"regime": None, "vix": None, "spread": None, "sp500_mom": None}


@st.cache_data(ttl=300)
def _load_prospective_picks() -> pd.DataFrame:
    """Load all new_pick rows from monthly_picks since 2026-06-01."""
    if not PICKS_DB.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(PICKS_DB) as conn:
            return pd.read_sql(
                "SELECT run_date, track, ticker, composite_score "
                "FROM monthly_picks "
                "WHERE recommendation = 'new_pick' AND run_date >= '2026-06-01' "
                "ORDER BY run_date ASC, track ASC, composite_score DESC",
                conn,
            )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def _fetch_performance_prices(tickers: tuple[str, ...], start_date: str) -> pd.DataFrame:
    """Download daily close prices from start_date to today for tickers + SPY."""
    all_t = list(dict.fromkeys(list(tickers) + ["SPY"]))  # dedup, preserve order
    try:
        if len(all_t) == 1:
            raw   = yf.download(all_t[0], start=start_date, progress=False, auto_adjust=True)
            close = raw["Close"].to_frame(name=all_t[0]) if "Close" in raw.columns else pd.DataFrame()
        else:
            raw = yf.download(all_t, start=start_date, progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex) and "Close" in raw.columns.get_level_values(0):
                close = raw["Close"]
            else:
                close = pd.DataFrame()
        if isinstance(close, pd.Series):
            close = close.to_frame(name=all_t[0])
        if not close.empty and close.index.tz is None:
            close.index = close.index.tz_localize("UTC")
        return close
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120)
def _build_portfolio_for_advice() -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Build portfolio eval DataFrame + shortlists for allocation_advice().

    Derives state from portfolio.csv, current prices, SQLite recommendation
    history, and the latest monthly .md shortlists — no EDGAR snapshot needed.
    """
    positions = _load_portfolio_csv()
    if positions.empty:
        return pd.DataFrame(), {}

    tickers = tuple(positions["ticker"].tolist())
    prices  = _fetch_current_prices(tickers)

    months = _available_months()
    shortlists: dict[int, pd.DataFrame] = {}
    if months:
        for track in (1, 2, 3):
            sl, _ = _load_track_picks(months[0], track)
            if not sl.empty:
                shortlists[track] = sl

    _REC_UP = {
        "hold": "HOLD", "reinforce": "REINFORCE", "review": "REVIEW",
        "exit_signal": "EXIT_SIGNAL", "data_missing": "DATA_MISSING",
    }

    rows = []
    for _, pos in positions.iterrows():
        tkr       = str(pos["ticker"])
        curr      = prices.get(tkr)
        ep        = float(pos["entry_price"])
        sh        = float(pos["shares"])
        cb        = ep * sh
        mv        = curr * sh if curr is not None else float("nan")
        ret_pct   = (curr / ep - 1.0) if curr is not None else float("nan")
        gain_loss = mv - cb if curr is not None else float("nan")

        raw_rec = (_latest_rec_from_db(tkr) or "").lower()
        rec     = _REC_UP.get(raw_rec, "DATA_MISSING")

        track_num      = int(pos["track"])
        rank_in_track  = float("nan")
        passes_filters = False
        if track_num in shortlists:
            sl = shortlists[track_num]
            if tkr in sl.index:
                rank_in_track  = float(sl.index.get_loc(tkr)) + 1.0
                passes_filters = True

        rows.append({
            "ticker":         tkr,
            "track":          track_num,
            "shares":         sh,
            "entry_price":    ep,
            "cost_basis":     cb,
            "market_value":   mv,
            "return_pct":     ret_pct,
            "gain_loss":      gain_loss,
            "recommendation": rec,
            "rank_in_track":  rank_in_track,
            "passes_filters": passes_filters,
        })

    if not rows:
        return pd.DataFrame(), shortlists

    return pd.DataFrame(rows).set_index("ticker"), shortlists


# ===========================================================================
# Helpers
# ===========================================================================

_REC_COLORS = {
    "REINFORCE":    "background-color: #d4edda; color: #155724",
    "HOLD":         "background-color: #cce5ff; color: #004085",
    "REVIEW":       "background-color: #fff3cd; color: #856404",
    "EXIT_SIGNAL":  "background-color: #f8d7da; color: #721c24",
    "DATA_MISSING": "background-color: #e2e3e5; color: #383d41",
}

_REGIME_BADGE = {
    "GROWTH":    ("🟢", "GROWTH",    "#155724", "#d4edda"),
    "DEFENSIVE": ("🟡", "DEFENSIVE", "#856404", "#fff3cd"),
    "HIGH_VOL":  ("🔴", "HIGH VOL",  "#721c24", "#f8d7da"),
}


def _style_rec(val: str) -> str:
    return _REC_COLORS.get(str(val).upper(), "")


def _style_return(val: float) -> str:
    if pd.isna(val):
        return ""
    return "color: #28a745; font-weight:600" if val >= 0 else "color: #dc3545; font-weight:600"


def _render_regime_badge(data: dict) -> None:
    """Render the coloured regime badge plus the three input metrics."""
    regime    = data.get("regime")
    vix       = data.get("vix")
    spread    = data.get("spread")
    sp500_mom = data.get("sp500_mom")

    if regime is None:
        st.caption("Market regime: unavailable (yfinance network error)")
        return

    icon, label, text_color, bg = _REGIME_BADGE.get(
        regime, ("⚪", regime or "UNKNOWN", "#383d41", "#e2e3e5")
    )
    st.markdown(
        f'<span style="display:inline-block; padding:5px 14px; border-radius:6px; '
        f'background:{bg}; color:{text_color}; font-weight:700; font-size:1.05rem;">'
        f"{icon}&nbsp;Market Regime: {label}</span>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "VIX",
        f"{vix:.1f}" if vix is not None else "—",
        help="HIGH_VOL when VIX > 25",
    )
    c2.metric(
        "10y – 2y yield spread",
        f"{spread:+.2f} pp" if spread is not None else "—",
        help="Inverted (< 0) + bearish SP500 → DEFENSIVE regime",
    )
    c3.metric(
        "SP500 12m momentum",
        f"{sp500_mom:+.1%}" if sp500_mom is not None else "—",
        help="12-month price return of S&P 500",
    )


def _latest_snapshot_lookup(ticker: str) -> dict[str, Any] | None:
    months = _available_months()
    if not months:
        return None
    for track in (1, 2, 3):
        _, full = _load_track_picks(months[0], track)
        if not full.empty and ticker in full.index:
            return full.loc[ticker].to_dict()
    return None


def _latest_rec_from_db(ticker: str) -> str | None:
    if not PICKS_DB.exists():
        return None
    try:
        with sqlite3.connect(PICKS_DB) as conn:
            row = conn.execute(
                "SELECT recommendation FROM monthly_picks "
                "WHERE ticker=? ORDER BY run_date DESC LIMIT 1",
                (ticker,),
            ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _save_portfolio_row(
    ticker: str,
    entry_price: float,
    entry_date: date,
    track: int,
    shares: float,
    entry_pfcf: float,
    entry_ps: float,
) -> None:
    row = pd.DataFrame([{
        "ticker":      ticker,
        "entry_price": entry_price,
        "entry_date":  entry_date.isoformat(),
        "track":       track,
        "shares":      shares,
        "entry_pfcf":  entry_pfcf if entry_pfcf > 0 else "",
        "entry_ps":    entry_ps   if entry_ps   > 0 else "",
    }])
    if not PORTFOLIO_CSV.exists() or PORTFOLIO_CSV.stat().st_size == 0:
        row.to_csv(PORTFOLIO_CSV, index=False)
    else:
        row.to_csv(PORTFOLIO_CSV, mode="a", header=False, index=False)
    _load_portfolio_csv.clear()


# ===========================================================================
# xStation5 CSV / XLSX helpers
# ===========================================================================

def _xtb_symbol_to_ticker(symbol: str) -> str:
    """Strip XTB market suffix from a symbol, e.g. AAPL.US → AAPL."""
    return re.sub(r"\.(US|DE|FR|UK|PL|HK|JP)(_\d+)?$", "", symbol, flags=re.IGNORECASE).upper()


# ===========================================================================
# Score columns (ordered for display)
# ===========================================================================

_SCORE_COLS = [
    "composite_score", "quality_score", "growth_quality_score",
    "momentum_score", "valuation_score", "value_score",
    "recovery_signal_score", "balance_sheet_score",
    "momentum_raw", "momentum_3m",
    "revenue_growth_yr1", "revenue_growth_yr2", "revenue_acceleration",
    "gross_margin_latest", "gross_margin_yr1_change",
    "fcf_positive_last2yr", "fcf_trajectory",
    "net_debt_ebitda", "roic_proxy_avg", "fcf_positive_years",
    "p_s", "p_fcf", "ev_ebitda", "p_e",
    "share_buyback_signal", "p_fcf_vs_history",
    "sector",
]

_TRACK_LABELS = {
    1: "Track 1 — Quality Compounder",
    2: "Track 2 — Growth Inflection",
    3: "Track 3 — Value Recovery",
}


# ===========================================================================
# Tab 1 — Monthly Picks
# ===========================================================================

def _tab_monthly_picks() -> None:
    months = _available_months()
    if not months:
        st.info("No monthly output found. Run `python scripts/run_monthly.py --track N` first.")
        return

    # ── Regime indicator ────────────────────────────────────────────────────
    with st.spinner("Loading market regime…"):
        regime_data = _regime_widget_data()
    _render_regime_badge(regime_data)
    st.divider()

    # ── Month selector + manifest ────────────────────────────────────────────
    col_sel, _ = st.columns([2, 5])
    selected_month = col_sel.selectbox("Month", months)

    manifest = _load_manifest(selected_month)
    if manifest:
        mc = st.columns(4)
        mc[0].metric("Universe",   manifest.get("universe", "—"))
        mc[1].metric("Run date",   (manifest.get("run_timestamp") or "")[:10])
        mc[2].metric("Git commit", (manifest.get("system_version") or "")[:8])
        n_dict = manifest.get("n_candidates_per_track", {})
        mc[3].metric("Candidates", "  ·  ".join(f"T{k}:{v}" for k, v in sorted(n_dict.items())) or "—")
        st.divider()

    for track in (1, 2, 3):
        shortlist, full_dump = _load_track_picks(selected_month, track)
        header = (
            f"**{_TRACK_LABELS[track]}** — {len(shortlist)} picks"
            if not shortlist.empty
            else f"{_TRACK_LABELS[track]} — no picks file"
        )
        with st.expander(header, expanded=(track == 2)):
            if shortlist.empty:
                st.caption("No picks file found for this track / month.")
                continue

            sort_cols = [c for c in _SCORE_COLS if c in shortlist.columns and c != "sector"]
            default_sort = "composite_score" if "composite_score" in sort_cols else sort_cols[0]

            sc1, sc2 = st.columns([4, 1])
            sort_by  = sc1.selectbox("Sort by", sort_cols,
                                     index=sort_cols.index(default_sort),
                                     key=f"sort_t{track}")
            sort_asc = sc2.checkbox("Ascending", value=False, key=f"asc_t{track}")

            disp_cols = [c for c in _SCORE_COLS if c in shortlist.columns]
            sorted_df = shortlist[disp_cols].sort_values(sort_by, ascending=sort_asc)
            st.dataframe(sorted_df, width="stretch", height=310)

            # Per-ticker metric detail
            if not full_dump.empty:
                st.caption("Company detail")
                available = [t for t in sorted_df.index if t in full_dump.index]
                if available:
                    chosen     = st.selectbox("Select ticker", available, key=f"det_t{track}")
                    detail_src = full_dump.loc[[chosen], [c for c in _SCORE_COLS if c in full_dump.columns]]
                    detail_df  = detail_src.T.rename(columns={chosen: "Value"})
                    detail_df["Value"] = detail_df["Value"].astype(str).replace("nan", "—")
                    st.dataframe(detail_df, width="stretch")

            st.download_button(
                label=f"Download Track {track} CSV",
                data=shortlist.reset_index().to_csv(index=False).encode(),
                file_name=f"track{track}_{selected_month}_picks.csv",
                mime="text/csv",
                key=f"dl_t{track}",
            )


# ===========================================================================
# Tab 2 — Portfolio
# ===========================================================================

def _tab_portfolio() -> None:
    positions = _load_portfolio_csv()

    st.subheader("Current positions")

    if positions.empty:
        st.info("No positions yet. Log your first purchase below.")
    else:
        tickers = tuple(positions["ticker"].tolist())
        prices  = _fetch_current_prices(tickers)

        rows = []
        for _, pos in positions.iterrows():
            tkr  = pos["ticker"]
            curr = prices.get(tkr)
            ep   = float(pos["entry_price"])
            sh   = float(pos["shares"])
            cb   = round(ep * sh, 2)
            mv   = round(curr * sh, 2)           if curr is not None else float("nan")
            ret  = round((curr / ep - 1) * 100, 2) if curr is not None else float("nan")
            gl   = round(mv - cb, 2)             if curr is not None else float("nan")
            rec  = (_latest_rec_from_db(tkr) or "—").upper()
            rows.append({
                "ticker":         tkr,
                "track":          int(pos["track"]),
                "shares":         sh,
                "entry_price":    ep,
                "entry_date":     pos["entry_date"].date() if pd.notna(pos.get("entry_date")) else "—",
                "current_price":  round(curr, 4) if curr is not None else float("nan"),
                "cost_basis":     cb,
                "market_value":   mv,
                "return_%":       ret,
                "gain_loss":      gl,
                "recommendation": rec,
            })

        disp = pd.DataFrame(rows).set_index("ticker")

        fmt = {
            "return_%":      "{:+.2f}",
            "gain_loss":     "{:+,.2f}",
            "market_value":  "{:,.2f}",
            "cost_basis":    "{:,.2f}",
            "current_price": "{:.4f}",
            "entry_price":   "{:.4f}",
        }
        styled = (
            disp.style
            .map(_style_rec, subset=["recommendation"])
            .format(fmt, na_rep="—")
        )
        st.dataframe(styled, width="stretch")

        total_cb  = sum(r["cost_basis"]  for r in rows)
        total_mv  = sum(r["market_value"] for r in rows if not np.isnan(r["market_value"]))
        total_ret = (total_mv / total_cb - 1) * 100 if total_cb > 0 else float("nan")
        total_gl  = total_mv - total_cb

        mc = st.columns(4)
        mc[0].metric("Positions",    len(positions))
        mc[1].metric("Total cost",   f"{total_cb:,.0f}")
        mc[2].metric("Market value", f"{total_mv:,.0f}")
        mc[3].metric("Total return", f"{total_ret:+.2f}%" if not np.isnan(total_ret) else "—",
                     delta=f"{total_gl:+.0f}")

        # ── Allocation advice ────────────────────────────────────────────────
        st.divider()
        st.subheader("Allocation Advice")
        budget = st.number_input(
            "Monthly budget (account currency)",
            min_value=0.0,
            value=100.0,
            step=10.0,
            format="%.0f",
            key="advice_budget",
        )
        portfolio_eval, shortlists = _build_portfolio_for_advice()
        if portfolio_eval.empty and not shortlists:
            st.info(
                "Run the monthly screener first to generate shortlists "
                "(`scripts/run_monthly.py --track 2 --budget 100`)."
            )
        else:
            advice_md = allocation_advice(portfolio_eval, float(budget), shortlists)
            st.markdown(advice_md)

    # ------------------------------------------------------------------
    # Log new purchase
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Log new purchase")

    # Auto-fill lookup lives outside the form so it can trigger a rerun
    lc1, lc2 = st.columns([3, 1])
    lookup_in = lc1.text_input("Look up ticker in latest snapshot", key="lookup_field").upper().strip()
    if lc2.button("Look up P/FCF & P/S", use_container_width=True):
        snap = _latest_snapshot_lookup(lookup_in)
        if snap is not None:
            st.session_state["_pfcf_pre"] = float(snap.get("p_fcf") or 0.0)
            st.session_state["_ps_pre"]   = float(snap.get("p_s")   or 0.0)
            st.success(
                f"{lookup_in}: P/FCF = {st.session_state['_pfcf_pre']:.2f}  ·  "
                f"P/S = {st.session_state['_ps_pre']:.3f}"
            )
        else:
            st.session_state["_pfcf_pre"] = 0.0
            st.session_state["_ps_pre"]   = 0.0
            st.warning(f"{lookup_in} not found in latest snapshot.")

    pfcf_pre = float(st.session_state.get("_pfcf_pre", 0.0))
    ps_pre   = float(st.session_state.get("_ps_pre",   0.0))

    with st.form("log_purchase", clear_on_submit=True):
        r1 = st.columns(4)
        new_ticker = r1[0].text_input("Ticker").upper().strip()
        new_track  = r1[1].selectbox("Track", [1, 2, 3])
        new_price  = r1[2].number_input("Entry price", min_value=0.0001, step=0.01, format="%.4f")
        new_shares = r1[3].number_input("Shares", min_value=0.001, step=0.001, format="%.3f")

        r2 = st.columns(4)
        new_date  = r2[0].date_input("Entry date", value=date.today())
        new_pfcf  = r2[1].number_input("P/FCF at entry (optional)", min_value=0.0,
                                        value=pfcf_pre, step=0.1, format="%.2f")
        new_ps    = r2[2].number_input("P/S at entry (optional)", min_value=0.0,
                                        value=ps_pre, step=0.01, format="%.3f")

        if st.form_submit_button("Log purchase", type="primary"):
            if not new_ticker:
                st.error("Ticker is required.")
            elif new_price <= 0:
                st.error("Entry price must be > 0.")
            elif new_shares <= 0:
                st.error("Shares must be > 0.")
            else:
                _save_portfolio_row(
                    new_ticker, new_price, new_date, new_track,
                    new_shares, new_pfcf, new_ps,
                )
                st.success(
                    f"Logged: {new_shares:.3f} × {new_ticker} @ {new_price:.4f} "
                    f"(Track {new_track})"
                )
                st.session_state.pop("_pfcf_pre", None)
                st.session_state.pop("_ps_pre",   None)


# ===========================================================================
# Tab 3 — Manual Import (xStation5 CSV / XLSX)
# ===========================================================================

def _parse_xstation_export(raw: bytes, filename: str) -> pd.DataFrame:
    """Parse an xStation5 account export (CSV or XLSX) into a normalised DataFrame.

    XLSX — two known formats are handled, detected by sheet names:
      • "Cash Operations" sheet (EUR_*.xlsx): derives net open positions by
        netting 'Stock purchase' buys against 'Stock sell' rows.  Volume and
        price are parsed from the Comment field ("OPEN BUY {vol} @ {price}").
      • "OPEN POSITION …" sheet (legacy account export): reads the open
        positions sheet directly after locating its header row.

    CSV — semicolon-delimited UTF-8/UTF-16; falls back to comma.

    All paths return: xtb_symbol, ticker, open_price, open_date, volume.
    Returns an empty DataFrame on parse failure.
    """
    if filename.lower().endswith((".xlsx", ".xls")):
        return _parse_xstation_xlsx(raw)
    return _parse_xstation_csv(raw)


# Matches "OPEN BUY 0.2004 @ 43.200" and partial fills "OPEN BUY 3/3.4397 @ 30.21"
_COMMENT_RE = re.compile(
    r"(?:OPEN|CLOSE)\s+BUY\s+([\d.]+)(?:/[\d.]+)?\s*@\s*([\d.]+)",
    re.IGNORECASE,
)


def _parse_xstation_xlsx(raw: bytes) -> pd.DataFrame:
    """Dispatch to the correct XLSX parser based on sheet names."""
    try:
        xl = pd.ExcelFile(io.BytesIO(raw))
    except Exception:
        return pd.DataFrame()

    names_upper = [s.strip().upper() for s in xl.sheet_names]

    if any("CASH OPERATIONS" in n for n in names_upper):
        return _parse_xstation_xlsx_cash_ops(xl)

    open_pos = next(
        (s for s in xl.sheet_names if s.strip().upper().startswith("OPEN POSITION")),
        None,
    )
    if open_pos:
        return _parse_xstation_xlsx_open_pos(xl, open_pos)

    return pd.DataFrame()


def _parse_xstation_xlsx_cash_ops(xl: pd.ExcelFile) -> pd.DataFrame:
    """Parse the EUR_*.xlsx format: net open positions from Cash Operations sheet.

    Derives open positions by summing buy lots per ticker and subtracting
    any 'Stock sell' volume.  Volume and price come from the Comment field.
    Entry price reported is the volume-weighted average across all buy lots.
    """
    sheet = next(s for s in xl.sheet_names if "CASH OPERATIONS" in s.strip().upper())

    # Header is in the first row that contains "Type" — scan to find it
    raw_df = pd.read_excel(xl, sheet_name=sheet, header=None)
    header_idx: int | None = None
    for idx, row in raw_df.iterrows():
        if "Type" in row.values:
            header_idx = int(idx)  # type: ignore[arg-type]
            break
    if header_idx is None:
        return pd.DataFrame()

    df = pd.read_excel(xl, sheet_name=sheet, header=header_idx)
    df.columns = [str(c).strip() for c in df.columns]

    # Only import individual stock picks — ETF/investment-plan purchases are excluded
    buys  = df[(df["Type"] == "Stock purchase") & (df["Product"] == "My Trades")]
    sells = df[df["Type"] == "Stock sell"]

    def _parse_comment(comment: Any) -> tuple[float | None, float | None]:
        m = _COMMENT_RE.search(str(comment))
        return (float(m.group(1)), float(m.group(2))) if m else (None, None)

    buy_rows: list[dict] = []
    for _, row in buys.iterrows():
        sym = str(row.get("Ticker", "")).strip()
        if not sym or sym.lower() == "nan":
            continue
        vol, price = _parse_comment(row.get("Comment", ""))
        if vol is None:
            continue
        buy_rows.append({"xtb_symbol": sym, "volume": vol,
                         "open_price": price, "time": row.get("Time")})

    sell_vols: dict[str, float] = {}
    for _, row in sells.iterrows():
        sym = str(row.get("Ticker", "")).strip()
        if not sym or sym.lower() == "nan":
            continue
        vol, _ = _parse_comment(row.get("Comment", ""))
        if vol is not None:
            sell_vols[sym] = sell_vols.get(sym, 0.0) + vol

    if not buy_rows:
        return pd.DataFrame()

    buy_df = pd.DataFrame(buy_rows)
    result: list[dict] = []

    for xtb_sym, group in buy_df.groupby("xtb_symbol"):
        total_bought = group["volume"].sum()
        net_vol      = total_bought - sell_vols.get(str(xtb_sym), 0.0)
        if net_vol < 1e-6:
            continue  # fully closed position

        avg_price = (group["volume"] * group["open_price"]).sum() / total_bought
        earliest  = pd.to_datetime(group["time"], errors="coerce").min()

        result.append({
            "xtb_symbol": xtb_sym,
            "ticker":     _xtb_symbol_to_ticker(str(xtb_sym)),
            "open_price": round(avg_price, 4),
            "volume":     round(net_vol, 6),
            "open_date":  earliest.date() if pd.notna(earliest) else date.today(),
        })

    return pd.DataFrame(result).reset_index(drop=True) if result else pd.DataFrame()


def _parse_xstation_xlsx_open_pos(xl: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    """Parse the legacy xStation5 XLSX format with an 'OPEN POSITION …' sheet."""
    raw_df = pd.read_excel(xl, sheet_name=sheet, header=None)

    header_idx: int | None = None
    for idx, row in raw_df.iterrows():
        if "Symbol" in row.values:
            header_idx = int(idx)  # type: ignore[arg-type]
            break
    if header_idx is None:
        return pd.DataFrame()

    df = pd.read_excel(xl, sheet_name=sheet, header=header_idx)
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    if not {"Symbol", "Open price"}.issubset(df.columns):
        return pd.DataFrame()

    df = df[df["Symbol"].notna() & (df["Symbol"].astype(str).str.strip() != "Total")]
    if "Type" in df.columns:
        df = df[df["Type"].astype(str).str.upper().str.contains("BUY", na=False)]

    return _normalise_positions(df, symbol_col="Symbol", price_col="Open price",
                                volume_col="Volume", time_col="Open time")


def _parse_xstation_csv(raw: bytes) -> pd.DataFrame:
    """Parse the xStation5 CSV transaction history export."""
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except (UnicodeDecodeError, ValueError):
            continue
    else:
        return pd.DataFrame()

    for sep in (";", ","):
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
            if len(df.columns) > 2:
                break
        except Exception:
            continue
    else:
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]
    if not {"Symbol", "Open price"}.issubset(df.columns):
        return pd.DataFrame()

    if "Type" in df.columns:
        df = df[df["Type"].str.upper().str.contains("BUY", na=False)]

    return _normalise_positions(df, symbol_col="Symbol", price_col="Open price",
                                volume_col="Volume", time_col="Open time")


def _normalise_positions(
    df: pd.DataFrame,
    symbol_col: str,
    price_col: str,
    volume_col: str,
    time_col: str,
) -> pd.DataFrame:
    """Map raw xStation5 columns to the internal importer schema."""
    out = pd.DataFrame()
    out["xtb_symbol"] = df[symbol_col].astype(str).str.strip()
    out["ticker"]     = out["xtb_symbol"].apply(_xtb_symbol_to_ticker)

    price_series = df[price_col]
    if price_series.dtype == object:
        price_series = price_series.str.replace(",", ".", regex=False)
    out["open_price"] = pd.to_numeric(price_series, errors="coerce")

    if volume_col in df.columns:
        vol_series = df[volume_col]
        if vol_series.dtype == object:
            vol_series = vol_series.str.replace(",", ".", regex=False)
        out["volume"] = pd.to_numeric(vol_series, errors="coerce")
    else:
        out["volume"] = float("nan")

    if time_col in df.columns:
        out["open_date"] = pd.to_datetime(df[time_col], errors="coerce").dt.date
    else:
        out["open_date"] = date.today()

    return out.dropna(subset=["ticker", "open_price"]).reset_index(drop=True)


def _tab_manual_import() -> None:
    st.subheader("Import positions from xStation5")
    st.caption(
        "XTB's programmatic API was discontinued on 14 March 2025. "
        "Export your transaction history from xStation5 (History → Transactions → Export CSV) "
        "and upload it here to import open BUY positions into the portfolio."
    )

    with st.expander("How to export from xStation5"):
        st.markdown(
            "**XLSX (recommended)** — exports all sheets including open positions:\n"
            "1. Log in to **xStation5** (web or desktop).\n"
            "2. Click the account icon → **Export to XLSX**.\n"
            "3. Upload the downloaded `.xlsx` file below.\n\n"
            "**CSV** — transaction history only:\n"
            "1. Go to **History** → **Transactions**.\n"
            "2. Set the date range to cover all open positions.\n"
            "3. Click **Export** → **CSV** and upload below."
        )

    uploaded = st.file_uploader("Upload xStation5 export", type=["csv", "txt", "xlsx", "xls"], key="xtb_csv_upload")
    if uploaded is None:
        return

    raw_df = _parse_xstation_export(uploaded.read(), uploaded.name)
    if raw_df.empty:
        st.error(
            "Could not parse the file. Make sure it is an xStation5 transaction history export "
            "(CSV semicolon-delimited or XLSX) with the default English column layout."
        )
        return

    existing = set(_load_portfolio_csv()["ticker"].tolist())
    raw_df["already_in_portfolio"] = raw_df["ticker"].isin(existing)

    st.dataframe(raw_df[["xtb_symbol", "ticker", "open_price", "open_date", "volume", "already_in_portfolio"]],
                 width="stretch", hide_index=True)

    new_only = raw_df[~raw_df["already_in_portfolio"]].copy()
    if new_only.empty:
        st.info("All positions from this export are already in the portfolio.")
        return

    st.write(f"**{len(new_only)}** new position(s) to import — review and adjust before confirming:")

    import_df = new_only[["ticker", "open_price", "open_date", "volume"]].copy()
    import_df["track"]      = 2
    import_df["entry_pfcf"] = 0.0
    import_df["entry_ps"]   = 0.0

    edited = st.data_editor(
        import_df,
        column_config={
            "ticker":     st.column_config.TextColumn("Ticker",           disabled=True),
            "open_price": st.column_config.NumberColumn("Entry price",    format="%.4f"),
            "open_date":  st.column_config.DateColumn("Entry date"),
            "volume":     st.column_config.NumberColumn("Shares",         format="%.3f"),
            "track":      st.column_config.SelectboxColumn("Track",       options=[1, 2, 3]),
            "entry_pfcf": st.column_config.NumberColumn("P/FCF at entry", format="%.2f"),
            "entry_ps":   st.column_config.NumberColumn("P/S at entry",   format="%.3f"),
        },
        width="stretch",
        hide_index=True,
        key="xtb_import_editor",
    )

    if st.button("Import into portfolio", type="primary"):
        count = 0
        for _, row in edited.iterrows():
            if not row["ticker"]:
                continue
            _save_portfolio_row(
                ticker=str(row["ticker"]).upper(),
                entry_price=float(row["open_price"]),
                entry_date=row["open_date"],
                track=int(row["track"]),
                shares=float(row["volume"]),
                entry_pfcf=float(row.get("entry_pfcf") or 0.0),
                entry_ps=float(row.get("entry_ps") or 0.0),
            )
            count += 1
        st.success(f"Imported {count} position(s).")
        st.rerun()


# ===========================================================================
# Tab 4 — History
# ===========================================================================

def _tab_history() -> None:
    st.subheader("Pick history")

    hist = _load_history_db()
    if hist.empty:
        st.info("No history yet. Run `python scripts/run_monthly.py --track N` to start logging.")
        return

    # Filters
    fc = st.columns(3)
    track_opts = ["All"] + [str(v) for v in sorted(hist["track"].unique())]
    rec_opts   = ["All"] + sorted(hist["recommendation"].unique().tolist())
    date_opts  = ["All"] + sorted(hist["run_date"].unique().tolist(), reverse=True)

    sel_track = fc[0].selectbox("Track",          track_opts)
    sel_rec   = fc[1].selectbox("Recommendation", rec_opts)
    sel_date  = fc[2].selectbox("Run date",        date_opts)

    filtered = hist.copy()
    if sel_track != "All":
        filtered = filtered[filtered["track"] == int(sel_track)]
    if sel_rec != "All":
        filtered = filtered[filtered["recommendation"] == sel_rec]
    if sel_date != "All":
        filtered = filtered[filtered["run_date"] == sel_date]

    disp_cols = [
        "run_date", "track", "ticker", "universe", "recommendation",
        "sector", "composite_score",
    ]
    for col in ("quality_score", "growth_quality_score", "momentum_score",
                "valuation_score", "momentum_raw", "momentum_3m", "p_fcf", "p_s"):
        if col in filtered.columns:
            disp_cols.append(col)
    disp_cols = [c for c in disp_cols if c in filtered.columns]

    styled = filtered[disp_cols].style.map(_style_rec, subset=["recommendation"])
    st.dataframe(styled, width="stretch", height=420)
    st.caption(f"{len(filtered):,} rows")

    # ------------------------------------------------------------------
    # Compute actual returns (past runs only)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Actual returns")
    st.caption(
        "For each past screener run, simulates buying at the closing price on the run date "
        "and holding to today. Useful after several months of logged picks to measure screener accuracy. "
        "Today's run is excluded — prices won't be available until the market closes."
    )

    today_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    new_picks = hist[
        (hist["recommendation"] == "new_pick") &
        (hist["run_date"] < today_str)
    ].copy()

    # Apply same track/date filters
    if sel_track != "All":
        new_picks = new_picks[new_picks["track"] == int(sel_track)]
    if sel_date != "All":
        new_picks = new_picks[new_picks["run_date"] == sel_date]

    if new_picks.empty:
        st.info(
            "No past picks to evaluate yet — this panel becomes useful once you have "
            "screener runs from previous months logged in the database."
        )
        return

    st.caption(f"{len(new_picks)} pick(s) from {new_picks['run_date'].nunique()} past run(s) in scope.")

    if not st.button("Compute returns", type="primary"):
        return

    unique_tickers = new_picks["ticker"].unique().tolist()
    min_date       = new_picks["run_date"].min()

    with st.spinner(f"Downloading price history for {len(unique_tickers)} ticker(s)…"):
        try:
            if len(unique_tickers) == 1:
                raw   = yf.download(unique_tickers[0], start=min_date, progress=False, auto_adjust=True)
                close = raw["Close"].to_frame(name=unique_tickers[0]) if "Close" in raw.columns else pd.DataFrame()
            else:
                raw   = yf.download(unique_tickers, start=min_date, progress=False, auto_adjust=True)
                close = (
                    raw["Close"]
                    if isinstance(raw.columns, pd.MultiIndex)
                    and "Close" in raw.columns.get_level_values(0)
                    else pd.DataFrame()
                )
            if isinstance(close, pd.Series):
                close = close.to_frame(name=unique_tickers[0])
            if not close.empty and close.index.tz is None:
                close.index = close.index.tz_localize("UTC")
        except Exception as exc:
            st.error(f"Price download failed: {exc}")
            return

    if close.empty:
        st.warning("No price data returned — check that tickers are valid US-listed symbols.")
        return

    ret_rows = []
    for _, row in new_picks.iterrows():
        tkr = row["ticker"]
        if tkr not in close.columns:
            continue
        series = close[tkr].dropna()
        if series.empty:
            continue
        try:
            run_ts   = pd.Timestamp(row["run_date"], tz="UTC")
            after    = series[series.index >= run_ts]
            entry_px = float(after.iloc[0]) if not after.empty else float("nan")
            curr_px  = float(series.iloc[-1])
            ret_pct  = (curr_px / entry_px - 1) * 100 if entry_px > 0 else float("nan")
        except Exception:
            entry_px = curr_px = ret_pct = float("nan")

        ret_rows.append({
            "run_date":        row["run_date"],
            "track":           row["track"],
            "ticker":          tkr,
            "entry_price":     round(entry_px, 4),
            "current_price":   round(curr_px, 4),
            "return_%":        round(ret_pct, 2),
            "composite_score": row.get("composite_score"),
            "sector":          row.get("sector"),
        })

    if not ret_rows:
        st.warning("Could not compute returns — no price data matched the logged tickers.")
        return

    ret_df  = pd.DataFrame(ret_rows).sort_values("run_date")
    avg_ret = ret_df["return_%"].dropna().mean()
    pos_hit = (ret_df["return_%"].dropna() > 0).mean()

    rc = st.columns(3)
    rc[0].metric("Picks",      len(ret_df))
    rc[1].metric("Avg return", f"{avg_ret:+.2f}%")
    rc[2].metric("Hit rate",   f"{pos_hit:.1%}")

    st.dataframe(ret_df, width="stretch")


# ===========================================================================
# Tab 5 — Performance
# ===========================================================================

def _tab_performance() -> None:
    st.subheader("Prospective Performance")
    st.caption(
        "All screener picks (new_pick) from **June 2026 onwards** — the prospective "
        "validation period. Entry price is the first close on or after the run date."
    )

    picks = _load_prospective_picks()

    if picks.empty:
        st.info(
            "No prospective picks logged yet. The prospective period begins June 2026. "
            "Once the first monthly run after June 1 2026 is logged, results will appear here."
        )
        return

    unique_tickers = tuple(sorted(picks["ticker"].unique().tolist()))
    min_date       = picks["run_date"].min()
    today_str      = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    if min_date >= today_str:
        st.info("Picks exist but all run dates are today or in the future — prices not yet available.")
        return

    with st.spinner(f"Fetching prices for {len(unique_tickers)} ticker(s) + SPY benchmark…"):
        close = _fetch_performance_prices(unique_tickers, start_date=min_date)

    if close.empty:
        st.warning("Could not download price data — check network connection.")
        return

    # Build per-pick return rows
    ret_rows: list[dict] = []
    for _, row in picks.iterrows():
        tkr     = row["ticker"]
        run_dt  = row["run_date"]
        if run_dt >= today_str:
            continue  # exclude today — market may not have closed
        if tkr not in close.columns:
            continue

        series = close[tkr].dropna()
        if series.empty:
            continue

        try:
            run_ts   = pd.Timestamp(run_dt, tz="UTC")
            after    = series[series.index >= run_ts]
            entry_px = float(after.iloc[0]) if not after.empty else float("nan")
            curr_px  = float(series.iloc[-1])
            ret_pct  = (curr_px / entry_px - 1) * 100 if entry_px > 0 else float("nan")
        except Exception:
            entry_px = curr_px = ret_pct = float("nan")

        ret_rows.append({
            "run_date":          run_dt,
            "track":             int(row["track"]),
            "ticker":            tkr,
            "score_at_pick":     round(float(row["composite_score"]), 3)
                                 if pd.notna(row.get("composite_score")) else float("nan"),
            "entry_price":       round(entry_px, 4),
            "current_price":     round(curr_px, 4),
            "return_%":          round(ret_pct, 2),
        })

    if not ret_rows:
        st.info("No completed pick periods yet — prices are fetched the day after the run date.")
        return

    ret_df = pd.DataFrame(ret_rows).sort_values(["run_date", "track", "return_%"],
                                                  ascending=[True, True, False])

    # ── Summary metrics ──────────────────────────────────────────────────────
    valid_rets = ret_df["return_%"].dropna()
    avg_pick   = valid_rets.mean()
    hit_rate   = (valid_rets > 0).mean()
    n_picks    = len(ret_df)

    # SP500 benchmark: return from earliest pick date to today (via SPY)
    sp500_ret: float | None = None
    if "SPY" in close.columns:
        spy = close["SPY"].dropna()
        try:
            spy_start = pd.Timestamp(min_date, tz="UTC")
            spy_after = spy[spy.index >= spy_start]
            if not spy_after.empty:
                sp500_ret = (float(spy.iloc[-1]) / float(spy_after.iloc[0]) - 1) * 100
        except Exception:
            pass

    mc = st.columns(4)
    mc[0].metric("Picks evaluated", n_picks)
    mc[1].metric("Average return",  f"{avg_pick:+.2f}%" if not np.isnan(avg_pick) else "—")
    mc[2].metric("Hit rate",        f"{hit_rate:.1%}" if len(valid_rets) > 0 else "—")
    mc[3].metric(
        "SP500 (same period)",
        f"{sp500_ret:+.2f}%" if sp500_ret is not None else "—",
        delta=f"{avg_pick - sp500_ret:+.2f}pp excess" if sp500_ret is not None and not np.isnan(avg_pick) else None,
    )

    st.divider()

    # ── Per-pick table with coloured return ──────────────────────────────────
    styled = (
        ret_df.style
        .map(_style_return, subset=["return_%"])
        .format(
            {
                "entry_price":   "{:.4f}",
                "current_price": "{:.4f}",
                "return_%":      "{:+.2f}",
                "score_at_pick": "{:.3f}",
            },
            na_rep="—",
        )
    )
    st.dataframe(styled, width="stretch", height=420, hide_index=True)

    # ── Track breakdown ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("By track")
    track_summary = (
        ret_df.groupby("track")["return_%"]
        .agg(picks="count", avg_return="mean", hit_rate=lambda x: (x > 0).mean())
        .reset_index()
    )
    track_summary.columns = ["track", "picks", "avg_return_%", "hit_rate"]
    track_summary["track_label"] = track_summary["track"].map(
        {1: "T1 Quality", 2: "T2 Growth", 3: "T3 Value"}
    )
    styled_track = track_summary[["track_label", "picks", "avg_return_%", "hit_rate"]].style.format(
        {"avg_return_%": "{:+.2f}", "hit_rate": "{:.1%}"}, na_rep="—"
    )
    st.dataframe(styled_track, width="stretch", hide_index=True)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    st.title("⚗️ Crucible — Monthly Stock Screener")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Monthly Picks", "Portfolio", "Manual Import", "History", "Performance"]
    )
    with tab1:
        _tab_monthly_picks()
    with tab2:
        _tab_portfolio()
    with tab3:
        _tab_manual_import()
    with tab4:
        _tab_history()
    with tab5:
        _tab_performance()


if __name__ == "__main__":
    main()
