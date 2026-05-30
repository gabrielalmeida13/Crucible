#!/usr/bin/env python3
"""Monthly screener entry point — all three tracks, with portfolio review.

Usage
-----
    python scripts/run_monthly.py --track 1
    python scripts/run_monthly.py --track 2
    python scripts/run_monthly.py --track 3
    python scripts/run_monthly.py --track 1 --universe SP500
    python scripts/run_monthly.py --track 2 --top-n 15 --budget 5000

Output
------
    stdout                                         — portfolio review + ranked shortlist
    data/monthly/{YYYY-MM}/track{N}_picks.md       — full metric dump (AI-debate format)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from crucible.alerts import check_monthly_reminder, check_portfolio_alerts, dispatch_alerts
from crucible.config import CrucibleConfig
from crucible.fetcher import _load_cik_mapping, fetch_russell1000_tickers, fetch_sp500_tickers
from crucible.portfolio import allocation_advice, evaluate_portfolio, load_portfolio
from crucible.regime import Regime, detect_regime, regime_allocation_hint
from crucible.snapshot import attach_momentum, build_snapshots_parallel
from crucible.store import log_monthly_picks
from crucible.tracks import track1_quality, track2_growth, track3_value

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EDGAR_DIR     = ROOT / "data" / "raw" / "edgar" / "companyfacts"
CIK_MAP_PATH  = ROOT / "data" / "raw" / "edgar" / "cik_mapping.json"
PRICE_WORKERS = 20


# ---------------------------------------------------------------------------
# Sector helpers
# ---------------------------------------------------------------------------


def _fetch_sector_map() -> dict[str, str]:
    """Best-effort bulk sector map from Wikipedia S&P 500 table."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        sp500_df = tables[0]
        return dict(zip(sp500_df["Symbol"], sp500_df["GICS Sector"]))
    except Exception:
        log.warning("Wikipedia sector fetch failed — will fall back to yfinance for shortlist")
        return {}


def _attach_sectors(df: pd.DataFrame, sector_map: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["sector"] = df.index.map(sector_map).fillna("Unknown")
    return df


def _fill_sectors_yfinance(df: pd.DataFrame, existing: dict[str, str]) -> pd.DataFrame:
    """For tickers still missing a sector, fetch from yfinance (shortlist only, ≤ 35 tickers)."""
    missing = [t for t in df.index if existing.get(t, "Unknown") == "Unknown"]
    if not missing:
        return df

    log.info("Fetching sectors via yfinance for %d tickers: %s", len(missing), missing)

    def _get_sector(ticker: str) -> tuple[str, str]:
        try:
            return ticker, yf.Ticker(ticker).info.get("sector", "Unknown") or "Unknown"
        except Exception:
            return ticker, "Unknown"

    filled: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        for ticker, sector in pool.map(_get_sector, missing):
            filled[ticker] = sector

    df = df.copy()
    df["sector"] = df.index.map(lambda t: filled.get(t, existing.get(t, "Unknown")))
    return df


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------


def _fetch_one_price(ticker: str, start: str, end: str) -> tuple[str, pd.Series]:
    label = "SP500" if ticker == "SPY" else ticker
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return label, pd.Series(dtype=float, name=label)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return label, close.resample("ME").last().rename(label)
    except Exception:
        log.warning("Price fetch failed for %s", ticker, exc_info=True)
        return label, pd.Series(dtype=float, name=label)


def _fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    all_tickers = list(tickers) + ["SPY"]
    series_map: dict[str, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=PRICE_WORKERS) as pool:
        futures = {pool.submit(_fetch_one_price, t, start, end): t for t in all_tickers}
        for future in as_completed(futures):
            label, s = future.result()
            if not s.empty:
                series_map[label] = s
    if not series_map:
        return pd.DataFrame()
    prices = pd.concat(series_map.values(), axis=1)
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize("UTC")
    return prices


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_SCORE_COLS = [
    "composite_score",
    "growth_quality_score", "momentum_score", "valuation_score",   # Track 2
    "value_score", "recovery_signal_score", "balance_sheet_score", # Track 3
    "quality_score",                                                # Track 1
    "momentum_raw", "momentum_3m",
    "revenue_growth_yr1", "revenue_growth_yr2", "revenue_acceleration",
    "gross_margin_latest", "gross_margin_yr1_change",
    "fcf_positive_last2yr", "fcf_trajectory",
    "net_debt_ebitda",
    "roic_proxy_avg", "fcf_positive_years",
    "p_s", "p_fcf", "ev_ebitda", "p_e",
    "share_buyback_signal", "p_fcf_vs_history",
    "asset_growth_yoy", "deferred_revenue_growth", "eps_surprise_last_q",
    "sector",
]


def _print_portfolio_review(
    eval_df: pd.DataFrame,
    advice: str,
    today: pd.Timestamp,
    regime: Regime | None = None,
) -> None:
    print(f"\n{'='*70}")
    print(f"  Portfolio Review  ({today.strftime('%Y-%m-%d')})")
    if regime is not None:
        print(f"  Market regime: {regime.value}")
    print(f"{'='*70}")

    if eval_df.empty:
        print("  No positions found in data/portfolio.csv")
        print()
        return

    reinforce    = eval_df[eval_df["recommendation"] == "REINFORCE"]
    hold         = eval_df[eval_df["recommendation"] == "HOLD"]
    review       = eval_df[eval_df["recommendation"] == "REVIEW"]
    exit_signal  = eval_df[eval_df["recommendation"] == "EXIT_SIGNAL"]
    data_missing = eval_df[eval_df["recommendation"] == "DATA_MISSING"]

    _DISP = ["track", "shares", "entry_price", "current_price",
             "return_pct", "gain_loss", "rank_in_track", "recommendation"]

    def _pct_fmt(v: float) -> str:
        return f"{v:+.1%}" if not np.isnan(v) else "—"

    sections = [
        ("🔼 REINFORCE — add to position",    reinforce),
        ("✓ HOLD",                            hold),
        ("↘ REVIEW",                          review),
        ("⚠ EXIT SIGNAL",                     exit_signal),
        ("ℹ DATA MISSING — hold, no action",  data_missing),
    ]
    for label, subset in sections:
        if subset.empty:
            continue
        print(f"\n  {label}")
        disp_cols = [c for c in _DISP if c in subset.columns]
        formatted = subset[disp_cols].copy()
        if "return_pct" in formatted.columns:
            formatted["return_pct"] = formatted["return_pct"].apply(_pct_fmt)
        print(formatted.to_string())

    print(f"\n{advice}")
    print()


def _print_shortlist(result: pd.DataFrame, track: int, top_n: int, today: pd.Timestamp) -> None:
    cols = [c for c in _SCORE_COLS if c in result.columns]
    top  = result.head(top_n)[cols]
    print(f"\n{'='*70}")
    print(f"  Track {track} — Top {top_n} candidates  ({today.strftime('%Y-%m')})")
    print(f"{'='*70}")
    print(top.to_string(float_format="{:.3f}".format))
    print()


def _write_picks_md(
    result: pd.DataFrame,
    track: int,
    top_n: int,
    output_dir: Path,
    today: pd.Timestamp,
    universe: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"track{track}_picks.md"
    top  = result.head(top_n)
    cols = [c for c in _SCORE_COLS if c in top.columns]

    lines = [
        f"# Track {track} Monthly Picks — {today.strftime('%Y-%m')}",
        "",
        f"**Run date:** {today.strftime('%Y-%m-%d')}  ",
        f"**Universe:** {universe}  ",
        f"**Track:** {track}  ",
        f"**Candidates after filters:** {len(result)}  ",
        "",
        "## Shortlist",
        "",
        top[cols].to_markdown(floatfmt=".3f"),
        "",
        "## Full metric dump (all candidates post-filter, for AI reasoning)",
        "",
        result[cols].to_markdown(floatfmt=".3f"),
    ]
    path.write_text("\n".join(lines))
    log.info("Picks written → %s", path)


# ---------------------------------------------------------------------------
# Prospective logging helpers
# ---------------------------------------------------------------------------

PICKS_DB = ROOT / "data" / "crucible_picks.db"


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _write_manifest_atomic(
    output_dir: Path,
    today: pd.Timestamp,
    universe: str,
    track: int,
    n_candidates: int,
) -> None:
    manifest = {
        "run_timestamp":        today.isoformat(),
        "universe":             universe,
        "tracks_run":           [track],
        "n_candidates_per_track": {str(track): n_candidates},
        "system_version":       _git_commit_hash(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "run_manifest.json"
    fd, tmp_path = tempfile.mkstemp(dir=output_dir, prefix=".manifest_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(manifest, fh, indent=2)
        os.replace(tmp_path, dest)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    log.info("Manifest written → %s", dest)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly Crucible screener with portfolio review")
    parser.add_argument("--track", choices=["1", "2", "3"], required=True,
                        help="Which track to run (1=Quality, 2=Growth, 3=Value)")
    parser.add_argument(
        "--universe",
        choices=["SP500", "RUSSELL1000"],
        default=os.getenv("CRUCIBLE_UNIVERSE", "RUSSELL1000"),
    )
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of candidates to display (default 10)")
    parser.add_argument("--budget", type=float, default=0.0,
                        help="Monthly investment budget for allocation advice (account currency)")
    parser.add_argument(
        "--month",
        default=None,
        metavar="YYYY-MM",
        help="Override output directory month (e.g. 2026-06). Defaults to current month.",
    )
    args = parser.parse_args()

    track  = int(args.track)
    top_n  = args.top_n
    budget = args.budget

    for path, label in ((CIK_MAP_PATH, "CIK mapping"), (EDGAR_DIR, "EDGAR data")):
        if not path.exists():
            log.error("%s not found at %s. Run scripts/download_edgar_bulk.py first.", label, path)
            sys.exit(1)

    config  = CrucibleConfig(account_currency=os.getenv("CRUCIBLE_ACCOUNT_CURRENCY", "EUR"))
    cik_map = _load_cik_mapping(CIK_MAP_PATH)

    # 1. Universe
    if args.universe == "RUSSELL1000":
        tickers = fetch_russell1000_tickers()
    else:
        tickers = fetch_sp500_tickers()
    log.info("%d tickers in universe", len(tickers))

    # 2. Snapshot date: last day of the current month so the cache is reused on
    #    every run within the same month regardless of which day it's executed.
    today     = pd.Timestamp.now(tz="UTC").normalize()
    snap_key  = (today + pd.offsets.MonthEnd(0)).normalize()
    snap_date = pd.DatetimeIndex([snap_key])
    output_month = args.month if args.month else today.strftime("%Y-%m")

    # 3. Prices: 15 months back so momentum_raw (12-1m) is computable
    price_end   = today.strftime("%Y-%m-%d")
    price_start = (today - pd.DateOffset(months=15)).strftime("%Y-%m-%d")
    log.info("Fetching prices %s → %s", price_start, price_end)
    prices = _fetch_prices(tickers, start=price_start, end=price_end)

    # 4. Build EDGAR snapshot (passes prices for inline valuation multiples)
    log.info("Building EDGAR snapshot for %s …", today.date())
    fund_by_date = build_snapshots_parallel(
        tickers=tickers,
        dates=snap_date,
        cik_map=cik_map,
        edgar_dir=EDGAR_DIR,
        prices=prices,
        workers=4,
    )

    # 5. Momentum
    attach_momentum(fund_by_date, prices)

    # Resolve the actual dict key: handles legacy caches built with exact-day keys
    # and any other mismatch between snap_key and what the cache stored.
    if snap_key in fund_by_date:
        data_key = snap_key
    else:
        data_key = max(fund_by_date.keys())
        log.warning(
            "Snapshot not found for %s, using most recent cached date %s",
            today.date(), data_key.date(),
        )

    # 6. Sectors (Wikipedia bulk → yfinance fallback for shortlist)
    sector_map = _fetch_sector_map()
    df = fund_by_date[data_key]
    df = _attach_sectors(df, sector_map)
    fund_by_date[data_key] = df  # write back so evaluate_portfolio sees sectors

    # 7. Portfolio review (runs before new picks)
    log.info("Detecting market regime …")
    try:
        regime = detect_regime()
    except Exception:
        log.warning("Regime detection failed — defaulting to GROWTH", exc_info=True)
        regime = Regime.GROWTH

    positions = load_portfolio()
    eval_df   = evaluate_portfolio(positions, fund_by_date, prices, config, monthly_budget=budget)

    # 8. Run the selected track
    if track == 1:
        result = track1_quality.run(df, config)

    elif track == 2:
        mom_mask = df["momentum_raw"].notna() & (df["momentum_raw"] > 0)
        result   = track2_growth.run(df[mom_mask], config)

    else:  # track == 3
        if "p_fcf_vs_history" not in df.columns:
            log.warning(
                "Track 3: p_fcf_vs_history not available in live snapshot "
                "(requires multi-month history). p_fcf_vs_history filter will be skipped."
            )
        result = track3_value.run(df, config)

    if result.empty:
        log.warning("No candidates survived filters — no output written.")
        advice = allocation_advice(eval_df, budget, {}) if budget > 0 or not eval_df.empty else ""
        if not eval_df.empty or advice:
            _print_portfolio_review(eval_df, advice, today, regime)
        output_dir = ROOT / "data" / "monthly" / output_month
        _write_manifest_atomic(output_dir, today, args.universe, track, 0)
        sys.exit(0)

    # 9. Fill remaining unknown sectors for shortlist tickers only
    result = _fill_sectors_yfinance(result, sector_map)

    # 10. Portfolio review output
    shortlists = {track: result} if budget > 0 or not eval_df.empty else {}
    advice     = allocation_advice(eval_df, budget, shortlists)
    regime_hint = regime_allocation_hint(regime)
    full_advice = f"{advice}\n\n  {regime_hint}" if advice else f"  {regime_hint}"
    _print_portfolio_review(eval_df, full_advice, today, regime)

    # 11. New picks output
    output_dir = ROOT / "data" / "monthly" / output_month
    _print_shortlist(result, track, top_n, today)
    _write_picks_md(result, track, top_n, output_dir, today, args.universe)

    # 12a. Portfolio alerts — before logging so SQLite still holds the previous state
    portfolio_alerts = check_portfolio_alerts(
        portfolio_csv=ROOT / "data" / "portfolio.csv",
        fund_by_date=fund_by_date,
        prices=prices,
        config=config,
        db_path=PICKS_DB,
    )

    # 12b. Prospective logging
    run_date = today.strftime("%Y-%m-%d")
    log_monthly_picks(
        db_path=PICKS_DB,
        run_date=run_date,
        track=track,
        universe=args.universe,
        result=result,
        portfolio_recs=eval_df if not eval_df.empty else None,
    )
    _write_manifest_atomic(output_dir, today, args.universe, track, len(result))

    # 12c. Monthly reminder — after logging so the current run is visible in SQLite
    all_alerts = list(portfolio_alerts)
    reminder = check_monthly_reminder(db_path=PICKS_DB)
    if reminder:
        all_alerts.append(reminder)
    if all_alerts:
        dispatch_alerts(all_alerts)


if __name__ == "__main__":
    main()
