#!/usr/bin/env python3
"""Monthly screener entry point — runs Track 1 or Track 2 against today's Russell 1000 snapshot.

Usage
-----
    python scripts/run_monthly.py --track 1
    python scripts/run_monthly.py --track 2
    python scripts/run_monthly.py --track 1 --universe SP500
    python scripts/run_monthly.py --track 2 --top-n 15

Output
------
    stdout                                         — ranked top-N shortlist
    data/monthly/{YYYY-MM}/track{N}_picks.md       — full metric dump (AI-debate format)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from crucible.config import CrucibleConfig, Track2ScoreWeights
from crucible.fetcher import _load_cik_mapping, fetch_russell1000_tickers, fetch_sp500_tickers
from crucible.snapshot import attach_momentum, build_snapshots_parallel, prices_at
from crucible.tracks import track1_quality, track2_growth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EDGAR_DIR    = ROOT / "data" / "raw" / "edgar" / "companyfacts"
CIK_MAP_PATH = ROOT / "data" / "raw" / "edgar" / "cik_mapping.json"
PRICE_WORKERS = 20


# ---------------------------------------------------------------------------
# Sector attachment
# ---------------------------------------------------------------------------

def _attach_sectors(df: pd.DataFrame, sector_map: dict[str, str]) -> pd.DataFrame:
    """Fill sector column from ticker → sector map; Unknown for missing."""
    df = df.copy()
    df["sector"] = df.index.map(sector_map).fillna("Unknown")
    return df


def _fetch_sector_map(tickers: list[str]) -> dict[str, str]:
    """Best-effort sector map: SP500 Wikipedia table, Unknown for anything else."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        sp500_df = tables[0]
        return dict(zip(sp500_df["Symbol"], sp500_df["GICS Sector"]))
    except Exception:
        log.warning("Could not fetch sector map — all sectors will be 'Unknown'")
        return {}


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
# Momentum filter for Track 2 (applied after price attachment)
# ---------------------------------------------------------------------------

def _filter_momentum_positive(df: pd.DataFrame) -> pd.DataFrame:
    """Track 2 requires positive 12-1m price momentum at entry."""
    mask = df["momentum_raw"].notna() & (df["momentum_raw"] > 0)
    return df[mask]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_SCORE_COLS = [
    "composite_score",
    "growth_quality_score", "momentum_score", "valuation_score",  # Track 2
    "quality_score",        # Track 1
    "momentum_raw", "momentum_3m",
    "revenue_growth_yr1", "revenue_growth_yr2", "revenue_acceleration",
    "gross_margin_latest", "gross_margin_yr1_change",
    "fcf_positive_last2yr", "fcf_trajectory",
    "net_debt_ebitda",
    "roic_proxy_avg", "fcf_positive_years",
    "p_s", "p_fcf", "ev_ebitda", "p_e",
]


def _print_shortlist(result: pd.DataFrame, track: int, top_n: int) -> None:
    cols = [c for c in _SCORE_COLS if c in result.columns]
    top = result.head(top_n)[cols]
    print(f"\n{'='*70}")
    print(f"  Track {track} — Top {top_n} candidates  ({pd.Timestamp.now().strftime('%Y-%m')})")
    print(f"{'='*70}")
    print(top.to_string(float_format="{:.3f}".format))
    print()


def _write_picks_md(result: pd.DataFrame, track: int, top_n: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"track{track}_picks.md"
    top = result.head(top_n)
    cols = [c for c in _SCORE_COLS if c in top.columns]

    lines = [
        f"# Track {track} Monthly Picks — {pd.Timestamp.now().strftime('%Y-%m')}",
        "",
        f"**Run date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
        f"**Universe:** {os.getenv('CRUCIBLE_UNIVERSE', 'RUSSELL1000')}  ",
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly Crucible screener")
    parser.add_argument("--track", choices=["1", "2"], required=True)
    parser.add_argument(
        "--universe",
        choices=["SP500", "RUSSELL1000"],
        default=os.getenv("CRUCIBLE_UNIVERSE", "RUSSELL1000"),
    )
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    track   = int(args.track)
    top_n   = args.top_n

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

    # 2. Snapshot date = today
    today = pd.Timestamp.now(tz="UTC").normalize()
    snap_date = pd.DatetimeIndex([today])

    # 3. Prices: fetch 15 months back so momentum_raw (12-1m) and momentum_3m can be computed
    price_end   = today.strftime("%Y-%m-%d")
    price_start = (today - pd.DateOffset(months=15)).strftime("%Y-%m-%d")
    log.info("Fetching prices %s → %s", price_start, price_end)
    prices = _fetch_prices(tickers, start=price_start, end=price_end)

    # 4. Build EDGAR snapshot (passes prices so valuation multiples are computed inline)
    log.info("Building EDGAR snapshot for %s …", today.date())
    fund_by_date = build_snapshots_parallel(
        tickers=tickers,
        dates=snap_date,
        cik_map=cik_map,
        edgar_dir=EDGAR_DIR,
        prices=prices,
        workers=4,
    )

    # 5. Attach momentum (adds momentum_raw and momentum_3m)
    attach_momentum(fund_by_date, prices)

    # 6. Attach sectors
    sector_map = _fetch_sector_map(tickers)
    df = fund_by_date[today]
    df = _attach_sectors(df, sector_map)

    # 7. Run track
    if track == 1:
        result = track1_quality.run(df, config)
    else:
        # Track 2: apply momentum filter after prices are attached
        df_filtered_by_momentum = _filter_momentum_positive(df)
        result = track2_growth.run(df_filtered_by_momentum, config)

    # 8. Output
    _print_shortlist(result, track, top_n)
    output_dir = ROOT / "data" / "monthly" / today.strftime("%Y-%m")
    _write_picks_md(result, track, top_n, output_dir)


if __name__ == "__main__":
    main()
