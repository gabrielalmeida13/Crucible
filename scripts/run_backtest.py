#!/usr/bin/env python3
"""3×2 matrix backtest: holding_months=[1, 3, 12] × universe=['SP500', 'RUSSELL1000'].

Usage
-----
  # Full 3×2 matrix (default)
  python scripts/run_backtest.py

  # Single universe, all holding periods
  python scripts/run_backtest.py --universe SP500

  # Single holding period, all universes
  python scripts/run_backtest.py --holding 1

  # Single combination
  python scripts/run_backtest.py --universe SP500 --holding 1

  Flags:
    --universe  SP500 | RUSSELL1000          (default: both)
    --holding   1 | 3 | 12                  (default: all three)

Walk-forward design
-------------------
  Train 24 months, walk forward 1 month at a time.
  Test period: 2012-01 → 2022-12 (119 test months).

Data sources
------------
  SEC EDGAR companyfacts/{CIK}.json  — fundamentals (point-in-time via filed date)
  yfinance                           — price data only

Memory management
-----------------
  EDGAR JSONs are loaded lazily on first access and kept in a 300-entry LRU cache
  (≈50 MB peak) so the full 1 000 company JSONs are never in memory simultaneously.

Parallelism
-----------
  yfinance price fetches:   ThreadPoolExecutor(max_workers=20) — I/O bound
  Monthly snapshot builds:  ThreadPoolExecutor(max_workers=4)  — memory-safe
  Matrix combinations:      sequential — no parallel backtest runs

Requirements
------------
  Run scripts/download_edgar_bulk.py once before running.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import psutil
import yfinance as yf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from crucible.backtest import (
    BacktestConfig,
    attach_momentum,
    generate_picks_csv,
    generate_report,
    generate_ticker_contribution,
    hit_rate,
    max_drawdown,
    run_backtest,
    run_sensitivity,
    sharpe_ratio,
    total_return,
)
from crucible.config import CrucibleConfig
from crucible.fetcher import (
    _load_cik_mapping,
    _load_dei_shares_cached,
    fetch_russell1000_tickers,
    fetch_sp500_tickers,
)
from crucible.filters import (
    filter_fcf_consistency,
    filter_gross_margin_stability,
    filter_leverage,
    filter_revenue_growth,
    filter_roic,
)
from crucible.snapshot import build_snapshots_parallel

# ---------------------------------------------------------------------------
# Matrix configuration
# ---------------------------------------------------------------------------

MATRIX_UNIVERSES      = ["SP500", "RUSSELL1000"]
MATRIX_HOLDING_MONTHS = [1, 3, 12]

BACKTEST_START    = pd.Timestamp("2010-01-31", tz="UTC")
BACKTEST_END      = pd.Timestamp("2022-12-31", tz="UTC")
PRICE_FETCH_START = "2009-01-01"
PRICE_FETCH_END   = "2024-01-31"   # covers 12-month hit-rate lookforward from Dec 2022

TRAIN_MONTHS = 24
TOP_N        = 20

RESULTS_DIR      = ROOT / "data" / "results"
DIAGNOSTICS_DIR  = ROOT / "data" / "diagnostics"
EDGAR_DIR        = ROOT / "data" / "raw" / "edgar" / "companyfacts"
CIK_MAP_PATH     = ROOT / "data" / "raw" / "edgar" / "cik_mapping.json"

PRICE_WORKERS    = 20    # I/O-bound yfinance
SNAPSHOT_WORKERS = 4     # memory-safe EDGAR snapshot workers
MEMORY_LOG_SECS  = 600   # log RSS every 10 minutes

# Enable for monthly screener runs only — too slow for full backtest (62k API calls).
ENABLE_INSIDER_FORM4 = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------


def _memory_monitor(stop_event: threading.Event) -> None:
    """Log process RSS every MEMORY_LOG_SECS. Runs in a daemon thread."""
    proc = psutil.Process()
    while not stop_event.wait(MEMORY_LOG_SECS):
        rss_mib = proc.memory_info().rss / 1024 / 1024
        log.info("[mem] RSS %.0f MiB", rss_mib)


# ---------------------------------------------------------------------------
# Insider / buyback signal helpers
# ---------------------------------------------------------------------------

import os as _os
import requests as _requests

_EDGAR_UA: str = _os.environ.get("EDGAR_USER_AGENT", "Crucible gabrielserens@gmail.com")

# Per-process caches — avoids redundant network round-trips within a single run.
_SUBMISSIONS_CACHE: dict[str, dict] = {}   # CIK10 → SEC submissions JSON
_FORM4_SIGNAL_CACHE: dict[tuple[str, str], float | None] = {}  # (CIK10, as_of) → signal


def _compute_buyback_signal(cik: str, as_of_date: pd.Timestamp) -> float | None:
    """Year-over-year change in shares outstanding from EDGAR DEI 10-K filings.

    Returns (prior_shares − current_shares) / prior_shares.
    Positive = net buyback; negative = net dilution.
    """
    records = _load_dei_shares_cached(cik.zfill(10), str(EDGAR_DIR))
    as_of_str = as_of_date.strftime("%Y-%m-%d")
    annual = sorted(
        [r for r in records if r["form"] in {"10-K", "10-K/A"} and r["filed"] <= as_of_str],
        key=lambda r: r["filed"],
        reverse=True,
    )
    if len(annual) < 2:
        return None
    current = annual[0]["val"]
    # Find the most recent annual filing approximately 1 year before the latest
    latest_dt = pd.Timestamp(annual[0]["filed"])
    window_lo = (latest_dt - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    window_hi = (latest_dt - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    prior_candidates = [r for r in annual[1:] if window_lo <= r["filed"] <= window_hi]
    if not prior_candidates:
        return None
    prior = prior_candidates[0]["val"]
    return (prior - current) / prior if prior > 0 else None


def _compute_insider_signal(
    cik: str,
    as_of_date: pd.Timestamp,
    shares_outstanding: float | None,
) -> float | None:
    """Net insider share acquisitions (buys − sells) via Form 4 / 6 months prior to as_of_date.

    Uses the EDGAR submissions JSON to list filings, then downloads Form 4 XMLs (up to 5)
    and parses them with edgartools. Returns net_shares / shares_outstanding, or None on
    any failure (network unavailable, parse error, no filings in window).
    """
    if not shares_outstanding or shares_outstanding <= 0:
        return None
    cik10 = cik.zfill(10)
    as_of_str = as_of_date.strftime("%Y-%m-%d")
    cache_key = (cik10, as_of_str)
    if cache_key in _FORM4_SIGNAL_CACHE:
        return _FORM4_SIGNAL_CACHE[cache_key]

    result: float | None = None
    try:
        window_start = (as_of_date - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
        # Fetch / reuse submissions JSON (one HTTP call per CIK per process lifetime)
        if cik10 not in _SUBMISSIONS_CACHE:
            resp = _requests.get(
                f"https://data.sec.gov/submissions/CIK{cik10}.json",
                headers={"User-Agent": _EDGAR_UA},
                timeout=10,
            )
            resp.raise_for_status()
            _SUBMISSIONS_CACHE[cik10] = resp.json()
        sub = _SUBMISSIONS_CACHE[cik10]
        recent = sub.get("filings", {}).get("recent", {})
        forms     = recent.get("form", [])
        dates     = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        prim_docs  = recent.get("primaryDocument", [])
        cik_int = str(int(cik10))

        entries = [
            (acc, pdoc)
            for form, fdate, acc, pdoc in zip(forms, dates, accessions, prim_docs)
            if form == "4" and window_start <= fdate <= as_of_str
        ][:5]

        if not entries:
            result = None
        else:
            from edgar.ownership import Form4 as _Form4
            net_shares = 0.0
            found = False
            for acc, pdoc in entries:
                try:
                    acc_clean = acc.replace("-", "")
                    xml_url = (
                        f"https://www.sec.gov/Archives/edgar/data/"
                        f"{cik_int}/{acc_clean}/{pdoc}"
                    )
                    r2 = _requests.get(
                        xml_url, headers={"User-Agent": _EDGAR_UA}, timeout=5
                    )
                    if r2.status_code != 200:
                        continue
                    f4 = _Form4.parse_xml(r2.text)
                    buys = f4.common_stock_purchases
                    if buys is not None and not buys.empty and "Shares" in buys.columns:
                        net_shares += float(buys["Shares"].sum())
                        found = True
                    sells = f4.common_stock_sales
                    if sells is not None and not sells.empty and "Shares" in sells.columns:
                        net_shares -= float(sells["Shares"].sum())
                        found = True
                except Exception:
                    continue
            if found:
                result = net_shares / shares_outstanding
    except Exception:
        log.debug("Form 4 signal failed for CIK %s at %s", cik, as_of_str, exc_info=True)
        result = None

    _FORM4_SIGNAL_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Parallel price fetching
# ---------------------------------------------------------------------------


def _fetch_one_price(ticker: str, start: str, end: str) -> tuple[str, pd.Series]:
    """Fetch monthly close prices for one ticker. Returns (label, series)."""
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


def _fetch_prices_parallel(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch prices for tickers + SPY benchmark in parallel (20 workers).

    SPY is renamed to SP500 in the returned DataFrame.
    """
    all_tickers = list(tickers) + ["SPY"]
    series_map: dict[str, pd.Series] = {}
    total = len(all_tickers)
    done = 0

    with ThreadPoolExecutor(max_workers=PRICE_WORKERS) as pool:
        futures = {pool.submit(_fetch_one_price, t, start, end): t for t in all_tickers}
        for future in as_completed(futures):
            label, s = future.result()
            done += 1
            if done % 50 == 0 or done == total:
                log.info("Prices: %d / %d fetched", done, total)
            if not s.empty:
                series_map[label] = s

    if not series_map:
        return pd.DataFrame()

    prices = pd.concat(series_map.values(), axis=1)
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize("UTC")
    log.info(
        "Price matrix: %d month-end rows × %d series",
        len(prices), len(prices.columns),
    )
    return prices


# ---------------------------------------------------------------------------
# Filter funnel diagnostics
# ---------------------------------------------------------------------------


def _write_filter_funnel(
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    config: CrucibleConfig,
    output_path: Path,
) -> None:
    """Write per-snapshot filter funnel to CSV for diagnostic analysis.

    Applies each filter sequentially so the counts reflect cumulative drop-off,
    matching the live apply_filters() pipeline. Covers only the SP500 universe.
    """
    th = config.filters
    rows: list[dict] = []

    for date in sorted(fund_by_date):
        df = fund_by_date[date]
        total = len(df)
        n_insufficient = int(df["insufficient_data"].astype(bool).sum())

        usable = df[~df["insufficient_data"].astype(bool)]
        n_roic_null = int(usable["roic_proxy_avg"].isna().sum())

        after_roic   = filter_roic(usable, th.roic_min)
        after_fcf    = filter_fcf_consistency(after_roic, th.fcf_positive_min_years)
        after_debt   = filter_leverage(after_fcf, th.net_debt_ebitda_max)
        after_growth = filter_revenue_growth(after_debt, th.revenue_growth_positive_min_years)
        after_margin = filter_gross_margin_stability(after_growth)

        rows.append({
            "date":             date.date(),
            "total_tickers":    total,
            "insufficient_data": n_insufficient,
            "roic_null":        n_roic_null,
            "passed_roic":      len(after_roic),
            "passed_fcf":       len(after_fcf),
            "passed_debt":      len(after_debt),
            "passed_growth":    len(after_growth),
            "passed_margin":    len(after_margin),
            "passed_all":       len(after_margin),
        })

        log.info(
            "[funnel] %s  total=%d  insuff=%d  roic_null=%d  "
            "→roic=%d →fcf=%d →debt=%d →growth=%d →margin=%d  passed=%d",
            date.date(), total, n_insufficient, n_roic_null,
            len(after_roic), len(after_fcf), len(after_debt),
            len(after_growth), len(after_margin), len(after_margin),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    log.info("Filter funnel diagnostics → %s", output_path)


# ---------------------------------------------------------------------------
# Matrix summary
# ---------------------------------------------------------------------------


def _generate_matrix_summary(
    summary_rows: list[dict],
    output_path: Path,
) -> None:
    """Write side-by-side comparison of all 6 matrix combinations."""

    def _pct(v: object) -> str:
        try:
            return f"{float(v):.2%}"   # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "—"

    def _f2(v: object) -> str:
        try:
            return f"{float(v):.2f}"   # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "—"

    header = (
        "| Universe | Holding | Total Return | Benchmark | Excess |"
        " Sharpe | Max DD | Hit Rate | Months |"
    )
    sep = (
        "|----------|---------|-------------|-----------|--------|"
        "--------|--------|----------|--------|"
    )
    table_rows = [header, sep]
    for row in summary_rows:
        table_rows.append(
            f"| {row['universe']} | {row['holding_months']}m"
            f" | {_pct(row['portfolio_return'])}"
            f" | {_pct(row['benchmark_return'])}"
            f" | {_pct(row['excess_return'])}"
            f" | {_f2(row['sharpe'])}"
            f" | {_pct(row['max_drawdown'])}"
            f" | {_pct(row['hit_rate'])}"
            f" | {int(row['n_months'])} |"
        )

    lines = [
        "# Crucible Matrix Backtest Summary",
        "",
        f"**Train window:** {TRAIN_MONTHS} months  ",
        f"**Test period:** {BACKTEST_START.date()} → {BACKTEST_END.date()}  ",
        f"**Portfolio size:** Top {TOP_N} per month  ",
        f"**Universes:** {', '.join(MATRIX_UNIVERSES)}  ",
        f"**Holding periods:** {', '.join(str(h) + 'm' for h in MATRIX_HOLDING_MONTHS)}  ",
        "",
        "---",
        "",
        "## Results",
        "",
        *table_rows,
        "",
        "---",
        "",
        "> Per-combination reports, picks CSVs, and contribution analyses saved alongside.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Matrix summary → %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Crucible walk-forward backtest matrix.")
    parser.add_argument(
        "--universe",
        choices=MATRIX_UNIVERSES,
        default=None,
        help="Restrict to one universe (default: all).",
    )
    parser.add_argument(
        "--holding",
        type=int,
        choices=MATRIX_HOLDING_MONTHS,
        default=None,
        help="Restrict to one holding period in months (default: all).",
    )
    args = parser.parse_args()

    universes = [args.universe] if args.universe else MATRIX_UNIVERSES
    holdings  = [args.holding]  if args.holding  else MATRIX_HOLDING_MONTHS

    if not CIK_MAP_PATH.exists():
        log.error(
            "CIK mapping not found at %s. Run scripts/download_edgar_bulk.py first.",
            CIK_MAP_PATH,
        )
        sys.exit(1)
    if not EDGAR_DIR.exists():
        log.error(
            "EDGAR companyfacts directory not found at %s. "
            "Run scripts/download_edgar_bulk.py first.",
            EDGAR_DIR,
        )
        sys.exit(1)

    cik_map   = _load_cik_mapping(CIK_MAP_PATH)
    monthly_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq="ME", tz="UTC")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Memory monitor (daemon thread — auto-stops at process exit) ──────────
    stop_mem = threading.Event()
    threading.Thread(
        target=_memory_monitor, args=(stop_mem,), daemon=True, name="mem-monitor"
    ).start()
    log.info("Memory monitor started (every %.0f min)", MEMORY_LOG_SECS / 60)

    summary_rows: list[dict] = []
    config = CrucibleConfig(account_currency="USD")

    try:
        for universe_id in universes:
            log.info("=" * 60)
            log.info("Universe: %s", universe_id)
            log.info("=" * 60)

            # ── Tickers ──────────────────────────────────────────────────────
            log.info("Fetching %s ticker list …", universe_id)
            if universe_id == "SP500":
                tickers = fetch_sp500_tickers()
            else:
                tickers = fetch_russell1000_tickers()
            log.info("Universe: %d tickers", len(tickers))

            # ── Prices (fetched once, shared across all holding periods) ──────
            log.info(
                "Fetching prices for %d tickers + SPY benchmark "
                "(parallel, %d workers) …",
                len(tickers), PRICE_WORKERS,
            )
            prices = _fetch_prices_parallel(tickers, PRICE_FETCH_START, PRICE_FETCH_END)
            if prices.empty:
                log.error("No price data for %s — skipping", universe_id)
                continue

            # ── Fundamentals snapshots (built once per universe) ──────────────
            log.info(
                "Building %d monthly snapshots from EDGAR (%d workers) …",
                len(monthly_dates), SNAPSHOT_WORKERS,
            )
            fund_by_date = build_snapshots_parallel(
                tickers=tickers,
                dates=monthly_dates,
                cik_map=cik_map,
                edgar_dir=EDGAR_DIR,
                prices=prices,
                workers=SNAPSHOT_WORKERS,
            )
            log.info("Snapshots complete: %d dates", len(fund_by_date))
            attach_momentum(fund_by_date, prices)
            log.info("Momentum attached to all snapshots")

            if universe_id == "SP500":
                _write_filter_funnel(
                    fund_by_date,
                    config,
                    DIAGNOSTICS_DIR / "filter_funnel.csv",
                )

            # ── Run each holding period sequentially ──────────────────────────
            for holding in holdings:
                combo = f"{universe_id}_{holding}m"
                log.info("-" * 50)
                log.info("Combination: %s", combo)

                bt_cfg = BacktestConfig(
                    train_months=TRAIN_MONTHS,
                    top_n=TOP_N,
                    holding_months=holding,
                    hit_rate_months=12,
                    risk_free_annual=0.04,
                    benchmark_col="SP500",
                )

                result = run_backtest(fund_by_date, prices, config, bt_cfg)
                log.info(
                    "%s: %d test months, %d hit-rate observations",
                    combo, len(result.monthly_results), len(result.hit_rate_returns),
                )

                if not result.monthly_results:
                    log.warning("%s: no test results — skipping outputs", combo)
                    continue

                sensitivity = run_sensitivity(
                    fund_by_date, prices, config, bt_cfg,
                    roic_thresholds=(0.10, 0.12, 0.15, 0.18, 0.20),
                )

                # Write per-combination outputs
                generate_report(
                    result, sensitivity,
                    RESULTS_DIR / f"{combo}_report.md",
                    config,
                )
                generate_picks_csv(
                    result, prices,
                    RESULTS_DIR / f"{combo}_picks.csv",
                )
                generate_ticker_contribution(
                    result,
                    RESULTS_DIR / f"{combo}_contributions.md",
                    roic_threshold=config.filters.roic_min,
                )
                log.info("%s outputs written to %s/", combo, RESULTS_DIR)

                port_rets  = result.portfolio_returns()
                bench_rets = result.benchmark_returns()
                port_total = total_return(port_rets)
                bench_total = total_return(bench_rets)
                summary_rows.append({
                    "universe":         universe_id,
                    "holding_months":   holding,
                    "portfolio_return": port_total,
                    "benchmark_return": bench_total,
                    "excess_return":    port_total - bench_total,
                    "sharpe":           sharpe_ratio(port_rets, bt_cfg.risk_free_annual),
                    "max_drawdown":     max_drawdown(port_rets),
                    "hit_rate":         hit_rate(result.hit_rate_returns),
                    "n_months":         len(result.monthly_results),
                })

    finally:
        stop_mem.set()

    if summary_rows:
        _generate_matrix_summary(summary_rows, RESULTS_DIR / "matrix_summary.md")
        print("\n" + "═" * 70)
        print((RESULTS_DIR / "matrix_summary.md").read_text())
    else:
        log.error("No combinations produced results.")


if __name__ == "__main__":
    main()
