#!/usr/bin/env python3
"""Track 1 (Quality Compounders) — SP500 universe, 2013-01-31 → 2024-12-31.

Derives the SP500 snapshot cache from the RUSSELL1000 cache by filtering tickers,
so no EDGAR rebuild is needed when the RUSSELL1000 cache is available.

Outputs
-------
  data/results/track1_SP500_2013_2024_1m_report.md
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from crucible.backtest import (
    BacktestConfig,
    BacktestResult,
    MonthlyResult,
    _advance,
    _benchmark_return,
    _single_return,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    total_return,
)
from crucible.config import CrucibleConfig
from crucible.fetcher import _load_cik_mapping, fetch_sp500_tickers
from crucible.snapshot import _CACHE_DIR, attach_momentum, build_snapshots_parallel
from crucible.tracks import track1_quality

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKTEST_START    = pd.Timestamp("2013-01-31", tz="UTC")
BACKTEST_END      = pd.Timestamp("2024-12-31", tz="UTC")
PRICE_FETCH_START = "2012-01-01"
PRICE_FETCH_END   = "2026-06-01"

TRAIN_MONTHS     = 24
TOP_N            = 20
HOLDING_MONTHS   = 1
HIT_RATE_MONTHS  = 12
RISK_FREE_ANNUAL = 0.04
BENCHMARK_COL    = "SP500"

RESULTS_DIR      = ROOT / "data" / "results"
EDGAR_DIR        = ROOT / "data" / "raw" / "edgar" / "companyfacts"
CIK_MAP_PATH     = ROOT / "data" / "raw" / "edgar" / "cik_mapping.json"

PRICE_WORKERS    = 20
SNAPSHOT_WORKERS = 4
MEMORY_LOG_SECS  = 600

SP500_CACHE      = _CACHE_DIR / "snapshots_SP500_201301_202412.pkl"
R1000_CACHE      = _CACHE_DIR / "snapshots_RUSSELL1000_201301_202412.pkl"

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
# Memory monitor
# ---------------------------------------------------------------------------


def _memory_monitor(stop_event: threading.Event) -> None:
    import psutil
    proc = psutil.Process()
    while not stop_event.wait(MEMORY_LOG_SECS):
        log.info("[mem] RSS %.0f MiB", proc.memory_info().rss / 1024 / 1024)


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


def _fetch_prices_parallel(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    all_tickers = list(tickers) + ["SPY"]
    series_map: dict[str, pd.Series] = {}
    done = 0
    total = len(all_tickers)

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
    log.info("Price matrix: %d rows × %d series", len(prices), len(prices.columns))
    return prices


# ---------------------------------------------------------------------------
# Snapshot loading — SP500 from RUSSELL1000 cache or fresh build
# ---------------------------------------------------------------------------


def _load_or_build_snapshots(
    tickers: list[str],
    use_cache: bool,
) -> dict[pd.Timestamp, pd.DataFrame]:
    if use_cache and SP500_CACHE.exists():
        log.info("Cache HIT — loading SP500 snapshots from %s", SP500_CACHE)
        return joblib.load(SP500_CACHE)

    # Derive from RUSSELL1000 cache if available — avoids full EDGAR rebuild
    if use_cache and R1000_CACHE.exists():
        log.info("Deriving SP500 snapshots from RUSSELL1000 cache (filtering tickers) …")
        r1000 = joblib.load(R1000_CACHE)
        sp500_set = set(tickers)
        fund_by_date: dict[pd.Timestamp, pd.DataFrame] = {
            date: df.loc[df.index.intersection(sp500_set)]
            for date, df in r1000.items()
        }
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(fund_by_date, SP500_CACHE, compress=3)
        log.info("SP500 snapshot cache saved → %s", SP500_CACHE)
        return fund_by_date

    # Full rebuild from EDGAR
    log.info("No cache found — rebuilding SP500 snapshots from EDGAR …")
    cik_map = _load_cik_mapping(CIK_MAP_PATH)
    monthly_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq="ME", tz="UTC")
    return build_snapshots_parallel(
        tickers=tickers,
        dates=monthly_dates,
        cik_map=cik_map,
        edgar_dir=EDGAR_DIR,
        prices=None,   # multiples skipped; track1 doesn't require them for filtering
        workers=SNAPSHOT_WORKERS,
        universe="SP500",
        use_cache=use_cache,
    )


# ---------------------------------------------------------------------------
# Walk-forward (Track 1)
# ---------------------------------------------------------------------------


def _run_backtest(
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    config: CrucibleConfig,
) -> BacktestResult:
    bt_config = BacktestConfig(
        train_months=TRAIN_MONTHS,
        top_n=TOP_N,
        holding_months=HOLDING_MONTHS,
        hit_rate_months=HIT_RATE_MONTHS,
        risk_free_annual=RISK_FREE_ANNUAL,
        benchmark_col=BENCHMARK_COL,
    )

    dates      = sorted(fund_by_date.keys())
    price_idx  = prices.index
    test_dates = dates[TRAIN_MONTHS::HOLDING_MONTHS]

    log.info(
        "Walk-forward: %d training months, %d test dates  (%s → %s)",
        TRAIN_MONTHS, len(test_dates),
        test_dates[0].date() if test_dates else "—",
        test_dates[-1].date() if test_dates else "—",
    )

    monthly_results: list[MonthlyResult] = []
    hit_rate_returns: list[float] = []

    for i, test_date in enumerate(test_dates, 1):
        if i % 12 == 0:
            log.info("Progress: %d / %d test months", i, len(test_dates))

        df = fund_by_date[test_date]
        try:
            filtered = track1_quality.apply_filters(df, config.filters)
        except Exception:
            log.warning("Filter error at %s — skipping", test_date, exc_info=True)
            continue

        if filtered.empty:
            continue

        scored = track1_quality.score(filtered, config)
        picks  = scored.head(TOP_N).index.tolist()

        next_month = _advance(test_date, price_idx, HOLDING_MONTHS)
        if next_month is not None and test_date in price_idx:
            tkr_rets = {
                t: r
                for t in picks
                for r in (_single_return(t, test_date, next_month, prices),)
                if r is not None
            }
            port_ret  = float(np.mean(list(tkr_rets.values()))) if tkr_rets else 0.0
            bench_ret = _benchmark_return(test_date, next_month, prices, BENCHMARK_COL)
            monthly_results.append(MonthlyResult(
                date=test_date,
                portfolio_return=port_ret,
                benchmark_return=bench_ret,
                n_picks=len(picks),
                tickers=picks,
                ticker_returns=tkr_rets,
            ))

        hit_date = _advance(test_date, price_idx, HIT_RATE_MONTHS)
        if hit_date is not None and test_date in price_idx:
            for ticker in picks:
                r = _single_return(ticker, test_date, hit_date, prices)
                if r is not None:
                    hit_rate_returns.append(r)

    log.info(
        "Backtest complete: %d test months with picks, %d hit-rate observations",
        len(monthly_results), len(hit_rate_returns),
    )
    return BacktestResult(
        monthly_results=monthly_results,
        hit_rate_returns=hit_rate_returns,
        bt_config=bt_config,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _generate_report(result: BacktestResult, output_path: Path) -> None:
    bt = result.bt_config

    port_rets   = result.portfolio_returns()
    bench_rets  = result.benchmark_returns()
    port_total  = total_return(port_rets)
    bench_total = total_return(bench_rets)
    port_sharpe = sharpe_ratio(port_rets, bt.risk_free_annual)
    hr          = hit_rate(result.hit_rate_returns)
    excess      = port_total - bench_total

    def _pct(v: float) -> str:
        return f"{v:.2%}" if not np.isnan(v) else "—"

    def _f2(v: float) -> str:
        return f"{v:.2f}" if not np.isnan(v) else "—"

    lines: list[str] = [
        "# Track 1 — SP500 Universe — Backtest Report",
        "",
        "**Track:** 1 — Quality Compounders  ",
        "**Universe:** SP500 (~503 tickers)  ",
        "**Holding period:** 1 month  ",
        f"**Snapshot window:** {BACKTEST_START.date()} → {BACKTEST_END.date()}  ",
        f"**First test month:** 2015-01-31 (after {bt.train_months}-month warm-up)  ",
        f"**Generated:** {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
        "## Performance Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total return (portfolio) | {_pct(port_total)} |",
        f"| Benchmark return (SP500) | {_pct(bench_total)} |",
        f"| Excess return | {_pct(excess)} |",
        f"| Annualised Sharpe | {_f2(port_sharpe)} |",
        f"| Hit rate (12m forward) | {_pct(hr)} |",
        f"| Test months with ≥ 1 pick | {len(result.monthly_results)} |",
        f"| Hit-rate observations | {len(result.hit_rate_returns)} |",
        "",
        "---",
        "",
        "> Fundamentals: SEC EDGAR (point-in-time). Prices: yfinance (OHLCV only). "
        "No look-ahead bias.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Report saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Track 1 SP500 backtest")
    parser.add_argument(
        "--no-cache", dest="no_cache", action="store_true",
        help="Force rebuild — skip all cached snapshots",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = CrucibleConfig(account_currency="USD")

    stop_mem = threading.Event()
    threading.Thread(
        target=_memory_monitor, args=(stop_mem,), daemon=True, name="mem-monitor"
    ).start()

    try:
        log.info("Fetching SP500 tickers …")
        tickers = fetch_sp500_tickers()
        log.info("%d tickers in SP500 universe", len(tickers))

        log.info("Fetching prices %s → %s …", PRICE_FETCH_START, PRICE_FETCH_END)
        prices = _fetch_prices_parallel(tickers, PRICE_FETCH_START, PRICE_FETCH_END)
        if prices.empty:
            log.error("No price data — aborting")
            sys.exit(1)

        fund_by_date = _load_or_build_snapshots(tickers, use_cache=not args.no_cache)
        log.info("Snapshots ready: %d dates", len(fund_by_date))

        attach_momentum(fund_by_date, prices)
        log.info("Momentum attached")

        log.info("Running Track 1 walk-forward …")
        result = _run_backtest(fund_by_date, prices, config)

        if not result.monthly_results:
            log.error("No test results — check filter thresholds or snapshot coverage")
            sys.exit(1)

        report_path = RESULTS_DIR / "track1_SP500_2013_2024_1m_report.md"
        _generate_report(result, report_path)

        port_rets   = result.portfolio_returns()
        bench_rets  = result.benchmark_returns()
        port_total  = total_return(port_rets)
        bench_total = total_return(bench_rets)

        print("\n" + "═" * 55)
        print("  Track 1 — SP500 — 2013-01-31 → 2024-12-31")
        print("═" * 55)
        print(f"  Total return:      {port_total:.2%}")
        print(f"  Benchmark (SP500): {bench_total:.2%}")
        print(f"  Excess return:     {port_total - bench_total:+.2%}")
        print(f"  Sharpe (ann.):     {sharpe_ratio(port_rets, RISK_FREE_ANNUAL):.2f}")
        print(f"  Hit rate (12m):    {hit_rate(result.hit_rate_returns):.2%}")
        print("═" * 55)
        print(f"\n  Report: {report_path}")
        print()

    finally:
        stop_mem.set()


if __name__ == "__main__":
    main()
