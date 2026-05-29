#!/usr/bin/env python3
"""Track 3 (Value Recovery) — SP500 universe, 2013-01-31 → 2024-12-31.

Uses the SP500 snapshot cache if available, otherwise derives it from the
RUSSELL1000 cache by filtering to SP500 tickers.

Outputs
-------
  data/results/track3_SP500_2013_2024_1m_report.md
  data/results/track3_SP500_2013_2024_1m_contributions.md
  data/results/track3_SP500_2013_2024_1m_picks.csv
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
import psutil
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
    generate_picks_csv,
    generate_ticker_contribution,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    total_return,
)
from crucible.config import CrucibleConfig
from crucible.fetcher import _load_cik_mapping, fetch_sp500_tickers
from crucible.snapshot import (
    _CACHE_DIR,
    attach_momentum,
    attach_p_fcf_history,
    build_snapshots_parallel,
)
from crucible.tracks import track3_value

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

RESULTS_DIR   = ROOT / "data" / "results"
EDGAR_DIR     = ROOT / "data" / "raw" / "edgar" / "companyfacts"
CIK_MAP_PATH  = ROOT / "data" / "raw" / "edgar" / "cik_mapping.json"

PRICE_WORKERS    = 20
SNAPSHOT_WORKERS = 4
MEMORY_LOG_SECS  = 600

SP500_CACHE  = _CACHE_DIR / "snapshots_SP500_201301_202412.pkl"
R1000_CACHE  = _CACHE_DIR / "snapshots_RUSSELL1000_201301_202412.pkl"

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
# Snapshot loading
# ---------------------------------------------------------------------------


def _load_or_build_snapshots(
    tickers: list[str],
    use_cache: bool,
) -> dict[pd.Timestamp, pd.DataFrame]:
    if use_cache and SP500_CACHE.exists():
        log.info("Cache HIT — loading SP500 snapshots from %s", SP500_CACHE)
        return joblib.load(SP500_CACHE)

    if use_cache and R1000_CACHE.exists():
        log.info("Deriving SP500 snapshots from RUSSELL1000 cache …")
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

    log.info("No cache found — rebuilding SP500 snapshots from EDGAR …")
    cik_map = _load_cik_mapping(CIK_MAP_PATH)
    monthly_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq="ME", tz="UTC")
    return build_snapshots_parallel(
        tickers=tickers,
        dates=monthly_dates,
        cik_map=cik_map,
        edgar_dir=EDGAR_DIR,
        prices=None,
        workers=SNAPSHOT_WORKERS,
        universe="SP500",
        use_cache=use_cache,
    )


# ---------------------------------------------------------------------------
# Walk-forward loop (Track 3)
# ---------------------------------------------------------------------------


def _run_track3_backtest(
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
            filtered = track3_value.apply_filters(df, config.track3_filters)
        except Exception:
            log.warning("Filter error at %s — skipping", test_date, exc_info=True)
            continue

        if filtered.empty:
            log.debug("No candidates at %s after Track 3 filters", test_date.date())
            continue

        scored = track3_value.score(filtered, config, config.track3_score_weights)
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
# Report generation
# ---------------------------------------------------------------------------


def _generate_report(
    result: BacktestResult,
    output_path: Path,
    config: CrucibleConfig,
) -> None:
    bt = result.bt_config

    port_rets   = result.portfolio_returns()
    bench_rets  = result.benchmark_returns()
    port_total  = total_return(port_rets)
    bench_total = total_return(bench_rets)
    port_sharpe = sharpe_ratio(port_rets, bt.risk_free_annual)
    port_mdd    = max_drawdown(port_rets)
    hr          = hit_rate(result.hit_rate_returns)
    excess      = port_total - bench_total

    all_tickers = {t for m in result.monthly_results for t in m.tickers}
    pick_counts = [m.n_picks for m in result.monthly_results]
    avg_picks   = float(np.mean(pick_counts)) if pick_counts else 0.0

    th = config.track3_filters
    sw = config.track3_score_weights

    def _pct(v: float) -> str:
        return f"{v:.2%}" if not np.isnan(v) else "—"

    def _f2(v: float) -> str:
        return f"{v:.2f}" if not np.isnan(v) else "—"

    if port_total > bench_total:
        verdict = (
            f"Track 3 (Value Recovery) **outperformed** the SP500 benchmark "
            f"({_pct(port_total)} vs {_pct(bench_total)}, excess {_pct(excess)})."
        )
    else:
        verdict = (
            f"Track 3 (Value Recovery) **underperformed** the SP500 benchmark "
            f"({_pct(port_total)} vs {_pct(bench_total)}, excess {_pct(excess)}). "
            "Review filter thresholds, scoring weights, and holding period before drawing conclusions."
        )

    sharpe_note = (
        f"Sharpe of **{_f2(port_sharpe)}** "
        + ("(above 0.5 — risk-adjusted return appears non-trivial)."
           if not np.isnan(port_sharpe) and port_sharpe > 0.5
           else "(below 0.5 — risk-adjusted return is weak).")
    )

    lines: list[str] = [
        "# Track 3 — SP500 Universe — Backtest Report",
        "",
        "**Track:** 3 — Value Recovery  ",
        "**Universe:** SP500 (~503 tickers)  ",
        "**Holding period:** 1 month  ",
        f"**Snapshot window:** {BACKTEST_START.date()} → {BACKTEST_END.date()}  ",
        f"**First test month:** 2015-01-31 (after {bt.train_months}-month warm-up)  ",
        f"**Generated:** {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
        "## Walk-forward Parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Training window | {bt.train_months} months |",
        f"| Portfolio size (top-N) | {bt.top_n} |",
        f"| Holding / rebalance | {bt.holding_months} month |",
        f"| Hit-rate measurement | {bt.hit_rate_months} months |",
        f"| Risk-free rate | {bt.risk_free_annual:.1%} p.a. |",
        f"| Benchmark | {bt.benchmark_col} (SPY) |",
        f"| Snapshot start | {BACKTEST_START.date()} |",
        f"| First test month | 2015-01-31 (month {bt.train_months + 1}) |",
        f"| Last snapshot | {BACKTEST_END.date()} |",
        "",
        "## Filter Thresholds (Layer 1)",
        "",
        "| # | Filter | Condition |",
        "|---|--------|-----------|",
        f"| 1 | ROIC proxy (avg) | > {th.roic_proxy_min:.0%} |",
        f"| 2 | P/FCF vs own 5yr history | (p_fcf_5yr_avg − p_fcf) / p_fcf_5yr_std ≥ {th.p_fcf_vs_history_min:.1f} σ |",
        f"| 3 | FCF positive (last 5yr) | ≥ {th.fcf_positive_min_years} of last 5 years |",
        "| 4 | Recovery signal (any one) | Buyback > 3% OR revenue inflection OR margin recovery |",
        f"|   | → Buyback signal | share_buyback_signal > {th.buyback_signal_min:.0%} |",
        "| | → Revenue inflection | revenue_growth_yr1 > 0 AND revenue_growth_yr2 < 0 |",
        f"| | → Margin recovery | gross_margin_yr1_change > {th.gm_recovery_change_min:.0%} AND trend_slope < 0 |",
        "",
        "## Score Weights (Layer 2)",
        "",
        "| Component | Weight | Sub-components |",
        "|-----------|--------|----------------|",
        f"| Value | {sw.value:.0%} | equal-weight p_fcf rank, ev_ebitda rank (ascending=False) |",
        f"| Recovery signal | {sw.recovery_signal:.0%} | equal-weight buyback rank, gm_yr1_change rank, rev_inflection rank |",
        f"| Balance sheet | {sw.balance_sheet:.0%} | equal-weight net_debt_ebitda rank (asc=False), interest_coverage rank |",
        "",
        "---",
        "",
        "## Performance Summary",
        "",
        "| Metric | Portfolio | Benchmark (SP500) |",
        "|--------|-----------|-------------------|",
        f"| Total return | {_pct(port_total)} | {_pct(bench_total)} |",
        f"| Excess return | {_pct(excess)} | — |",
        f"| Annualised Sharpe | {_f2(port_sharpe)} | — |",
        f"| Maximum drawdown | {_pct(port_mdd)} | — |",
        f"| Hit rate (12m forward) | {_pct(hr)} | — |",
        f"| Avg picks / month | {avg_picks:.1f} | — |",
        f"| Unique tickers ever picked | {len(all_tickers)} | — |",
        f"| Test months with ≥ 1 pick | {len(result.monthly_results)} | — |",
        f"| Hit-rate observations | {len(result.hit_rate_returns)} | — |",
        "",
        "---",
        "",
        "## Conclusion",
        "",
        verdict,
        "",
        f"Hit rate: **{_pct(hr)}** across {len(result.hit_rate_returns)} individual 12-month pick observations.",
        "",
        sharpe_note,
        "",
        "**Regime caveat:** Track 3 (Value Recovery) is contrarian by design. The 2013–2024",
        "window is predominantly a growth-led bull market where mean-reversion strategies",
        "tend to underperform. Value recovery requires patient capital: a company trading",
        "1 std below its own P/FCF history may remain cheap for months before a catalyst",
        "triggers re-rating. The 1-month holding period may be too short to capture the",
        "full value recovery cycle. Compare Track 3 results with a 3-month or 6-month",
        "holding period before drawing conclusions about the strategy's viability.",
        "",
        "---",
        "",
        "> **Data integrity:** Fundamentals from SEC EDGAR (point-in-time, filed date only).",
        "> Prices from yfinance (OHLCV; not used for fundamentals). No look-ahead bias.",
        "> p_fcf_vs_history computed from trailing snapshots — no future data leakage.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Report saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Track 3 SP500 backtest")
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
    log.info("Memory monitor started (logs every %.0f min)", MEMORY_LOG_SECS / 60)

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

        attach_p_fcf_history(fund_by_date)
        log.info("P/FCF history attached to all snapshots")

        log.info("Running Track 3 walk-forward …")
        result = _run_track3_backtest(fund_by_date, prices, config)

        if not result.monthly_results:
            log.error(
                "No test results — check filter thresholds or snapshot coverage. "
                "Run diagnose_funnel.py --universe SP500 --track 3 for details."
            )
            sys.exit(1)

        report_path = RESULTS_DIR / "track3_SP500_2013_2024_1m_report.md"
        _generate_report(result, report_path, config)

        generate_ticker_contribution(
            result,
            RESULTS_DIR / "track3_SP500_2013_2024_1m_contributions.md",
            roic_threshold=config.track3_filters.roic_proxy_min,
        )

        generate_picks_csv(
            result, prices,
            RESULTS_DIR / "track3_SP500_2013_2024_1m_picks.csv",
        )

        port_rets   = result.portfolio_returns()
        bench_rets  = result.benchmark_returns()
        port_total  = total_return(port_rets)
        bench_total = total_return(bench_rets)
        all_tickers = {t for m in result.monthly_results for t in m.tickers}
        avg_n       = float(np.mean([m.n_picks for m in result.monthly_results]))

        print("\n" + "═" * 55)
        print("  Track 3 — SP500 — 2013-01-31 → 2024-12-31")
        print("═" * 55)
        print(f"  Total return:      {port_total:.2%}")
        print(f"  Benchmark (SP500): {bench_total:.2%}")
        print(f"  Excess return:     {port_total - bench_total:+.2%}")
        print(f"  Sharpe (ann.):     {sharpe_ratio(port_rets, RISK_FREE_ANNUAL):.2f}")
        print(f"  Max drawdown:      {max_drawdown(port_rets):.2%}")
        print(f"  Hit rate (12m):    {hit_rate(result.hit_rate_returns):.2%}")
        print(f"  Avg picks/month:   {avg_n:.1f}")
        print(f"  Unique tickers:    {len(all_tickers)}")
        print(f"  Test months:       {len(result.monthly_results)}")
        print("═" * 55)
        print(f"\n  Report:        {report_path}")
        print(f"  Contributions: {RESULTS_DIR}/track3_SP500_2013_2024_1m_contributions.md")
        print(f"  Picks CSV:     {RESULTS_DIR}/track3_SP500_2013_2024_1m_picks.csv")
        print()

    finally:
        stop_mem.set()


if __name__ == "__main__":
    main()
