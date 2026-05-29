#!/usr/bin/env python3
"""Track 2 (Growth Inflection) walk-forward backtest — RUSSELL1000, 1-month holding.

Usage
-----
    python scripts/run_backtest_track2.py

Walk-forward design
-------------------
  Train 24 months, walk forward 1 month at a time.
  Snapshot window: 2013-01-31 → 2024-12-31  (2010-2012 skipped — EDGAR XBRL
  coverage is too sparse: 880/653/456 out of 903 tickers have insufficient data).
  First test month: 2015-01-31  (after 24-month training warm-up).

Outputs
-------
  data/results/track2_SP500_1m_report.md
  data/results/track2_SP500_1m_contributions.md
  data/results/track2_SP500_1m_picks.csv
  data/diagnostics/track2_momentum_impact.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
from crucible.fetcher import _load_cik_mapping, fetch_russell1000_tickers
from crucible.snapshot import attach_momentum, build_snapshots_parallel
from crucible.tracks import track2_growth

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKTEST_START    = pd.Timestamp("2013-01-31", tz="UTC")
BACKTEST_END      = pd.Timestamp("2024-12-31", tz="UTC")
PRICE_FETCH_START = "2012-01-01"   # 13 months before first snapshot — covers momentum_raw
PRICE_FETCH_END   = "2026-06-01"   # covers 12-month hit-rate forward from Dec 2024

TRAIN_MONTHS     = 24
TOP_N            = 20
HOLDING_MONTHS   = 1
HIT_RATE_MONTHS  = 12
RISK_FREE_ANNUAL = 0.04
BENCHMARK_COL    = "SP500"

RESULTS_DIR      = ROOT / "data" / "results"
DIAGNOSTICS_DIR  = ROOT / "data" / "diagnostics"
EDGAR_DIR        = ROOT / "data" / "raw" / "edgar" / "companyfacts"
CIK_MAP_PATH     = ROOT / "data" / "raw" / "edgar" / "cik_mapping.json"

PRICE_WORKERS    = 20
SNAPSHOT_WORKERS = 4
MEMORY_LOG_SECS  = 600

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
# Memory monitor (daemon — auto-stops at process exit)
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
    log.info(
        "Price matrix: %d month-end rows × %d series",
        len(prices), len(prices.columns),
    )
    return prices


# ---------------------------------------------------------------------------
# Walk-forward loop (Track 2)
# ---------------------------------------------------------------------------


def _run_track2_backtest(
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    config: CrucibleConfig,
) -> tuple[BacktestResult, list[dict]]:
    """Walk-forward backtest using Track 2 filters + momentum gate.

    Returns (BacktestResult, momentum_impact_rows).
    momentum_impact_rows tracks per-test-month how many companies pass the 5
    fundamental filters vs how many remain after the momentum gate.
    """
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
    momentum_impact: list[dict] = []

    for i, test_date in enumerate(test_dates, 1):
        if i % 12 == 0:
            log.info("Progress: %d / %d test months", i, len(test_dates))

        df = fund_by_date[test_date]

        # Stage 1 — 5 fundamental filters
        try:
            filtered = track2_growth.apply_filters(df, config.track2_filters)
        except Exception:
            log.warning("Filter error at %s — skipping", test_date, exc_info=True)
            continue

        n_after_fundamental = len(filtered)

        # Stage 2 — Momentum gate (Track 2 requires positive 12-1m momentum)
        mom_mask = filtered["momentum_raw"].notna() & (filtered["momentum_raw"] > 0)
        filtered = filtered[mom_mask]
        n_after_momentum = len(filtered)

        momentum_impact.append({
            "date":                test_date.date(),
            "after_5_filters":     n_after_fundamental,
            "after_momentum":      n_after_momentum,
            "dropped_by_momentum": n_after_fundamental - n_after_momentum,
        })

        if filtered.empty:
            log.debug("No candidates at %s after momentum gate", test_date.date())
            continue

        scored = track2_growth.score(filtered, config, config.track2_score_weights)
        picks  = scored.head(TOP_N).index.tolist()

        # 1-month portfolio and benchmark returns
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

        # 12-month returns for hit rate (one observation per pick per month)
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
    ), momentum_impact


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _generate_report(
    result: BacktestResult,
    momentum_impact: list[dict],
    output_path: Path,
    config: CrucibleConfig,
) -> None:
    """Write Track 2 Markdown backtest report."""
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

    # Momentum impact stats
    mom_df = pd.DataFrame(momentum_impact) if momentum_impact else pd.DataFrame()
    if not mom_df.empty:
        avg_before_mom = float(mom_df["after_5_filters"].mean())
        avg_after_mom  = float(mom_df["after_momentum"].mean())
        avg_dropped    = float(mom_df["dropped_by_momentum"].mean())
        months_zero    = int((mom_df["after_momentum"] == 0).sum())
        pct_drop       = (avg_dropped / avg_before_mom * 100.0) if avg_before_mom > 0 else 0.0
    else:
        avg_before_mom = avg_after_mom = avg_dropped = pct_drop = 0.0
        months_zero = 0

    th = config.track2_filters

    def _pct(v: float) -> str:
        return f"{v:.2%}" if not np.isnan(v) else "—"

    def _f2(v: float) -> str:
        return f"{v:.2f}" if not np.isnan(v) else "—"

    if pct_drop >= 30:
        mom_assessment = (
            f"**SIGNIFICANT cut** — momentum eliminates {pct_drop:.0f}% of post-fundamental "
            "candidates on average. The pool is meaningfully thinned; recovery-phase companies "
            "with strong fundamentals but lagging price may be excluded."
        )
    elif pct_drop >= 10:
        mom_assessment = (
            f"**MODERATE effect** — {pct_drop:.0f}% reduction. Adds useful trend confirmation "
            "without excessively cutting the investable pool."
        )
    else:
        mom_assessment = (
            f"**MINOR effect** — {pct_drop:.0f}% reduction. Most companies passing the 5 "
            "fundamental filters already carry positive momentum; the gate is nearly redundant."
        )

    if port_total > bench_total:
        verdict = (
            f"Track 2 (Growth Inflection) **outperformed** the SP500 benchmark "
            f"({_pct(port_total)} vs {_pct(bench_total)}, excess {_pct(excess)})."
        )
    else:
        verdict = (
            f"Track 2 (Growth Inflection) **underperformed** the SP500 benchmark "
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
        "# Crucible Track 2 Backtest Report",
        "",
        "**Track:** 2 — Growth Inflection  ",
        "**Universe:** RUSSELL1000  ",
        "**Holding period:** 1 month  ",
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
        "## Filter Thresholds",
        "",
        "| # | Filter | Condition |",
        "|---|--------|-----------|",
        f"| 1 | Revenue growth | yr1 and yr2 both > {th.revenue_growth_min_pct:.0%} |",
        f"| 2 | Revenue acceleration | YoY growth rate increasing (> 0) |",
        f"| 3 | Gross margin | ≥ {th.gross_margin_min:.0%} OR expanding vs prior year |",
        f"| 4 | FCF positive | ≥ {th.fcf_positive_last2yr_min} of last 2 years |",
        f"| 5 | Leverage (soft) | Net Debt/EBITDA < {th.net_debt_ebitda_soft_max:.1f} OR fcf_trajectory > 0 |",
        f"| 6 | Momentum gate | 12-1m price momentum > 0 (applied after fundamentals) |",
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
        "## Momentum Filter Impact",
        "",
        "The momentum gate is applied **after** the 5 fundamental filters,",
        "so its cost is measured relative to that post-fundamental pool.",
        "",
        "| Stage | Avg candidates across test months |",
        "|-------|-----------------------------------|",
        f"| After 5 fundamental filters | {avg_before_mom:.1f} |",
        f"| After momentum gate | {avg_after_mom:.1f} |",
        f"| Dropped by momentum | {avg_dropped:.1f} ({pct_drop:.0f}% of post-fundamental pool) |",
        f"| Test months with zero candidates post-momentum | {months_zero} |",
        "",
        f"**Assessment:** {mom_assessment}",
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
        "**Regime caveat:** The 2013–2024 window includes a strong growth-factor cycle",
        "(2013–2021) and a sharp reversal (2022). Track 2 targets companies with",
        "accelerating revenue and expanding margins, which cluster in Technology and",
        "Healthcare. Results are sensitive to the growth-vs-value factor regime and",
        "should not be extrapolated naively into a value-dominant environment.",
        "",
        "---",
        "",
        "> **Data integrity:** Fundamentals from SEC EDGAR (point-in-time, filed date only).",
        "> Prices from yfinance (OHLCV; not used for fundamentals). No look-ahead bias.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Report saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Track 2 walk-forward backtest")
    parser.add_argument(
        "--no-cache", dest="no_cache", action="store_true",
        help="Force rebuild — skip loading cached snapshots",
    )
    args = parser.parse_args()

    for path, label in (
        (CIK_MAP_PATH, "CIK mapping"),
        (EDGAR_DIR,    "EDGAR companyfacts directory"),
    ):
        if not path.exists():
            log.error(
                "%s not found at %s. Run scripts/download_edgar_bulk.py first.",
                label, path,
            )
            sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    config  = CrucibleConfig(account_currency="USD")
    cik_map = _load_cik_mapping(CIK_MAP_PATH)

    stop_mem = threading.Event()
    threading.Thread(
        target=_memory_monitor, args=(stop_mem,), daemon=True, name="mem-monitor"
    ).start()
    log.info("Memory monitor started (logs every %.0f min)", MEMORY_LOG_SECS / 60)

    try:
        # ── Tickers ──────────────────────────────────────────────────────────
        log.info("Fetching Russell 1000 tickers …")
        tickers = fetch_russell1000_tickers()
        log.info("%d tickers in universe", len(tickers))

        # ── Prices ───────────────────────────────────────────────────────────
        log.info(
            "Fetching prices %s → %s (%d workers) …",
            PRICE_FETCH_START, PRICE_FETCH_END, PRICE_WORKERS,
        )
        prices = _fetch_prices_parallel(tickers, PRICE_FETCH_START, PRICE_FETCH_END)
        if prices.empty:
            log.error("No price data — aborting")
            sys.exit(1)

        # ── Fundamental snapshots ─────────────────────────────────────────────
        monthly_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq="ME", tz="UTC")
        log.info(
            "Building %d monthly EDGAR snapshots (%d workers) …",
            len(monthly_dates), SNAPSHOT_WORKERS,
        )
        fund_by_date = build_snapshots_parallel(
            tickers=tickers,
            dates=monthly_dates,
            cik_map=cik_map,
            edgar_dir=EDGAR_DIR,
            prices=prices,
            workers=SNAPSHOT_WORKERS,
            universe="RUSSELL1000",
            use_cache=not args.no_cache,
        )
        log.info("Snapshots complete: %d dates", len(fund_by_date))

        attach_momentum(fund_by_date, prices)
        log.info("Momentum attached to all snapshots")

        # ── Walk-forward ──────────────────────────────────────────────────────
        log.info("Running Track 2 walk-forward backtest …")
        result, momentum_impact = _run_track2_backtest(fund_by_date, prices, config)

        if not result.monthly_results:
            log.error(
                "No test results — check filter thresholds or snapshot coverage. "
                "Run diagnose_funnel.py --universe RUSSELL1000 --track 2 for details."
            )
            sys.exit(1)

        # ── Save outputs ──────────────────────────────────────────────────────
        mom_csv = DIAGNOSTICS_DIR / "track2_momentum_impact.csv"
        pd.DataFrame(momentum_impact).to_csv(mom_csv, index=False)
        log.info("Momentum impact CSV → %s", mom_csv)

        _generate_report(
            result, momentum_impact,
            RESULTS_DIR / "track2_SP500_1m_report.md",
            config,
        )

        # generate_ticker_contribution expects a roic_threshold for its header label;
        # Track 2 has no ROIC filter, so 0.0 is passed — the bar chart and table are correct.
        generate_ticker_contribution(
            result,
            RESULTS_DIR / "track2_SP500_1m_contributions.md",
            roic_threshold=0.0,
        )

        generate_picks_csv(
            result, prices,
            RESULTS_DIR / "track2_SP500_1m_picks.csv",
        )

        # ── Console summary ───────────────────────────────────────────────────
        port_rets  = result.portfolio_returns()
        bench_rets = result.benchmark_returns()
        port_total = total_return(port_rets)
        bench_total = total_return(bench_rets)
        all_tickers = {t for m in result.monthly_results for t in m.tickers}
        avg_n = np.mean([m.n_picks for m in result.monthly_results])

        print("\n" + "═" * 65)
        print("  Track 2 Backtest — RUSSELL1000 — 1-month holding")
        print("═" * 65)
        print(f"  Total return:      {port_total:.2%}")
        print(f"  Benchmark (SP500): {bench_total:.2%}")
        print(f"  Excess return:     {port_total - bench_total:+.2%}")
        print(f"  Sharpe (ann.):     {sharpe_ratio(port_rets, RISK_FREE_ANNUAL):.2f}")
        print(f"  Max drawdown:      {max_drawdown(port_rets):.2%}")
        print(f"  Hit rate (12m):    {hit_rate(result.hit_rate_returns):.2%}")
        print(f"  Avg picks/month:   {avg_n:.1f}")
        print(f"  Unique tickers:    {len(all_tickers)}")
        print(f"  Test months:       {len(result.monthly_results)}")
        print()

        if momentum_impact:
            mom_df = pd.DataFrame(momentum_impact)
            avg_b  = float(mom_df["after_5_filters"].mean())
            avg_a  = float(mom_df["after_momentum"].mean())
            pct_d  = (avg_b - avg_a) / avg_b * 100.0 if avg_b > 0 else 0.0
            zeros  = int((mom_df["after_momentum"] == 0).sum())
            print(f"  Momentum filter:   {avg_b:.1f} → {avg_a:.1f} avg candidates ({pct_d:.0f}% cut, {zeros} months with 0 left)")

        print("═" * 65)
        print(f"\n  Report:        {RESULTS_DIR}/track2_SP500_1m_report.md")
        print(f"  Contributions: {RESULTS_DIR}/track2_SP500_1m_contributions.md")
        print(f"  Picks CSV:     {RESULTS_DIR}/track2_SP500_1m_picks.csv")
        print(f"  Momentum CSV:  {DIAGNOSTICS_DIR}/track2_momentum_impact.csv")
        print()

    finally:
        stop_mem.set()


if __name__ == "__main__":
    main()
