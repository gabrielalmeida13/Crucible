#!/usr/bin/env python3
"""Held-out validation — Track 2 (Growth Inflection) v2, SP500, 2025-01-31 → 2026-05-31.

Phase 4.7 features active in scorer:
  - asset_growth_yoy penalty      (Fama-French CMA proxy)
  - deferred_revenue_growth       (book-to-bill proxy)
  - eps_surprise_last_q           (earnings beat strength, 10% weight)

Baseline for comparison (v1, without eps_surprise_last_q):
  Total return: 40.17%  |  Excess: +14.48%  |  Sharpe: 1.11  |  Hit rate: 50.82%

Outputs
-------
  data/results/heldout_track2_SP500_v2_report.md
  data/results/heldout_track2_SP500_v2_contributions.md

Usage
-----
  python scripts/run_heldout_track2_sp500_v2.py [--no-cache]
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
    build_snapshots_parallel,
)
from crucible.tracks import track2_growth

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HELDOUT_START     = pd.Timestamp("2025-01-31", tz="UTC")
HELDOUT_END       = pd.Timestamp("2026-05-31", tz="UTC")
PRICE_FETCH_START = "2024-01-01"
PRICE_FETCH_END   = "2026-06-15"

TRAIN_MONTHS     = 0
TOP_N            = 20
HOLDING_MONTHS   = 1
HIT_RATE_MONTHS  = 12
RISK_FREE_ANNUAL = 0.04
BENCHMARK_COL    = "SP500"

# Baseline from v1 (asset_growth_yoy + deferred_revenue_growth, no eps_surprise)
BASELINE_V1 = dict(
    total_return=0.4017,
    excess=0.1448,
    sharpe=1.11,
    hit_rate=0.5082,
    benchmark=0.2569,
)

RESULTS_DIR      = ROOT / "data" / "results"
EDGAR_DIR        = ROOT / "data" / "raw" / "edgar" / "companyfacts"
CIK_MAP_PATH     = ROOT / "data" / "raw" / "edgar" / "cik_mapping.json"

PRICE_WORKERS    = 20
SNAPSHOT_WORKERS = 4
MEMORY_LOG_SECS  = 600

# Reuse the same heldout snapshot cache as run_heldout_three_tracks.py
HELDOUT_CACHE = _CACHE_DIR / "snapshots_SP500_202501_202605.pkl"

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


def _fetch_prices_parallel(tickers: list[str]) -> pd.DataFrame:
    all_tickers = list(tickers) + ["SPY"]
    series_map: dict[str, pd.Series] = {}
    done = 0
    total = len(all_tickers)

    with ThreadPoolExecutor(max_workers=PRICE_WORKERS) as pool:
        futures = {pool.submit(_fetch_one_price, t, PRICE_FETCH_START, PRICE_FETCH_END): t
                   for t in all_tickers}
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
# Walk-forward loop
# ---------------------------------------------------------------------------


def _run_walkforward(
    heldout_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    pick_fn,
) -> BacktestResult:
    bt_config = BacktestConfig(
        train_months=TRAIN_MONTHS,
        top_n=TOP_N,
        holding_months=HOLDING_MONTHS,
        hit_rate_months=HIT_RATE_MONTHS,
        risk_free_annual=RISK_FREE_ANNUAL,
        benchmark_col=BENCHMARK_COL,
    )

    dates      = sorted(heldout_by_date.keys())
    price_idx  = prices.index
    test_dates = dates[TRAIN_MONTHS:] if TRAIN_MONTHS == 0 else dates[TRAIN_MONTHS::HOLDING_MONTHS]

    log.info(
        "Walk-forward: %d test dates  (%s → %s)",
        len(test_dates),
        test_dates[0].date() if test_dates else "—",
        test_dates[-1].date() if test_dates else "—",
    )

    monthly_results: list[MonthlyResult] = []
    hit_rate_returns: list[float] = []

    for test_date in test_dates:
        df = heldout_by_date[test_date]
        try:
            scored = pick_fn(df)
        except Exception:
            log.warning("Error at %s — skipping", test_date, exc_info=True)
            continue

        if scored.empty:
            continue

        picks = scored.head(TOP_N).index.tolist()

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
        "Walk-forward done: %d months with picks, %d hit-rate observations",
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


def _write_report(result: BacktestResult, output_path: Path) -> None:
    port_rets   = result.portfolio_returns()
    bench_rets  = result.benchmark_returns()
    port_total  = total_return(port_rets)
    bench_total = total_return(bench_rets)
    port_sharpe = sharpe_ratio(port_rets, RISK_FREE_ANNUAL)
    port_mdd    = max_drawdown(port_rets)
    hr          = hit_rate(result.hit_rate_returns)
    excess      = port_total - bench_total

    all_tickers = {t for m in result.monthly_results for t in m.tickers}
    pick_counts = [m.n_picks for m in result.monthly_results]
    avg_picks   = float(np.mean(pick_counts)) if pick_counts else 0.0
    n_months    = len(result.monthly_results)

    def _pct(v: float) -> str:
        return f"{v:.2%}" if not np.isnan(v) else "—"

    def _f2(v: float) -> str:
        return f"{v:.2f}" if not np.isnan(v) else "—"

    def _delta(v2: float, v1: float) -> str:
        d = v2 - v1
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.2%}"

    b = BASELINE_V1

    if not result.monthly_results:
        conclusion = (
            "*No test months produced picks — check EDGAR coverage for 2025-2026 "
            "or adjust filter thresholds for the held-out period.*"
        )
    elif port_total > bench_total and (np.isnan(hr) or hr >= 0.5):
        conclusion = (
            f"Track 2 v2 outperformed the benchmark by **{_pct(excess)}** "
            f"over the held-out period. Hit rate: **{_pct(hr)}**. "
            f"The addition of `eps_surprise_last_q` "
            f"({'improved' if port_total > b['total_return'] else 'did not improve'} "
            f"total return vs v1 baseline of {_pct(b['total_return'])})."
        )
    elif port_total > bench_total:
        conclusion = (
            f"Track 2 v2 outperformed the benchmark by **{_pct(excess)}** on "
            f"a total-return basis, but the hit rate of **{_pct(hr)}** is below 50%. "
            "Positive total return may be driven by a small number of large winners."
        )
    else:
        conclusion = (
            f"Track 2 v2 underperformed the benchmark by **{_pct(abs(excess))}** "
            f"over the held-out period (hit rate: **{_pct(hr)}**). "
            "The 2025-2026 market regime may differ from 2013-2024 training conditions. "
            "Review sector concentration and filter passage rates before drawing conclusions."
        )

    lines: list[str] = [
        "# Held-Out Validation — Track 2 v2 (Growth Inflection + EPS Surprise) — SP500",
        "",
        "> **HELD-OUT VALIDATION.** Parameters frozen at end of 2013-2024 backtest.",
        "> Do not re-tune after reading these results.",
        "",
        "**Track:** 2 v2 — Growth Inflection (Phase 4.7 features active)  ",
        "**Universe:** SP500 (~503 tickers)  ",
        "**Holding period:** 1 month  ",
        f"**Test window:** {HELDOUT_START.date()} → {HELDOUT_END.date()}  ",
        "**Burn-in:** none (TRAIN_MONTHS=0 — every month is a test point)  ",
        f"**Generated:** {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "**Phase 4.7 scorer changes vs v1:**",
        "- `revenue_acceleration` weight: 20% → 10%",
        "- `eps_surprise_last_q` added: 10% weight (earnings beat strength)",
        "- `deferred_revenue_growth` (10%) and `asset_growth_yoy` penalty (−10%): unchanged",
        "",
        "---",
        "",
        "## Performance Summary",
        "",
        "| Metric | Portfolio v2 | Benchmark (SP500) | v1 Baseline | Δ vs v1 |",
        "|--------|-------------|-------------------|-------------|---------|",
        f"| Total return | {_pct(port_total)} | {_pct(bench_total)} | {_pct(b['total_return'])} | {_delta(port_total, b['total_return'])} |",
        f"| Excess return | {_pct(excess)} | — | {_pct(b['excess'])} | {_delta(excess, b['excess'])} |",
        f"| Annualised Sharpe | {_f2(port_sharpe)} | — | {b['sharpe']:.2f} | {_delta(port_sharpe, b['sharpe'])} |",
        f"| Maximum drawdown | {_pct(port_mdd)} | — | — | — |",
        f"| Hit rate (12m forward) | {_pct(hr)} | — | {_pct(b['hit_rate'])} | {_delta(hr if not np.isnan(hr) else 0, b['hit_rate'])} |",
        f"| Avg picks / month | {avg_picks:.1f} | — | — | — |",
        f"| Unique tickers picked | {len(all_tickers)} | — | — | — |",
        f"| Test months with ≥ 1 pick | {n_months} | — | — | — |",
        f"| Hit-rate observations | {len(result.hit_rate_returns)} | — | — | — |",
        "",
        "---",
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        f"*Hit rate covers {len(result.hit_rate_returns)} observations where a "
        f"12-month forward price was available. Months from mid-2025 onwards may "
        f"have partial or no 12m forward coverage given the evaluation date.*",
        "",
        "---",
        "",
        "> **Data integrity:** Fundamentals from SEC EDGAR (point-in-time, `filed` ≤ snapshot date).",
        "> Prices from yfinance (OHLCV only). Heldout window not seen during backtest development.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Report saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Track 2 v2 SP500 held-out (Phase 4.7 features)")
    parser.add_argument(
        "--no-cache", dest="no_cache", action="store_true",
        help="Force rebuild of heldout snapshots from EDGAR",
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
    config  = CrucibleConfig(account_currency="USD")
    cik_map = _load_cik_mapping(CIK_MAP_PATH)

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
        prices = _fetch_prices_parallel(tickers)
        if prices.empty:
            log.error("No price data — aborting")
            sys.exit(1)

        heldout_dates = pd.date_range(HELDOUT_START, HELDOUT_END, freq="ME", tz="UTC")
        log.info("Building %d heldout EDGAR snapshots …", len(heldout_dates))
        heldout_by_date = build_snapshots_parallel(
            tickers=tickers,
            dates=heldout_dates,
            cik_map=cik_map,
            edgar_dir=EDGAR_DIR,
            prices=prices,
            workers=SNAPSHOT_WORKERS,
            universe="SP500",
            use_cache=not args.no_cache,
        )
        log.info("Heldout snapshots ready: %d dates", len(heldout_by_date))

        attach_momentum(heldout_by_date, prices)
        log.info("Momentum attached to heldout snapshots")

        def _t2_pick_fn(df: pd.DataFrame) -> pd.DataFrame:
            filtered = track2_growth.apply_filters(df, config.track2_filters)
            if filtered.empty:
                return filtered
            mom_mask = filtered["momentum_raw"].notna() & (filtered["momentum_raw"] > 0)
            filtered = filtered[mom_mask]
            if filtered.empty:
                return filtered
            return track2_growth.score(filtered, config, config.track2_score_weights)

        result   = _run_walkforward(heldout_by_date, prices, _t2_pick_fn)
        report   = RESULTS_DIR / "heldout_track2_SP500_v2_report.md"
        contrib  = RESULTS_DIR / "heldout_track2_SP500_v2_contributions.md"

        if not result.monthly_results:
            log.warning("Track 2 v2: no test results — check filter thresholds or EDGAR coverage")
        else:
            _write_report(result, report)
            generate_ticker_contribution(result, contrib, roic_threshold=0.0)

        port_rets   = result.portfolio_returns()
        bench_rets  = result.benchmark_returns()
        port_total  = total_return(port_rets)
        bench_total = total_return(bench_rets)
        b = BASELINE_V1

        print("\n" + "═" * 65)
        print("  Track 2 v2 — Growth Inflection + EPS Surprise — SP500 Held-Out")
        print(f"  {HELDOUT_START.date()} → {HELDOUT_END.date()}")
        print("═" * 65)
        print(f"  Total return:      {port_total:.2%}  (v1: {b['total_return']:.2%}  Δ {port_total - b['total_return']:+.2%})")
        print(f"  Benchmark (SP500): {bench_total:.2%}")
        print(f"  Excess return:     {port_total - bench_total:+.2%}  (v1: {b['excess']:.2%}  Δ {(port_total - bench_total) - b['excess']:+.2%})")
        print(f"  Sharpe (ann.):     {sharpe_ratio(port_rets, RISK_FREE_ANNUAL):.2f}  (v1: {b['sharpe']:.2f})")
        print(f"  Max drawdown:      {max_drawdown(port_rets):.2%}")
        print(f"  Hit rate (12m):    {hit_rate(result.hit_rate_returns):.2%}  (v1: {b['hit_rate']:.2%})")
        print(f"  Avg picks/month:   {float(np.mean([m.n_picks for m in result.monthly_results])):.1f}" if result.monthly_results else "  Avg picks/month:   —")
        print(f"  Unique tickers:    {len({t for m in result.monthly_results for t in m.tickers})}")
        print(f"  Test months:       {len(result.monthly_results)}")
        print("═" * 65)
        print(f"  Report:      {report}")
        print(f"  Contribs:    {contrib}")
        print()

    finally:
        stop_mem.set()


if __name__ == "__main__":
    main()
