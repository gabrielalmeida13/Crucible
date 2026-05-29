#!/usr/bin/env python3
"""Held-out validation — all three tracks, SP500, 2025-01-31 → 2026-05-31.

Design
------
  TRAIN_MONTHS = 0: parameters were frozen at end of 2013-2024 backtest.
  Every month in the heldout window is a live test point.

  Heldout snapshots are built from EDGAR and cached to
  data/cache/snapshots_SP500_202501_202605.pkl on first run.

  Track 3 requires p_fcf_vs_history, which needs up to 60 months of prior
  P/FCF data.  The 2013-2024 training cache is loaded read-only and merged
  with the heldout snapshots before calling attach_p_fcf_history — no
  data leakage because the walk-forward only evaluates heldout dates.

  Hit rate (12m forward): picks from 2025-01 → 2026-05 need prices through
  2027-05.  Only months where a 12m forward price exists are included.
  Expect partial coverage for months from mid-2025 onwards.

Outputs
-------
  data/results/heldout_track1_SP500_1m_report.md
  data/results/heldout_track2_SP500_1m_report.md
  data/results/heldout_track3_SP500_1m_report.md

Usage
-----
  python scripts/run_heldout_three_tracks.py [--no-cache]
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
from crucible.tracks import track1_quality, track2_growth, track3_value

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HELDOUT_START     = pd.Timestamp("2025-01-31", tz="UTC")
HELDOUT_END       = pd.Timestamp("2026-05-31", tz="UTC")
# 13 months before first snapshot covers the 12-1m momentum lookback
PRICE_FETCH_START = "2024-01-01"
PRICE_FETCH_END   = "2026-06-15"

TRAIN_MONTHS     = 0    # parameters pre-specified — every month is a test point
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

TRAINING_CACHE = _CACHE_DIR / "snapshots_SP500_201301_202412.pkl"

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
# Walk-forward loop (generic — accepts a filter+score callable)
# ---------------------------------------------------------------------------


def _run_walkforward(
    heldout_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    pick_fn,  # callable(df) -> scored DataFrame sorted by composite_score
) -> BacktestResult:
    """Generic walk-forward evaluator for any track's pick_fn."""
    bt_config = BacktestConfig(
        train_months=TRAIN_MONTHS,
        top_n=TOP_N,
        holding_months=HOLDING_MONTHS,
        hit_rate_months=HIT_RATE_MONTHS,
        risk_free_annual=RISK_FREE_ANNUAL,
        benchmark_col=BENCHMARK_COL,
    )

    dates     = sorted(heldout_by_date.keys())
    price_idx = prices.index
    # TRAIN_MONTHS=0 → all dates are test dates
    test_dates = dates[TRAIN_MONTHS:] if TRAIN_MONTHS == 0 else dates[TRAIN_MONTHS::HOLDING_MONTHS]

    log.info(
        "Walk-forward: %d test dates  (%s → %s)",
        len(test_dates),
        test_dates[0].date() if test_dates else "—",
        test_dates[-1].date() if test_dates else "—",
    )

    monthly_results: list[MonthlyResult] = []
    hit_rate_returns: list[float] = []

    for i, test_date in enumerate(test_dates, 1):
        df = heldout_by_date[test_date]

        try:
            scored = pick_fn(df)
        except Exception:
            log.warning("Error at %s — skipping", test_date, exc_info=True)
            continue

        if scored.empty:
            log.debug("No candidates at %s", test_date.date())
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


def _write_report(
    result: BacktestResult,
    output_path: Path,
    track_num: int,
    track_name: str,
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
    n_months    = len(result.monthly_results)

    def _pct(v: float) -> str:
        return f"{v:.2%}" if not np.isnan(v) else "—"

    def _f2(v: float) -> str:
        return f"{v:.2f}" if not np.isnan(v) else "—"

    if not result.monthly_results:
        conclusion = (
            "*No test months produced picks — check EDGAR coverage for 2025-2026 "
            "or adjust filter thresholds for the held-out period.*"
        )
    elif port_total > bench_total and (np.isnan(hr) or hr >= 0.5):
        conclusion = (
            f"Track {track_num} outperformed the benchmark by **{_pct(excess)}** "
            f"over the held-out period. Hit rate: **{_pct(hr)}**. "
            f"This is an encouraging out-of-sample result, but the window is short "
            f"(~{n_months} months) — interpret with appropriate caution."
        )
    elif port_total > bench_total:
        conclusion = (
            f"Track {track_num} outperformed the benchmark by **{_pct(excess)}** on "
            f"a total-return basis, but the hit rate of **{_pct(hr)}** is below 50%. "
            "Positive total return may be driven by a small number of large winners."
        )
    else:
        conclusion = (
            f"Track {track_num} underperformed the benchmark by **{_pct(abs(excess))}** "
            f"over the held-out period (hit rate: **{_pct(hr)}**). "
            f"The 2025-2026 market regime may differ from 2013-2024 training conditions. "
            "Review sector concentration and filter passage rates before drawing conclusions."
        )

    hr_note = (
        f"Hit rate covers {len(result.hit_rate_returns)} observations where a "
        f"12-month forward price was available. Months from mid-2025 onwards may "
        f"have partial or no 12m forward coverage given the evaluation date."
    )

    lines: list[str] = [
        f"# Held-Out Validation — Track {track_num} ({track_name}) — SP500",
        "",
        "> **HELD-OUT VALIDATION.** Parameters frozen at end of 2013-2024 backtest.",
        "> Do not re-tune after reading these results.",
        "",
        f"**Track:** {track_num} — {track_name}  ",
        "**Universe:** SP500 (~503 tickers)  ",
        "**Holding period:** 1 month  ",
        f"**Test window:** {HELDOUT_START.date()} → {HELDOUT_END.date()}  ",
        "**Burn-in:** none (TRAIN_MONTHS=0 — every month is a test point)  ",
        f"**Generated:** {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}",
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
        f"| Unique tickers picked | {len(all_tickers)} | — |",
        f"| Test months with ≥ 1 pick | {n_months} | — |",
        f"| Hit-rate observations | {len(result.hit_rate_returns)} | — |",
        "",
        "---",
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        f"*{hr_note}*",
        "",
        "---",
        "",
        "> **Data integrity:** Fundamentals from SEC EDGAR (point-in-time, `filed` ≤ snapshot date).",
        "> Prices from yfinance (OHLCV only). Heldout window not seen during backtest development.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Report saved: %s", output_path)


def _print_summary(
    result: BacktestResult,
    track_num: int,
    track_name: str,
    report_path: Path,
) -> None:
    port_rets   = result.portfolio_returns()
    bench_rets  = result.benchmark_returns()
    port_total  = total_return(port_rets)
    bench_total = total_return(bench_rets)
    all_tickers = {t for m in result.monthly_results for t in m.tickers}
    avg_n       = float(np.mean([m.n_picks for m in result.monthly_results])) if result.monthly_results else 0.0

    print("\n" + "═" * 60)
    print(f"  Track {track_num} ({track_name}) — SP500 Held-Out")
    print(f"  {HELDOUT_START.date()} → {HELDOUT_END.date()}")
    print("═" * 60)
    print(f"  Total return:      {port_total:.2%}")
    print(f"  Benchmark (SP500): {bench_total:.2%}")
    print(f"  Excess return:     {port_total - bench_total:+.2%}")
    print(f"  Sharpe (ann.):     {sharpe_ratio(port_rets, RISK_FREE_ANNUAL):.2f}")
    print(f"  Max drawdown:      {max_drawdown(port_rets):.2%}")
    print(f"  Hit rate (12m):    {hit_rate(result.hit_rate_returns):.2%}")
    print(f"  Avg picks/month:   {avg_n:.1f}")
    print(f"  Unique tickers:    {len(all_tickers)}")
    print(f"  Test months:       {len(result.monthly_results)}")
    print("═" * 60)
    print(f"  Report: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-track SP500 held-out validation")
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
        # ── Tickers ──────────────────────────────────────────────────────────
        log.info("Fetching SP500 tickers …")
        tickers = fetch_sp500_tickers()
        log.info("%d tickers in SP500 universe", len(tickers))

        # ── Prices (2024-01 → 2026-06) ───────────────────────────────────────
        log.info("Fetching prices %s → %s …", PRICE_FETCH_START, PRICE_FETCH_END)
        prices = _fetch_prices_parallel(tickers)
        if prices.empty:
            log.error("No price data — aborting")
            sys.exit(1)

        # ── Heldout snapshots (2025-01 → 2026-05) ────────────────────────────
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

        # ── Momentum (operates only on heldout_by_date in-place) ─────────────
        attach_momentum(heldout_by_date, prices)
        log.info("Momentum attached to heldout snapshots")

        # ====================================================================
        # Track 1 — Quality Compounders
        # ====================================================================
        log.info("=" * 55)
        log.info("Running Track 1 (Quality Compounders) held-out …")
        log.info("=" * 55)

        def _t1_pick_fn(df: pd.DataFrame) -> pd.DataFrame:
            filtered = track1_quality.apply_filters(df, config.filters)
            if filtered.empty:
                return filtered
            return track1_quality.score(filtered, config)

        t1_result = _run_walkforward(heldout_by_date, prices, _t1_pick_fn)
        t1_path   = RESULTS_DIR / "heldout_track1_SP500_1m_report.md"

        if not t1_result.monthly_results:
            log.warning("Track 1: no test results — check filter thresholds or EDGAR coverage")
        _write_report(t1_result, t1_path, track_num=1, track_name="Quality Compounders")
        _print_summary(t1_result, 1, "Quality Compounders", t1_path)

        # ====================================================================
        # Track 2 — Growth Inflection
        # ====================================================================
        log.info("=" * 55)
        log.info("Running Track 2 (Growth Inflection) held-out …")
        log.info("=" * 55)

        def _t2_pick_fn(df: pd.DataFrame) -> pd.DataFrame:
            filtered = track2_growth.apply_filters(df, config.track2_filters)
            if filtered.empty:
                return filtered
            # Momentum gate: must have positive 12-1m momentum
            mom_mask = filtered["momentum_raw"].notna() & (filtered["momentum_raw"] > 0)
            filtered = filtered[mom_mask]
            if filtered.empty:
                return filtered
            return track2_growth.score(filtered, config, config.track2_score_weights)

        t2_result = _run_walkforward(heldout_by_date, prices, _t2_pick_fn)
        t2_path   = RESULTS_DIR / "heldout_track2_SP500_1m_report.md"

        if not t2_result.monthly_results:
            log.warning("Track 2: no test results — check filter thresholds or EDGAR coverage")
        _write_report(t2_result, t2_path, track_num=2, track_name="Growth Inflection")
        _print_summary(t2_result, 2, "Growth Inflection", t2_path)

        # ====================================================================
        # Track 3 — Value Recovery
        # (needs p_fcf_vs_history → merge training cache + heldout, then compute)
        # ====================================================================
        log.info("=" * 55)
        log.info("Running Track 3 (Value Recovery) held-out …")
        log.info("=" * 55)

        if TRAINING_CACHE.exists():
            log.info("Loading training cache for p_fcf history: %s", TRAINING_CACHE)
            training_by_date: dict[pd.Timestamp, pd.DataFrame] = joblib.load(TRAINING_CACHE)
            log.info("Training cache: %d snapshot dates loaded", len(training_by_date))
            # Merge: training provides historical p_fcf; heldout has the test data
            combined_by_date = {**training_by_date, **heldout_by_date}
        else:
            log.warning(
                "Training cache not found at %s — p_fcf_vs_history will use heldout "
                "window only (insufficient history; Track 3 filter will likely find 0 picks).",
                TRAINING_CACHE,
            )
            combined_by_date = dict(heldout_by_date)

        attach_p_fcf_history(combined_by_date)
        log.info("P/FCF history attached (combined dict)")

        # Walk-forward evaluates ONLY the heldout dates
        heldout_with_history: dict[pd.Timestamp, pd.DataFrame] = {
            date: combined_by_date[date]
            for date in heldout_by_date
        }

        def _t3_pick_fn(df: pd.DataFrame) -> pd.DataFrame:
            filtered = track3_value.apply_filters(df, config.track3_filters)
            if filtered.empty:
                return filtered
            return track3_value.score(filtered, config, config.track3_score_weights)

        t3_result = _run_walkforward(heldout_with_history, prices, _t3_pick_fn)
        t3_path   = RESULTS_DIR / "heldout_track3_SP500_1m_report.md"

        if not t3_result.monthly_results:
            log.warning("Track 3: no test results — check filter thresholds or EDGAR coverage")
        _write_report(t3_result, t3_path, track_num=3, track_name="Value Recovery")
        _print_summary(t3_result, 3, "Value Recovery", t3_path)

        # ── Final summary ─────────────────────────────────────────────────────
        print("\n" + "═" * 60)
        print("  Held-Out Complete — Three-Track SP500")
        print("═" * 60)
        for num, name, result, path in [
            (1, "Quality Compounders", t1_result, t1_path),
            (2, "Growth Inflection",   t2_result, t2_path),
            (3, "Value Recovery",      t3_result, t3_path),
        ]:
            if result.monthly_results:
                port  = total_return(result.portfolio_returns())
                bench = total_return(result.benchmark_returns())
                print(f"  T{num} {name:<22} port={port:+.2%}  bench={bench:+.2%}  excess={port-bench:+.2%}")
            else:
                print(f"  T{num} {name:<22} — no picks")
        print("═" * 60)
        print()

    finally:
        stop_mem.set()


if __name__ == "__main__":
    main()
