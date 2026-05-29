#!/usr/bin/env python3
"""Held-out validation — Protocol B (50% T2 / 30% T3 / 20% T1), SP500.

Protocol B allocation
---------------------
  50 %  Track 2 — Growth Inflection (Phase 4.7 scorer active)
  30 %  Track 3 — Value Recovery
  20 %  Track 1 — Quality Compounders

Combined monthly return = 0.50 * r_T2 + 0.30 * r_T3 + 0.20 * r_T1

Each track runs its own equal-weighted TOP_N=20 portfolio independently.
If a track produces no picks for a given month its weight is redistributed
proportionally to the other tracks that do have picks.

Baselines for comparison
------------------------
  Benchmark (SP500):  25.69%
  Track 2 alone:      40.17%  (+14.48% excess, Sharpe 1.11, hit 50.82%)

Snapshot / price data reuse
----------------------------
  Heldout EDGAR snapshots: data/cache/snapshots_SP500_202501_202605.pkl
  Training EDGAR cache:     data/cache/snapshots_SP500_201301_202412.pkl
  Both are written by run_heldout_three_tracks.py — run that first if missing.

Outputs
-------
  data/results/heldout_combined_protocolB_report.md

Usage
-----
  python scripts/run_heldout_protocol_b.py [--no-cache]
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
PRICE_FETCH_START = "2024-01-01"
PRICE_FETCH_END   = "2026-06-15"

TRAIN_MONTHS     = 0
TOP_N            = 20
HOLDING_MONTHS   = 1
HIT_RATE_MONTHS  = 12
RISK_FREE_ANNUAL = 0.04
BENCHMARK_COL    = "SP500"

# Protocol B allocation weights (must sum to 1.0)
WEIGHTS = {"t1": 0.20, "t2": 0.50, "t3": 0.30}

# Baselines
BENCHMARK_TOTAL  = 0.2569
TRACK2_TOTAL     = 0.4017
TRACK2_EXCESS    = 0.1448
TRACK2_SHARPE    = 1.11
TRACK2_HIT       = 0.5082

RESULTS_DIR      = ROOT / "data" / "results"
EDGAR_DIR        = ROOT / "data" / "raw" / "edgar" / "companyfacts"
CIK_MAP_PATH     = ROOT / "data" / "raw" / "edgar" / "cik_mapping.json"

PRICE_WORKERS    = 20
SNAPSHOT_WORKERS = 4
MEMORY_LOG_SECS  = 600

HELDOUT_CACHE  = _CACHE_DIR / "snapshots_SP500_202501_202605.pkl"
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
        futures = {
            pool.submit(_fetch_one_price, t, PRICE_FETCH_START, PRICE_FETCH_END): t
            for t in all_tickers
        }
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
# Pick functions (one per track)
# ---------------------------------------------------------------------------


def _make_pick_fns(config: CrucibleConfig):
    def _t1(df: pd.DataFrame) -> pd.DataFrame:
        filtered = track1_quality.apply_filters(df, config.filters)
        if filtered.empty:
            return filtered
        return track1_quality.score(filtered, config)

    def _t2(df: pd.DataFrame) -> pd.DataFrame:
        filtered = track2_growth.apply_filters(df, config.track2_filters)
        if filtered.empty:
            return filtered
        mom_mask = filtered["momentum_raw"].notna() & (filtered["momentum_raw"] > 0)
        filtered = filtered[mom_mask]
        if filtered.empty:
            return filtered
        return track2_growth.score(filtered, config, config.track2_score_weights)

    def _t3(df: pd.DataFrame) -> pd.DataFrame:
        filtered = track3_value.apply_filters(df, config.track3_filters)
        if filtered.empty:
            return filtered
        return track3_value.score(filtered, config, config.track3_score_weights)

    return _t1, _t2, _t3


# ---------------------------------------------------------------------------
# Protocol B walk-forward
# ---------------------------------------------------------------------------


def _run_protocol_b(
    heldout_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    pick_fns: tuple,
) -> BacktestResult:
    """Walk-forward combining three tracks with Protocol B weights.

    Monthly return = w_t1 * r_T1 + w_t2 * r_T2 + w_t3 * r_T3.
    If a track has no picks for a month its weight is redistributed
    proportionally to the remaining tracks.
    Hit rate pools all individual 12m returns across tracks, weighted by
    allocation so T2 picks count for 50% of the pool.
    """
    bt_config = BacktestConfig(
        train_months=TRAIN_MONTHS,
        top_n=TOP_N,
        holding_months=HOLDING_MONTHS,
        hit_rate_months=HIT_RATE_MONTHS,
        risk_free_annual=RISK_FREE_ANNUAL,
        benchmark_col=BENCHMARK_COL,
    )

    _t1_fn, _t2_fn, _t3_fn = pick_fns
    dates      = sorted(heldout_by_date.keys())
    price_idx  = prices.index
    test_dates = dates  # TRAIN_MONTHS=0, every date is a test point

    log.info(
        "Protocol B walk-forward: %d test dates (%s → %s)",
        len(test_dates),
        test_dates[0].date() if test_dates else "—",
        test_dates[-1].date() if test_dates else "—",
    )

    monthly_results: list[MonthlyResult] = []
    # Weighted hit-rate list: (return, weight) pairs
    hit_rate_weighted: list[tuple[float, float]] = []

    track_labels = ("t1", "t2", "t3")
    track_fns    = (_t1_fn, _t2_fn, _t3_fn)

    for test_date in test_dates:
        df = heldout_by_date[test_date]

        # Run all three pick functions
        track_picks: dict[str, list[str]] = {}
        for label, fn in zip(track_labels, track_fns):
            try:
                scored = fn(df)
                if not scored.empty:
                    track_picks[label] = scored.head(TOP_N).index.tolist()
            except Exception:
                log.warning("Pick error [%s] at %s — skipping track", label, test_date, exc_info=True)

        if not track_picks:
            log.debug("No picks from any track at %s", test_date.date())
            continue

        # Redistribute weights if any track has no picks
        active_labels = [l for l in track_labels if l in track_picks]
        raw_weights   = {l: WEIGHTS[l] for l in active_labels}
        total_w       = sum(raw_weights.values())
        eff_weights   = {l: w / total_w for l, w in raw_weights.items()}

        # 1-month portfolio return
        next_month = _advance(test_date, price_idx, HOLDING_MONTHS)
        if next_month is not None and test_date in price_idx:
            track_returns: dict[str, float] = {}
            all_ticker_returns: dict[str, float] = {}
            total_picks = 0

            for label in active_labels:
                picks = track_picks[label]
                tkr_rets = {
                    t: r
                    for t in picks
                    for r in (_single_return(t, test_date, next_month, prices),)
                    if r is not None
                }
                if tkr_rets:
                    track_returns[label] = float(np.mean(list(tkr_rets.values())))
                    all_ticker_returns.update(tkr_rets)
                    total_picks += len(picks)
                else:
                    # No priceable picks — drop this track from the month
                    del eff_weights[label]

            if not track_returns:
                continue

            # Re-normalise after any priceless-track drops
            sum_ew = sum(eff_weights[l] for l in track_returns)
            if sum_ew < 1e-9:
                continue
            norm_weights = {l: eff_weights[l] / sum_ew for l in track_returns}

            port_ret  = sum(norm_weights[l] * track_returns[l] for l in track_returns)
            bench_ret = _benchmark_return(test_date, next_month, prices, BENCHMARK_COL)

            # Represent n_picks as the weighted-average track size
            n_picks = int(round(sum(
                norm_weights[l] * len(track_picks[l])
                for l in track_returns
            )))

            # Collect all tickers (deduplicated) for the monthly record
            all_tickers_month: list[str] = []
            seen: set[str] = set()
            for label in active_labels:
                for t in track_picks.get(label, []):
                    if t not in seen:
                        all_tickers_month.append(t)
                        seen.add(t)

            monthly_results.append(MonthlyResult(
                date=test_date,
                portfolio_return=port_ret,
                benchmark_return=bench_ret,
                n_picks=n_picks,
                tickers=all_tickers_month,
                ticker_returns=all_ticker_returns,
            ))

        # 12-month hit rate (weighted by track allocation)
        hit_date = _advance(test_date, price_idx, HIT_RATE_MONTHS)
        if hit_date is not None and test_date in price_idx:
            for label in active_labels:
                w = eff_weights.get(label, WEIGHTS[label])
                for ticker in track_picks.get(label, []):
                    r = _single_return(ticker, test_date, hit_date, prices)
                    if r is not None:
                        hit_rate_weighted.append((r, w))

    # Flatten weighted hit-rate: duplicate each observation proportionally
    # so that the standard hit_rate() function gives a weighted result.
    hit_rate_returns: list[float] = []
    if hit_rate_weighted:
        # Normalise weights so they sum to len(hit_rate_weighted) on average
        total_w_obs = sum(w for _, w in hit_rate_weighted)
        scale = len(hit_rate_weighted) / total_w_obs if total_w_obs > 0 else 1.0
        for r, w in hit_rate_weighted:
            # Repeat each observation proportional to its weight
            repeats = max(1, round(w * scale))
            hit_rate_returns.extend([r] * repeats)

    log.info(
        "Protocol B walk-forward done: %d months, %d hit-rate obs",
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


def _write_report(result: BacktestResult, output_path: Path) -> None:
    port_rets   = result.portfolio_returns()
    bench_rets  = result.benchmark_returns()
    port_total  = total_return(port_rets)
    bench_total = total_return(bench_rets)
    port_sharpe = sharpe_ratio(port_rets, RISK_FREE_ANNUAL)
    port_mdd    = max_drawdown(port_rets)
    hr          = hit_rate(result.hit_rate_returns)
    excess      = port_total - bench_total

    all_tickers  = {t for m in result.monthly_results for t in m.tickers}
    avg_picks    = float(np.mean([m.n_picks for m in result.monthly_results])) if result.monthly_results else 0.0
    n_months     = len(result.monthly_results)

    def _pct(v: float) -> str:
        return f"{v:.2%}" if not np.isnan(v) else "—"

    def _f2(v: float) -> str:
        return f"{v:.2f}" if not np.isnan(v) else "—"

    def _delta(v_new: float, v_old: float) -> str:
        d = v_new - v_old
        return f"{d:+.2%}"

    if not result.monthly_results:
        conclusion = (
            "*No test months produced picks — check EDGAR coverage for 2025-2026.*"
        )
    elif port_total > TRACK2_TOTAL:
        conclusion = (
            f"Protocol B outperformed both the benchmark (**{_pct(excess)}** excess) "
            f"and Track 2 alone (**{_delta(port_total, TRACK2_TOTAL)}** vs Track 2). "
            f"Hit rate: **{_pct(hr)}**. The diversification across three tracks "
            f"improved total return in this held-out window — interpret with caution "
            f"given the short test period (~{n_months} months)."
        )
    elif port_total > bench_total:
        conclusion = (
            f"Protocol B outperformed the benchmark by **{_pct(excess)}** "
            f"but trailed Track 2 alone by **{_delta(TRACK2_TOTAL, port_total)}**. "
            f"Hit rate: **{_pct(hr)}**. The T3/T1 allocation diluted the stronger "
            f"Track 2 signal in this held-out window."
        )
    else:
        conclusion = (
            f"Protocol B underperformed the benchmark by **{_pct(abs(excess))}** "
            f"over the held-out period (hit rate: **{_pct(hr)}**). "
            "Review track weighting and filter passage rates before drawing conclusions."
        )

    lines: list[str] = [
        "# Held-Out Validation — Protocol B (50% T2 / 30% T3 / 20% T1) — SP500",
        "",
        "> **HELD-OUT VALIDATION.** Parameters frozen at end of 2013-2024 backtest.",
        "> Do not re-tune after reading these results.",
        "",
        "**Protocol:** B — combined three-track portfolio  ",
        "**Allocation:** 50% Track 2 (Growth Inflection) / 30% Track 3 (Value Recovery) / 20% Track 1 (Quality Compounders)  ",
        "**Track 2 scorer:** Phase 4.7 active (asset_growth_yoy penalty, deferred_revenue_growth, eps_surprise_last_q)  ",
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
        "| Metric | Protocol B | Benchmark (SP500) | Track 2 alone | Δ vs T2 |",
        "|--------|-----------|-------------------|---------------|---------|",
        f"| Total return | {_pct(port_total)} | {_pct(bench_total)} | {_pct(TRACK2_TOTAL)} | {_delta(port_total, TRACK2_TOTAL)} |",
        f"| Excess return | {_pct(excess)} | — | {_pct(TRACK2_EXCESS)} | {_delta(excess, TRACK2_EXCESS)} |",
        f"| Annualised Sharpe | {_f2(port_sharpe)} | — | {TRACK2_SHARPE:.2f} | {_delta(port_sharpe, TRACK2_SHARPE)} |",
        f"| Maximum drawdown | {_pct(port_mdd)} | — | -6.79% | — |",
        f"| Hit rate (12m forward) | {_pct(hr)} | — | {_pct(TRACK2_HIT)} | {_delta(hr if not np.isnan(hr) else 0.0, TRACK2_HIT)} |",
        f"| Avg picks / month (eff.) | {avg_picks:.1f} | — | 11.1 | — |",
        f"| Unique tickers picked | {len(all_tickers)} | — | 25 | — |",
        f"| Test months with ≥ 1 pick | {n_months} | — | 16 | — |",
        f"| Hit-rate observations | {len(result.hit_rate_returns)} | — | 61 | — |",
        "",
        "---",
        "",
        "## Individual track inputs (from three_tracks held-out, same window)",
        "",
        "| Track | Total return | Excess | Sharpe | Hit rate |",
        "|-------|-------------|--------|--------|----------|",
        "| Track 1 — Quality Compounders (20%) | 8.48% | -17.22% | 0.25 | 59.00% |",
        "| Track 2 — Growth Inflection (50%) | 40.17% | +14.48% | 1.11 | 50.82% |",
        "| Track 3 — Value Recovery (30%) | 19.44% | -6.25% | 0.80 | 67.06% |",
        "| **Protocol B weighted expectation** | "
        f"**{0.20*0.0848 + 0.50*0.4017 + 0.30*0.1944:.2%}** | "
        f"**{0.20*(-0.1722) + 0.50*0.1448 + 0.30*(-0.0625):.2%}** | — | — |",
        "",
        "---",
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        f"*Hit rate covers {len(result.hit_rate_returns)} observations (weighted by track allocation). "
        f"Months from mid-2025 onwards may have partial or no 12m forward coverage given the evaluation date.*",
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
    parser = argparse.ArgumentParser(description="Protocol B SP500 held-out validation")
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

    try:
        # ── Tickers ──────────────────────────────────────────────────────────
        log.info("Fetching SP500 tickers …")
        tickers = fetch_sp500_tickers()
        log.info("%d tickers in SP500 universe", len(tickers))

        # ── Prices ───────────────────────────────────────────────────────────
        log.info("Fetching prices %s → %s …", PRICE_FETCH_START, PRICE_FETCH_END)
        prices = _fetch_prices_parallel(tickers)
        if prices.empty:
            log.error("No price data — aborting")
            sys.exit(1)

        # ── Heldout snapshots ─────────────────────────────────────────────────
        heldout_dates = pd.date_range(HELDOUT_START, HELDOUT_END, freq="ME", tz="UTC")
        log.info("Building / loading %d heldout EDGAR snapshots …", len(heldout_dates))
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
        log.info("Momentum attached")

        # ── Track 3 needs p_fcf history (merge training + heldout) ───────────
        if TRAINING_CACHE.exists():
            log.info("Loading training cache for p_fcf history: %s", TRAINING_CACHE)
            training_by_date: dict[pd.Timestamp, pd.DataFrame] = joblib.load(TRAINING_CACHE)
            log.info("Training cache: %d snapshot dates loaded", len(training_by_date))
            combined_by_date = {**training_by_date, **heldout_by_date}
        else:
            log.warning(
                "Training cache not found at %s — Track 3 p_fcf_vs_history will use "
                "heldout window only (likely zero T3 picks).",
                TRAINING_CACHE,
            )
            combined_by_date = dict(heldout_by_date)

        attach_p_fcf_history(combined_by_date)
        log.info("P/FCF history attached")

        # Restrict walk-forward to heldout dates only
        heldout_with_history: dict[pd.Timestamp, pd.DataFrame] = {
            date: combined_by_date[date]
            for date in heldout_by_date
        }

        # ── Protocol B walk-forward ───────────────────────────────────────────
        log.info("=" * 60)
        log.info("Running Protocol B walk-forward …")
        log.info("  Weights: T1=%.0f%%  T2=%.0f%%  T3=%.0f%%",
                 WEIGHTS["t1"] * 100, WEIGHTS["t2"] * 100, WEIGHTS["t3"] * 100)
        log.info("=" * 60)

        pick_fns = _make_pick_fns(config)
        result   = _run_protocol_b(heldout_with_history, prices, pick_fns)

        report_path = RESULTS_DIR / "heldout_combined_protocolB_report.md"
        _write_report(result, report_path)

        # ── Console summary ───────────────────────────────────────────────────
        port_rets   = result.portfolio_returns()
        bench_rets  = result.benchmark_returns()
        port_total  = total_return(port_rets)
        bench_total = total_return(bench_rets)
        hr          = hit_rate(result.hit_rate_returns)

        print("\n" + "═" * 65)
        print("  Protocol B (50% T2 / 30% T3 / 20% T1) — SP500 Held-Out")
        print(f"  {HELDOUT_START.date()} → {HELDOUT_END.date()}")
        print("═" * 65)
        print(f"  Total return:      {port_total:.2%}  (T2 alone: {TRACK2_TOTAL:.2%}  Δ {port_total - TRACK2_TOTAL:+.2%})")
        print(f"  Benchmark (SP500): {bench_total:.2%}")
        print(f"  Excess return:     {port_total - bench_total:+.2%}  (T2 alone: {TRACK2_EXCESS:.2%}  Δ {(port_total - bench_total) - TRACK2_EXCESS:+.2%})")
        print(f"  Sharpe (ann.):     {sharpe_ratio(port_rets, RISK_FREE_ANNUAL):.2f}  (T2 alone: {TRACK2_SHARPE:.2f})")
        print(f"  Max drawdown:      {max_drawdown(port_rets):.2%}")
        print(f"  Hit rate (12m):    {hr:.2%}  (T2 alone: {TRACK2_HIT:.2%})")
        print(f"  Test months:       {len(result.monthly_results)}")
        print(f"  Unique tickers:    {len({t for m in result.monthly_results for t in m.tickers})}")
        print("═" * 65)
        print(f"  Report: {report_path}")
        print()

    finally:
        stop_mem.set()


if __name__ == "__main__":
    main()
