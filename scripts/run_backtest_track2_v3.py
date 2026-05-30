#!/usr/bin/env python3
"""Track 2 v3 — SP500, quarterly EDGAR features active.

Rebuilds both snapshot caches from scratch (quarterly features now included),
runs the 2013-2024 backtest and the 2025-01→2026-05 held-out, then writes
a comparison table against the v2 baselines.

v2 Baselines (Phase 4.7 scorer, annual snapshots)
--------------------------------------------------
Backtest  2013-2024:  407.14% total | +147.99% excess | 0.71 Sharpe | 68.81% HR
Held-out  2025-2026:   40.17% total |  +14.48% excess | 1.11 Sharpe | 50.82% HR

v3 Changes (Phase 5 quarterly features)
----------------------------------------
- revenue_growth_q1yoy   replaces revenue_growth_yr1 in the Layer 1 filter
  (threshold 6% quarterly YoY; falls back to 8% annual when column absent)
- revenue_accel_quarterly added to growth_quality sub-score (weight 10%)
- Snapshots rebuilt from EDGAR to populate the new quarterly columns

Outputs
-------
  data/results/track2_SP500_2013_2024_v3_report.md   — backtest
  data/results/heldout_track2_SP500_v3_report.md      — held-out
  stdout                                              — comparison table

Usage
-----
  python scripts/run_backtest_track2_v3.py
"""
from __future__ import annotations

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
from crucible.snapshot import _CACHE_DIR, attach_momentum, build_snapshots_parallel
from crucible.tracks import track2_growth

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKTEST_START    = pd.Timestamp("2013-01-31", tz="UTC")
BACKTEST_END      = pd.Timestamp("2024-12-31", tz="UTC")
HELDOUT_START     = pd.Timestamp("2025-01-31", tz="UTC")
HELDOUT_END       = pd.Timestamp("2026-05-31", tz="UTC")

PRICE_FETCH_START = "2012-01-01"
PRICE_FETCH_END   = "2026-06-15"

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

BT_CACHE  = _CACHE_DIR / "snapshots_SP500_201301_202412.pkl"
HO_CACHE  = _CACHE_DIR / "snapshots_SP500_202501_202605.pkl"

# v2 baselines for comparison
V2_BACKTEST = dict(total_return=4.0714, excess=1.4799, sharpe=0.71, hit_rate=0.6881)
V2_HELDOUT  = dict(total_return=0.4017, excess=0.1448, sharpe=1.11, hit_rate=0.5082,
                   benchmark=0.2569, mdd=float("nan"))

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


def _fetch_one_price(ticker: str) -> tuple[str, pd.Series]:
    label = BENCHMARK_COL if ticker == "SPY" else ticker
    try:
        df = yf.download(ticker, start=PRICE_FETCH_START, end=PRICE_FETCH_END,
                         progress=False, auto_adjust=True)
        if df.empty:
            return label, pd.Series(dtype=float, name=label)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return label, close.resample("ME").last().rename(label)
    except Exception:
        log.warning("Price fetch failed for %s", ticker, exc_info=True)
        return label, pd.Series(dtype=float, name=label)


def _fetch_prices(tickers: list[str]) -> pd.DataFrame:
    all_tickers = list(tickers) + ["SPY"]
    series_map: dict[str, pd.Series] = {}
    done, total = 0, len(all_tickers)
    with ThreadPoolExecutor(max_workers=PRICE_WORKERS) as pool:
        futures = {pool.submit(_fetch_one_price, t): t for t in all_tickers}
        for future in as_completed(futures):
            label, s = future.result()
            done += 1
            if done % 100 == 0 or done == total:
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
# Walk-forward loop (shared by backtest and held-out)
# ---------------------------------------------------------------------------


def _run_walkforward(
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    config: CrucibleConfig,
    skip_first: int = 24,
) -> BacktestResult:
    """Run the Track 2 walk-forward.

    skip_first=24 for the full backtest (24-month warm-up).
    skip_first=0 for the held-out (every month is a test point).
    """
    bt_config = BacktestConfig(
        train_months=skip_first,
        top_n=TOP_N,
        holding_months=HOLDING_MONTHS,
        hit_rate_months=HIT_RATE_MONTHS,
        risk_free_annual=RISK_FREE_ANNUAL,
        benchmark_col=BENCHMARK_COL,
    )
    dates      = sorted(fund_by_date.keys())
    price_idx  = prices.index
    test_dates = dates[skip_first:]

    log.info(
        "Walk-forward: %d test dates  [%s → %s]",
        len(test_dates),
        test_dates[0].date() if test_dates else "—",
        test_dates[-1].date() if test_dates else "—",
    )

    monthly_results: list[MonthlyResult] = []
    hit_rate_returns: list[float] = []

    for i, test_date in enumerate(test_dates, 1):
        if i % 12 == 0:
            log.info("Progress: %d / %d", i, len(test_dates))

        df = fund_by_date[test_date]

        try:
            filtered = track2_growth.apply_filters(df, config.track2_filters)
        except Exception:
            log.warning("Filter error at %s — skipping", test_date, exc_info=True)
            continue

        mom_mask = (
            filtered["momentum_raw"].notna() & (filtered["momentum_raw"] > 0)
        )
        filtered = filtered[mom_mask]
        if filtered.empty:
            continue

        scored = track2_growth.score(filtered, config, config.track2_score_weights)
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
        "Walk-forward done: %d months with picks, %d HR observations",
        len(monthly_results), len(hit_rate_returns),
    )
    return BacktestResult(
        monthly_results=monthly_results,
        hit_rate_returns=hit_rate_returns,
        bt_config=bt_config,
    )


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _pct(v: float) -> str:
    return f"{v:.2%}" if not np.isnan(v) else "—"


def _f2(v: float) -> str:
    return f"{v:.2f}" if not np.isnan(v) else "—"


def _delta_pct(v3: float, v2: float) -> str:
    if np.isnan(v3) or np.isnan(v2):
        return "—"
    d = v3 - v2
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2%}"


def _delta_f(v3: float, v2: float) -> str:
    if np.isnan(v3) or np.isnan(v2):
        return "—"
    d = v3 - v2
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f}"


def _extract_metrics(result: BacktestResult) -> dict:
    port_rets  = result.portfolio_returns()
    bench_rets = result.benchmark_returns()
    pt         = total_return(port_rets)
    bt         = total_return(bench_rets)
    return {
        "total_return": pt,
        "benchmark":    bt,
        "excess":       pt - bt,
        "sharpe":       sharpe_ratio(port_rets, RISK_FREE_ANNUAL),
        "mdd":          max_drawdown(port_rets),
        "hit_rate":     hit_rate(result.hit_rate_returns),
        "avg_picks":    float(np.mean([m.n_picks for m in result.monthly_results]))
                        if result.monthly_results else 0.0,
        "n_months":     len(result.monthly_results),
        "n_tickers":    len({t for m in result.monthly_results for t in m.tickers}),
        "n_hr_obs":     len(result.hit_rate_returns),
    }


def _write_backtest_report(
    result: BacktestResult,
    m: dict,
    output_path: Path,
) -> None:
    b = V2_BACKTEST
    lines = [
        "# Track 2 v3 — SP500 Backtest Report (2013-2024)",
        "",
        "> **v3 changes:** quarterly EDGAR features active.",
        "> `revenue_growth_q1yoy` (> 6% QoQ-YoY) replaces annual revenue growth filter.",
        "> `revenue_accel_quarterly` (weight 10%) added to growth_quality sub-score.",
        "> Snapshots rebuilt from EDGAR; v2 cache deleted before this run.",
        "",
        f"**Generated:** {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Universe:** SP500 (~503 tickers)  ",
        f"**Period:** {BACKTEST_START.date()} → {BACKTEST_END.date()} (24-month warm-up)  ",
        "",
        "---",
        "",
        "## Performance vs v2 Baseline",
        "",
        "| Metric | v3 (quarterly) | v2 baseline | Δ |",
        "|--------|---------------|-------------|---|",
        f"| Total return | {_pct(m['total_return'])} | {_pct(b['total_return'])} | {_delta_pct(m['total_return'], b['total_return'])} |",
        f"| Benchmark (SP500) | {_pct(m['benchmark'])} | — | — |",
        f"| Excess return | {_pct(m['excess'])} | {_pct(b['excess'])} | {_delta_pct(m['excess'], b['excess'])} |",
        f"| Annualised Sharpe | {_f2(m['sharpe'])} | {b['sharpe']:.2f} | {_delta_f(m['sharpe'], b['sharpe'])} |",
        f"| Max drawdown | {_pct(m['mdd'])} | — | — |",
        f"| Hit rate (12m) | {_pct(m['hit_rate'])} | {_pct(b['hit_rate'])} | {_delta_pct(m['hit_rate'], b['hit_rate'])} |",
        f"| Avg picks / month | {m['avg_picks']:.1f} | — | — |",
        f"| Unique tickers | {m['n_tickers']} | — | — |",
        f"| Test months | {m['n_months']} | — | — |",
        f"| Hit-rate observations | {m['n_hr_obs']} | — | — |",
        "",
        "---",
        "",
        "> Fundamentals: SEC EDGAR (point-in-time). Prices: yfinance (OHLCV only).",
        "> Quarterly features (revenue_growth_q1yoy, revenue_accel_quarterly, gross_margin_q_latest,",
        "> fcf_q_last2) computed inline from 10-Q filings during snapshot build.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Backtest report saved: %s", output_path)


def _write_heldout_report(
    result: BacktestResult,
    m: dict,
    output_path: Path,
) -> None:
    b = V2_HELDOUT

    if not result.monthly_results:
        conclusion = (
            "*No test months produced picks — check EDGAR coverage for 2025-2026.*"
        )
    elif m["total_return"] > m["benchmark"] and m["total_return"] >= b["total_return"]:
        conclusion = (
            f"Track 2 v3 **outperformed** both the benchmark ({_pct(m['benchmark'])}) "
            f"and the v2 held-out baseline ({_pct(b['total_return'])}). "
            f"The quarterly features appear to add value in the prospective window."
        )
    elif m["total_return"] > m["benchmark"]:
        conclusion = (
            f"Track 2 v3 outperformed the benchmark by **{_pct(m['excess'])}** "
            f"but returned {_delta_pct(m['total_return'], b['total_return'])} vs v2 baseline. "
            f"Quarterly features changed the outcome without clear improvement."
        )
    else:
        conclusion = (
            f"Track 2 v3 underperformed the benchmark by **{_pct(abs(m['excess']))}** "
            f"over the held-out period. Hit rate: **{_pct(m['hit_rate'])}**. "
            f"Review whether the quarterly filter threshold is appropriate for the 2025-2026 regime."
        )

    lines = [
        "# Track 2 v3 — SP500 Held-Out (2025-01 → 2026-05)",
        "",
        "> **HELD-OUT VALIDATION.** Parameters frozen at end of 2013-2024 backtest.",
        "> Do not re-tune after reading these results.",
        "",
        "> **v3 changes vs v2:** quarterly EDGAR features active in filter and scorer.",
        "",
        f"**Generated:** {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Universe:** SP500 (~503 tickers)  ",
        f"**Period:** {HELDOUT_START.date()} → {HELDOUT_END.date()} (no warm-up)  ",
        "",
        "---",
        "",
        "## Performance vs v2 Baseline",
        "",
        "| Metric | v3 (quarterly) | v2 baseline | Δ |",
        "|--------|---------------|-------------|---|",
        f"| Total return | {_pct(m['total_return'])} | {_pct(b['total_return'])} | {_delta_pct(m['total_return'], b['total_return'])} |",
        f"| Benchmark (SP500) | {_pct(m['benchmark'])} | {_pct(b['benchmark'])} | — |",
        f"| Excess return | {_pct(m['excess'])} | {_pct(b['excess'])} | {_delta_pct(m['excess'], b['excess'])} |",
        f"| Annualised Sharpe | {_f2(m['sharpe'])} | {b['sharpe']:.2f} | {_delta_f(m['sharpe'], b['sharpe'])} |",
        f"| Max drawdown | {_pct(m['mdd'])} | — | — |",
        f"| Hit rate (12m) | {_pct(m['hit_rate'])} | {_pct(b['hit_rate'])} | {_delta_pct(m['hit_rate'], b['hit_rate'])} |",
        f"| Avg picks / month | {m['avg_picks']:.1f} | — | — |",
        f"| Unique tickers | {m['n_tickers']} | — | — |",
        f"| Test months | {m['n_months']} | — | — |",
        "",
        "---",
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        f"*Hit rate covers {m['n_hr_obs']} observations with 12-month forward price available.",
        "Months from mid-2025 onward have partial or no 12m forward coverage.*",
        "",
        "---",
        "",
        "> Data: EDGAR point-in-time fundamentals + yfinance OHLCV prices.",
        "> Held-out window not seen during development of quarterly features.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Held-out report saved: %s", output_path)


def _print_comparison(bt_m: dict, ho_m: dict) -> None:
    W = 72
    b_bt = V2_BACKTEST
    b_ho = V2_HELDOUT

    print("\n" + "═" * W)
    print("  Track 2 v3 vs v2 — Full Comparison")
    print("  Quarterly EDGAR features: revenue_growth_q1yoy + revenue_accel_quarterly")
    print("═" * W)

    def _row(label, v3, v2, fmt="pct"):
        if fmt == "pct":
            v3s = f"{v3:.2%}" if not np.isnan(v3) else "—"
            v2s = f"{v2:.2%}" if not np.isnan(v2) else "—"
            d   = v3 - v2 if not (np.isnan(v3) or np.isnan(v2)) else float("nan")
            ds  = f"{d:+.2%}" if not np.isnan(d) else "—"
        else:
            v3s = f"{v3:.2f}" if not np.isnan(v3) else "—"
            v2s = f"{v2:.2f}" if not np.isnan(v2) else "—"
            d   = v3 - v2 if not (np.isnan(v3) or np.isnan(v2)) else float("nan")
            ds  = f"{d:+.2f}" if not np.isnan(d) else "—"
        tick = "✅" if not np.isnan(d) and d >= 0 else ("❌" if not np.isnan(d) else " ")
        print(f"  {label:<26} {v3s:>9}  {v2s:>9}  {ds:>9}  {tick}")

    print(f"\n  {'Metric':<26} {'v3':>9}  {'v2':>9}  {'delta':>9}")
    print(f"  {'───────────────────────────':26} {'────────':>9}  {'────────':>9}  {'────────':>9}")
    print(f"  BACKTEST 2013-2024")
    _row("Total return",       bt_m["total_return"],  b_bt["total_return"])
    _row("Excess vs SP500",    bt_m["excess"],        b_bt["excess"])
    _row("Sharpe",             bt_m["sharpe"],        b_bt["sharpe"],       fmt="f")
    _row("Hit rate (12m)",     bt_m["hit_rate"],      b_bt["hit_rate"])
    print(f"  {'───────────────────────────':26} {'────────':>9}  {'────────':>9}  {'────────':>9}")
    print(f"  HELD-OUT 2025-01 → 2026-05")
    _row("Total return",       ho_m["total_return"],  b_ho["total_return"])
    _row("Excess vs SP500",    ho_m["excess"],        b_ho["excess"])
    _row("Sharpe",             ho_m["sharpe"],        b_ho["sharpe"],       fmt="f")
    _row("Hit rate (12m)",     ho_m["hit_rate"],      b_ho["hit_rate"])
    print("═" * W)

    bt_ok = not np.isnan(bt_m["total_return"]) and bt_m["total_return"] >= b_bt["total_return"]
    ho_ok = not np.isnan(ho_m["total_return"]) and ho_m["total_return"] >= b_ho["total_return"]

    verdict = "BOTH windows better" if bt_ok and ho_ok else \
              "BACKTEST better only" if bt_ok else \
              "HELD-OUT better only" if ho_ok else \
              "BOTH windows worse — quarterly features did not improve performance"
    print(f"\n  Verdict: {verdict}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    for path, label in (
        (CIK_MAP_PATH, "CIK mapping"),
        (EDGAR_DIR,    "EDGAR companyfacts directory"),
    ):
        if not path.exists():
            log.error("%s not found at %s.", label, path)
            sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config  = CrucibleConfig(account_currency="USD")
    cik_map = _load_cik_mapping(CIK_MAP_PATH)

    stop_mem = threading.Event()
    threading.Thread(
        target=_memory_monitor, args=(stop_mem,), daemon=True, name="mem-monitor"
    ).start()

    try:
        # ── 1. Delete old SP500 caches ────────────────────────────────────────
        for cache_path in (BT_CACHE, HO_CACHE):
            if cache_path.exists():
                cache_path.unlink()
                log.info("Deleted old cache: %s", cache_path)

        # ── 2. Fetch prices (covers both backtest and held-out) ───────────────
        log.info("Fetching SP500 tickers …")
        tickers = fetch_sp500_tickers()
        log.info("%d tickers in SP500 universe", len(tickers))

        log.info("Fetching prices %s → %s …", PRICE_FETCH_START, PRICE_FETCH_END)
        prices = _fetch_prices(tickers)
        if prices.empty:
            log.error("No price data — aborting")
            sys.exit(1)

        # ── 3. Build backtest snapshots 2013-2024 ─────────────────────────────
        log.info("Building backtest snapshots (144 months) — this takes ~40-60 min …")
        bt_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq="ME", tz="UTC")
        bt_fund = build_snapshots_parallel(
            tickers=tickers,
            dates=bt_dates,
            cik_map=cik_map,
            edgar_dir=EDGAR_DIR,
            prices=prices,
            workers=SNAPSHOT_WORKERS,
            universe="SP500",
            use_cache=True,   # will save new cache after build (old one deleted above)
        )
        log.info("Backtest snapshots: %d dates", len(bt_fund))
        attach_momentum(bt_fund, prices)

        # ── 4. Run backtest ───────────────────────────────────────────────────
        log.info("Running Track 2 v3 backtest (2013-2024) …")
        bt_result = _run_walkforward(bt_fund, prices, config, skip_first=24)
        if not bt_result.monthly_results:
            log.error("Backtest produced no results — check EDGAR coverage")
            sys.exit(1)

        bt_m = _extract_metrics(bt_result)
        bt_report = RESULTS_DIR / "track2_SP500_2013_2024_v3_report.md"
        _write_backtest_report(bt_result, bt_m, bt_report)

        # ── 5. Build held-out snapshots 2025-2026 ─────────────────────────────
        log.info("Building held-out snapshots (2025-01 → 2026-05) …")
        ho_dates = pd.date_range(HELDOUT_START, HELDOUT_END, freq="ME", tz="UTC")
        ho_fund = build_snapshots_parallel(
            tickers=tickers,
            dates=ho_dates,
            cik_map=cik_map,
            edgar_dir=EDGAR_DIR,
            prices=prices,
            workers=SNAPSHOT_WORKERS,
            universe="SP500",
            use_cache=True,
        )
        log.info("Held-out snapshots: %d dates", len(ho_fund))
        attach_momentum(ho_fund, prices)

        # ── 6. Run held-out ───────────────────────────────────────────────────
        log.info("Running Track 2 v3 held-out (2025-01 → 2026-05) …")
        ho_result = _run_walkforward(ho_fund, prices, config, skip_first=0)

        ho_m = _extract_metrics(ho_result)
        ho_report = RESULTS_DIR / "heldout_track2_SP500_v3_report.md"
        _write_heldout_report(ho_result, ho_m, ho_report)

        # ── 7. Print comparison ───────────────────────────────────────────────
        _print_comparison(bt_m, ho_m)

        print(f"  Backtest report:  {bt_report}")
        print(f"  Held-out report:  {ho_report}")
        print()

    finally:
        stop_mem.set()


if __name__ == "__main__":
    main()
