#!/usr/bin/env python3
"""Combined rotation backtest — two protocols vs Track 2 alone baseline.

Protocols
---------
  A) Deterministic rotation  T1 → T2 → T3 → T1 → …  (one track per month)
  B) Weighted blend          50% T2 / 30% T3 / 20% T1 (equal-weighted within
     each track's shortlist; blend weights renormalised if a track has no picks)

Baseline: Track 2 alone (same walk-forward, top 20, equal-weight).

All three use the SP500 snapshot cache (2013-01-31 → 2024-12-31) built by
run_backtest_track2_sp500.py.  First test month is 2015-01 (after the
24-month warm-up).

Output
------
  data/results/combined_SP500_2013_2024_rotation_report.md
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
import yfinance as yf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from crucible.backtest import (
    BacktestConfig,
    MonthlyResult,
    BacktestResult,
    _advance,
    _benchmark_return,
    _single_return,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    total_return,
)
from crucible.config import CrucibleConfig
from crucible.fetcher import fetch_sp500_tickers
from crucible.snapshot import _CACHE_DIR, attach_momentum
from crucible.tracks import track1_quality, track2_growth, track3_value

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
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

PRICE_WORKERS    = 20
MEMORY_LOG_SECS  = 600

SP500_CACHE  = _CACHE_DIR / "snapshots_SP500_201301_202412.pkl"
RESULTS_DIR  = ROOT / "data" / "results"

# Weights for Protocol B (weighted blend)
WEIGHT_T1 = 0.20
WEIGHT_T2 = 0.50
WEIGHT_T3 = 0.30


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
# Track picker helpers
# ---------------------------------------------------------------------------


def _picks_t1(df: pd.DataFrame, config: CrucibleConfig) -> list[str]:
    try:
        filtered = track1_quality.apply_filters(df, config.filters)
        if filtered.empty:
            return []
        scored = track1_quality.score(filtered, config)
        return scored.head(TOP_N).index.tolist()
    except Exception:
        log.debug("Track 1 filter/score error", exc_info=True)
        return []


def _picks_t2(df: pd.DataFrame, config: CrucibleConfig) -> list[str]:
    try:
        filtered = track2_growth.apply_filters(df, config.track2_filters)
        mom_mask = filtered["momentum_raw"].notna() & (filtered["momentum_raw"] > 0)
        filtered = filtered[mom_mask]
        if filtered.empty:
            return []
        scored = track2_growth.score(filtered, config, config.track2_score_weights)
        return scored.head(TOP_N).index.tolist()
    except Exception:
        log.warning("Track 2 filter/score error", exc_info=True)
        return []


def _picks_t3(df: pd.DataFrame, config: CrucibleConfig) -> list[str]:
    try:
        filtered = track3_value.apply_filters(df, config.track3_filters)
        if filtered.empty:
            return []
        scored = track3_value.score(filtered, config, config.track3_score_weights)
        return scored.head(TOP_N).index.tolist()
    except Exception:
        log.debug("Track 3 filter/score error", exc_info=True)
        return []


def _equal_weight_return(
    picks: list[str],
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    prices: pd.DataFrame,
) -> tuple[float, dict[str, float]]:
    """Return (portfolio_return, {ticker: return}) for an equal-weighted portfolio."""
    tkr_rets = {}
    for t in picks:
        r = _single_return(t, t0, t1, prices)
        if r is not None:
            tkr_rets[t] = r
    port_ret = float(np.mean(list(tkr_rets.values()))) if tkr_rets else 0.0
    return port_ret, tkr_rets


# ---------------------------------------------------------------------------
# Walk-forward: Protocol A (deterministic rotation)
# ---------------------------------------------------------------------------


def _run_protocol_a(
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    config: CrucibleConfig,
    bt_config: BacktestConfig,
) -> BacktestResult:
    """Deterministic T1 → T2 → T3 → T1 → … rotation."""
    dates      = sorted(fund_by_date.keys())
    price_idx  = prices.index
    test_dates = dates[TRAIN_MONTHS::HOLDING_MONTHS]

    log.info("Protocol A: %d test dates", len(test_dates))

    monthly_results: list[MonthlyResult] = []
    hit_rate_returns: list[float] = []
    track_sequence  = [1, 2, 3]

    for i, test_date in enumerate(test_dates):
        df     = fund_by_date[test_date]
        track  = track_sequence[i % 3]

        if track == 1:
            picks = _picks_t1(df, config)
        elif track == 2:
            picks = _picks_t2(df, config)
        else:
            picks = _picks_t3(df, config)

        next_month = _advance(test_date, price_idx, HOLDING_MONTHS)
        if next_month is not None and test_date in price_idx:
            port_ret, tkr_rets = _equal_weight_return(picks, test_date, next_month, prices)
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

    return BacktestResult(
        monthly_results=monthly_results,
        hit_rate_returns=hit_rate_returns,
        bt_config=bt_config,
    )


# ---------------------------------------------------------------------------
# Walk-forward: Protocol B (weighted blend)
# ---------------------------------------------------------------------------


def _run_protocol_b(
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    config: CrucibleConfig,
    bt_config: BacktestConfig,
) -> BacktestResult:
    """Weighted blend: 50% T2 / 30% T3 / 20% T1 (renormalised if any track is empty)."""
    dates      = sorted(fund_by_date.keys())
    price_idx  = prices.index
    test_dates = dates[TRAIN_MONTHS::HOLDING_MONTHS]

    log.info("Protocol B: %d test dates", len(test_dates))

    monthly_results: list[MonthlyResult] = []
    hit_rate_returns: list[float] = []

    for test_date in test_dates:
        df = fund_by_date[test_date]

        picks_t1 = _picks_t1(df, config)
        picks_t2 = _picks_t2(df, config)
        picks_t3 = _picks_t3(df, config)

        next_month = _advance(test_date, price_idx, HOLDING_MONTHS)
        if next_month is not None and test_date in price_idx:
            ret_t1, tkr_t1 = _equal_weight_return(picks_t1, test_date, next_month, prices)
            ret_t2, tkr_t2 = _equal_weight_return(picks_t2, test_date, next_month, prices)
            ret_t3, tkr_t3 = _equal_weight_return(picks_t3, test_date, next_month, prices)

            # Renormalise weights if any track is empty
            w1 = WEIGHT_T1 if picks_t1 else 0.0
            w2 = WEIGHT_T2 if picks_t2 else 0.0
            w3 = WEIGHT_T3 if picks_t3 else 0.0
            total_w = w1 + w2 + w3
            if total_w > 0:
                w1, w2, w3 = w1 / total_w, w2 / total_w, w3 / total_w
            port_ret = w1 * ret_t1 + w2 * ret_t2 + w3 * ret_t3

            all_tkr_rets = {**tkr_t1, **tkr_t2, **tkr_t3}
            bench_ret    = _benchmark_return(test_date, next_month, prices, BENCHMARK_COL)
            all_picks    = list(dict.fromkeys(picks_t1 + picks_t2 + picks_t3))  # preserve order, dedupe

            monthly_results.append(MonthlyResult(
                date=test_date,
                portfolio_return=port_ret,
                benchmark_return=bench_ret,
                n_picks=len(all_picks),
                tickers=all_picks,
                ticker_returns=all_tkr_rets,
            ))

        # Hit rate: union of all picks (12m forward)
        hit_date     = _advance(test_date, price_idx, HIT_RATE_MONTHS)
        all_picks_hr = list(dict.fromkeys(picks_t1 + picks_t2 + picks_t3))
        if hit_date is not None and test_date in price_idx:
            for ticker in all_picks_hr:
                r = _single_return(ticker, test_date, hit_date, prices)
                if r is not None:
                    hit_rate_returns.append(r)

    return BacktestResult(
        monthly_results=monthly_results,
        hit_rate_returns=hit_rate_returns,
        bt_config=bt_config,
    )


# ---------------------------------------------------------------------------
# Walk-forward: Track 2 baseline
# ---------------------------------------------------------------------------


def _run_track2_baseline(
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    config: CrucibleConfig,
    bt_config: BacktestConfig,
) -> BacktestResult:
    """Track 2 alone — same walk-forward as run_backtest_track2_sp500.py."""
    dates      = sorted(fund_by_date.keys())
    price_idx  = prices.index
    test_dates = dates[TRAIN_MONTHS::HOLDING_MONTHS]

    log.info("Track 2 baseline: %d test dates", len(test_dates))

    monthly_results: list[MonthlyResult] = []
    hit_rate_returns: list[float] = []

    for test_date in test_dates:
        df    = fund_by_date[test_date]
        picks = _picks_t2(df, config)

        if not picks:
            continue

        next_month = _advance(test_date, price_idx, HOLDING_MONTHS)
        if next_month is not None and test_date in price_idx:
            port_ret, tkr_rets = _equal_weight_return(picks, test_date, next_month, prices)
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

    return BacktestResult(
        monthly_results=monthly_results,
        hit_rate_returns=hit_rate_returns,
        bt_config=bt_config,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _summary_row(
    label: str,
    result: BacktestResult,
    bench_total: float,
) -> dict:
    port_rets = result.portfolio_returns()
    bench_rets = result.benchmark_returns()
    p_total   = total_return(port_rets)
    p_sharpe  = sharpe_ratio(port_rets, RISK_FREE_ANNUAL)
    p_mdd     = max_drawdown(port_rets)
    p_hr      = hit_rate(result.hit_rate_returns)
    pick_counts = [m.n_picks for m in result.monthly_results]
    avg_picks = float(np.mean(pick_counts)) if pick_counts else 0.0
    unique    = len({t for m in result.monthly_results for t in m.tickers})
    return {
        "protocol": label,
        "total_return": p_total,
        "excess_vs_sp500": p_total - bench_total,
        "sharpe": p_sharpe,
        "max_drawdown": p_mdd,
        "hit_rate_12m": p_hr,
        "avg_picks_month": avg_picks,
        "unique_tickers": unique,
        "test_months": len(result.monthly_results),
    }


def _generate_report(
    r_a: BacktestResult,
    r_b: BacktestResult,
    r_t2: BacktestResult,
    output_path: Path,
) -> None:
    bench_rets  = r_t2.benchmark_returns()
    bench_total = total_return(bench_rets)

    rows = [
        _summary_row("A — Deterministic T1→T2→T3 rotation", r_a, bench_total),
        _summary_row("B — Weighted blend (50%T2/30%T3/20%T1)", r_b, bench_total),
        _summary_row("Track 2 alone (baseline)", r_t2, bench_total),
        {
            "protocol": "SP500 benchmark",
            "total_return": bench_total,
            "excess_vs_sp500": 0.0,
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "hit_rate_12m": float("nan"),
            "avg_picks_month": float("nan"),
            "unique_tickers": 0,
            "test_months": len(r_t2.monthly_results),
        },
    ]

    def _pct(v: float) -> str:
        if np.isnan(v):
            return "—"
        return f"{v:+.2%}" if abs(v) < 10 else f"{v:.2%}"

    def _f2(v: float) -> str:
        return "—" if np.isnan(v) else f"{v:.2f}"

    def _f1(v: float) -> str:
        return "—" if np.isnan(v) else f"{v:.1f}"

    def _i(v: float) -> str:
        return "—" if np.isnan(v) else str(int(v))

    header = (
        "| Protocol | Total Return | Excess vs SP500 | Sharpe | "
        "Max Drawdown | Hit Rate (12m) | Avg Picks/Mo | Unique Tickers |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    table_rows = []
    for r in rows:
        table_rows.append(
            f"| {r['protocol']} "
            f"| {_pct(r['total_return'])} "
            f"| {_pct(r['excess_vs_sp500'])} "
            f"| {_f2(r['sharpe'])} "
            f"| {_pct(r['max_drawdown'])} "
            f"| {_pct(r['hit_rate_12m'])} "
            f"| {_f1(r['avg_picks_month'])} "
            f"| {_i(float(r['unique_tickers']))} |"
        )

    # --- interpretation ---
    rot_vs_t2 = rows[0]["total_return"] - rows[2]["total_return"]
    blend_vs_t2 = rows[1]["total_return"] - rows[2]["total_return"]

    interp_lines = []
    if rot_vs_t2 > 0:
        interp_lines.append(
            f"Protocol A (rotation) **outperformed** Track 2 alone by {rot_vs_t2:+.2%}."
        )
    else:
        interp_lines.append(
            f"Protocol A (rotation) **underperformed** Track 2 alone by {rot_vs_t2:+.2%}. "
            "Cyclically spending 2/3 of months in T1/T3 diluted growth alpha."
        )

    if blend_vs_t2 > 0:
        interp_lines.append(
            f"Protocol B (weighted blend) **outperformed** Track 2 alone by {blend_vs_t2:+.2%}."
        )
    else:
        interp_lines.append(
            f"Protocol B (weighted blend) **underperformed** Track 2 alone by {blend_vs_t2:+.2%}. "
            "T1 and T3 exposures reduced return without proportional drawdown benefit."
        )

    # Compare max drawdown
    mdd_a  = rows[0]["max_drawdown"]
    mdd_b  = rows[1]["max_drawdown"]
    mdd_t2 = rows[2]["max_drawdown"]
    if not np.isnan(mdd_a) and not np.isnan(mdd_t2):
        if mdd_a > mdd_t2:
            interp_lines.append(
                f"Rotation (A) improved max drawdown to {mdd_a:.2%} vs T2 alone ({mdd_t2:.2%})."
            )
        else:
            interp_lines.append(
                f"Rotation (A) did not improve max drawdown ({mdd_a:.2%} vs T2 {mdd_t2:.2%})."
            )
    if not np.isnan(mdd_b) and not np.isnan(mdd_t2):
        if mdd_b > mdd_t2:
            interp_lines.append(
                f"Weighted blend (B) improved max drawdown to {mdd_b:.2%} vs T2 alone ({mdd_t2:.2%})."
            )

    lines = [
        "# Combined Rotation Backtest — SP500 Universe",
        "",
        f"**Run date:** {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Universe:** SP500 (~503 tickers)  ",
        f"**Training warm-up:** {TRAIN_MONTHS} months (2013-01 → 2014-12)  ",
        f"**Test window:** 2015-01 → 2024-12  ",
        f"**Holding period:** {HOLDING_MONTHS} month  ",
        f"**Portfolio size (per track):** top {TOP_N}  ",
        "",
        "---",
        "",
        "## Protocol definitions",
        "",
        "**Protocol A — Deterministic rotation:**  ",
        "Each month, picks come exclusively from one track in the sequence T1→T2→T3→T1→…  ",
        "Month 1 = Track 1, month 2 = Track 2, month 3 = Track 3, month 4 = Track 1, etc.",
        "",
        "**Protocol B — Weighted blend:**  ",
        "Each month, picks come from all three tracks simultaneously.  ",
        "Portfolio return = 20% × equal-weight T1 return + 50% × T2 return + 30% × T3 return.  ",
        "Weights renormalised proportionally if any track produces no picks.",
        "",
        "**Track 2 alone (baseline):**  ",
        "Standard Track 2 Growth Inflection walk-forward, top 20 equal-weight, 1-month hold.",
        "",
        "---",
        "",
        "## Results",
        "",
        header,
        sep,
    ] + table_rows + [
        "",
        "---",
        "",
        "## Interpretation",
        "",
    ] + [f"- {l}" for l in interp_lines] + [
        "",
        "---",
        "",
        "> **Data integrity:** Fundamentals from SEC EDGAR (point-in-time, filed-date filtered).  ",
        "> Prices from yfinance (OHLCV only). No look-ahead bias.  ",
        "> Past backtest performance does not guarantee future results.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Report saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not SP500_CACHE.exists():
        log.error(
            "SP500 snapshot cache not found at %s. "
            "Run scripts/run_backtest_track2_sp500.py first to build the cache.",
            SP500_CACHE,
        )
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config    = CrucibleConfig(account_currency="USD")
    bt_config = BacktestConfig(
        train_months=TRAIN_MONTHS,
        top_n=TOP_N,
        holding_months=HOLDING_MONTHS,
        hit_rate_months=HIT_RATE_MONTHS,
        risk_free_annual=RISK_FREE_ANNUAL,
        benchmark_col=BENCHMARK_COL,
    )

    stop_mem = threading.Event()
    threading.Thread(
        target=_memory_monitor, args=(stop_mem,), daemon=True, name="mem-monitor"
    ).start()

    try:
        log.info("Loading SP500 snapshot cache …")
        fund_by_date: dict[pd.Timestamp, pd.DataFrame] = joblib.load(SP500_CACHE)
        log.info("Snapshots loaded: %d dates", len(fund_by_date))

        log.info("Fetching SP500 tickers for price download …")
        tickers = fetch_sp500_tickers()

        log.info("Fetching prices %s → %s …", PRICE_FETCH_START, PRICE_FETCH_END)
        prices = _fetch_prices_parallel(tickers, PRICE_FETCH_START, PRICE_FETCH_END)
        if prices.empty:
            log.error("No price data — aborting")
            sys.exit(1)

        attach_momentum(fund_by_date, prices)
        log.info("Momentum attached to all snapshots")

        log.info("Running Track 2 baseline …")
        r_t2 = _run_track2_baseline(fund_by_date, prices, config, bt_config)

        log.info("Running Protocol A (deterministic rotation) …")
        r_a  = _run_protocol_a(fund_by_date, prices, config, bt_config)

        log.info("Running Protocol B (weighted blend) …")
        r_b  = _run_protocol_b(fund_by_date, prices, config, bt_config)

        report_path = RESULTS_DIR / "combined_SP500_2013_2024_rotation_report.md"
        _generate_report(r_a, r_b, r_t2, report_path)

        # --- console summary ---
        bench_rets  = r_t2.benchmark_returns()
        bench_total = total_return(bench_rets)

        def _row(label: str, result: BacktestResult) -> None:
            port_rets = result.portfolio_returns()
            p_total   = total_return(port_rets)
            p_sharpe  = sharpe_ratio(port_rets, RISK_FREE_ANNUAL)
            p_mdd     = max_drawdown(port_rets)
            p_hr      = hit_rate(result.hit_rate_returns)
            print(
                f"  {label:<40}  return={p_total:+.2%}  "
                f"excess={p_total-bench_total:+.2%}  "
                f"sharpe={p_sharpe:.2f}  "
                f"mdd={p_mdd:.2%}  "
                f"hr={p_hr:.2%}"
            )

        print("\n" + "═" * 80)
        print("  Combined Rotation Backtest — SP500 — 2015-01 → 2024-12")
        print("═" * 80)
        _row("Protocol A (T1→T2→T3 rotation)", r_a)
        _row("Protocol B (50%T2/30%T3/20%T1)", r_b)
        _row("Track 2 alone (baseline)", r_t2)
        print(f"  {'SP500 benchmark':<40}  return={bench_total:+.2%}")
        print("═" * 80)
        print(f"\n  Report: {report_path}")
        print()

    finally:
        stop_mem.set()


if __name__ == "__main__":
    main()
