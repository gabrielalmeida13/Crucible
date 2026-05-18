#!/usr/bin/env python3
"""3×2 matrix backtest: holding_months=[1, 3, 12] × universe=['SP500', 'RUSSELL1000'].

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
    _to_float,
    fetch_financials,
    fetch_russell1000_tickers,
    fetch_sp500_tickers,
)

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

RESULTS_DIR  = ROOT / "data" / "results"
EDGAR_DIR    = ROOT / "data" / "raw" / "edgar" / "companyfacts"
CIK_MAP_PATH = ROOT / "data" / "raw" / "edgar" / "cik_mapping.json"

IWB_URL = (
    "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
)

PRICE_WORKERS    = 20    # I/O-bound yfinance
SNAPSHOT_WORKERS = 4     # memory-safe EDGAR snapshot workers
MEMORY_LOG_SECS  = 600   # log RSS every 10 minutes

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
# Snapshot helpers
# ---------------------------------------------------------------------------


def _linear_slope(values: list[float]) -> float | None:
    """OLS slope for a list of evenly-spaced values."""
    n = len(values)
    if n < 2:
        return None
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((i - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


def _compute_snapshot_row(ticker: str, pivoted: dict[str, pd.Series]) -> dict:
    """Derive the processed row that apply_filters() and score() expect."""
    rev    = pivoted.get("Total Revenue",             pd.Series(dtype=float))
    gp     = pivoted.get("Gross Profit",              pd.Series(dtype=float))
    ni     = pivoted.get("Net Income",                pd.Series(dtype=float))
    fcf    = pivoted.get("Free Cash Flow",            pd.Series(dtype=float))
    td     = pivoted.get("Total Debt",                pd.Series(dtype=float))
    cash   = pivoted.get("Cash And Cash Equivalents", pd.Series(dtype=float))
    ebitda = pivoted.get("EBITDA",                    pd.Series(dtype=float))
    equity = pivoted.get("Total Equity",              pd.Series(dtype=float))

    data_years = len(rev)

    roic_vals: list[float] = []
    for fy in ni.index:
        ni_v = _to_float(ni.get(fy))
        eq_v = _to_float(equity.get(fy))
        td_v = _to_float(td.get(fy))
        if ni_v is not None and eq_v is not None and td_v is not None:
            denom = eq_v + td_v
            if denom > 0:
                roic_vals.append(ni_v / denom)
    roic_avg = sum(roic_vals) / len(roic_vals) if roic_vals else None

    fcf_vals   = [_to_float(v) for v in fcf.values]
    fcf_pos    = float(sum(1 for v in fcf_vals if v is not None and v > 0))
    fcf_latest = _to_float(fcf.iloc[-1]) if not fcf.empty else None

    td_last = _to_float(td.iloc[-1])     if not td.empty     else None
    ca_last = _to_float(cash.iloc[-1])   if not cash.empty   else None
    eb_last = _to_float(ebitda.iloc[-1]) if not ebitda.empty else None
    nd_eb: float | None = None
    if td_last is not None and ca_last is not None and eb_last and eb_last > 0:
        nd_eb = (td_last - ca_last) / eb_last

    rev_vals = [_to_float(v) for v in rev.values if _to_float(v) is not None]
    rev_growth = (
        float(sum(1 for i in range(1, len(rev_vals)) if rev_vals[i] > rev_vals[i - 1]))
        if len(rev_vals) >= 2 else None
    )

    gm_vals: list[float] = []
    for fy in rev.index:
        r = _to_float(rev.get(fy))
        g = _to_float(gp.get(fy))
        if r and r > 0 and g is not None:
            gm_vals.append(g / r)

    return {
        "ticker":                        ticker,
        "sector":                        None,
        "sub_industry":                  None,
        "currency":                      "USD",
        "p_e":                           None,
        "p_fcf":                         None,
        "ev_ebitda":                     None,
        "data_years":                    data_years,
        "insufficient_data":             data_years < 3,
        "roic_proxy_avg":                roic_avg,
        "fcf_latest":                    fcf_latest,
        "fcf_positive_years":            fcf_pos if not fcf.empty else None,
        "net_debt_ebitda":               nd_eb,
        "revenue_growth_positive_years": rev_growth,
        "gross_margin_latest":           gm_vals[-1] if gm_vals else None,
        "gross_margin_avg":              sum(gm_vals) / len(gm_vals) if gm_vals else None,
        "gross_margin_trend_slope":      _linear_slope(gm_vals),
    }


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
# Parallel monthly snapshot building
# ---------------------------------------------------------------------------


def _build_one_snapshot(
    date: pd.Timestamp,
    tickers: list[str],
    edgar_dir: Path,
    cik_map: dict[str, str],
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Build a point-in-time fundamentals snapshot for one monthly date."""
    panel = fetch_financials(tickers, date, edgar_dir, cik_map)

    # Pre-group by ticker to avoid O(n²) repeated DataFrame filtering
    if not panel.empty:
        panel_by_ticker: dict[str, pd.DataFrame] = {
            t: g for t, g in panel.groupby("ticker")
        }
    else:
        panel_by_ticker = {}

    empty = pd.DataFrame(columns=["ticker", "fiscal_year", "metric", "value"])
    rows = []
    for ticker in tickers:
        ticker_df = panel_by_ticker.get(ticker, empty)
        pivoted = (
            {
                metric: grp.set_index("fiscal_year")["value"].sort_index()
                for metric, grp in ticker_df.groupby("metric")
            }
            if not ticker_df.empty else {}
        )
        rows.append(_compute_snapshot_row(ticker, pivoted))

    df = pd.DataFrame(rows).set_index("ticker")
    n_ok = int((~df["insufficient_data"]).sum())
    log.debug("%s  %d/%d tickers with sufficient data", date.date(), n_ok, len(tickers))
    return date, df


def _build_fundamentals_parallel(
    tickers: list[str],
    monthly_dates: pd.DatetimeIndex,
    edgar_dir: Path,
    cik_map: dict[str, str],
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Build monthly fundamentals snapshots with 4 workers (memory-safe).

    4 workers keep peak concurrent memory at ~4× per-snapshot overhead while the
    300-entry LRU cache prevents simultaneous loading of all company JSONs.
    """
    result: dict[pd.Timestamp, pd.DataFrame] = {}
    total = len(monthly_dates)
    done = 0

    with ThreadPoolExecutor(max_workers=SNAPSHOT_WORKERS) as pool:
        futures = {
            pool.submit(_build_one_snapshot, d, tickers, edgar_dir, cik_map): d
            for d in monthly_dates
        }
        for future in as_completed(futures):
            date, df = future.result()
            result[date] = df
            done += 1
            if done % 12 == 0 or done == total:
                log.info("Snapshots: %d / %d built", done, total)

    return result


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
        for universe_id in MATRIX_UNIVERSES:
            log.info("=" * 60)
            log.info("Universe: %s", universe_id)
            log.info("=" * 60)

            # ── Tickers ──────────────────────────────────────────────────────
            log.info("Fetching %s ticker list …", universe_id)
            if universe_id == "SP500":
                tickers = fetch_sp500_tickers()
            else:
                tickers = fetch_russell1000_tickers(IWB_URL)
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
            fund_by_date = _build_fundamentals_parallel(
                tickers, monthly_dates, EDGAR_DIR, cik_map
            )
            log.info("Snapshots complete: %d dates", len(fund_by_date))

            # ── Run each holding period sequentially ──────────────────────────
            for holding in MATRIX_HOLDING_MONTHS:
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
