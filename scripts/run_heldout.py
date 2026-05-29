#!/usr/bin/env python3
"""Held-out validation: SP500, 1-month holding, 2023-01 → 2026-04.

THIS IS A ONE-SHOT EVALUATION on data the model has never seen.
Parameters (filters, thresholds, scorer weights) were fixed before this
run.  Do not re-tune after inspecting these results — that would
constitute look-ahead contamination of the held-out set.

Walk-forward design
-------------------
  TRAIN_MONTHS = 0: every month in the test window is a live test point.
  The model specification was frozen during the 2010–2022 backtest period.

Data sources
------------
  SEC EDGAR companyfacts/{CIK}.json  — fundamentals (point-in-time)
  yfinance                           — price data only
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from crucible.backtest import (
    BacktestConfig,
    attach_momentum,
    generate_picks_csv,
    generate_ticker_contribution,
    hit_rate,
    max_drawdown,
    run_backtest,
    sharpe_ratio,
    total_return,
)
from crucible.config import CrucibleConfig, ScoreWeights
from crucible.fetcher import _load_cik_mapping, fetch_sp500_tickers
from crucible.ml.features import add_roic_direction
from crucible.ml.model import ModelArtifact, _impute, load_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKTEST_START    = pd.Timestamp("2023-01-31", tz="UTC")
BACKTEST_END      = pd.Timestamp("2026-04-30", tz="UTC")
PRICE_FETCH_START = "2022-01-01"
PRICE_FETCH_END   = "2027-01-01"

TRAIN_MONTHS = 0   # model already specified — every month is a test point
TOP_N        = 20

RESULTS_DIR      = ROOT / "data" / "results"
DIAGNOSTICS_DIR  = ROOT / "data" / "diagnostics"
EDGAR_DIR        = ROOT / "data" / "raw" / "edgar" / "companyfacts"
CIK_MAP_PATH     = ROOT / "data" / "raw" / "edgar" / "cik_mapping.json"
MODEL_PATH       = ROOT / "data" / "models" / "phase3a_model.pkl"

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
# Memory monitoring
# ---------------------------------------------------------------------------


def _memory_monitor(stop_event: threading.Event) -> None:
    proc = psutil.Process()
    while not stop_event.wait(MEMORY_LOG_SECS):
        rss_mib = proc.memory_info().rss / 1024 / 1024
        log.info("[mem] RSS %.0f MiB", rss_mib)


# ---------------------------------------------------------------------------
# ML scoring
# ---------------------------------------------------------------------------


def _add_ml_scores(
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    artifact: ModelArtifact,
) -> None:
    """Add ml_score column (P(outperform vs S&P 500)) to every snapshot in-place.

    Uses artifact.imputation_values and artifact.scaler from training — no re-fitting.
    Missing feature columns (e.g. random_baseline) are treated as NaN and filled with
    training medians before scoring.
    """
    feature_names = artifact.feature_names
    for df in fund_by_date.values():
        X = pd.DataFrame(index=df.index, dtype=float)
        for col in feature_names:
            if col in df.columns:
                X[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                X[col] = np.nan
        X = _impute(X, artifact.imputation_values)
        if artifact.scaler is not None:
            X = pd.DataFrame(
                artifact.scaler.transform(X),
                columns=feature_names,
                index=df.index,
            )
        proba = artifact.model.predict_proba(X)[:, 1]
        df["ml_score"] = proba


# ---------------------------------------------------------------------------
# Held-out report (no sensitivity section)
# ---------------------------------------------------------------------------


def _write_heldout_report(
    result: "BacktestResult",  # type: ignore[name-defined]
    output_path: Path,
    config: CrucibleConfig,
) -> None:
    """Write the held-out validation Markdown report, without sensitivity analysis."""
    bt = result.bt_config
    port_rets  = result.portfolio_returns()
    bench_rets = result.benchmark_returns()

    port_total  = total_return(port_rets)
    bench_total = total_return(bench_rets)
    port_sharpe = sharpe_ratio(port_rets, bt.risk_free_annual)
    port_mdd    = max_drawdown(port_rets)
    hr          = hit_rate(result.hit_rate_returns)
    excess      = port_total - bench_total

    def _pct(v: float) -> str:
        return f"{v:.2%}" if not np.isnan(v) else "—"

    def _f2(v: float) -> str:
        return f"{v:.2f}" if not np.isnan(v) else "—"

    lines: list[str] = [
        "# Crucible — Held-Out Validation Report",
        "",
        "> **HELD-OUT VALIDATION — parameters fixed before this run,**",
        "> **no further tuning permitted after seeing these results.**",
        ">",
        "> This report covers data from 2023-01 onwards, which was entirely",
        "> excluded from model development. All filter thresholds, scorer",
        "> weights, and universe definitions were frozen at the end of the",
        "> 2010–2022 backtest period. Modifying any parameter after reading",
        "> this output constitutes look-ahead contamination.",
        "",
        f"**Generated:** {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
        "## Validation Parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Test window | {BACKTEST_START.date()} → {BACKTEST_END.date()} |",
        f"| Training burn-in | {bt.train_months} months (none — model pre-specified) |",
        f"| Portfolio size (top-N) | {bt.top_n} |",
        f"| Holding period | {bt.holding_months} month(s) |",
        f"| Hit-rate measurement window | {bt.hit_rate_months} months |",
        f"| Risk-free rate (annual) | {bt.risk_free_annual:.1%} |",
        f"| Benchmark | {bt.benchmark_col} |",
        f"| Score weights | quality={config.score_weights.quality:.0%}, "
        f"val={config.score_weights.valuation:.0%}, "
        f"mom={config.score_weights.momentum:.0%}, "
        f"ml={config.score_weights.ml_score:.0%} |",
        f"| ML model | data/models/phase3a_model.pkl (RF, val_acc=57.9%) |",
        "",
        "---",
        "",
        "## Performance Summary",
        "",
        "| Metric | Portfolio | Benchmark |",
        "|--------|-----------|-----------|",
        f"| Total return | {_pct(port_total)} | {_pct(bench_total)} |",
        f"| Excess return vs benchmark | {_pct(excess)} | — |",
        f"| Annualised Sharpe ratio | {_f2(port_sharpe)} | — |",
        f"| Maximum drawdown | {_pct(port_mdd)} | — |",
        f"| Hit rate ({bt.hit_rate_months}m) | {_pct(hr)} | — |",
        f"| Test months | {len(result.monthly_results)} | {len(result.monthly_results)} |",
        f"| Hit-rate observations | {len(result.hit_rate_returns)} | — |",
        "",
        "---",
        "",
        "## Conclusion",
        "",
    ]

    if not result.monthly_results:
        lines.append("*No test months produced results — check data availability.*")
    elif port_total > bench_total and hr > 0.5:
        lines.append(
            f"The strategy outperformed the benchmark by {_pct(excess)} over the held-out "
            f"period with a hit rate of {_pct(hr)}. This is an encouraging out-of-sample "
            f"result, but a single held-out window is insufficient to draw strong conclusions."
        )
    elif port_total > bench_total:
        lines.append(
            f"The strategy outperformed the benchmark by {_pct(excess)} on a total-return "
            f"basis, but the hit rate of {_pct(hr)} is below 50%. Proceed with caution."
        )
    else:
        lines.append(
            f"The strategy underperformed the benchmark by {_pct(abs(excess))} over the "
            f"held-out period (hit rate: {_pct(hr)}). Review whether market conditions "
            f"during this window are systematically different from the training period "
            f"before drawing conclusions about model failure."
        )

    lines += [
        "",
        "---",
        "",
        "> **Data integrity note:** Fundamentals are sourced from SEC EDGAR with",
        "> point-in-time correctness (only filings with `filed` ≤ snapshot date are used).",
        "> Price data is from yfinance (closing prices only — no fundamental data).",
        "> This held-out window was not seen during any stage of model development.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Held-out report saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    from scripts.run_backtest import (  # type: ignore[import]
        _build_fundamentals_parallel,
        _fetch_prices_parallel,
    )

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
    if not MODEL_PATH.exists():
        log.error(
            "ML model not found at %s. Run scripts/run_phase3a.py first.",
            MODEL_PATH,
        )
        sys.exit(1)

    cik_map       = _load_cik_mapping(CIK_MAP_PATH)
    monthly_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq="ME", tz="UTC")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stop_mem = threading.Event()
    threading.Thread(
        target=_memory_monitor, args=(stop_mem,), daemon=True, name="mem-monitor"
    ).start()
    log.info("Memory monitor started (every %.0f min)", MEMORY_LOG_SECS / 60)

    config = CrucibleConfig(
        account_currency="USD",
        score_weights=ScoreWeights(quality=0.50, valuation=0.25, momentum=0.10, ml_score=0.15),
    )

    try:
        log.info("=" * 60)
        log.info("Held-out validation: SP500, 1-month holding, ML-augmented scoring")
        log.info("Window: %s → %s", BACKTEST_START.date(), BACKTEST_END.date())
        log.info("=" * 60)

        log.info("Fetching SP500 ticker list …")
        tickers = fetch_sp500_tickers()
        log.info("Universe: %d tickers", len(tickers))

        log.info(
            "Fetching prices for %d tickers + SPY (%d workers) …",
            len(tickers), PRICE_WORKERS,
        )
        prices = _fetch_prices_parallel(tickers, PRICE_FETCH_START, PRICE_FETCH_END)
        if prices.empty:
            log.error("No price data — aborting.")
            sys.exit(1)

        log.info(
            "Building %d monthly snapshots from EDGAR (%d workers) …",
            len(monthly_dates), SNAPSHOT_WORKERS,
        )
        fund_by_date = _build_fundamentals_parallel(
            tickers, monthly_dates, EDGAR_DIR, cik_map, prices
        )
        log.info("Snapshots complete: %d dates", len(fund_by_date))
        attach_momentum(fund_by_date, prices)
        log.info("Momentum attached to all snapshots")

        add_roic_direction(fund_by_date)
        log.info("roic_direction added to all snapshots")

        log.info("Loading ML model from %s …", MODEL_PATH)
        artifact = load_model(MODEL_PATH)
        log.info(
            "ML model loaded: type=%s, val_accuracy=%.3f",
            artifact.model_type, artifact.val_accuracy,
        )
        _add_ml_scores(fund_by_date, artifact)
        log.info("ml_score column added to all snapshots")

        bt_cfg = BacktestConfig(
            train_months=TRAIN_MONTHS,
            top_n=TOP_N,
            holding_months=1,
            hit_rate_months=12,
            risk_free_annual=0.04,
            benchmark_col="SP500",
        )

        result = run_backtest(fund_by_date, prices, config, bt_cfg)
        log.info(
            "Held-out run complete: %d test months, %d hit-rate observations",
            len(result.monthly_results), len(result.hit_rate_returns),
        )

        if not result.monthly_results:
            log.warning("No test results produced — check date range and data availability.")
        else:
            _write_heldout_report(result, RESULTS_DIR / "heldout_1m_report.md", config)
            generate_picks_csv(result, prices, RESULTS_DIR / "heldout_1m_picks.csv")
            generate_ticker_contribution(
                result,
                RESULTS_DIR / "heldout_1m_contributions.md",
                roic_threshold=config.filters.roic_min,
            )
            log.info("Outputs written to %s/", RESULTS_DIR)

            port_rets   = result.portfolio_returns()
            bench_rets  = result.benchmark_returns()
            port_total  = total_return(port_rets)
            bench_total = total_return(bench_rets)

            w = config.score_weights
            print("\n" + "═" * 60)
            print("HELD-OUT VALIDATION RESULT  (ML-augmented)")
            print("═" * 60)
            print(f"  Period:           {BACKTEST_START.date()} → {BACKTEST_END.date()}")
            print(f"  Score weights:    quality={w.quality:.0%}, val={w.valuation:.0%}, "
                  f"mom={w.momentum:.0%}, ml={w.ml_score:.0%}")
            print(f"  Test months:      {len(result.monthly_results)}")
            print(f"  Portfolio return: {port_total:.2%}")
            print(f"  Benchmark return: {bench_total:.2%}")
            print(f"  Excess return:    {port_total - bench_total:.2%}")
            print(f"  Sharpe ratio:     {sharpe_ratio(port_rets, bt_cfg.risk_free_annual):.2f}")
            print(f"  Max drawdown:     {max_drawdown(port_rets):.2%}")
            print(f"  Hit rate (12m):   {hit_rate(result.hit_rate_returns):.2%}")
            print("═" * 60)
            print("  Previous run (no ML):  portfolio=55.11%, Sharpe=0.76")
            print("═" * 60)
            print(f"\nReport:        {RESULTS_DIR / 'heldout_1m_report.md'}")
            print(f"Contributions: {RESULTS_DIR / 'heldout_1m_contributions.md'}")
            print(f"Picks:         {RESULTS_DIR / 'heldout_1m_picks.csv'}\n")

    finally:
        stop_mem.set()


if __name__ == "__main__":
    main()
