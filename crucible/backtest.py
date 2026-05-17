"""Walk-forward backtest engine for the Crucible screener.

IMPORTANT: All historical fundamentals and prices must come from FMP
point-in-time snapshots. Using yfinance data will introduce look-ahead
bias because yfinance retroactively rewrites restated financials.

Walk-forward design
-------------------
Months 1..train_months  — warm-up / calibration window (no trades placed)
Month train_months+1    — first live test point
  ...advance one month, repeat until data exhausted...

Portfolio construction at each test month T
-------------------------------------------
1. Apply Layer 1 filters to the fundamentals snapshot as of T.
2. Score the survivors; take the top `top_n` by composite_score.
3. Equal-weight the portfolio; hold for `holding_months`.

Metrics computed
----------------
- Total return (compound) vs S&P 500 benchmark
- Annualised Sharpe ratio (excess over risk-free rate)
- Maximum peak-to-trough drawdown
- Hit rate: % of individual picks with a positive 12-month forward return

Sensitivity analysis
--------------------
`run_sensitivity` re-runs the backtest across a grid of ROIC thresholds and
reports how metrics change, giving an honest view of threshold fragility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from crucible.config import CrucibleConfig, FilterThresholds
from crucible.filters import apply_filters
from crucible.scorer import score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """Parameters governing walk-forward execution and metric calculation."""

    train_months: int = 24
    top_n: int = 20
    holding_months: int = 1
    hit_rate_months: int = 12
    risk_free_annual: float = 0.04
    benchmark_col: str = "SP500"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class MonthlyResult:
    """Outcome of a single monthly test step."""

    date: pd.Timestamp
    portfolio_return: float
    benchmark_return: float
    n_picks: int
    tickers: list[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Full walk-forward result with individual observations and summary metrics."""

    monthly_results: list[MonthlyResult]
    hit_rate_returns: list[float]
    bt_config: BacktestConfig

    def portfolio_returns(self) -> list[float]:
        return [m.portfolio_return for m in self.monthly_results]

    def benchmark_returns(self) -> list[float]:
        return [m.benchmark_return for m in self.monthly_results]

    def to_dataframe(self) -> pd.DataFrame:
        """Monthly results as a tidy DataFrame."""
        if not self.monthly_results:
            return pd.DataFrame(columns=["date", "portfolio_return",
                                         "benchmark_return", "n_picks"])
        return pd.DataFrame([
            {
                "date": m.date,
                "portfolio_return": m.portfolio_return,
                "benchmark_return": m.benchmark_return,
                "n_picks": m.n_picks,
            }
            for m in self.monthly_results
        ])


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_backtest(
    fundamentals_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    crucible_config: CrucibleConfig,
    bt_config: BacktestConfig | None = None,
) -> BacktestResult:
    """Execute the walk-forward backtest.

    Parameters
    ----------
    fundamentals_by_date:
        Mapping of month-end date → processed fundamentals DataFrame (same
        schema as cleaner.py output).  Must be point-in-time (FMP).
    prices:
        Monthly price series.  Rows = dates, columns = tickers + benchmark.
        Use total-return prices where possible.
    crucible_config:
        Live CrucibleConfig (filters, scoring weights, FX settings).
    bt_config:
        Walk-forward parameters.  Defaults to BacktestConfig().
    """
    bt_config = bt_config or BacktestConfig()
    dates = sorted(fundamentals_by_date.keys())
    price_idx = prices.index

    if len(dates) <= bt_config.train_months:
        raise ValueError(
            f"Need more than {bt_config.train_months} months of fundamentals data "
            f"(got {len(dates)})"
        )

    monthly_results: list[MonthlyResult] = []
    hit_rate_returns: list[float] = []

    test_dates = dates[bt_config.train_months:]
    logger.info(
        "Walk-forward: %d training months, %d test months",
        bt_config.train_months, len(test_dates),
    )

    for test_date in test_dates:
        fundamentals = fundamentals_by_date[test_date]

        try:
            filtered = apply_filters(fundamentals, crucible_config.filters)
        except Exception:
            logger.warning("Filter error at %s — skipping", test_date, exc_info=True)
            continue

        if filtered.empty:
            logger.debug("No picks at %s — all tickers eliminated by filters", test_date)
            continue

        scored = score(filtered, crucible_config)
        picks = scored.head(bt_config.top_n).index.tolist()

        # 1-month portfolio and benchmark returns
        next_month = _advance(test_date, price_idx, bt_config.holding_months)
        if next_month is not None and test_date in price_idx:
            port_ret = _portfolio_return(picks, test_date, next_month, prices)
            bench_ret = _benchmark_return(
                test_date, next_month, prices, bt_config.benchmark_col
            )
            monthly_results.append(MonthlyResult(
                date=test_date,
                portfolio_return=port_ret,
                benchmark_return=bench_ret,
                n_picks=len(picks),
                tickers=picks,
            ))

        # 12-month returns for hit rate (one observation per pick per month)
        hit_date = _advance(test_date, price_idx, bt_config.hit_rate_months)
        if hit_date is not None and test_date in price_idx:
            for ticker in picks:
                r = _single_return(ticker, test_date, hit_date, prices)
                if r is not None:
                    hit_rate_returns.append(r)

    logger.info(
        "Backtest complete: %d monthly periods, %d hit-rate observations",
        len(monthly_results), len(hit_rate_returns),
    )
    return BacktestResult(
        monthly_results=monthly_results,
        hit_rate_returns=hit_rate_returns,
        bt_config=bt_config,
    )


def run_sensitivity(
    fundamentals_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    crucible_config: CrucibleConfig,
    bt_config: BacktestConfig | None = None,
    roic_thresholds: tuple[float, ...] = (0.10, 0.12, 0.15, 0.18, 0.20),
) -> pd.DataFrame:
    """Re-run the backtest for each ROIC threshold; return a comparison table.

    All other filter thresholds and config fields are held constant.
    """
    bt_config = bt_config or BacktestConfig()
    rows = []

    for roic_min in roic_thresholds:
        modified_cfg = replace(
            crucible_config,
            filters=replace(crucible_config.filters, roic_min=roic_min),
        )
        result = run_backtest(fundamentals_by_date, prices, modified_cfg, bt_config)
        port_rets = result.portfolio_returns()
        bench_rets = result.benchmark_returns()

        avg_n = (
            float(np.mean([m.n_picks for m in result.monthly_results]))
            if result.monthly_results else 0.0
        )
        rows.append({
            "roic_min": roic_min,
            "n_test_months": len(result.monthly_results),
            "avg_picks": round(avg_n, 1),
            "portfolio_total_return": total_return(port_rets),
            "benchmark_total_return": total_return(bench_rets),
            "sharpe_ratio": sharpe_ratio(port_rets, bt_config.risk_free_annual),
            "max_drawdown": max_drawdown(port_rets),
            "hit_rate": hit_rate(result.hit_rate_returns),
        })

    return pd.DataFrame(rows)


def generate_report(
    result: BacktestResult,
    sensitivity: pd.DataFrame,
    output_path: Path,
    crucible_config: CrucibleConfig | None = None,
) -> None:
    """Write a Markdown backtest report to output_path."""
    bt = result.bt_config
    port_rets = result.portfolio_returns()
    bench_rets = result.benchmark_returns()

    port_total = total_return(port_rets)
    bench_total = total_return(bench_rets)
    port_sharpe = sharpe_ratio(port_rets, bt.risk_free_annual)
    port_mdd = max_drawdown(port_rets)
    hr = hit_rate(result.hit_rate_returns)
    excess = port_total - bench_total

    def _pct(v: float) -> str:
        return f"{v:.2%}" if not np.isnan(v) else "—"

    def _f2(v: float) -> str:
        return f"{v:.2f}" if not np.isnan(v) else "—"

    lines: list[str] = [
        "# Crucible Backtest Report",
        "",
        f"**Generated:** {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
        "## Walk-forward Parameters",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Training window | {bt.train_months} months |",
        f"| Portfolio size (top-N) | {bt.top_n} |",
        f"| Rebalance / holding period | {bt.holding_months} month(s) |",
        f"| Hit-rate measurement window | {bt.hit_rate_months} months |",
        f"| Risk-free rate (annual) | {bt.risk_free_annual:.1%} |",
        f"| Benchmark | {bt.benchmark_col} |",
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
        "## ROIC Threshold Sensitivity",
        "",
        "How sensitive are results to the ROIC filter threshold?",
        "All other parameters held constant.",
        "",
        _sensitivity_table(sensitivity),
        "",
        "---",
        "",
        "## Conclusion",
        "",
        _conclusion(port_total, bench_total, hr, port_sharpe),
        "",
        "---",
        "",
        "> **Data integrity note:** This backtest requires FMP point-in-time financial",
        "> statements. Results are only valid if the fundamentals snapshots were built",
        "> from FMP data with no look-ahead (Q1 reports available after their filing",
        "> date, not their fiscal quarter end). Past backtest performance does not",
        "> guarantee future results.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Backtest report saved: %s", output_path)


# ---------------------------------------------------------------------------
# Metric functions (public — used directly in tests and report)
# ---------------------------------------------------------------------------


def total_return(monthly_returns: list[float]) -> float:
    """Compound total return from a sequence of periodic returns."""
    if not monthly_returns:
        return 0.0
    return float(np.prod([1.0 + r for r in monthly_returns]) - 1.0)


def sharpe_ratio(
    monthly_returns: list[float],
    risk_free_annual: float = 0.04,
) -> float:
    """Annualised Sharpe ratio (excess return / volatility × √12).

    Returns NaN when fewer than 2 observations are available.
    """
    if len(monthly_returns) < 2:
        return float("nan")
    monthly_rf = (1.0 + risk_free_annual) ** (1.0 / 12) - 1.0
    excess = np.array(monthly_returns, dtype=float) - monthly_rf
    std = float(excess.std(ddof=1))
    if std < 1e-12:
        return float("nan")
    return float((excess.mean() / std) * np.sqrt(12))


def max_drawdown(monthly_returns: list[float]) -> float:
    """Maximum peak-to-trough drawdown (≤ 0).

    Computed on the cumulative wealth index derived from monthly_returns.
    Returns 0.0 for an empty series.
    """
    if not monthly_returns:
        return 0.0
    wealth = np.cumprod([1.0 + r for r in monthly_returns])
    peaks = np.maximum.accumulate(wealth)
    drawdowns = (wealth - peaks) / peaks
    return float(drawdowns.min())


def hit_rate(returns: list[float]) -> float:
    """Fraction of picks with a strictly positive return.

    Returns NaN for an empty list.
    """
    if not returns:
        return float("nan")
    return float(sum(1 for r in returns if r > 0) / len(returns))


def cumulative_return_series(monthly_returns: list[float]) -> pd.Series:
    """Cumulative wealth index (base 1.0) from monthly return list."""
    if not monthly_returns:
        return pd.Series([1.0])
    return pd.Series(np.cumprod([1.0 + r for r in monthly_returns]))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _advance(
    date: pd.Timestamp,
    index: pd.DatetimeIndex,
    steps: int,
) -> pd.Timestamp | None:
    """Return the date `steps` positions ahead of `date` in `index`."""
    locs = index.searchsorted(date)
    target = locs + steps
    if target >= len(index):
        return None
    return index[target]


def _single_return(
    ticker: str,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    prices: pd.DataFrame,
) -> float | None:
    """Simple price return for one ticker between two dates. None if unavailable."""
    if ticker not in prices.columns:
        return None
    try:
        p0 = prices.at[t0, ticker]
        p1 = prices.at[t1, ticker]
    except KeyError:
        return None
    if pd.isna(p0) or pd.isna(p1) or p0 <= 0:
        return None
    return float(p1 / p0 - 1.0)


def _portfolio_return(
    picks: list[str],
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    prices: pd.DataFrame,
) -> float:
    """Equal-weighted return of picks from t0 to t1. 0.0 if none are priceable."""
    rets = [_single_return(t, t0, t1, prices) for t in picks]
    valid = [r for r in rets if r is not None]
    return float(np.mean(valid)) if valid else 0.0


def _benchmark_return(
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    prices: pd.DataFrame,
    benchmark_col: str,
) -> float:
    """Return of the benchmark column; fallback to equal-weight of all columns."""
    if benchmark_col in prices.columns:
        r = _single_return(benchmark_col, t0, t1, prices)
        if r is not None:
            return r
    rets = [_single_return(c, t0, t1, prices) for c in prices.columns]
    valid = [r for r in rets if r is not None]
    return float(np.mean(valid)) if valid else 0.0


def _sensitivity_table(df: pd.DataFrame) -> str:
    """Format a DataFrame as a Markdown table without external dependencies."""
    if df.empty:
        return "_No sensitivity data available._"

    fmt: dict[str, str] = {
        "roic_min": ".0%",
        "n_test_months": "d",
        "avg_picks": ".1f",
        "portfolio_total_return": ".2%",
        "benchmark_total_return": ".2%",
        "sharpe_ratio": ".2f",
        "max_drawdown": ".2%",
        "hit_rate": ".2%",
    }

    def _format_cell(col: str, val: object) -> str:
        spec = fmt.get(col, "")
        if spec == "d":
            return str(int(val)) if not (isinstance(val, float) and np.isnan(val)) else "—"
        try:
            return format(float(val), spec)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return str(val)

    headers = list(df.columns)
    rows = [
        [_format_cell(c, row[c]) for c in headers]
        for _, row in df.iterrows()
    ]

    col_widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]

    def _row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, col_widths)) + " |"

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    return "\n".join([_row(headers), sep] + [_row(r) for r in rows])


def _conclusion(
    port_total: float,
    bench_total: float,
    hr: float,
    sharpe: float,
) -> str:
    excess = port_total - bench_total
    if port_total > bench_total:
        verdict = (
            f"The Crucible screener **outperformed** the benchmark over the test period "
            f"({port_total:.2%} vs {bench_total:.2%}, excess {excess:+.2%})."
        )
    else:
        verdict = (
            f"The Crucible screener **underperformed** the benchmark over the test period "
            f"({port_total:.2%} vs {bench_total:.2%}, excess {excess:+.2%}). "
            "This is an honest outcome. Review filter thresholds, scoring weights, "
            "and the holding period before drawing conclusions."
        )

    hr_str = (
        f" The hit rate of **{hr:.2%}** means {hr:.0%} of individual 12-month picks "
        "were profitable."
        if not np.isnan(hr) else ""
    )

    sharpe_str = (
        f" The annualised Sharpe ratio of **{sharpe:.2f}** "
        + ("is above 0.5, suggesting the return was not purely noise."
           if sharpe > 0.5 else "is below 0.5, indicating poor risk-adjusted return.")
        if not np.isnan(sharpe) else ""
    )

    caveat = (
        "\n\n**Important caveats:** The test window must be long enough to span multiple "
        "market regimes (bull, bear, sideways). A short backtest with favourable timing "
        "is not evidence of a good strategy. The sensitivity table above shows whether "
        "results are robust to small threshold changes — fragile results are a red flag."
    )

    return verdict + hr_str + sharpe_str + caveat
