# Crucible Track 1 Backtest Report

**Track:** 1 — Quality Compounders  
**Universe:** RUSSELL1000  
**Holding period:** 1 month  
**Generated:** 2026-05-22 17:45 UTC

---

## Walk-forward Parameters

| Parameter | Value |
|-----------|-------|
| Training window | 24 months |
| Portfolio size (top-N) | 20 |
| Holding / rebalance | 1 month |
| Hit-rate measurement | 12 months |
| Risk-free rate | 4.0% p.a. |
| Benchmark | SP500 (SPY) |
| Snapshot start | 2013-01-31 |
| First test month | 2015-01-31 (month 25) |
| Last snapshot | 2024-12-31 |

## Filter Thresholds (Layer 1)

| # | Filter | Condition |
|---|--------|-----------|
| 1 | ROIC (5yr avg) | > 15% |
| 2 | FCF positive | ≥ 4 of last 5 years |
| 3 | Net Debt / EBITDA | < 3.0x |
| 4 | Revenue growth positive | ≥ 3 of last 5 years |
| 5 | Gross margin slope | ≥ -0.005 (stable or improving) |

## Score Weights (Layer 2)

| Component | Weight |
|-----------|--------|
| Quality | 60% |
| Valuation | 30% |
| Momentum | 10% |

---

## Performance Summary

| Metric | Portfolio | Benchmark (SP500) |
|--------|-----------|-------------------|
| Total return | 227.87% | 259.15% |
| Excess return | -31.28% | — |
| Annualised Sharpe | 0.53 | — |
| Maximum drawdown | -26.05% | — |
| Hit rate (12m forward) | 65.05% | — |
| Avg picks / month | 17.7 | — |
| Unique tickers ever picked | 48 | — |
| Test months with ≥ 1 pick | 120 | — |
| Hit-rate observations | 2120 | — |

---

## Conclusion

Track 1 (Quality Compounders) **underperformed** the SP500 benchmark (227.87% vs 259.15%, excess -31.28%). Review filter thresholds, scoring weights, and holding period before drawing conclusions.

Hit rate: **65.05%** across 2120 individual 12-month pick observations.

Sharpe of **0.53** (above 0.5 — risk-adjusted return appears non-trivial).

**Regime caveat:** The 2013–2024 window includes a strong growth-factor cycle
(2013–2021) and a sharp reversal (2022). Track 1 targets companies with proven
multi-year quality (ROIC > 15%, consistent FCF, stable margins), which are
predominantly mature compounders in Consumer Staples, Industrials, and select
Technology. These companies tend to lag in momentum-driven bull markets but
show lower drawdowns in corrections.

---

> **Data integrity:** Fundamentals from SEC EDGAR (point-in-time, filed date only).
> Prices from yfinance (OHLCV; not used for fundamentals). No look-ahead bias.
