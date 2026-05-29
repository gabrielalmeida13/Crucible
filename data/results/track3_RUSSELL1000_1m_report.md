# Crucible Track 3 Backtest Report

**Track:** 3 — Value Recovery  
**Universe:** RUSSELL1000  
**Holding period:** 1 month  
**Generated:** 2026-05-22 17:57 UTC

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
| 1 | ROIC proxy (avg) | > 8% |
| 2 | P/FCF vs own 5yr history | (p_fcf_5yr_avg − p_fcf) / p_fcf_5yr_std ≥ 1.0 σ |
| 3 | FCF positive (last 5yr) | ≥ 2 of last 5 years |
| 4 | Recovery signal (any one) | Buyback > 3% OR revenue inflection OR margin recovery |
|   | → Buyback signal | share_buyback_signal > 3% |
| | → Revenue inflection | revenue_growth_yr1 > 0 AND revenue_growth_yr2 < 0 |
| | → Margin recovery | gross_margin_yr1_change > 2% AND trend_slope < 0 |

## Score Weights (Layer 2)

| Component | Weight | Sub-components |
|-----------|--------|----------------|
| Value | 50% | equal-weight p_fcf rank, ev_ebitda rank (ascending=False) |
| Recovery signal | 30% | equal-weight buyback rank, gm_yr1_change rank, rev_inflection rank |
| Balance sheet | 20% | equal-weight net_debt_ebitda rank (asc=False), interest_coverage rank |

---

## Performance Summary

| Metric | Portfolio | Benchmark (SP500) |
|--------|-----------|-------------------|
| Total return | 218.91% | 259.15% |
| Excess return | -40.24% | — |
| Annualised Sharpe | 0.47 | — |
| Maximum drawdown | -26.16% | — |
| Hit rate (12m forward) | 68.77% | — |
| Avg picks / month | 16.1 | — |
| Unique tickers ever picked | 161 | — |
| Test months with ≥ 1 pick | 120 | — |
| Hit-rate observations | 1934 | — |

---

## Conclusion

Track 3 (Value Recovery) **underperformed** the SP500 benchmark (218.91% vs 259.15%, excess -40.24%). Review filter thresholds, scoring weights, and holding period before drawing conclusions.

Hit rate: **68.77%** across 1934 individual 12-month pick observations.

Sharpe of **0.47** (below 0.5 — risk-adjusted return is weak).

**Regime caveat:** Track 3 (Value Recovery) is contrarian by design. The 2013–2024
window is predominantly a growth-led bull market where mean-reversion strategies
tend to underperform. Value recovery requires patient capital: a company trading
1 std below its own P/FCF history may remain cheap for months before a catalyst
triggers re-rating. The 1-month holding period may be too short to capture the
full value recovery cycle. Compare Track 3 results with a 3-month or 6-month
holding period before drawing conclusions about the strategy's viability.

---

> **Data integrity:** Fundamentals from SEC EDGAR (point-in-time, filed date only).
> Prices from yfinance (OHLCV; not used for fundamentals). No look-ahead bias.
> p_fcf_vs_history computed from trailing snapshots — no future data leakage.
