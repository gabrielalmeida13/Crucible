# Crucible Track 2 Backtest Report

**Track:** 2 — Growth Inflection  
**Universe:** RUSSELL1000  
**Holding period:** 1 month  
**Generated:** 2026-05-22 12:48 UTC

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

## Filter Thresholds

| # | Filter | Condition |
|---|--------|-----------|
| 1 | Revenue growth | yr1 and yr2 both > 8% |
| 2 | Revenue acceleration | YoY growth rate increasing (> 0) |
| 3 | Gross margin | ≥ 30% OR expanding vs prior year |
| 4 | FCF positive | ≥ 1 of last 2 years |
| 5 | Leverage (soft) | Net Debt/EBITDA < 8.0 OR fcf_trajectory > 0 |
| 6 | Momentum gate | 12-1m price momentum > 0 (applied after fundamentals) |

---

## Performance Summary

| Metric | Portfolio | Benchmark (SP500) |
|--------|-----------|-------------------|
| Total return | 342.84% | 259.15% |
| Excess return | 83.70% | — |
| Annualised Sharpe | 0.64 | — |
| Maximum drawdown | -29.23% | — |
| Hit rate (12m forward) | 64.67% | — |
| Avg picks / month | 13.8 | — |
| Unique tickers ever picked | 115 | — |
| Test months with ≥ 1 pick | 120 | — |
| Hit-rate observations | 1656 | — |

---

## Momentum Filter Impact

The momentum gate is applied **after** the 5 fundamental filters,
so its cost is measured relative to that post-fundamental pool.

| Stage | Avg candidates across test months |
|-------|-----------------------------------|
| After 5 fundamental filters | 26.8 |
| After momentum gate | 16.8 |
| Dropped by momentum | 10.0 (37% of post-fundamental pool) |
| Test months with zero candidates post-momentum | 0 |

**Assessment:** **SIGNIFICANT cut** — momentum eliminates 37% of post-fundamental candidates on average. The pool is meaningfully thinned; recovery-phase companies with strong fundamentals but lagging price may be excluded.

---

## Conclusion

Track 2 (Growth Inflection) **outperformed** the SP500 benchmark (342.84% vs 259.15%, excess 83.70%).

Hit rate: **64.67%** across 1656 individual 12-month pick observations.

Sharpe of **0.64** (above 0.5 — risk-adjusted return appears non-trivial).

**Regime caveat:** The 2013–2024 window includes a strong growth-factor cycle
(2013–2021) and a sharp reversal (2022). Track 2 targets companies with
accelerating revenue and expanding margins, which cluster in Technology and
Healthcare. Results are sensitive to the growth-vs-value factor regime and
should not be extrapolated naively into a value-dominant environment.

---

> **Data integrity:** Fundamentals from SEC EDGAR (point-in-time, filed date only).
> Prices from yfinance (OHLCV; not used for fundamentals). No look-ahead bias.
