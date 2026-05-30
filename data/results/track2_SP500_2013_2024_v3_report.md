# Track 2 v3 — SP500 Backtest Report (2013-2024)

> **v3 changes:** quarterly EDGAR features active.
> `revenue_growth_q1yoy` (> 6% QoQ-YoY) replaces annual revenue growth filter.
> `revenue_accel_quarterly` (weight 10%) added to growth_quality sub-score.
> Snapshots rebuilt from EDGAR; v2 cache deleted before this run.

**Generated:** 2026-05-30 14:03 UTC  
**Universe:** SP500 (~503 tickers)  
**Period:** 2013-01-31 → 2024-12-31 (24-month warm-up)  

---

## Performance vs v2 Baseline

| Metric | v3 (quarterly) | v2 baseline | Δ |
|--------|---------------|-------------|---|
| Total return | 477.40% | 407.14% | +70.26% |
| Benchmark (SP500) | 259.15% | — | — |
| Excess return | 218.25% | 147.99% | +70.26% |
| Annualised Sharpe | 0.88 | 0.71 | +0.17 |
| Max drawdown | -21.89% | — | — |
| Hit rate (12m) | 70.03% | 68.81% | +1.22% |
| Avg picks / month | 19.0 | — | — |
| Unique tickers | 129 | — | — |
| Test months | 120 | — | — |
| Hit-rate observations | 2282 | — | — |

---

> Fundamentals: SEC EDGAR (point-in-time). Prices: yfinance (OHLCV only).
> Quarterly features (revenue_growth_q1yoy, revenue_accel_quarterly, gross_margin_q_latest,
> fcf_q_last2) computed inline from 10-Q filings during snapshot build.
