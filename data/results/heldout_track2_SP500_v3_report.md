# Track 2 v3 — SP500 Held-Out (2025-01 → 2026-05)

> **HELD-OUT VALIDATION.** Parameters frozen at end of 2013-2024 backtest.
> Do not re-tune after reading these results.

> **v3 changes vs v2:** quarterly EDGAR features active in filter and scorer.

**Generated:** 2026-05-30 14:15 UTC  
**Universe:** SP500 (~503 tickers)  
**Period:** 2025-01-31 → 2026-05-31 (no warm-up)  

---

## Performance vs v2 Baseline

| Metric | v3 (quarterly) | v2 baseline | Δ |
|--------|---------------|-------------|---|
| Total return | 45.76% | 40.17% | +5.59% |
| Benchmark (SP500) | 27.52% | 25.69% | — |
| Excess return | 18.24% | 14.48% | +3.76% |
| Annualised Sharpe | 1.54 | 1.11 | +0.43 |
| Max drawdown | -6.14% | — | — |
| Hit rate (12m) | 57.00% | 50.82% | +6.18% |
| Avg picks / month | 20.0 | — | — |
| Unique tickers | 50 | — | — |
| Test months | 16 | — | — |

---

## Conclusion

Track 2 v3 **outperformed** both the benchmark (27.52%) and the v2 held-out baseline (40.17%). The quarterly features appear to add value in the prospective window.

*Hit rate covers 100 observations with 12-month forward price available.
Months from mid-2025 onward have partial or no 12m forward coverage.*

---

> Data: EDGAR point-in-time fundamentals + yfinance OHLCV prices.
> Held-out window not seen during development of quarterly features.
