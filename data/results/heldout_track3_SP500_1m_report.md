# Held-Out Validation — Track 3 (Value Recovery) — SP500

> **HELD-OUT VALIDATION.** Parameters frozen at end of 2013-2024 backtest.
> Do not re-tune after reading these results.

**Track:** 3 — Value Recovery  
**Universe:** SP500 (~503 tickers)  
**Holding period:** 1 month  
**Test window:** 2025-01-31 → 2026-05-31  
**Burn-in:** none (TRAIN_MONTHS=0 — every month is a test point)  
**Generated:** 2026-05-25 11:32 UTC

---

## Performance Summary

| Metric | Portfolio | Benchmark (SP500) |
|--------|-----------|-------------------|
| Total return | 19.44% | 25.69% |
| Excess return | -6.25% | — |
| Annualised Sharpe | 0.80 | — |
| Maximum drawdown | -7.72% | — |
| Hit rate (12m forward) | 67.06% | — |
| Avg picks / month | 15.6 | — |
| Unique tickers picked | 39 | — |
| Test months with ≥ 1 pick | 16 | — |
| Hit-rate observations | 85 | — |

---

## Conclusion

Track 3 underperformed the benchmark by **6.25%** over the held-out period (hit rate: **67.06%**). The 2025-2026 market regime may differ from 2013-2024 training conditions. Review sector concentration and filter passage rates before drawing conclusions.

*Hit rate covers 85 observations where a 12-month forward price was available. Months from mid-2025 onwards may have partial or no 12m forward coverage given the evaluation date.*

---

> **Data integrity:** Fundamentals from SEC EDGAR (point-in-time, `filed` ≤ snapshot date).
> Prices from yfinance (OHLCV only). Heldout window not seen during backtest development.
