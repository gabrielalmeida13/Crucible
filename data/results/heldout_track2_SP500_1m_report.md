# Held-Out Validation — Track 2 (Growth Inflection) — SP500

> **HELD-OUT VALIDATION.** Parameters frozen at end of 2013-2024 backtest.
> Do not re-tune after reading these results.

**Track:** 2 — Growth Inflection  
**Universe:** SP500 (~503 tickers)  
**Holding period:** 1 month  
**Test window:** 2025-01-31 → 2026-05-31  
**Burn-in:** none (TRAIN_MONTHS=0 — every month is a test point)  
**Generated:** 2026-05-25 11:32 UTC

---

## Performance Summary

| Metric | Portfolio | Benchmark (SP500) |
|--------|-----------|-------------------|
| Total return | 40.17% | 25.69% |
| Excess return | 14.48% | — |
| Annualised Sharpe | 1.11 | — |
| Maximum drawdown | -6.79% | — |
| Hit rate (12m forward) | 50.82% | — |
| Avg picks / month | 11.1 | — |
| Unique tickers picked | 25 | — |
| Test months with ≥ 1 pick | 16 | — |
| Hit-rate observations | 61 | — |

---

## Conclusion

Track 2 outperformed the benchmark by **14.48%** over the held-out period. Hit rate: **50.82%**. This is an encouraging out-of-sample result, but the window is short (~16 months) — interpret with appropriate caution.

*Hit rate covers 61 observations where a 12-month forward price was available. Months from mid-2025 onwards may have partial or no 12m forward coverage given the evaluation date.*

---

> **Data integrity:** Fundamentals from SEC EDGAR (point-in-time, `filed` ≤ snapshot date).
> Prices from yfinance (OHLCV only). Heldout window not seen during backtest development.
