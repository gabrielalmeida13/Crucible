# Held-Out Validation — Protocol B (50% T2 / 30% T3 / 20% T1) — SP500

> **HELD-OUT VALIDATION.** Parameters frozen at end of 2013-2024 backtest.
> Do not re-tune after reading these results.

**Protocol:** B — combined three-track portfolio  
**Allocation:** 50% Track 2 (Growth Inflection) / 30% Track 3 (Value Recovery) / 20% Track 1 (Quality Compounders)  
**Track 2 scorer:** Phase 4.7 active (asset_growth_yoy penalty, deferred_revenue_growth, eps_surprise_last_q)  
**Universe:** SP500 (~503 tickers)  
**Holding period:** 1 month  
**Test window:** 2025-01-31 → 2026-05-31  
**Burn-in:** none (TRAIN_MONTHS=0 — every month is a test point)  
**Generated:** 2026-05-26 17:36 UTC

---

## Performance Summary

| Metric | Protocol B | Benchmark (SP500) | Track 2 alone | Δ vs T2 |
|--------|-----------|-------------------|---------------|---------|
| Total return | 28.94% | 26.24% | 40.17% | -11.23% |
| Excess return | 2.70% | — | 14.48% | -11.78% |
| Annualised Sharpe | 1.16 | — | 1.11 | +4.77% |
| Maximum drawdown | -4.64% | — | -6.79% | — |
| Hit rate (12m forward) | 57.65% | — | 50.82% | +6.83% |
| Avg picks / month (eff.) | 14.1 | — | 11.1 | — |
| Unique tickers picked | 85 | — | 25 | — |
| Test months with ≥ 1 pick | 16 | — | 16 | — |
| Hit-rate observations | 307 | — | 61 | — |

---

## Individual track inputs (from three_tracks held-out, same window)

| Track | Total return | Excess | Sharpe | Hit rate |
|-------|-------------|--------|--------|----------|
| Track 1 — Quality Compounders (20%) | 8.48% | -17.22% | 0.25 | 59.00% |
| Track 2 — Growth Inflection (50%) | 40.17% | +14.48% | 1.11 | 50.82% |
| Track 3 — Value Recovery (30%) | 19.44% | -6.25% | 0.80 | 67.06% |
| **Protocol B weighted expectation** | **27.61%** | **1.92%** | — | — |

---

## Conclusion

Protocol B outperformed the benchmark by **2.70%** but trailed Track 2 alone by **+11.23%**. Hit rate: **57.65%**. The T3/T1 allocation diluted the stronger Track 2 signal in this held-out window.

*Hit rate covers 307 observations (weighted by track allocation). Months from mid-2025 onwards may have partial or no 12m forward coverage given the evaluation date.*

---

> **Data integrity:** Fundamentals from SEC EDGAR (point-in-time, `filed` ≤ snapshot date).
> Prices from yfinance (OHLCV only). Heldout window not seen during backtest development.
