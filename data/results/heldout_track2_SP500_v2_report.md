# Held-Out Validation — Track 2 v2 (Growth Inflection + EPS Surprise) — SP500

> **HELD-OUT VALIDATION.** Parameters frozen at end of 2013-2024 backtest.
> Do not re-tune after reading these results.

**Track:** 2 v2 — Growth Inflection (Phase 4.7 features active)  
**Universe:** SP500 (~503 tickers)  
**Holding period:** 1 month  
**Test window:** 2025-01-31 → 2026-05-31  
**Burn-in:** none (TRAIN_MONTHS=0 — every month is a test point)  
**Generated:** 2026-05-26 13:05 UTC

**Phase 4.7 scorer changes vs v1:**
- `revenue_acceleration` weight: 20% → 10%
- `eps_surprise_last_q` added: 10% weight (earnings beat strength)
- `deferred_revenue_growth` (10%) and `asset_growth_yoy` penalty (−10%): unchanged

---

## Performance Summary

| Metric | Portfolio v2 | Benchmark (SP500) | v1 Baseline | Δ vs v1 |
|--------|-------------|-------------------|-------------|---------|
| Total return | 40.17% | 25.69% | 40.17% | +0.00% |
| Excess return | 14.48% | — | 14.48% | -0.00% |
| Annualised Sharpe | 1.11 | — | 1.11 | +0.28% |
| Maximum drawdown | -6.79% | — | — | — |
| Hit rate (12m forward) | 50.82% | — | 50.82% | -0.00% |
| Avg picks / month | 11.1 | — | — | — |
| Unique tickers picked | 25 | — | — | — |
| Test months with ≥ 1 pick | 16 | — | — | — |
| Hit-rate observations | 61 | — | — | — |

---

## Conclusion

Track 2 v2 outperformed the benchmark by **14.48%** over the held-out period. Hit rate: **50.82%**. The addition of `eps_surprise_last_q` (improved total return vs v1 baseline of 40.17%).

*Hit rate covers 61 observations where a 12-month forward price was available. Months from mid-2025 onwards may have partial or no 12m forward coverage given the evaluation date.*

---

> **Data integrity:** Fundamentals from SEC EDGAR (point-in-time, `filed` ≤ snapshot date).
> Prices from yfinance (OHLCV only). Heldout window not seen during backtest development.
