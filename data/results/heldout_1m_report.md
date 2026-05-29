# Crucible — Held-Out Validation Report

> **HELD-OUT VALIDATION — parameters fixed before this run,**
> **no further tuning permitted after seeing these results.**
>
> This report covers data from 2023-01 onwards, which was entirely
> excluded from model development. All filter thresholds, scorer
> weights, and universe definitions were frozen at the end of the
> 2010–2022 backtest period. Modifying any parameter after reading
> this output constitutes look-ahead contamination.

**Generated:** 2026-05-19 21:37 UTC

---

## Validation Parameters

| Parameter | Value |
|-----------|-------|
| Test window | 2023-01-31 → 2026-04-30 |
| Training burn-in | 0 months (none — model pre-specified) |
| Portfolio size (top-N) | 20 |
| Holding period | 1 month(s) |
| Hit-rate measurement window | 12 months |
| Risk-free rate (annual) | 4.0% |
| Benchmark | SP500 |
| Score weights | quality=50%, val=25%, mom=10%, ml=15% |
| ML model | data/models/phase3a_model.pkl (RF, val_acc=57.9%) |

---

## Performance Summary

| Metric | Portfolio | Benchmark |
|--------|-----------|-----------|
| Total return | 46.25% | 88.29% |
| Excess return vs benchmark | -42.04% | — |
| Annualised Sharpe ratio | 0.62 | — |
| Maximum drawdown | -8.76% | — |
| Hit rate (12m) | 71.21% | — |
| Test months | 40 | 40 |
| Hit-rate observations | 580 | — |

---

## Conclusion

The strategy underperformed the benchmark by 42.04% over the held-out period (hit rate: 71.21%). Review whether market conditions during this window are systematically different from the training period before drawing conclusions about model failure.

---

> **Data integrity note:** Fundamentals are sourced from SEC EDGAR with
> point-in-time correctness (only filings with `filed` ≤ snapshot date are used).
> Price data is from yfinance (closing prices only — no fundamental data).
> This held-out window was not seen during any stage of model development.
