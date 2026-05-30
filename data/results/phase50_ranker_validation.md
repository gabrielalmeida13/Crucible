# Phase 5.0 — LambdaMART Ranker: Expanding-Window Comparison

**Generated:** 2026-05-29 20:21 UTC  
**Validation method:** 10-fold expanding window (min 24 months training, 12 months validation each fold)  
**Forward return window:** 3 months  
**Ranking metric:** NDCG@5, top-1 hit rate (positive 3m return)  
**Blend weight:** 50% score + 50% ML rank positions  

---

## Summary

Four combinations vs the pure score-based baseline:

| Combination | NDCG@5 mean ± std | HR@1 mean ± std | Δ vs baseline | 3pp gate |
|-------------|------------------|----------------|---------------|---------|
| Score baseline | 0.6908 ±0.0837 | 65.0% ±17.9 | baseline | — |
| A. Full ML (13 features) | 0.6496 ±0.0737 | 54.2% ±11.3 | -10.8 pp | ❌ |
| B. Full + Blend 50/50 | 0.6676 ±0.0733 | 59.2% ±21.3 | -5.8 pp | ❌ |
| C. Economic ML (4 features) | 0.6579 ±0.0873 | 59.2% ±20.2 | -5.8 pp | ❌ |
| D. Economic + Blend 50/50 | 0.6735 ±0.0969 | 57.5% ±23.1 | -7.5 pp | ❌ |

*10 folds × 12 validation months each. Mean ± std across folds (each fold's value is itself a mean across its validation months).*

---

## Gate Result

**Historical gate (≥ 3pp HR@1 improvement vs score baseline):**

- **No combination** clears the ≥ 3pp gate on the historical expanding-window CV.

> **Important:** the historical gate is indicative only. The deployment gate
> requires ≥ 3pp improvement on the **prospective held-out**
> (June 2026 → December 2026 — zero-iteration data collected after model training).
> Re-evaluate in December 2026.

---

## Feature Sets

**Full features (13):** `composite_score`, `growth_quality_score`, `momentum_score`, `valuation_score`, `momentum_raw`, `momentum_3m`, `revenue_growth_yr1`, `revenue_acceleration`, `gross_margin_latest`, `fcf_trajectory`, `asset_growth_yoy`, `deferred_revenue_growth`, `eps_surprise_last_q`

**Economic features (4) — no scorer-derived quantities:**  
`momentum_3m`, `revenue_growth_yr1`, `gross_margin_latest`, `revenue_acceleration`

The economic features avoid the circular-learning risk: composite_score and
scorer sub-components are derived from the same fundamentals the model would
learn from, so including them may teach the model to replicate the scorer
rather than discover new signal. Economic features are a cleaner test.

---

## Context & Caveats

- Group sizes are small (9–22 per month): NDCG and HR estimates are noisy.
- Expanding window uses the same snapshot data that informed Track 2 scorer
  design → results are **not** independent of researcher choices.
- Phase 4.7 features (asset_growth_yoy, deferred_revenue_growth,
  eps_surprise_last_q) are absent from the pre-2026 cache and imputed to 0.
- The score baseline is itself well-calibrated: beating it consistently
  across 10 folds is a high bar.

---


### Per-Fold Detail (Full ML features)

| Fold | Train window | Val window | Groups | Score HR@1 | ML HR@1 | Blend HR@1 | ML Δ |
|------|-------------|-----------|--------|-----------|---------|-----------|------|
| 0 | 2013-01→2014-12 | 2015-01→2015-12 | 24 | 33.3% | 33.3% | 33.3% | 0.0 pp |
| 1 | 2013-01→2015-12 | 2016-01→2016-12 | 36 | 66.7% | 58.3% | 50.0% | -8.3 pp |
| 2 | 2013-01→2016-12 | 2017-01→2017-12 | 48 | 58.3% | 66.7% | 75.0% | +8.3 pp |
| 3 | 2013-01→2017-12 | 2018-01→2018-12 | 60 | 58.3% | 58.3% | 58.3% | 0.0 pp |
| 4 | 2013-01→2018-12 | 2019-01→2019-12 | 72 | 75.0% | 41.7% | 41.7% | -33.3 pp |
| 5 | 2013-01→2019-12 | 2020-01→2020-12 | 84 | 91.7% | 58.3% | 75.0% | -33.3 pp |
| 6 | 2013-01→2020-12 | 2021-01→2021-12 | 96 | 75.0% | 58.3% | 66.7% | -16.7 pp |
| 7 | 2013-01→2021-12 | 2022-01→2022-12 | 108 | 41.7% | 41.7% | 25.0% | 0.0 pp |
| 8 | 2013-01→2022-12 | 2023-01→2023-12 | 120 | 83.3% | 58.3% | 91.7% | -25.0 pp |
| 9 | 2013-01→2023-12 | 2024-01→2024-12 | 132 | 66.7% | 66.7% | 75.0% | 0.0 pp |

### Per-Fold Detail (Economic features)

| Fold | Train window | Val window | Groups | Score HR@1 | ML HR@1 | Blend HR@1 | ML Δ |
|------|-------------|-----------|--------|-----------|---------|-----------|------|
| 0 | 2013-01→2014-12 | 2015-01→2015-12 | 24 | 33.3% | 50.0% | 33.3% | +16.7 pp |
| 1 | 2013-01→2015-12 | 2016-01→2016-12 | 36 | 66.7% | 33.3% | 33.3% | -33.3 pp |
| 2 | 2013-01→2016-12 | 2017-01→2017-12 | 48 | 58.3% | 75.0% | 58.3% | +16.7 pp |
| 3 | 2013-01→2017-12 | 2018-01→2018-12 | 60 | 58.3% | 58.3% | 58.3% | 0.0 pp |
| 4 | 2013-01→2018-12 | 2019-01→2019-12 | 72 | 75.0% | 83.3% | 83.3% | +8.3 pp |
| 5 | 2013-01→2019-12 | 2020-01→2020-12 | 84 | 91.7% | 75.0% | 100.0% | -16.7 pp |
| 6 | 2013-01→2020-12 | 2021-01→2021-12 | 96 | 75.0% | 50.0% | 41.7% | -25.0 pp |
| 7 | 2013-01→2021-12 | 2022-01→2022-12 | 108 | 41.7% | 25.0% | 33.3% | -16.7 pp |
| 8 | 2013-01→2022-12 | 2023-01→2023-12 | 120 | 83.3% | 58.3% | 58.3% | -25.0 pp |
| 9 | 2013-01→2023-12 | 2024-01→2024-12 | 132 | 66.7% | 83.3% | 75.0% | +16.7 pp |
