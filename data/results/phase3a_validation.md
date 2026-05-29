# Phase 3a ML Validation Report

**Generated:** 2026-05-19 20:19 UTC

---

## Training configuration

| Parameter | Value |
|-----------|-------|
| Train window | 2010-01-01 – 2021-12-31 |
| Validation window | 2022-01-01 – 2023-12-31 |
| Held-out | 2024-01-01 – present (not evaluated) |
| Features | 17 real + 1 random_baseline |
| Model selected | random_forest |
| Train rows | 66868 |
| Train label balance | 53.5% outperform |
| Val rows | 11928 |
| Val label balance | 42.1% outperform |

---

## Model comparison (2022–2023 validation)

All three models are always trained. The highest validation accuracy wins.
Threshold (55.0%) is informational only.

| Model | Accuracy | vs threshold | |
|-------|----------|--------------|---|
| logistic_regression | 0.548 (54.8%) | below |  |
| random_forest | 0.579 (57.9%) | ABOVE | **selected** |
| xgboost | 0.502 (50.2%) | below |  |

---

## Validation accuracy (selected model)

**Accuracy on 2022–2023 validation set: 0.579 (57.9%)**

Selected model: `random_forest`
Threshold (informational): 55.0% — Result: ABOVE

---

## Confusion matrix (2022–2023 validation)

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 5355 | 1546 |
| **Actual 1** | 3473 | 1554 |

---

## Feature importances (model-derived)

Model type: `random_forest` — importances are absolute coefficients for LR, mean decrease in impurity for RF/XGBoost. `random_baseline` is pure N(0,1) noise; real features below its importance are flagged.

| Rank | Feature | Importance | Notes |
|------|---------|------------|-------|
| 1 | p_fcf | 0.1406 |  |
| 2 | ev_ebitda | 0.1362 |  |
| 3 | roic_proxy_avg | 0.1221 |  |
| 4 | capex_intensity | 0.1096 |  |
| 5 | share_buyback_signal | 0.0879 |  |
| 6 | gross_margin_avg | 0.0572 |  |
| 7 | momentum_raw | 0.0568 |  |
| 8 | p_e | 0.0479 |  |
| 9 | cfo_to_ni | 0.0424 |  |
| 10 | interest_coverage | 0.0413 |  |
| 11 | fcf_positive_years | 0.0388 |  |
| 12 | net_debt_ebitda | 0.0363 |  |
| 13 | operating_margin_trend | 0.0334 |  |
| 14 | revenue_acceleration | 0.0333 |  |
| 15 | revenue_growth_positive_years | 0.0116 |  |
| 16 | roic_direction | 0.0030 |  |
| 17 | **random_baseline** | **0.0015** | ← threshold |
| 18 | insider_buy_ratio | 0.0000 | candidate for removal |

---

## Feature ranking by Mutual Information (model-independent)

MI is computed on the imputed training set (2010–2021) against the binary outperform label. `random_baseline` is pure N(0,1) noise; real features below its MI score are flagged.

| Rank | Feature | MI Score | Notes |
|------|---------|----------|-------|
| 1 | cfo_to_ni | 0.1893 |  |
| 2 | revenue_acceleration | 0.1741 |  |
| 3 | operating_margin_trend | 0.1590 |  |
| 4 | net_debt_ebitda | 0.1460 |  |
| 5 | interest_coverage | 0.1457 |  |
| 6 | capex_intensity | 0.1422 |  |
| 7 | roic_proxy_avg | 0.1213 |  |
| 8 | gross_margin_avg | 0.0997 |  |
| 9 | share_buyback_signal | 0.0989 |  |
| 10 | p_fcf | 0.0073 |  |
| 11 | p_e | 0.0054 |  |
| 12 | fcf_positive_years | 0.0043 |  |
| 13 | revenue_growth_positive_years | 0.0023 |  |
| 14 | roic_direction | 0.0018 |  |
| 15 | momentum_raw | 0.0015 |  |
| 16 | insider_buy_ratio | 0.0014 |  |
| 17 | ev_ebitda | 0.0005 |  |
| 18 | **random_baseline** | **0.0000** | ← threshold |

---

## Notes

- `insider_buy_ratio` is NaN for all training/validation rows (ENABLE_INSIDER_FORM4=False).
  Its imputed value is 0.0 from training medians. It will carry real weight only after
  live monthly runs compute it for the shortlist.
- Imputation medians are computed from the training window only — no leakage.
- The held-out 2024+ window has NOT been evaluated. It is the final performance gate.
