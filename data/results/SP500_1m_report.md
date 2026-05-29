# Crucible Backtest Report

**Generated:** 2026-05-19 14:57 UTC

---

## Walk-forward Parameters

| Parameter | Value |
|-----------|-------|
| Training window | 24 months |
| Portfolio size (top-N) | 20 |
| Rebalance / holding period | 1 month(s) |
| Hit-rate measurement window | 12 months |
| Risk-free rate (annual) | 4.0% |
| Benchmark | SP500 |

---

## Performance Summary

| Metric | Portfolio | Benchmark |
|--------|-----------|-----------|
| Total return | 305.98% | 228.74% |
| Excess return vs benchmark | 77.24% | — |
| Annualised Sharpe ratio | 0.68 | — |
| Maximum drawdown | -23.65% | — |
| Hit rate (12m) | 70.41% | — |
| Test months | 124 | 124 |
| Hit-rate observations | 1602 | — |

---

## ROIC Threshold Sensitivity

How sensitive are results to the ROIC filter threshold?
All other parameters held constant.

| roic_min | n_test_months | avg_picks | portfolio_total_return | benchmark_total_return | sharpe_ratio | max_drawdown | hit_rate |
| -------- | ------------- | --------- | ---------------------- | ---------------------- | ------------ | ------------ | -------- |
| 10%      | 132           | 17.3      | 424.43%                | 280.96%                | 0.82         | -18.31%      | 72.99%   |
| 12%      | 132           | 15.9      | 443.64%                | 280.96%                | 0.82         | -19.39%      | 72.92%   |
| 15%      | 124           | 12.9      | 305.98%                | 228.74%                | 0.68         | -23.65%      | 70.41%   |
| 18%      | 119           | 8.4       | 225.48%                | 222.81%                | 0.60         | -18.78%      | 69.79%   |
| 20%      | 119           | 7.3       | 227.78%                | 222.81%                | 0.59         | -22.31%      | 71.02%   |

---

## Conclusion

The Crucible screener **outperformed** the benchmark over the test period (305.98% vs 228.74%, excess +77.24%). The hit rate of **70.41%** means 70% of individual 12-month picks were profitable. The annualised Sharpe ratio of **0.68** is above 0.5, suggesting the return was not purely noise.

**Important caveats:** The test window must be long enough to span multiple market regimes (bull, bear, sideways). A short backtest with favourable timing is not evidence of a good strategy. The sensitivity table above shows whether results are robust to small threshold changes — fragile results are a red flag.

---

> **Data integrity note:** This backtest requires FMP point-in-time financial
> statements. Results are only valid if the fundamentals snapshots were built
> from FMP data with no look-ahead (Q1 reports available after their filing
> date, not their fiscal quarter end). Past backtest performance does not
> guarantee future results.