# Crucible Backtest Report

**Generated:** 2026-05-18 23:03 UTC

---

## Walk-forward Parameters

| Parameter | Value |
|-----------|-------|
| Training window | 24 months |
| Portfolio size (top-N) | 20 |
| Rebalance / holding period | 3 month(s) |
| Hit-rate measurement window | 12 months |
| Risk-free rate (annual) | 4.0% |
| Benchmark | SP500 |

---

## Performance Summary

| Metric | Portfolio | Benchmark |
|--------|-----------|-----------|
| Total return | 5582.81% | 3106.72% |
| Excess return vs benchmark | 2476.09% | — |
| Annualised Sharpe ratio | 1.44 | — |
| Maximum drawdown | -43.28% | — |
| Hit rate (12m) | 71.47% | — |
| Test months | 124 | 124 |
| Hit-rate observations | 1602 | — |

---

## ROIC Threshold Sensitivity

How sensitive are results to the ROIC filter threshold?
All other parameters held constant.

| roic_min | n_test_months | avg_picks | portfolio_total_return | benchmark_total_return | sharpe_ratio | max_drawdown | hit_rate |
| -------- | ------------- | --------- | ---------------------- | ---------------------- | ------------ | ------------ | -------- |
| 10%      | 132           | 17.3      | 15165.01%              | 4749.12%               | 1.91         | -31.68%      | 74.35%   |
| 12%      | 132           | 15.9      | 14191.11%              | 4749.12%               | 1.76         | -33.83%      | 73.88%   |
| 15%      | 124           | 12.9      | 5582.81%               | 3106.72%               | 1.44         | -43.28%      | 71.47%   |
| 18%      | 119           | 8.4       | 3673.20%               | 2919.37%               | 1.51         | -34.24%      | 69.69%   |
| 20%      | 119           | 7.3       | 3932.17%               | 2919.37%               | 1.50         | -34.24%      | 71.02%   |

---

## Conclusion

The Crucible screener **outperformed** the benchmark over the test period (5582.81% vs 3106.72%, excess +2476.09%). The hit rate of **71.47%** means 71% of individual 12-month picks were profitable. The annualised Sharpe ratio of **1.44** is above 0.5, suggesting the return was not purely noise.

**Important caveats:** The test window must be long enough to span multiple market regimes (bull, bear, sideways). A short backtest with favourable timing is not evidence of a good strategy. The sensitivity table above shows whether results are robust to small threshold changes — fragile results are a red flag.

---

> **Data integrity note:** This backtest requires FMP point-in-time financial
> statements. Results are only valid if the fundamentals snapshots were built
> from FMP data with no look-ahead (Q1 reports available after their filing
> date, not their fiscal quarter end). Past backtest performance does not
> guarantee future results.