# Crucible Backtest Report

**Generated:** 2026-05-17 23:45 UTC

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
| Total return | 536.07% | 222.81% |
| Excess return vs benchmark | 313.25% | — |
| Annualised Sharpe ratio | 0.94 | — |
| Maximum drawdown | -25.73% | — |
| Hit rate (12m) | 69.99% | — |
| Test months | 119 | 119 |
| Hit-rate observations | 753 | — |

---

## ROIC Threshold Sensitivity

How sensitive are results to the ROIC filter threshold?
All other parameters held constant.

| roic_min | n_test_months | avg_picks | portfolio_total_return | benchmark_total_return | sharpe_ratio | max_drawdown | hit_rate |
| -------- | ------------- | --------- | ---------------------- | ---------------------- | ------------ | ------------ | -------- |
| 10%      | 119           | 10.5      | 354.28%                | 222.81%                | 0.87         | -21.45%      | 69.70%   |
| 12%      | 119           | 8.8       | 367.44%                | 222.81%                | 0.87         | -23.26%      | 71.29%   |
| 15%      | 119           | 6.3       | 536.07%                | 222.81%                | 0.94         | -25.73%      | 69.99%   |
| 18%      | 119           | 4.6       | 695.80%                | 222.81%                | 0.97         | -29.58%      | 73.16%   |
| 20%      | 119           | 3.9       | 809.84%                | 222.81%                | 1.01         | -27.35%      | 75.92%   |

---

## Conclusion

The Crucible screener **outperformed** the benchmark over the test period (536.07% vs 222.81%, excess +313.25%). The hit rate of **69.99%** means 70% of individual 12-month picks were profitable. The annualised Sharpe ratio of **0.94** is above 0.5, suggesting the return was not purely noise.

**Important caveats:** The test window must be long enough to span multiple market regimes (bull, bear, sideways). A short backtest with favourable timing is not evidence of a good strategy. The sensitivity table above shows whether results are robust to small threshold changes — fragile results are a red flag.

---

> **Data integrity note:** This backtest requires FMP point-in-time financial
> statements. Results are only valid if the fundamentals snapshots were built
> from FMP data with no look-ahead (Q1 reports available after their filing
> date, not their fiscal quarter end). Past backtest performance does not
> guarantee future results.