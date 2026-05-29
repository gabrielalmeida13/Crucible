# Crucible Backtest Report

**Generated:** 2026-05-18 23:03 UTC

---

## Walk-forward Parameters

| Parameter | Value |
|-----------|-------|
| Training window | 24 months |
| Portfolio size (top-N) | 20 |
| Rebalance / holding period | 12 month(s) |
| Hit-rate measurement window | 12 months |
| Risk-free rate (annual) | 4.0% |
| Benchmark | SP500 |

---

## Performance Summary

| Metric | Portfolio | Benchmark |
|--------|-----------|-----------|
| Total return | 1160958806.75% | 158677833.08% |
| Excess return vs benchmark | 1002280973.67% | — |
| Annualised Sharpe ratio | 3.36 | — |
| Maximum drawdown | -32.33% | — |
| Hit rate (12m) | 71.47% | — |
| Test months | 124 | 124 |
| Hit-rate observations | 1602 | — |

---

## ROIC Threshold Sensitivity

How sensitive are results to the ROIC filter threshold?
All other parameters held constant.

| roic_min | n_test_months | avg_picks | portfolio_total_return | benchmark_total_return | sharpe_ratio | max_drawdown | hit_rate |
| -------- | ------------- | --------- | ---------------------- | ---------------------- | ------------ | ------------ | -------- |
| 10%      | 132           | 17.3      | 221379873196.90%       | 941218372.52%          | 4.71         | -19.97%      | 74.35%   |
| 12%      | 132           | 15.9      | 193942146609.93%       | 941218372.52%          | 4.39         | -31.17%      | 73.88%   |
| 15%      | 124           | 12.9      | 1160958806.75%         | 158677833.08%          | 3.36         | -32.33%      | 71.47%   |
| 18%      | 119           | 8.4       | 704118587.31%          | 74701029.36%           | 4.43         | -16.75%      | 69.69%   |
| 20%      | 119           | 7.3       | 1099612500.10%         | 74701029.36%           | 4.41         | -13.32%      | 71.02%   |

---

## Conclusion

The Crucible screener **outperformed** the benchmark over the test period (1160958806.75% vs 158677833.08%, excess +1002280973.67%). The hit rate of **71.47%** means 71% of individual 12-month picks were profitable. The annualised Sharpe ratio of **3.36** is above 0.5, suggesting the return was not purely noise.

**Important caveats:** The test window must be long enough to span multiple market regimes (bull, bear, sideways). A short backtest with favourable timing is not evidence of a good strategy. The sensitivity table above shows whether results are robust to small threshold changes — fragile results are a red flag.

---

> **Data integrity note:** This backtest requires FMP point-in-time financial
> statements. Results are only valid if the fundamentals snapshots were built
> from FMP data with no look-ahead (Q1 reports available after their filing
> date, not their fiscal quarter end). Past backtest performance does not
> guarantee future results.