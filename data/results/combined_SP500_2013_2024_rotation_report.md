# Combined Rotation Backtest — SP500 Universe

**Run date:** 2026-05-26 12:05 UTC  
**Universe:** SP500 (~503 tickers)  
**Training warm-up:** 24 months (2013-01 → 2014-12)  
**Test window:** 2015-01 → 2024-12  
**Holding period:** 1 month  
**Portfolio size (per track):** top 20  

---

## Protocol definitions

**Protocol A — Deterministic rotation:**  
Each month, picks come exclusively from one track in the sequence T1→T2→T3→T1→…  
Month 1 = Track 1, month 2 = Track 2, month 3 = Track 3, month 4 = Track 1, etc.

**Protocol B — Weighted blend:**  
Each month, picks come from all three tracks simultaneously.  
Portfolio return = 20% × equal-weight T1 return + 50% × T2 return + 30% × T3 return.  
Weights renormalised proportionally if any track produces no picks.

**Track 2 alone (baseline):**  
Standard Track 2 Growth Inflection walk-forward, top 20 equal-weight, 1-month hold.

---

## Results

| Protocol | Total Return | Excess vs SP500 | Sharpe | Max Drawdown | Hit Rate (12m) | Avg Picks/Mo | Unique Tickers |
|---|---|---|---|---|---|---|---|
| A — Deterministic T1→T2→T3 rotation | +298.02% | +38.87% | 0.64 | -24.48% | +70.05% | 15.1 | 131 |
| B — Weighted blend (50%T2/30%T3/20%T1) | +433.90% | +174.75% | 0.82 | -23.26% | +69.74% | 41.0 | 138 |
| Track 2 alone (baseline) | +407.14% | +147.99% | 0.71 | -30.18% | +68.46% | 9.5 | 62 |
| SP500 benchmark | +259.15% | +0.00% | — | — | — | — | 0 |

---

## Interpretation

- Protocol A (rotation) **underperformed** Track 2 alone by -109.12%. Cyclically spending 2/3 of months in T1/T3 diluted growth alpha.
- Protocol B (weighted blend) **outperformed** Track 2 alone by +26.76%.
- Rotation (A) improved max drawdown to -24.48% vs T2 alone (-30.18%).
- Weighted blend (B) improved max drawdown to -23.26% vs T2 alone (-30.18%).

---

> **Data integrity:** Fundamentals from SEC EDGAR (point-in-time, filed-date filtered).  
> Prices from yfinance (OHLCV only). No look-ahead bias.  
> Past backtest performance does not guarantee future results.
