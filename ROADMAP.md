# Crucible — ROADMAP.md

> Current status: **Phase 4 complete — Phase 5 in progress (ML experimental + Options)**
> Update the status line above every time the phase changes.

---

## Overview

```
Phase 0 → Phase 1 → Phase 2 → Phase 2.5 → Phase 3 → Phase 4 → Phase 5
Setup     Pipeline   Backtest   Scorer      3-Track   Operational  ML v2 +
          + Filters  + EDGAR    complete    System    ✓ complete   Options

  ✓ done    ✓ done    ✓ done    ✓ done      ✓ done    ✓ done      ← here
```

---

## Phase 4 completion summary — May 2026 ✓

Everything built and operational:

- `crucible/portfolio.py` — HOLD / REINFORCE / REVIEW / EXIT_SIGNAL / DATA_MISSING
- `crucible/regime.py` — GROWTH / DEFENSIVE / HIGH_VOL regime detection
- `crucible/alerts.py` — Telegram + email alerts for portfolio changes
- `crucible/store.py` — SQLite prospective logging
- `crucible/tracks/track1_quality.py` — Quality Compounders
- `crucible/tracks/track2_growth.py` — Growth Inflection (primary engine)
- `crucible/tracks/track3_value.py` — Value Recovery
- `scripts/run_monthly.py` — three tracks, portfolio review, allocation advice, alerts
- `scripts/run_combined_backtest.py` — Protocol A and B rotation simulation
- `scripts/check_alerts.py` — lightweight daily cron
- `app/dashboard.py` — Monthly Picks, Portfolio, Manual Import, History, Performance tabs
- Phase 4.7 features: asset_growth_yoy, deferred_revenue_growth, eps_surprise_last_q

---

## System state — May 2026

### Validated decisions (frozen for prospective protocol)

**Universe:** SP500 (503 tickers) for all three tracks.
**Holding period:** 1 month.
**Primary engine:** Track 2 (Growth Inflection) — beats benchmark in held-out.
**Rotation protocol:** Protocol B (50% T2 / 30% T3 / 20% T1) beats Track 2 alone
on Sharpe and drawdown but sacrifices 11% return in current regime.
Given high risk tolerance: Track 2 pure for monthly picks.

### Backtest results (2013–2024, SP500)

| Track | Total return | Excess vs SP500 | Sharpe | Hit rate |
|-------|-------------|-----------------|--------|----------|
| 1 — Quality | 361.03% | +101.88% | 0.79 | 70.36% |
| 2 — Growth | 420.95% | +161.80% | 0.72 | 68.81% |
| 3 — Value | 298.94% | +39.80% | 0.61 | 73.48% |
| Protocol B blend | 433.90% | +174.75% | 0.82 | — |
| SP500 benchmark | 259.15% | — | — | — |

### Held-out results (2025-01 to 2026-05, SP500)

| Track | Total return | Excess | Sharpe | Hit rate | Max DD |
|-------|-------------|--------|--------|----------|--------|
| 1 — Quality | 8.48% | -17.22% | 0.25 | 59.00% | -7.43% |
| 2 — Growth | 40.17% | +14.48% | 1.11 | 50.82% | -6.79% |
| 3 — Value | 19.44% | -6.25% | 0.80 | 67.06% | -7.72% |
| Protocol B | 28.94% | +2.70% | 1.16 | 57.65% | -4.64% |
| SP500 benchmark | 25.69% | — | — | — | — |

### Real portfolio (from May 2026)

| Ticker | Track | Entry date | Entry price | Entry P/FCF | Entry P/S |
|--------|-------|-----------|-------------|-------------|-----------|
| APH | 2 | 2026-05-27 | (fill after execution) | 77.733 | 7.845 |
| NVDA | 2 | (historical) | (fill) | — | — |
| INTC | 2 | (historical) | (fill) | — | — |

---

## Prospective validation protocol (MANDATORY — June 2026 onwards)

**System frozen as of June 2026. No production parameter changes permitted.**

- Monthly run on 1st of each month — `run_monthly.py --track 2 --budget 100`
- All picks logged to SQLite — never overwritten
- Results reviewed May 2027 vs actual prices
- Bug fixes: document, fix, rerun without looking at results first
- Any production parameter change restarts the prospective clock

The ML experimental branch and options module are SEPARATE from the
production system — they do not affect prospective logging or validation.

---

## Phase 5 — Parallel workstreams (active now)

### 5.0 — ML experimental: LightGBM LambdaMART (start now, validate December)

**Why now, not December:** Development and historical backtesting can happen
immediately. The December gate is for production deployment, which requires
6 months of prospective data to validate on truly clean data. Build now,
validate in December.

**The key difference from Phase 3a (which failed):**
- Phase 3a: classify 500 companies as outperform/underperform (hard problem)
- Phase 5.0: rank 9-22 companies within the Track 2 shortlist (simpler problem,
  more signal per observation, less noise)

**Implementation:**
- [x] Create `crucible/ml/ranker.py` with LightGBM LambdaMART (`lambdarank` objective)
- [x] Training data: monthly Track 2 shortlists (9–22 companies), 3m forward
      return → quintile labels 0–4 (within-group percentile rank)
- [x] Features: 13 features (all Track 2 scorer components + raw metrics +
      Phase 4.7 signals; 4.7 features absent from pre-2026 cache → imputed)
- [x] Walk-forward: train 2013-2021 (20% internal val for early stopping),
      validate 2022-2024; metrics: NDCG@5 + hit-rate improvement
- [x] `scripts/run_phase50_ranker.py` — execution script; saves
      `data/results/phase50_ranker_validation.md` and
      `data/models/phase50_ranker.pkl`
- [ ] Run validation script (requires prices download ~15-30 min)
- [ ] December 2026: run held-out on prospective data 2026-06 to 2026-12
- [ ] Exit criterion: model ranking improves hit rate by ≥ 3pp vs score-based
      on prospective held-out. If not met, document and do not deploy.

### 5.1 — Options module (start now — XTB has options in Portugal)

**Context:** XTB launched American-style options on 110 US stocks in Portugal
in May 2026 (buy-only: calls and puts). Max expiry ~231 days. Standard
contracts of 100 shares.

**Practical capital constraint:** Options on high-price stocks (APH $137,
AMD $110+, FIX $350+) require $500-1500 minimum premium per contract.
With €100/month budget, options are not viable for monthly picks.
Viable uses with current capital:

1. **Protective puts on large winners** — INTC (+500%) is the primary candidate.
   A put at current price costs ~$200-400 for 6 months, protecting accumulated
   gains without selling and triggering tax.

2. **Leveraged calls when conviction is high** — accumulate 5-15 months of
   budget (~€500-1500) before buying a call on a Track 2 pick instead of shares.
   Deep ITM calls (delta ~0.8) give similar exposure to 100 shares for less capital.

3. **Index options** — SPY/QQQ options have lower nominal prices per contract
   and can be used to hedge the overall portfolio direction.

**Implementation:**
- [x] `crucible/options.py` — `suggest_options_strategy(ticker, action, current_price,
      budget_eur, expiry_days=180)` and `check_iv_rank(ticker, lookback_days=252)`;
      real option chain via `yf.Ticker(ticker).option_chain(date)`; outputs:
      strike, premium in EUR, contracts affordable, breakeven, max loss, payoff table
      at ±10%/±25%/±50%; IV rank badge; liquidity spread warning
- [x] Options tab in dashboard: ticker from portfolio/shortlist/manual; action
      selector; budget + expiry inputs; live option chain fetch; IV badge +
      payoff comparison table (option vs equivalent shares investment)
- [ ] Integrate with portfolio module: flag "consider protective put" in
      allocation_advice for positions with return > 100% and market_value > €500

**Learning path (before building the module):**
- Understand delta, theta, implied volatility — these determine option pricing
- Key insight for Track 2: growth companies have high IV (expensive options)
  because the market prices in uncertainty. This works against buying calls.
  Low IV = cheaper options = better risk/reward for long calls.
- For APH specifically: check IV rank before buying any call. If IV > 50th
  percentile of its own history, options are expensive — buy shares instead.

### 5.2 — Regime-adjusted scoring (low effort, real impact)

The regime module (`crucible/regime.py`) already detects GROWTH/DEFENSIVE/HIGH_VOL.
Currently it only shows a badge in the dashboard. The next step is using it
to dynamically adjust scorer weights — already proven to improve Sharpe in
combined backtest research.

- [ ] In HIGH_VOL regime: momentum weight -5pp, quality weight +5pp
- [ ] In DEFENSIVE regime: Track 1 gets priority in allocation advice
- [ ] Backtest on training period to confirm improvement before deploying
- [ ] No held-out test needed — this is a scorer improvement, not a new model

### 5.3 — EPS revisions (requires paid data — evaluate at Phase 5 mid-point)

Analyst EPS revision direction is one of the strongest documented alpha signals
(Robeco research: 158% IR improvement when combined with fundamentals). Not
currently implementable for free.

Options at Phase 5 mid-point (November 2026):
- Alpha Vantage (limited free tier)
- Quandl/Nasdaq Data Link (academic pricing)
- Evaluate cost vs expected signal strength with 6 months of prospective data

### 5.4 — Universe expansion to Europe (Phase 5 end)

> EDGAR covers US only. European data requires SimFin or FMP paid.
> Evaluate budget at Phase 5 end (May 2027).

---

## Monthly workflow (production — June 2026 onwards)

```
Day 1 of each month:
1. python scripts/run_monthly.py --track 2 --budget 100
2. Open dashboard → analyse shortlist
3. Export track2_picks.md → debate top 3 with AI assistant
4. Execute purchase on XTB
5. Register in dashboard (ticker, price, track, shares, P/FCF, P/S)
6. python scripts/check_alerts.py → confirm no alerts pending

Results reviewed: May 2027 (12 months prospective)
```

---

## Cross-cutting principles

**Point-in-time data:** EDGAR filings filtered by `filed` ≤ snapshot date always.

**Sector normalisation:** all metrics compared within GICS sector peer groups.

**The model is a tool, not an oracle:** monthly output is a starting point
for human investigation, not a buy instruction. Debate top candidates
with an AI assistant before deciding.

**Prospective validation is the only clean validation:** the June 2026
prospective clock is the definitive test. Production system is frozen.
Experimental branches (ML, options) are separate and do not affect it.

**Crucible is a living project:** refined continuously as prospective data
accumulates and new data sources become available. Every production change
requires a new held-out validation before deployment.

**Investor profile:** age 20, high risk tolerance, long time horizon.
Track 2 (Growth Inflection) is the primary engine. Monthly investment:
~€100 in stocks + occasional options when capital accumulates or for
protective hedging of large existing gains (INTC +500%).
Base portfolio: S&P 500 ETF + QVDE ETF as core, individual picks as
satellite positions built methodically over years.