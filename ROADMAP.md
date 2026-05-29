# Crucible — ROADMAP.md

> Current status: **Phase 4 complete — prospective validation active from June 2026**
> Update the status line above every time the phase changes.

---

## Overview

```
Phase 0 → Phase 1 → Phase 2 → Phase 2.5 → Phase 3 → Phase 4 → Phase 5
Setup     Pipeline   Backtest   Scorer      3-Track   Operational  ML v2 +
          + Filters  + EDGAR    complete    System    + Portfolio   Expansion

  ✓ done    ✓ done    ✓ done    ✓ done      ✓ done    ✓ done      ← next
```

---

## System state — May 2026

### Validated decisions (do not reopen without new evidence)

**Universe:** SP500 (503 tickers) for all three tracks.
Russell 1000 was tested and rejected — mid-caps dilute quality in Track 1
and reduce Sharpe in Tracks 2 and 3 despite more candidates per month.

**Holding period:** 1 month for all tracks.

**Rotation:** deterministic (Track 1 → Track 2 → Track 3 → repeat) or
weighted based on regime (see Phase 4.6). Given investor profile (20 years
old, high risk tolerance), Track 2 is the primary engine.

**ML (Phase 3a):** Random Forest at 57.9% validation accuracy was not
deployed. The model improved in-sample but degraded in held-out. Root cause:
validation set was iterated too many times. ML is not discarded — retested
in Phase 5 with more prospective data and fresh held-out.

### Backtest results (2013–2024, SP500)

| Track | Total return | Excess vs SP500 | Sharpe | Hit rate | Unique tickers |
|-------|-------------|-----------------|--------|----------|----------------|
| 1 — Quality | 361.03% | +101.88% | 0.79 | 70.36% | 42 |
| 2 — Growth | 420.95% | +161.80% | 0.72 | 68.81% | 62 |
| 3 — Value | 298.94% | +39.80% | 0.61 | 73.48% | 116 |
| SP500 benchmark | 259.15% | — | — | — | — |

### Held-out results (2025-01 to 2026-05, SP500, ~16 months)

| Track | Total return | Excess vs SP500 | Sharpe | Hit rate | Max drawdown |
|-------|-------------|-----------------|--------|----------|--------------|
| 1 — Quality | 8.48% | -17.22% | 0.25 | 59.00% | -7.43% |
| 2 — Growth | 40.17% | +14.48% | 1.11 | 50.82% | -6.79% |
| 3 — Value | 19.44% | -6.25% | 0.80 | 67.06% | -7.72% |
| SP500 benchmark | 25.69% | — | — | — | — |

**Interpretation:** Track 2 is the only track beating the benchmark in the
held-out. The 2025-2026 regime continues to favour growth over quality and
value — structurally consistent with training period. Track 1 and Track 3
are not broken; they are regime-dependent and will recover in rotations.

### First real pick — May 2026
**APH (Amphenol Corporation)** — Track 2, Growth Inflection.
Entry rationale: 58% revenue growth YoY Q1 2026, record orders (book-to-bill
1.24x), 27.3% adjusted operating margin, AI data center pick-and-shovel play,
P/E forward ~26 vs sector — reasonable for the growth profile.
Entry P/FCF: 77.733 | Entry P/S: 7.845

---

## Prospective validation protocol (MANDATORY from June 2026)

The system is frozen as of June 2026. No parameter changes permitted.

- Every monthly run produces a timestamped output in `data/monthly/{YYYY-MM}/`
- All picks are logged to SQLite — never overwritten
- Results reviewed May 2027 against actual prices
- If a bug is found: document, fix, rerun — but do not look at new results
  before the fix is applied
- Any parameter change after June 2026 restarts the prospective clock

---

## Phase 4 — Operational system ✓ complete

### Phase 4 completion summary — May 2026

Everything below was built, tested, and is running in production as of May 2026.
The system is frozen for prospective validation from June 2026.

**Core operational pipeline**
- [x] `crucible/portfolio.py` — position evaluator: HOLD / REINFORCE / REVIEW / EXIT_SIGNAL / DATA_MISSING
- [x] `scripts/run_monthly.py` — unified entry point for all three tracks with portfolio review and allocation advice
- [x] SQLite prospective logging (`data/crucible_picks.db`) with atomic JSON manifest per run

**Three-track screener (4.1–4.4)**
- [x] `crucible/tracks/track1_quality.py` — Quality Compounder: ROIC, FCF consistency, debt, margin stability
- [x] `crucible/tracks/track2_growth.py` — Growth Inflection: revenue acceleration, margin expansion, momentum gate
- [x] `crucible/tracks/track3_value.py` — Value Recovery: P/FCF vs history, balance sheet health, recovery signals
- [x] Track 2 earnings quality signals: `asset_growth_yoy`, `deferred_revenue_growth`, `eps_surprise_last_q` (4.7)

**Regime detection (4.6)**
- [x] `crucible/regime.py` — three-state rules-based regime (GROWTH / DEFENSIVE / HIGH_VOL)
  using VIX, 10y–2y yield curve spread, and SP500 12-month momentum (all free via yfinance)
- [x] Regime allocation hint integrated into `run_monthly.py` CLI output

**Alerts (4.5)**
- [x] `crucible/alerts.py` — EXIT_SIGNAL transitions, consecutive negative momentum (2+ months),
  HOLD→REVIEW downgrades; Telegram (preferred) and email (fallback) channels
- [x] `scripts/check_alerts.py` — lightweight SQLite-only daily cron script (no EDGAR needed)
- [x] Monthly reminder on the 1st of each month if screener not yet run
- [x] Alert dispatch integrated into `scripts/run_monthly.py` (step 12a–12c)

**Dashboard (4.8)**
- [x] Streamlit dashboard — Monthly Picks, Portfolio, Manual Import (XLSX+CSV), History, Performance
- [x] Regime indicator widget on Monthly Picks tab: coloured badge + VIX / spread / SP500 momentum
- [x] Allocation advice in Portfolio tab: budget widget, calls `allocation_advice()`, renders as Markdown
- [x] Performance tab: prospective picks since June 2026, return_pct coloured green/red,
  SP500 benchmark comparison, per-track breakdown
- [x] Manual Import tab replaces XTB API (discontinued March 2025)

---

## Phase 5 — ML v2 + Expansion

> Do not start until Phase 4.5-4.7 are complete AND at least 6 months of
> prospective data exist (December 2026 earliest).

### 5.1 — ML for Track 2 (learning-to-rank within shortlist)

Previous approach (Phase 3a) tried to classify outperformers across all 500 stocks.
New approach: rank companies within the already-filtered Track 2 shortlist.
This is a narrower, more tractable problem with cleaner signal.

- [ ] Training window: 2013–2025 (includes more regime diversity than before)
- [ ] Held-out: prospective data from June 2026 onwards (truly clean)
- [ ] Target: rank within Track 2 shortlist by 3-month forward return
- [ ] Features: all Track 2 scorer components + momentum_3m + insider_buy_ratio
      + eps_surprise_last_q + deferred_revenue_growth (new from Phase 4.7)
- [ ] Model: LightGBM LambdaMART (learning-to-rank, not binary classification)
- [ ] Exit criterion: model improves hit rate by ≥ 3pp vs score-based ranking
      on prospective held-out — not on backtested data

### 5.2 — EPS revisions signal (requires external data)

Academic research (MSCI, Robeco) consistently identifies analyst EPS revision
direction as one of the strongest alpha signals. Not currently implementable
for free — requires a data provider with consensus estimates.

Options to evaluate at Phase 5 start:
- Alpha Vantage (limited free tier, paid for full coverage)
- Quandl/Nasdaq Data Link (academic pricing available)
- Evaluate cost vs. signal strength with 6+ months of prospective data first

- [ ] Evaluate data source cost and coverage
- [ ] If viable: add `eps_revision_direction` (up/flat/down over last 30 days)
      as a binary modifier to Track 2 composite score

### 5.3 — Universe expansion to Europe

> EDGAR covers US only. European data requires SimFin (free, limited) or
> FMP paid (~$20/month). Evaluate budget at Phase 5 start.

- [ ] Add `EUROPE_LARGE` universe (LSE, Deutsche Börse, Euronext Paris)
- [ ] Enable IFRS normalisation in scorer
- [ ] Calibrate thresholds per region
- [ ] Add FX cost penalty in `fx.py` (XTB charges 0.5% conversion)
- [ ] Full backtest before production use

### 5.4 — Options/hedging (research only)

> Not for real capital until validated over ≥ 12 months prospectively.

- Protective puts on Track 2 positions with large unrealised gains (>100%)
- Covered calls on Track 1 positions for income generation
- Track 4 (Short candidates): inverse of Track 1 filters as put targets

---

## Cross-cutting principles

**Point-in-time data:** EDGAR filings filtered by `filed` ≤ snapshot date always.

**Sector normalisation:** all metrics compared within GICS sector peer groups.

**The model is a tool, not an oracle:** monthly output is a starting point
for human investigation, not a buy instruction. Debate top candidates with
an AI assistant using the full reasoning export before deciding.

**Prospective validation is the only clean validation:** the June 2026
prospective clock is the definitive test. Any modification after June 2026
restarts the clock.

**Crucible is a living project:** refined continuously as prospective data
accumulates, regimes change, and new sources become available. Every major
parameter change requires a new held-out before deployment.

**Investor profile:** age 20, high risk tolerance, long time horizon.
Track 2 (Growth Inflection) is the primary engine. Track 1 and Track 3
provide diversification and drawdown protection. Monthly investment: ~€100.
Strategy: build individual positions methodically over years, parallel to
core ETF allocation (S&P 500, QVDE).