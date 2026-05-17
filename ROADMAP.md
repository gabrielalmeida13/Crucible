# Crucible — ROADMAP.md

> Current status: **Phase 2 in progress — EDGAR data migration underway**
> Update the status line above every time the phase changes.

---

## Overview

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4
Setup     Pipeline   Backtest   ML        Continuous
          + Filters  + Validation layer   refinement

  ✓ done    ✓ done    ← here
```

Each phase must be **complete and validated** before moving to the next.

---

## Phase 0 — Project setup ✓
**Status: Complete**

- [x] Create Git repository
- [x] Configure Python environment with `uv` and `pyproject.toml`
- [x] Create folder structure as defined in `CLAUDE.md`
- [x] Configure `.env` and `.env.example`
- [x] Set up global `logging`
- [x] Create `config.py` with default values (thresholds, universe = `SP500`)
- [x] First `pytest` run
- [x] `CLAUDE.md` and `ROADMAP.md` committed to repository

---

## Phase 1 — Data pipeline + Fundamental filters ✓
**Status: Complete**

Filters implemented and functional. Initial backtest was run on a limited universe
(large-cap subset) using FMP free tier data (Phase 1 only — not point-in-time).

**Results from initial validation (Jan 2024 – Mar 2025, limited universe):**
- Portfolio total return: 15.41% vs SPY 14.84% (+0.57% excess)
- Sharpe ratio: 0.70 | Max drawdown: -6.13% | Hit rate: 79.49%
- Average picks per month: 2.6 (limited by API data gaps — not statistically robust)

**Important caveat:** 2.6 picks/month over 15 months is insufficient sample size.
The direction of the thesis is promising but not confirmed. This is why Phase 2
focuses on expanding sample size through a broader universe and longer history.

### Filters implemented (Layer 1 — hard rules)

| Metric                     | Threshold                 |
|----------------------------|---------------------------|
| 5-year average ROIC        | > 15%                     |
| FCF positive               | ≥ 4 of last 5 years       |
| Net Debt / EBITDA          | < 3x                      |
| Revenue growth             | Positive in ≥ 3 of 5 years|
| Gross margin               | Stable or growing         |

**Sensitivity test results (ROIC threshold):**
- ROIC > 10%: return 4.68%, Sharpe 0.02, hit rate 55% — significantly worse
- ROIC > 12%: return 4.17%, Sharpe -0.02, hit rate 55% — worse than index
- ROIC > 15%: return 15.41%, Sharpe 0.70, hit rate 79.49% — strong signal
- Conclusion: 15% threshold validated; do not lower without strong evidence

---

## Phase 2 — Data migration to SEC EDGAR + Full backtest ← current
**Status: In progress — 2.1 EDGAR migration underway**

> **Why SEC EDGAR instead of FMP:**
> The SEC EDGAR API is the primary US government data source. It is free,
> unlimited, requires no API key, and provides true point-in-time data via
> filing timestamps. XBRL structured data is available from 2009 (SEC mandate).
> EDGAR also contains all historical filers including delisted companies,
> which solves survivorship bias natively. FMP free tier was used for Phase 1
> development only and has been replaced entirely.

### 2.1 — EDGAR migration
- [x] Add `edgartools` dependency; update `pyproject.toml` and `.env.example`
- [x] Implement `download_edgar_bulk.py` for one-time `companyfacts.zip` download
- [x] Rewrite `fetcher.py`: XBRL taxonomy map + `_parse_edgar_json` point-in-time parser
- [ ] Use `yfinance` for price data only (fundamentals must come from EDGAR)
- [ ] Validate EDGAR data: spot-check revenue/net income against known values for 5 tickers
- [ ] Confirm all historical S&P 500 filers are present including delisted companies
- [ ] Update `tests/test_fetcher.py` for new EDGAR-based interface

### 2.2 — Universe expansion for statistical robustness
- [ ] Expand from S&P 500 to Russell 1000 (or broader) to increase picks per month
- [ ] Target: average ≥ 10–15 picks per monthly scan (vs. 2.6 in Phase 1)
- [ ] Rerun filters on expanded universe and validate shortlist quality

### 2.3 — Walk-forward backtest (2009–present)
- [ ] Historical range: 2009–present (~15 years, full XBRL coverage)
- [ ] Walk-forward implementation:
  - Train window: months 1–24
  - Test: month 25
  - Advance 1 month, repeat
- [ ] Comparison metrics: total return vs. S&P 500, Sharpe, max drawdown, hit rate
- [ ] Threshold sensitivity retest on full universe
- [ ] Survivorship bias check: delisted companies included in historical universe
- [ ] Point-in-time check: no filing used before its `filed` date
- [ ] Backtest report documented

---

## Phase 3 — ML layer
**Goal:** improve the composite score using a trained model.

> **Data available for ML:** 2009–present via EDGAR, all US public companies.
> Walk-forward validation is mandatory throughout — no exceptions.

### Overfitting mitigation strategy

With 15 years of data, overfitting is a real risk. Mitigations in place:

1. **Walk-forward only** — model never sees test period during training
2. **Feature count discipline** — start with fewer than 15 features;
   add only with clear economic rationale, not because it improves backtest
3. **Out-of-sample held-out period** — reserve 2023–present as final test set;
   do not touch it until the model is fully specified
4. **Benchmark comparison** — a model that barely beats the rules-based score
   from Phase 1 is not worth the added complexity; set a minimum improvement bar
5. **Regime awareness** — test separately on pre-2020, 2020–2022, and 2022–present
   to check if the model degrades in specific macro regimes

### 3a — Fundamental quality classifier (start here)
- [ ] Define feature set from EDGAR fundamentals:
  - ROIC level and YoY direction
  - FCF margin and consistency
  - Gross margin level and trend
  - Revenue growth rate and acceleration
  - Net debt / EBITDA and direction
  - Asset turnover
  - Accruals ratio (earnings quality signal)
- [ ] Target: binary — will ROIC be higher in N+1 than in N?
  - Rationale: directional momentum is far more tractable than predicting
    absolute values 3 years out; markets and macro cycles make N+3 targets noisy
- [ ] Feature engineering in `ml/features.py`
- [ ] Training pipeline with walk-forward in `ml/model.py`
- [ ] Model progression: Logistic Regression → Random Forest → XGBoost
  - Start simple; only add complexity if simpler model fails clearly
- [ ] Compare performance vs. rules-based score from Phase 1
- [ ] Integrate into dashboard as additional column — do not replace rules score

### 3b — Ranking score (optional, after 3a validated)
- [ ] Replace weighted composite score with learning-to-rank model (e.g., LambdaMART)
- [ ] Target: relative return vs. S&P 500 benchmark at 12 months
- [ ] Requires larger pick sample — validate universe size before starting

### 3c — Accounting red flag detection (optional, after 3a validated)
- [ ] Beneish M-score variant with EDGAR-derived features
- [ ] Additional signals: accruals, revenue/receivables divergence, asset growth anomalies
- [ ] Output: binary flag for earnings quality risk
- [ ] Integrate as negative score modifier, not hard filter

**Exit criterion:** ML model improves hit rate by ≥ 3 percentage points vs.
rules-based score, validated on out-of-sample data (2023–present held-out set).
If this bar is not met, document why and do not force the model into production.

---

## Phase 4 — Universe expansion + Continuous refinement
**Goal:** expand to European and broader XTB universe; add operational features.

### 4.1 — European universe expansion (staged)

> **Data source decision required before starting this phase.**
> SEC EDGAR covers US-listed companies only. European companies require a separate
> source. Options to evaluate at Phase 4 start:
> - **SimFin** (free, covers major European large-caps, limited history)
> - **FMP paid** (~$20/month, broader coverage, reasonable point-in-time)
> - **Open source scraping** (Stockanalysis.com, company IR pages — fragile)
> Evaluate based on coverage of target universe and data quality at that time.

| Step | Universe ID    | Action                                                            |
|------|----------------|-------------------------------------------------------------------|
| 4.1a | `EUROPE_LARGE` | Add LSE, Deutsche Börse, Euronext Paris (~150 companies)          |
| 4.1b | `JAPAN_ADR`    | Add Japanese large-caps via ADRs / European dual listings (~80)   |
| 4.1c | `XTB_FULL`     | Expand to all XTB-available stocks with sufficient data           |

For each expansion step:
- [ ] Confirm data source availability and point-in-time quality for new tickers
- [ ] Enable region-aware normalization in `scorer.py` (IFRS ≠ US GAAP)
- [ ] Calibrate Layer 1 thresholds per region
- [ ] Add FX cost penalty in `fx.py` for non-EUR denominated stocks (XTB charges 0.5%)
- [ ] Rerun backtest on expanded universe before using in production

### 4.2 — Operational features
- [ ] Automatic alerts (e.g., a shortlist company drops out of filters mid-month)
- [ ] Insider transactions signal (available free via EDGAR Form 4)
- [ ] Short interest as a negative scoring factor
- [ ] Institutional holdings changes (EDGAR 13F filings — also free)
- [ ] Monthly notification (email or Telegram bot)

---

## Cross-cutting notes

**Survivorship bias**
The historical universe for each scan must be the universe *of that month*, not
the current one. EDGAR contains all historical filers regardless of current listing
status — companies that were delisted, acquired, or went bankrupt are still in
the database with their historical filings. This is one of EDGAR's key advantages
over yfinance and free FMP tiers.

**Point-in-time data**
Every filing in EDGAR has a `filed` date. A 10-K for fiscal year ending December 2020
filed in February 2021 must not be used in a January 2021 backtest scan.
Always filter on `filed` ≤ scan date. Never use period end date as a proxy.

**Data range**
XBRL is available from 2009. This is the hard floor for fundamental data.
Earlier filings exist in EDGAR but are unstructured (HTML, PDF) and inconsistent
between companies. Do not attempt to parse pre-2009 fundamentals.

**Accounting comparability (US-only phase)**
Within the US universe all companies use US GAAP — direct comparison is valid.
Sector normalization is still mandatory: compare ROIC, margin, and valuation
metrics within the same GICS sector, not across the full universe.

**Overfitting with long history**
More historical data is better for robustness, but only if the walk-forward
protocol is strictly followed. The held-out period (2023–present) must not be
touched until the model specification is finalized in Phase 3. Any parameter
tuning that uses the held-out period invalidates it.

**The model is a tool, not an oracle**
The monthly output is a starting point for further investigation, not a buy
instruction. The final decision is always human.