# Crucible — ROADMAP.md

> Current status: **Phase 0 — Setup**
> Update the status line above every time the phase changes.

---

## Overview

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4
Setup     Pipeline   Backtest   ML        Continuous
          + Filters  + Validation layer   refinement
```

Each phase must be **complete and validated** before moving to the next.
Resist the temptation to jump to ML before having clean data and honest backtests.

---

## Phase 0 — Project setup
**Goal:** functional repository, configured environment, folder structure in place.

- [ ] Create Git repository
- [ ] Configure Python environment with `uv` and `pyproject.toml`
- [ ] Create folder structure as defined in `CLAUDE.md`
- [ ] Configure `.env` and `.env.example`
- [ ] Set up global `logging`
- [ ] Create `config.py` with default values (thresholds, universe = `SP500`)
- [ ] First `pytest` run (even with no real tests yet)
- [ ] `CLAUDE.md` and `ROADMAP.md` committed to repository

**Exit criterion:** `python scripts/run_scan.py` runs without errors (even with no real logic yet).

---

## Phase 1 — Data pipeline + Fundamental filters
**Goal:** given a universe of tickers, produce a filtered shortlist with reliable data.

### 1.1 — Fetcher
- [ ] Pull S&P 500 ticker list (Wikipedia via `pandas.read_html` or `yfinance`)
- [ ] Pull fundamentals per ticker via `yfinance`:
  - ROIC (or proxy: Net Income / (Total Assets − Current Liabilities))
  - Free Cash Flow (Operating CF − CapEx)
  - Net Debt / EBITDA
  - Revenue growth (YoY, 5 years)
  - Gross margin (5 years)
  - P/FCF, EV/EBITDA, P/E
  - Denomination currency (required by `fx.py`)
  - GICS Sector and GICS Sub-Industry
- [ ] Save raw data to `data/raw/` with timestamp
- [ ] Log tickers with critical missing data (do not impute)

> **Note:** yfinance is used here for pipeline development only.
> It does not provide point-in-time data and must not be used for backtesting.
> FMP migration is required before Phase 2.

### 1.2 — Cleaner + Validator
- [ ] Normalize column names
- [ ] Detect and explicitly mark `NaN` values
- [ ] Detect obvious outliers (e.g., ROIC of 10,000% = data error)
- [ ] Save processed data to `data/processed/`
- [ ] Define Pandera schemas in `validator.py` for all major DataFrames
- [ ] Enforce schema validation at the exit of `cleaner.py` — fail loudly on violations

### 1.3 — Filters (Layer 1 — hard rules)

| Metric                     | Default threshold         | Configurable? |
|----------------------------|---------------------------|---------------|
| 5-year average ROIC        | > 15%                     | Yes           |
| FCF positive               | ≥ 4 of last 5 years       | Yes           |
| Net Debt / EBITDA          | < 3x                      | Yes           |
| Revenue growth             | Positive in ≥ 3 of 5 years| Yes           |
| Gross margin               | Stable or growing         | Yes           |

- [ ] Implement each filter as a pure function in `filters.py`
- [ ] Unit tests for each filter (at least one positive and one negative case)
- [ ] Output: DataFrame with companies that passed all filters

### 1.4 — Scorer (Layer 2 — composite score)

**Sector and region normalization (mandatory from this phase)**
All metrics must be scored relative to peers in the **same GICS sector AND
the same accounting region** (US GAAP / IFRS / Japanese GAAP), not against
the full universe. Use percentile rank within peer group.

- [ ] Quality score: ROIC, FCF consistency, gross margin (configurable weights)
- [ ] Valuation score: P/FCF vs. company's own historical average, EV/EBITDA vs. historical
- [ ] FX penalty: integrate `fx.py` output — apply score penalty for stocks requiring
      currency conversion (default: −0.5 points, configurable)
- [ ] Composite score: weighted average (default: 60% quality, 40% valuation)
- [ ] Output: shortlist ordered by score, with all fundamentals visible

### 1.5 — Persistence
- [ ] SQLite schema: tables `scans`, `companies`, `scores`
- [ ] Save each monthly scan with timestamp and universe ID
- [ ] Query to compare shortlists between months

### 1.6 — Basic dashboard (Streamlit)
- [ ] Interactive table with current month's shortlist
- [ ] Sidebar filters to adjust thresholds in real time
- [ ] Comparison with previous scan
- [ ] CSV export
- [ ] FX conversion flag visible in the table

**Exit criterion:** given the S&P 500 universe, the pipeline produces a shortlist of
15–40 companies with a composite score in under 10 minutes, saves the result to the
database, and the Pandera schemas pass on every run.

---

## Phase 2 — Data migration + Backtesting + Validation
**Goal:** migrate to point-in-time data and honestly assess whether the filters
have any predictive value.

### 2.1 — Migrate to FMP (Financial Modeling Prep)
- [ ] Obtain FMP API key and add to `.env`
- [ ] Rewrite `fetcher.py` to use FMP as primary source
- [ ] Validate that FMP data matches yfinance data on current fundamentals
      (discrepancies are expected — document them)
- [ ] Confirm that FMP provides point-in-time financials for historical scans
- [ ] Confirm that FMP includes delisted tickers (survivorship bias requirement)

> **Why this must happen before backtesting:**
> yfinance retroactively overwrites restated financials. If a company reexpresses
> Q1 earnings in Q3, yfinance replaces the original Q1 figure. Using this data
> in a backtest means the model had information that didn't exist at the time —
> look-ahead bias that makes results completely unreliable.

### 2.2 — Walk-forward backtest
- [ ] Reconstruct historical point-in-time universe per month (critical — use FMP)
- [ ] Implement walk-forward validation:
  - Train window: months 1–24
  - Test: month 25
  - Advance 1 month, repeat
- [ ] Comparison metrics:
  - Total return vs. S&P 500 (benchmark)
  - Sharpe ratio
  - Maximum drawdown
  - Hit rate (% of picks with positive return at 12 months)
- [ ] Threshold sensitivity test: does the result change significantly if ROIC
      threshold moves from 15% to 12% or 18%?
- [ ] Document results honestly — including if performance is worse than the index

**Exit criterion:** documented backtest report with explicit comparison to S&P 500,
and an honest conclusion about filter value before adding ML complexity.

---

## Phase 3 — ML layer
**Goal:** improve the composite score using a trained model.

> **Do not start this phase until Phase 2 is validated.**

### ML problems to address (in order of increasing complexity)

**3a — Fundamental quality classifier**
- Input: company fundamentals in year N
- Target: binary — will ROIC be higher in N+1 than in N? (directional, not absolute)
- Rationale: predicting direction of fundamental momentum is far more tractable
  than predicting an absolute threshold 3 years out. Markets and macroeconomic
  cycles make N+3 absolute targets extremely noisy.
- Models to test: Logistic Regression → Random Forest → XGBoost
- Validation: walk-forward mandatory (no data leakage)

**3b — Ranking score (optional)**
- Replace the weighted composite score with a learning-to-rank model (e.g., LambdaMART)
- Target: relative return vs. benchmark at 12 months

**3c — Accounting red flag detection (optional)**
- Beneish M-score variant with additional ML features
- Output: binary flag for risk of earnings manipulation or accounting irregularities

Implementation steps:
- [ ] Feature engineering in `ml/features.py`
- [ ] Training pipeline with walk-forward in `ml/model.py`
- [ ] Compare ML model performance vs. rules-based score from Phase 1
- [ ] Integrate into dashboard as an additional column — do not replace the rules score

**Exit criterion:** ML model improves hit rate by at least 3 percentage points
vs. rules-based score, validated on out-of-sample data.

---

## Phase 4 — Universe expansion + Continuous refinement
**Goal:** expand to the full XTB universe and add ongoing maintenance features.

### 4.1 — Universe expansion (staged)

| Step | Universe ID    | Action                                                          |
|------|----------------|-----------------------------------------------------------------|
| 4.1a | `EUROPE_LARGE` | Add LSE, Deutsche Börse, Euronext Paris (~150 companies)        |
| 4.1b | `JAPAN_ADR`    | Add Japanese large-caps via ADRs / European dual listings (~80) |
| 4.1c | `XTB_FULL`     | Expand to all XTB-available stocks with sufficient FMP data     |

For each expansion step:
- [ ] Confirm FMP data availability for the new tickers
- [ ] Enable region-aware normalization in `scorer.py` for the new accounting standard
- [ ] Calibrate Layer 1 thresholds per region (IFRS companies typically show
      different ROIC distributions than US GAAP)
- [ ] Rerun backtest on expanded universe before using in production

### 4.2 — Additional features
- [ ] Automatic alerts (e.g., a shortlist company drops out of filters mid-month)
- [ ] Insider transactions signal integration
- [ ] Short interest as a negative scoring factor
- [ ] Monthly notification (email or Telegram bot)

---

## Cross-cutting notes

**Survivorship bias**
The historical universe for each monthly scan must be the universe *of that month*,
not the current one. Companies that left the index through bankruptcy, acquisition,
or removal must be included. Ignoring this makes any backtest useless.
FMP handles this; yfinance does not reliably.

**Point-in-time data**
Financials are reported with a delay. A Q1 report published in May cannot be used
as a feature for a March scan. yfinance does not guarantee this — it is the primary
risk of look-ahead bias in this project, and the reason FMP is mandatory for Phase 2.

**Accounting comparability**
Never compare ROIC or gross margin across regions without normalization.
A 40% gross margin is exceptional in retail but poor in software. A 15% ROIC
is typical for a US tech company but high for a Japanese industrial.
All comparisons must be within GICS sector + accounting region peer groups.

**The model is a tool, not an oracle**
The monthly output is a starting point for further investigation, not a buy
instruction. The final decision is always human.