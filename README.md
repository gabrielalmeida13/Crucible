# Crucible

> Point-in-time fundamental stock screener and portfolio allocation engine.

Crucible is a rigorous, data-driven equity screening tool designed for long-term investment decisions. Rather than attempting to predict short-term price movements, Crucible filters broad market universes (currently the S&P 500) using purely fundamental criteria derived from SEC EDGAR filings. 

By enforcing strict point-in-time correctness and sector normalization, it identifies high-quality, growth-inflection, and value-recovery opportunities.

---

## Core Philosophy

* **Fundamental First:** The core engine relies exclusively on accounting metrics (Income Statement, Balance Sheet, Cash Flow). Machine Learning is used solely as an experimental ranking layer, never to replace hard financial logic.
* **Point-in-Time Correctness:** Backtesting uses the exact SEC filing timestamp (filed date). No look-ahead bias. No restatement retroactive overwriting.
* **Tool, Not an Oracle:** Crucible generates a monthly shortlist. Final allocation decisions are made after human review and qualitative debate.

## System Architecture

The engine evaluates companies across three distinct quantitative tracks, adapting to different market regimes:

* **Track 1 - Quality Compounders:** Focuses on high ROIC, strong free cash flow generation, and low debt.
* **Track 2 - Growth Inflection (Primary Engine):** Identifies companies with accelerating quarterly revenue (revenue_accel_quarterly) and expanding margins. Currently the production baseline.
* **Track 3 - Value Recovery:** Targets fundamentally sound companies temporarily mispriced by the market.

### Current Performance (Track 2 v3 Baseline)
*Validated: May 2026*
* **Backtest (2013-2024):** 477.40% Total Return (+218.25% vs SP500) | 0.88 Sharpe
* **Held-out (2025-2026):** 45.76% Total Return (+18.24% vs SP500) | 1.54 Sharpe

## Tech Stack

* **Data Pipeline:** edgartools (SEC EDGAR API for fundamentals), yfinance (price data only).
* **Processing & Validation:** pandas, numpy, pandera (strict schema enforcement).
* **Persistence:** sqlite3 via SQLAlchemy.
* **Interface:** streamlit (Monthly dashboard for portfolio review and manual input).
* **Machine Learning (Phase 5):** scikit-learn, LightGBM (LambdaMART for within-group ranking).
* **Environment:** Python 3.11+, managed via uv.

## Project Structure

    crucible/
    |-- app/                (+) Streamlit dashboard application
    |-- crucible/           (+) Main package (fetcher, cleaner, scorer, filters)
    |   |-- ml/             (+) Experimental LightGBM ranking module
    |-- data/               (+) Local database, raw API caches, and processed sets
    |-- scripts/            (+) CLI tools for backtesting, bulk downloads, and monthly runs
    |-- tests/              (+) Pytest suite for fundamental filters and schemas

## Quick Start

### 1. Environment Setup

Clone the repository and install dependencies using uv:

    git clone https://github.com/gabrielalmeida13/crucible.git
    cd crucible
    uv sync

Configure your environment variables by copying the example file:

    cp .env.example .env

*(Ensure EDGAR_USER_AGENT is properly set in your .env to comply with SEC API guidelines).*

### 2. Data Initialization

For backtesting, download the bulk EDGAR facts archive (one-time operation):

    python scripts/download_edgar_bulk.py

### 3. Running the Engine

Execute the monthly prospective scan (e.g., for June 2026):

    python scripts/run_monthly.py --track 2 --budget 100 --month 2026-06

Launch the dashboard to review the generated shortlist and current portfolio:

    streamlit run app/dashboard.py

## Roadmap & Current Status

Crucible is currently in **Phase 5 (Operational System & ML Integration)**.

* (x) **Phase 0-4:** Pipeline setup, SEC EDGAR integration, backtesting engine, multi-track scoring, and baseline deployment.
* (x) **Phase 5.Q:** Quarterly EDGAR features deployed to Track 2.
* ( ) **Phase 5.0:** LightGBM LambdaMART ranking model (Under validation until Dec 2026).
* ( ) **Phase 5.1:** Options module integration (Protective puts and leveraged calls).
* ( ) **Phase 5.2:** Regime-adjusted scoring (Dynamic weight shifts based on volatility).
* ( ) **Phase 5.4:** Universe expansion to European markets (Requires non-EDGAR data source).

## Author
**Gabriel Almeida** - gabrielalmeida13 (GitHub)

## Disclaimer
*Crucible is an open-source research tool, not a financial advisor. The code and its outputs do not constitute investment advice. Any financial decisions should be made with proper risk management and independent research.*