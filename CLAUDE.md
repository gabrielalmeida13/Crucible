# Crucible — CLAUDE.md

> This file is the working contract between the developer and Claude Code.
> Read it entirely before touching any file in the project.

---

## What is Crucible

Crucible is a monthly quality screener for stocks across the US market,
starting with the S&P 500 and expanding progressively to the Russell 1000
and Russell 3000 universes. European and Asian markets (available through
the XTB brokerage) are a future expansion — Phase 4.

The goal is not to predict prices. It is to filter a broad universe of companies
using rigorous fundamental criteria, rank the resulting shortlist with a composite
score, and produce one actionable output per month.

An ML layer (Phase 3) will be added later to improve scoring — never to replace
the fundamental filters.

---

## Stack

| Component        | Technology                       | Notes                                                    |
|------------------|----------------------------------|----------------------------------------------------------|
| Data (primary)   | SEC EDGAR API + `edgartools`     | Free, unlimited, official source, true point-in-time     |
| Data (bulk)      | EDGAR `companyfacts.zip`         | Single download for all companies — use for backtesting  |
| Prices           | `yfinance`                       | Price data only — not fundamentals                       |
| Processing       | `pandas`, `numpy`                | Core of the entire pipeline                              |
| Data validation  | `pandera`                        | Schema enforcement on every DataFrame entering processing |
| Persistence      | `sqlite3` via `SQLAlchemy`       | Stores monthly scan history                              |
| Dashboard        | `streamlit`                      | Fast UI, aesthetics are not a priority                   |
| ML (future)      | `scikit-learn`                   | Phase 3+ only                                            |
| Environment      | Python 3.11+, `uv`               | Always prefer `uv` over `pip`                            |

---

## Data source architecture

### SEC EDGAR — primary source for all fundamentals

The SEC EDGAR API (`data.sec.gov`) is the official US government source.
It is free, has no authentication, has no rate limits beyond 10 req/s,
and provides true point-in-time data via filing submission timestamps.

Key endpoints:
- `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` — all XBRL facts per company
- `https://data.sec.gov/api/xbrl/frames/` — cross-company data per concept per period
- Bulk: `https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip`

Use `edgartools` as the primary Python interface. For bulk backtest work,
download `companyfacts.zip` once and query locally — do not loop over 1000+
individual API calls.

XBRL structured data is available from 2009 onwards (SEC mandate). Filings
before 2009 exist but are unstructured (HTML/PDF) and inconsistent between
companies — do not use pre-2009 data for fundamental metrics.

### Point-in-time correctness

This is non-negotiable for backtesting. EDGAR provides it natively:
every filing has a `filed` date (the exact date the SEC received the document).
When reconstructing a historical scan for month M, only use filings with
`filed` date ≤ last day of month M. Never use the period end date —
a 10-K for fiscal year ending December 2020 may be filed in February 2021.

### yfinance — price data only

`yfinance` is used exclusively for historical price data (OHLCV).
It must never be used for fundamental metrics (income statement, balance sheet,
cash flow). Its fundamentals are not point-in-time and are retroactively
overwritten on restatements.

---

## Folder structure

```
crucible/
├── CLAUDE.md
├── ROADMAP.md
├── README.md
├── pyproject.toml
├── .env.example
│
├── data/
│   ├── raw/              # Data pulled directly from APIs — never edited manually
│   │   ├── edgar/        # EDGAR bulk downloads and per-company JSONs
│   │   └── prices/       # yfinance price data
│   ├── processed/        # Data after cleaning and feature engineering
│   └── crucible.db       # SQLite database with scan history
│
├── crucible/             # Main package
│   ├── __init__.py
│   ├── config.py         # Thresholds, parameters, universe definitions
│   ├── fetcher.py        # Fundamentals from EDGAR; prices from yfinance
│   ├── cleaner.py        # Cleaning, normalization, missing value detection
│   ├── validator.py      # Pandera schemas — enforced at cleaner output
│   ├── filters.py        # Layer 1: fundamental filters (hard rules)
│   ├── scorer.py         # Layer 2: composite quality + valuation score
│   ├── fx.py             # Currency conversion cost adjustments (Phase 4)
│   ├── store.py          # SQLite read/write
│   └── ml/               # Phase 3+ only
│       ├── __init__.py
│       ├── features.py
│       └── model.py
│
├── app/
│   └── dashboard.py      # Streamlit app
│
├── scripts/
│   ├── run_scan.py           # Entry point: runs the full monthly scan
│   └── download_edgar_bulk.py # One-time bulk download of companyfacts.zip
│
└── tests/
    ├── test_filters.py
    ├── test_scorer.py
    ├── test_fetcher.py
    └── test_validator.py
```

---

## Universe definition

The screener universe expands in stages. The active universe is controlled by
`CRUCIBLE_UNIVERSE` in `.env`. Do not mix universes within a single scan run.

| Stage | Universe ID    | Coverage                                            | Approx. companies |
|-------|----------------|-----------------------------------------------------|-------------------|
| 1     | `SP500`        | S&P 500 — large-cap US                              | ~500              |
| 2     | `RUSSELL1000`  | Russell 1000 — large + mid cap US                   | ~1,000            |
| 3     | `RUSSELL3000`  | Russell 3000 — large + mid + small cap US           | ~3,000            |
| 4     | `EUROPE_LARGE` | LSE, Deutsche Börse, Euronext Paris (future)        | +150              |
| 5     | `XTB_FULL`     | All XTB-available stocks with sufficient data (future) | ~2,000–3,000  |

All US stages use SEC EDGAR as data source and are available from 2009.
Stages 4 and 5 require a separate data source decision (Phase 4).

---

## Accounting standards awareness

For the current US-only focus, all companies use US GAAP — this simplifies
comparison significantly. Sector normalization is still required because
a 40% gross margin is exceptional in retail but poor in software.

When Stage 4+ is active, region normalization must also be added:

| Region  | Standard       | Key impact on metrics                                  |
|---------|----------------|--------------------------------------------------------|
| USA     | US GAAP        | Current baseline — all companies comparable            |
| Europe  | IFRS           | R&D capitalization differs; lease treatment differs    |
| Japan   | Japanese GAAP  | Conservative revenue recognition; lower ROICs typical  |

---

## Code conventions

### Code style
- Static typing on all functions (`def foo(x: pd.DataFrame) -> pd.DataFrame`)
- Short docstrings on all public functions (one line is enough)
- No obvious comments — code should be self-explanatory
- No `print()` in production — use `logging` with appropriate levels
- Pure functions wherever possible; side effects isolated in `store.py` and `fetcher.py`

### Data rules
- `raw/` is sacred: **never write to or transform data inside this folder**
- All DataFrames entering `filters.py` and `scorer.py` must have index = ticker (string)
- Missing values must always be explicit (`NaN`), never filled with zero without documentation
- Dates always in `datetime64[ns]` UTC
- Every DataFrame exiting `cleaner.py` must pass its Pandera schema before proceeding
- Point-in-time rule: when filtering EDGAR filings for a historical scan at date D,
  only include filings with `filed` ≤ D — never use fiscal period end date as proxy

### Pandera schemas
- Defined in `validator.py`, one schema per major DataFrame type
- Schema must specify: column names, dtypes, nullable flags, and value ranges where applicable
- If a schema check fails, raise immediately — never pass dirty data downstream silently

### Git
- Commits in English, imperative: `Add ROIC filter`, `Fix missing value in cleaner`
- One commit per logical feature/fix
- Never commit `crucible.db`, data in `raw/` or `processed/`, `.env`, or
  the EDGAR bulk zip file (too large)

### Tests
- Each filter in `filters.py` has at least one positive and one negative test
- Use fixtures with synthetic data — never depend on real API calls in tests
- Run `pytest` before any commit touching `filters.py` or `scorer.py`

---

## What Claude Code must NOT do

- **Do not use yfinance for fundamental data** — prices only; fundamentals come from EDGAR
- **Do not use pre-2009 EDGAR data** — XBRL was not mandated before 2009; earlier data
  is unstructured and inconsistent between companies
- **Do not use fiscal period end date as the filing date** — always use the `filed`
  timestamp from EDGAR for point-in-time correctness
- **Do not loop over individual EDGAR API calls for 1000+ companies** — download
  `companyfacts.zip` and query locally for bulk operations
- **Do not add dependencies** without updating `pyproject.toml` and justifying in the commit
- **Do not change thresholds** in `config.py` without explicit developer instruction
- **Do not fill missing data** with estimates — mark as `NaN` and log
- **Do not build the ML layer** until Phase 2 is validated (see ROADMAP.md)
- **Do not optimize the dashboard** before the data pipeline is stable
- **Do not expand the universe** beyond the current stage without explicit developer instruction
- **Do not compare metrics across sectors** without sector normalization active

---

## Entry point

```bash
# Install dependencies
uv sync

# One-time: download EDGAR bulk data for backtesting
python scripts/download_edgar_bulk.py

# Run monthly scan
python scripts/run_scan.py

# Launch dashboard
streamlit run app/dashboard.py
```

---

## Environment variables (.env)

```
CRUCIBLE_UNIVERSE=SP500              # SP500 | RUSSELL1000 | RUSSELL3000
CRUCIBLE_DB_PATH=data/crucible.db
CRUCIBLE_LOG_LEVEL=INFO
CRUCIBLE_ACCOUNT_CURRENCY=EUR        # For future FX cost calculation (Phase 4)
EDGAR_USER_AGENT=Crucible your@email.com  # Required by SEC — identify your app
```

> The SEC requires a User-Agent header identifying your application and contact
> email on all requests to data.sec.gov. Set EDGAR_USER_AGENT and pass it in
> every request header. This is not optional — repeated requests without it
> may result in IP blocking.

---

## Financial context notes (so Claude Code does not make naive decisions)

- The goal is to **support long-term investment decisions**, not trading
- The final score is a **relative ranking**, not an absolute recommendation
- Survivorship bias: the monthly universe must be the universe **of that month**,
  not the current one — companies delisted, acquired, or removed must be included
  in historical scans; EDGAR contains all historical filers regardless of current status
- Backtesting is only valid with walk-forward validation — never train and test
  on the same period; a model trained on 2009–2020 and tested on 2021–2025 is
  not the same as testing on 2009–2025 with in-sample data
- The XBRL data cutoff is 2009 — do not attempt to extend fundamentals before this
- The ML model is a tool, not an oracle — the monthly output is a starting point
  for further research, not a buy instruction