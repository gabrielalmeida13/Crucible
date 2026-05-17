# Crucible — CLAUDE.md

> This file is the working contract between the developer and Claude Code.
> Read it entirely before touching any file in the project.

---

## What is Crucible

Crucible is a monthly quality screener for stocks across multiple global exchanges,
starting with the S&P 500 and expanding progressively to European and Asian markets
available through the XTB brokerage platform.

The goal is not to predict prices. It is to filter a broad universe of companies
using rigorous fundamental criteria, rank the resulting shortlist with a composite
score, and produce one actionable output per month.

An ML layer (Phase 3) will be added later to improve scoring — never to replace
the fundamental filters.

---

## Stack

| Component        | Technology                    | Notes                                              |
|------------------|-------------------------------|----------------------------------------------------|
| Data (dev)       | `yfinance`                    | Free, sufficient for Phase 1 pipeline development  |
| Data (backtest)  | Financial Modeling Prep (FMP) | Required before any backtest — point-in-time data  |
| Processing       | `pandas`, `numpy`             | Core of the entire pipeline                        |
| Data validation  | `pandera`                     | Schema enforcement on every DataFrame entering processing |
| Persistence      | `sqlite3` via `SQLAlchemy`    | Stores monthly scan history                        |
| Dashboard        | `streamlit`                   | Fast UI, aesthetics are not a priority             |
| ML (future)      | `scikit-learn`                | Phase 3+ only                                      |
| Environment      | Python 3.11+, `uv`            | Always prefer `uv` over `pip`                      |

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
│   ├── processed/        # Data after cleaning and feature engineering
│   └── crucible.db       # SQLite database with scan history
│
├── crucible/             # Main package
│   ├── __init__.py
│   ├── config.py         # Thresholds, parameters, universe definitions
│   ├── fetcher.py        # Data extraction (yfinance / FMP)
│   ├── cleaner.py        # Cleaning, normalization, missing value detection
│   ├── validator.py      # Pandera schemas — enforced at cleaner output
│   ├── filters.py        # Layer 1: fundamental filters (hard rules)
│   ├── scorer.py         # Layer 2: composite quality + valuation score
│   ├── fx.py             # Currency conversion cost adjustments
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
│   └── run_scan.py       # Entry point: runs the full monthly scan
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

| Stage | Universe ID       | Exchanges                                           | Approx. companies |
|-------|-------------------|-----------------------------------------------------|-------------------|
| 1     | `SP500`           | NYSE, NASDAQ                                        | ~500              |
| 2     | `EUROPE_LARGE`    | LSE, Deutsche Börse (Frankfurt), Euronext Paris     | +150              |
| 3     | `JAPAN_ADR`       | Japanese large-caps via ADRs / European listings    | +80               |
| 4     | `XTB_FULL`        | All XTB-available stocks with sufficient data       | ~2,000–3,000      |

When running multi-exchange scans (Stage 2+), **sector normalization must also
include regional normalization** — see scorer.py notes below.

---

## Accounting standards awareness

Different regions use different accounting standards. Metrics must never be
compared raw across regions without normalization:

| Region  | Standard       | Key impact on metrics                                  |
|---------|----------------|--------------------------------------------------------|
| USA     | US GAAP        | Baseline for most financial databases                  |
| Europe  | IFRS           | R&D capitalization differs; lease treatment differs    |
| Japan   | Japanese GAAP  | Conservative revenue recognition; lower ROICs typical  |

The scorer must always compare companies **within the same GICS sector AND
the same accounting region**. Absolute threshold filters (Layer 1) may need
region-specific calibration once Stage 2+ is active.

---

## Currency conversion cost

XTB applies a 0.5% FX conversion cost when buying stocks denominated in a
currency different from the account currency. This is a real transaction cost
and must be factored into the composite score.

`fx.py` is responsible for:
- Identifying the denomination currency of each ticker
- Flagging stocks that require conversion from the user's account currency
- Applying a configurable score penalty (default: -0.5 points) for stocks
  requiring FX conversion, to reflect the additional transaction cost

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

### Pandera schemas
- Defined in `validator.py`, one schema per major DataFrame type
- Schema must specify: column names, dtypes, nullable flags, and value ranges where applicable
- If a schema check fails, raise immediately — never pass dirty data downstream silently

### Git
- Commits in English, imperative: `Add ROIC filter`, `Fix missing value in cleaner`
- One commit per logical feature/fix
- Never commit `crucible.db`, data in `raw/` or `processed/`, or `.env` files

### Tests
- Each filter in `filters.py` has at least one positive and one negative test
- Use fixtures with synthetic data — never depend on real API calls in tests
- Run `pytest` before any commit touching `filters.py` or `scorer.py`

---

## What Claude Code must NOT do

- **Do not add dependencies** without updating `pyproject.toml` and justifying in the commit
- **Do not change thresholds** in `config.py` without explicit developer instruction
- **Do not fill missing data** with estimates — mark as `NaN` and log
- **Do not build the ML layer** until Phase 2 is validated (see ROADMAP.md)
- **Do not optimize the dashboard** before the data pipeline is stable
- **Do not run backtests using yfinance data** — yfinance does not provide point-in-time
  data and will produce look-ahead bias; FMP must be configured before Phase 2 begins
- **Do not compare metrics across regions** without sector + region normalization active
- **Do not expand the universe** beyond the current stage without explicit developer instruction

---

## Entry point

```bash
# Install dependencies
uv sync

# Run monthly scan
python scripts/run_scan.py

# Launch dashboard
streamlit run app/dashboard.py
```

---

## Environment variables (.env)

```
CRUCIBLE_UNIVERSE=SP500            # SP500 | EUROPE_LARGE | JAPAN_ADR | XTB_FULL
CRUCIBLE_DB_PATH=data/crucible.db
CRUCIBLE_LOG_LEVEL=INFO
CRUCIBLE_FMP_API_KEY=              # Required before Phase 2
CRUCIBLE_ACCOUNT_CURRENCY=EUR      # Used by fx.py for conversion cost calculation
```

---

## Financial context notes (so Claude Code does not make naive decisions)

- The goal is to **support long-term investment decisions**, not trading
- The final score is a **relative ranking**, not an absolute recommendation
- Survivorship bias: the monthly universe must be the universe **of that month**,
  not the current one — companies delisted, acquired, or removed must be included
  in historical scans
- Backtesting is only valid with walk-forward validation — never train and test
  on the same period
- The ML model is a tool, not an oracle — the monthly output is a starting point
  for further research, not a buy instruction
- FX conversion costs are real and must not be ignored when comparing
  cross-currency stocks in the scorer