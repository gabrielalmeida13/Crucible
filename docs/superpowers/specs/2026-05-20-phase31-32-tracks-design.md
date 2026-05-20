# Phase 3.1 & 3.2 — Track 1 (Quality) and Track 2 (Growth) Design

**Date:** 2026-05-20  
**Status:** Approved — ready for implementation planning

---

## Context

Phase 3.0 confirmed Russell 1000 produces ≥ 15 picks/month in mature EDGAR years (16.7 avg 2015–2021). The three-track system can now be built. This spec covers Phase 3.1 (Track 1 extraction) and Phase 3.2 (Track 2 implementation) in parallel.

---

## Architecture decision: Approach A — thin wrappers over shared engine

- `crucible/snapshot.py` is the single source of truth for all snapshot computation
- `track1_quality.py` delegates to existing `filters.py` + `scorer.py` unchanged
- `track2_growth.py` defines new filter functions and a new scorer reusing `_peer_rank` from `scorer.py`
- No business logic is duplicated between tracks

---

## 1. `crucible/snapshot.py` — new shared snapshot module

Migrate all snapshot computation out of `diagnose_funnel.py` and `run_backtest.py` into this module. Both scripts import from it. No behavior changes to existing code paths.

### Public API

```python
load_raw_annual_facts(padded_cik, edgar_dir_str) -> dict[str, list[dict]]
get_raw_pivoted(cik, as_of_date, edgar_dir) -> dict[str, pd.Series]
compute_snapshot_row(ticker, pivoted, cik=None, as_of_date=None) -> dict
build_snapshot(date, tickers, cik_map) -> tuple[pd.Timestamp, pd.DataFrame]
build_snapshots_parallel(tickers, dates, cik_map, workers=4) -> dict[pd.Timestamp, pd.DataFrame]
attach_momentum(fund_by_date, prices) -> None  # adds momentum_raw and momentum_3m in-place
```

### New columns added to `compute_snapshot_row()` output

| Column | Source | Description |
|--------|--------|-------------|
| `revenue_growth_yr1` | EDGAR | YoY revenue growth rate, most recent fiscal year (float, e.g. 0.15 = 15%) |
| `revenue_growth_yr2` | EDGAR | YoY revenue growth rate, year prior to `revenue_growth_yr1` |
| `fcf_positive_last2yr` | EDGAR | Count of FCF-positive years in the last 2 fiscal years (0, 1, or 2) |
| `fcf_trajectory` | EDGAR | OLS slope of FCF values — direction of change, not level |
| `gross_margin_yr1_change` | EDGAR | Gross margin (latest year) minus gross margin (prior year). `gross_margin_latest` already exists; this adds the delta. |

### New columns added by `attach_momentum()`

| Column | Source | Description |
|--------|--------|-------------|
| `momentum_raw` | yfinance | 12-1 month price return (existing) |
| `momentum_3m` | yfinance | 3-month simple price return (new) |

### `p_s` (price-to-sales)

Populated in the price-attach layer (same step as `p_fcf`, `ev_ebitda`, `p_e`), not in `compute_snapshot_row`. Requires market cap = price × shares (EDGAR DEI shares) and annual revenue (already in snapshot). NaN when revenue is zero or unavailable.

---

## 2. `crucible/config.py` additions

```python
@dataclass(frozen=True)
class Track2FilterThresholds:
    revenue_growth_min_pct: float = 0.10      # both last 2 years must exceed this
    gross_margin_min: float = 0.30            # OR expanding YoY (either condition passes)
    fcf_positive_last2yr_min: int = 1         # ≥ 1 of last 2 years FCF-positive
    net_debt_ebitda_max: float = 5.0          # relaxed vs Track 1's 3.0

@dataclass(frozen=True)
class Track2ScoreWeights:
    growth_quality: float = 0.50
    momentum: float = 0.30
    valuation: float = 0.20
    # __post_init__ validates sum == 1.0
```

`CrucibleConfig` gains optional `track2_filters: Track2FilterThresholds` and `track2_score_weights: Track2ScoreWeights` fields with default_factory.

---

## 3. `crucible/tracks/__init__.py`

Empty file. Creates the `crucible.tracks` package.

---

## 4. `crucible/tracks/track1_quality.py` — thin wrapper

```python
def apply_filters(df, thresholds: FilterThresholds) -> pd.DataFrame
def score(df, config: CrucibleConfig) -> pd.DataFrame
def run(df, config: CrucibleConfig) -> pd.DataFrame  # filter then score
```

All three functions are one-liners that delegate to `crucible.filters` and `crucible.scorer`. No logic here. The value is a consistent interface across tracks.

---

## 5. `crucible/tracks/track2_growth.py` — new logic

### Filter stack (5 fundamental filters, no momentum in funnel)

Applied in sequence via `apply_filters()`. Each filter is a pure function on a DataFrame.

| # | Function | Pass condition |
|---|----------|----------------|
| 1 | `filter_revenue_growth_10pct` | `revenue_growth_yr1 > 0.10` AND `revenue_growth_yr2 > 0.10` |
| 2 | `filter_revenue_acceleration` | `revenue_acceleration > 0` (positive acceleration) |
| 3 | `filter_gross_margin_growth` | `gross_margin_latest >= 0.30` OR `gross_margin_yr1_change > 0` |
| 4 | `filter_fcf_positive_last2yr` | `fcf_positive_last2yr >= 1` |
| 5 | `filter_leverage` | `net_debt_ebitda < 5.0` |

NaN on any filter metric → fail (missing data cannot confirm growth thesis).

### Scorer

```python
def score(df, config: CrucibleConfig, weights: Track2ScoreWeights) -> pd.DataFrame
```

All metrics sector-normalised within GICS peer groups via `_peer_rank` (imported from `crucible.scorer`).

**Growth quality (50%):** equal-weight (1/3 each) average of:
- `revenue_acceleration` rank (ascending=True, NaN → 0.0)
- `operating_margin_trend` rank (ascending=True, NaN → 0.0)
- `fcf_trajectory` rank (ascending=True, NaN → 0.0)

**Momentum (30%):** average of:
- `momentum_raw` rank (12-1m, ascending=True, NaN → 0.5 neutral)
- `momentum_3m` rank (ascending=True, NaN → 0.5 neutral)

**Valuation (20%):** per-ticker fallback logic:
- Compute `p_s` peer-rank (ascending=False — lower P/S = higher rank); keep NaN where unavailable (do not fill with 0.0)
- Compute `p_fcf` peer-rank (ascending=False, NaN → 0.0 worst rank)
- Final valuation rank = `p_s` rank where available, else `p_fcf` rank
- Both ranked against sector median peers within same GICS sector

### `run(df, config, weights) -> pd.DataFrame`

Convenience: `apply_filters()` then `score()`. Returns sorted descending by `composite_score`.

---

## 6. `scripts/run_monthly.py`

### Usage

```bash
python scripts/run_monthly.py --track 1
python scripts/run_monthly.py --track 2
```

### Flow

1. Parse `--track` argument (1 or 2)
2. Fetch current universe tickers (controlled by `CRUCIBLE_UNIVERSE` env var, default Russell 1000)
3. Build today's point-in-time EDGAR snapshot via `crucible.snapshot.build_snapshots_parallel()` (single date: today)
4. Fetch prices via yfinance; call `attach_momentum()` for `momentum_raw` + `momentum_3m`
5. Attach price-based valuation ratios (`p_fcf`, `p_s`, `ev_ebitda`, `p_e`) using EDGAR shares × price
6. Attach sector data from the ticker list
7. Dispatch to `track1_quality.run(df, config)` or `track2_growth.run(df, config, weights)`
8. Print ranked top-10 shortlist to stdout with all score components visible
9. Write `data/monthly/{YYYY-MM}/track{N}_picks.md` — full metric dump per candidate, designed for AI-assisted reasoning (ROADMAP Phase 3.4)

### Output format (`track{N}_picks.md`)

Markdown table, one row per company in the top 10. Columns: rank, ticker, sector, composite_score, score components, every filter metric value. Plus a header block with: run date, universe, track, filter thresholds used.

---

## 7. `scripts/diagnose_funnel.py` changes

Add `--track` argument:

```bash
python scripts/diagnose_funnel.py --universe RUSSELL1000 --track 1   # default, existing behavior
python scripts/diagnose_funnel.py --universe RUSSELL1000 --track 2   # new
```

Track 2 funnel (no momentum — no prices in diagnostic):

| Column | Filter |
|--------|--------|
| `→rev_growth_10pct` | both `revenue_growth_yr1` and `revenue_growth_yr2` > 10% |
| `→rev_acceleration` | `revenue_acceleration > 0` |
| `→gross_margin` | `gross_margin_latest >= 0.30` OR `gross_margin_yr1_change > 0` |
| `→fcf_last2yr` | `fcf_positive_last2yr >= 1` |
| `→leverage` | `net_debt_ebitda < 5.0` |

Output CSV: `data/diagnostics/filter_funnel_track2_russell1000.csv`

Diagnosis note printed to stdout: "Momentum filter (price momentum > 0) not shown — no price data in diagnostic. Live Track 2 output will cut further."

Both tracks import snapshot computation from `crucible/snapshot.py`.

---

## Migration plan for existing scripts

`diagnose_funnel.py` and `run_backtest.py` currently define `_load_raw_annual_facts`, `_get_raw_pivoted`, `_linear_slope`, `_compute_snapshot_row`, `_build_snapshot`, `_build_snapshots_parallel` independently. After this work:

- Both scripts delete their local definitions of these functions
- Both import from `crucible.snapshot` instead
- Behavior is identical — the only change is the source of the functions

---

## What this spec does NOT cover

- Track 3 (Value Recovery) — Phase 3.3, separate spec
- Combined backtest / held-out — Phase 3.5, after all three tracks are validated
- Phase 3.4 monthly output enhancements (reasoning.json, SQLite logging) — deferred
- Dashboard changes — Phase 4.1
