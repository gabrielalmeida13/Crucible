#!/usr/bin/env python3
"""Filter-funnel diagnostic for the EDGAR pipeline.

Builds fundamentals snapshots for 12 annual dates (Jan 31, 2010–2021) and
shows exactly how many companies survive each filter stage. No prices, no
backtest, no sensitivity. Typical runtime: 2–8 minutes depending on disk speed.

Usage
-----
    python scripts/diagnose_funnel.py [--universe SP500|RUSSELL1000]

Output
------
    data/diagnostics/filter_funnel_{universe}.csv   — machine-readable funnel per date
    stdout                                          — formatted summary table
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from crucible.config import CrucibleConfig, FilterThresholds
from crucible.fetcher import (
    _load_cik_mapping,
    fetch_russell1000_tickers,
    fetch_sp500_tickers,
)
from crucible.filters import (
    filter_fcf_consistency,
    filter_gross_margin_stability,
    filter_leverage,
    filter_revenue_growth,
    filter_roic,
)
from crucible.snapshot import build_snapshots_parallel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EDGAR_DIR    = ROOT / "data" / "raw" / "edgar" / "companyfacts"
CIK_MAP_PATH = ROOT / "data" / "raw" / "edgar" / "cik_mapping.json"

SNAPSHOT_WORKERS = 4

# One snapshot per year: Jan 31 of 2010–2021
DIAG_DATES = pd.DatetimeIndex(
    [pd.Timestamp(f"{year}-01-31", tz="UTC") for year in range(2010, 2022)]
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Funnel computation
# ---------------------------------------------------------------------------


def _compute_funnel(
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    thresholds: FilterThresholds,
) -> list[dict]:
    rows: list[dict] = []
    for date in sorted(fund_by_date):
        df = fund_by_date[date]
        total        = len(df)
        n_insuff     = int(df["insufficient_data"].astype(bool).sum())
        usable       = df[~df["insufficient_data"].astype(bool)]
        n_roic_null  = int(usable["roic_proxy_avg"].isna().sum())

        after_roic   = filter_roic(usable, thresholds.roic_min)
        after_fcf    = filter_fcf_consistency(after_roic, thresholds.fcf_positive_min_years)
        after_debt   = filter_leverage(after_fcf, thresholds.net_debt_ebitda_max)
        after_growth = filter_revenue_growth(after_debt, thresholds.revenue_growth_positive_min_years)
        after_margin = filter_gross_margin_stability(after_growth)

        rows.append({
            "date":              date.date(),
            "total_tickers":     total,
            "insufficient_data": n_insuff,
            "roic_null":         n_roic_null,
            "passed_roic":       len(after_roic),
            "passed_fcf":        len(after_fcf),
            "passed_debt":       len(after_debt),
            "passed_growth":     len(after_growth),
            "passed_margin":     len(after_margin),
            "passed_all":        len(after_margin),
        })
    return rows


# ---------------------------------------------------------------------------
# Stdout table
# ---------------------------------------------------------------------------


def _print_table(rows: list[dict], thresholds: FilterThresholds, universe: str = "SP500") -> None:
    cols = [
        ("date",        "date",              10),
        ("total",       "total_tickers",      6),
        ("insuff",      "insufficient_data",  6),
        ("no_roic",     "roic_null",          7),
        ("→roic",       "passed_roic",        6),
        ("→fcf",        "passed_fcf",         5),
        ("→debt",       "passed_debt",        6),
        ("→growth",     "passed_growth",      7),
        ("→margin",     "passed_margin",      8),
        ("passed",      "passed_all",         6),
    ]

    print()
    print(f"Filter Funnel — {universe}, annual snapshots Jan 2010–2021")
    print(
        f"Thresholds:  ROIC ≥ {thresholds.roic_min:.0%}  |  "
        f"FCF positive ≥ {thresholds.fcf_positive_min_years} yrs  |  "
        f"Net Debt/EBITDA < {thresholds.net_debt_ebitda_max:.1f}  |  "
        f"Rev growth positive ≥ {thresholds.revenue_growth_positive_min_years} yrs  |  "
        f"Gross margin slope ≥ 0"
    )
    print()

    header = "  ".join(f"{label:>{width}}" for label, _, width in cols)
    sep    = "  ".join("-" * width for _, _, width in cols)
    print(header)
    print(sep)

    for row in rows:
        line = "  ".join(
            f"{str(row[key]):>{width}}" for _, key, width in cols
        )
        print(line)

    print(sep)

    # Averages row
    avg_row = {
        "date": "AVG",
        **{
            key: round(sum(r[key] for r in rows) / len(rows))
            for _, key, _ in cols[1:]
        },
    }
    print("  ".join(f"{str(avg_row[key]):>{width}}" for _, key, width in cols))
    print()

    # Narrative diagnosis
    avg_total  = avg_row["total_tickers"]
    avg_insuff = avg_row["insufficient_data"]
    avg_usable = avg_total - avg_insuff
    avg_passed = avg_row["passed_all"]

    pct = lambda n, d: f"{n/d:.0%}" if d else "n/a"

    print("Diagnosis")
    print(f"  Average universe size:         {avg_total}")
    print(f"  Insufficient data (< 3 yrs):   {avg_insuff}  ({pct(avg_insuff, avg_total)} of total)")
    print(f"  Usable after data gate:        {avg_usable}  ({pct(avg_usable, avg_total)} of total)")
    print(f"  Pass ROIC filter:              {avg_row['passed_roic']}  ({pct(avg_row['passed_roic'], avg_usable)} of usable)")
    print(f"  Pass FCF filter:               {avg_row['passed_fcf']}  ({pct(avg_row['passed_fcf'], avg_usable)} of usable)")
    print(f"  Pass leverage filter:          {avg_row['passed_debt']}  ({pct(avg_row['passed_debt'], avg_usable)} of usable)")
    print(f"  Pass revenue growth filter:    {avg_row['passed_growth']}  ({pct(avg_row['passed_growth'], avg_usable)} of usable)")
    print(f"  Pass gross margin filter:      {avg_row['passed_margin']}  ({pct(avg_row['passed_margin'], avg_usable)} of usable)")
    print(f"  Pass ALL filters:              {avg_passed}  ({pct(avg_passed, avg_usable)} of usable, {pct(avg_passed, avg_total)} of total)")
    print()
    output_path = ROOT / "data" / "diagnostics" / f"filter_funnel_{universe.lower()}.csv"
    print(f"  CSV written to: {output_path}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter-funnel diagnostic for the EDGAR pipeline")
    parser.add_argument(
        "--universe",
        choices=["SP500", "RUSSELL1000"],
        default="SP500",
        help="Universe to analyse (default: SP500)",
    )
    args = parser.parse_args()
    universe: str = args.universe

    for path, label in (
        (CIK_MAP_PATH, "CIK mapping"),
        (EDGAR_DIR,    "EDGAR companyfacts directory"),
    ):
        if not path.exists():
            log.error(
                "%s not found at %s. Run scripts/download_edgar_bulk.py first.",
                label, path,
            )
            sys.exit(1)

    config = CrucibleConfig(account_currency="USD")
    cik_map = _load_cik_mapping(CIK_MAP_PATH)

    if universe == "RUSSELL1000":
        log.info("Fetching Russell 1000 ticker list …")
        tickers = fetch_russell1000_tickers()
        log.info("%d Russell 1000 tickers", len(tickers))
    else:
        log.info("Fetching SP500 ticker list …")
        tickers = fetch_sp500_tickers()
        log.info("%d SP500 tickers", len(tickers))

    log.info(
        "Building %d annual snapshots (%d workers) …",
        len(DIAG_DATES), SNAPSHOT_WORKERS,
    )
    fund_by_date = build_snapshots_parallel(
        tickers=tickers,
        dates=DIAG_DATES,
        cik_map=cik_map,
        edgar_dir=EDGAR_DIR,
        prices=None,
        workers=SNAPSHOT_WORKERS,
    )

    log.info("Computing filter funnel …")
    rows = _compute_funnel(fund_by_date, config.filters)

    output_path = ROOT / "data" / "diagnostics" / f"filter_funnel_{universe.lower()}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    log.info("CSV written → %s", output_path)

    _print_table(rows, config.filters, universe)
    _print_column_coverage(fund_by_date)


def _print_column_coverage(fund_by_date: dict[pd.Timestamp, pd.DataFrame]) -> None:
    """Print non-null coverage % for each new feature column across all snapshots."""
    new_cols = [
        "interest_coverage",
        "cfo_to_ni",
        "capex_intensity",
        "operating_margin_trend",
        "revenue_acceleration",
        "share_buyback_signal",
        "insider_buy_ratio",
    ]
    # Use the latest snapshot for a single-date coverage summary
    latest_date = max(fund_by_date)
    df = fund_by_date[latest_date]
    total = len(df)
    print(f"Column coverage at {latest_date.date()} (n={total}):")
    for col in new_cols:
        if col in df.columns:
            n_valid = int(df[col].notna().sum())
            print(f"  {col:<28}  {n_valid:>4} / {total}  ({n_valid/total:.0%})")
        else:
            print(f"  {col:<28}  column missing")
    print()


if __name__ == "__main__":
    main()
