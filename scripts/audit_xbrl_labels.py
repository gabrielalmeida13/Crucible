#!/usr/bin/env python3
"""Audit XBRL label coverage across EDGAR companyfacts JSONs.

For each fundamental concept, searches all us-gaap keys in the sampled files
for case-insensitive substring matches, reports frequency, and flags companies
with no match at all.

Usage
-----
    python scripts/audit_xbrl_labels.py [--seed N] [--n N] [--universe sp500]

    --universe sp500  Sample from S&P 500 constituents (fetched from Wikipedia)
                      mapped to CIKs via cik_mapping.json, instead of random JSONs.

Output: markdown printed to stdout (redirect to a file to save).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPANYFACTS_DIR = ROOT / "data" / "raw" / "edgar" / "companyfacts"
CIK_MAPPING_PATH = ROOT / "data" / "raw" / "edgar" / "cik_mapping.json"

# Allow `from crucible.fetcher import ...` when run as a script
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SAMPLE = 20
DEFAULT_SEED   = 42

# ---------------------------------------------------------------------------
# Concept → search substrings (case-insensitive substring match on us-gaap keys)
# Ordered to mirror the fetcher XBRL taxonomy.
# ---------------------------------------------------------------------------

CONCEPTS: dict[str, list[str]] = {
    "Revenue": [
        "revenuefromcontractwithcustomer",
        "revenues",                         # plain Revenues
        "salesrevenue",                     # SalesRevenueNet, SalesRevenueGoodsNet
        "revenuesnetofinterestexpense",     # banks
    ],
    "Gross Profit": [
        "grossprofit",
    ],
    "Net Income": [
        "netincomeloss",
        "profitloss",
    ],
    "Operating Cash Flow": [
        "netcashprovidedbyusedInoperatingactivities",
        "operatingactivities",              # broader fallback
    ],
    "CapEx": [
        "paymentstoacquirepropertyplant",
        "capitalexpenditures",
    ],
    "Total Debt": [
        "longtermdebt",                     # LongTermDebt, LongTermDebtNoncurrent, LongTermDebtCurrent
        "debtcurrent",                      # DebtCurrent
        "shorttermborrow",                  # ShortTermBorrowings
        "notespayable",                     # NotesPayableCurrent, LongTermNotesPayable
    ],
    "Cash & Equivalents": [
        "cashandcashequivalentsatcarrying",  # CashAndCashEquivalentsAtCarryingValue
        "cashcashequivalents",              # CashCashEquivalentsAndShortTermInvestments, CashCashEquivalentsRestrictedCash…
    ],
    "Stockholders Equity": [
        "stockholdersequity",
    ],
    "Interest Expense": [
        "interestexpense",
    ],
    "Depreciation": [
        "depreciationdepletionandamortization",
        "depreciationandamortization",
        "depreciation",                     # plain Depreciation
    ],
    "Tax Expense": [
        "incometaxexpensebenefit",          # IncomeTaxExpenseBenefit (the primary P&L line)
        "currentincometaxexpense",
        "deferredincometaxexpense",
    ],
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _key_matches(key: str, substrings: list[str]) -> bool:
    k = key.lower()
    return any(sub.lower() in k for sub in substrings)


def load_sample(
    companyfacts_dir: Path, n: int, seed: int
) -> list[tuple[str, set[str]]]:
    """Return [(entity_name, set_of_us_gaap_keys), …] for n random files."""
    files = sorted(companyfacts_dir.glob("CIK*.json"))
    if not files:
        print(f"ERROR: no CIK*.json files in {companyfacts_dir}", file=sys.stderr)
        sys.exit(1)

    chosen = random.Random(seed).sample(files, min(n, len(files)))
    results: list[tuple[str, set[str]]] = []
    for path in chosen:
        try:
            raw = json.loads(path.read_bytes())
            name    = raw.get("entityName") or path.stem
            us_gaap = raw.get("facts", {}).get("us-gaap", {})
            results.append((name, set(us_gaap.keys())))
        except Exception as exc:
            print(f"WARN: skipping {path.name}: {exc}", file=sys.stderr)
    return results


def load_sp500_sample(
    companyfacts_dir: Path, n: int, seed: int
) -> list[tuple[str, set[str]]]:
    """Return [(entity_name, set_of_us_gaap_keys), …] for n S&P 500 companies.

    Fetches the current S&P 500 constituent list from Wikipedia, maps tickers to
    CIKs via cik_mapping.json, and samples n from those with a local JSON file.
    """
    from crucible.fetcher import fetch_sp500_tickers  # noqa: PLC0415

    if not CIK_MAPPING_PATH.exists():
        print(
            f"ERROR: CIK mapping not found at {CIK_MAPPING_PATH}. "
            "Run scripts/download_edgar_bulk.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    raw_mapping: dict = json.loads(CIK_MAPPING_PATH.read_bytes())
    ticker_to_cik: dict[str, str] = {
        entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
        for entry in raw_mapping.values()
        if "ticker" in entry and "cik_str" in entry
    }

    sp500_tickers = fetch_sp500_tickers()
    print(f"Fetched {len(sp500_tickers)} S&P 500 tickers", file=sys.stderr)

    available: list[tuple[str, Path]] = []
    missing_cik: list[str] = []
    missing_file: list[str] = []
    for ticker in sp500_tickers:
        cik = ticker_to_cik.get(ticker.upper())
        if not cik:
            missing_cik.append(ticker)
            continue
        path = companyfacts_dir / f"CIK{cik}.json"
        if not path.exists():
            missing_file.append(ticker)
            continue
        available.append((ticker, path))

    if missing_cik:
        print(f"WARN: {len(missing_cik)} tickers have no CIK in mapping", file=sys.stderr)
    if missing_file:
        print(f"WARN: {len(missing_file)} tickers have no local JSON file", file=sys.stderr)

    if not available:
        print(
            f"ERROR: no S&P 500 CIK JSON files found in {companyfacts_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"{len(available)} S&P 500 companies have local EDGAR JSON files", file=sys.stderr)

    chosen = random.Random(seed).sample(available, min(n, len(available)))
    results: list[tuple[str, set[str]]] = []
    for ticker, path in chosen:
        try:
            raw = json.loads(path.read_bytes())
            name = raw.get("entityName") or ticker
            us_gaap = raw.get("facts", {}).get("us-gaap", {})
            results.append((name, set(us_gaap.keys())))
        except Exception as exc:
            print(f"WARN: skipping {path.name}: {exc}", file=sys.stderr)
    return results


def analyse(
    sample: list[tuple[str, set[str]]],
    concepts: dict[str, list[str]],
) -> dict[str, dict]:
    """Return per-concept dicts with label counts and no-match company lists."""
    report: dict[str, dict] = {}
    for concept, substrings in concepts.items():
        label_counter: Counter[str] = Counter()
        no_match: list[str] = []

        for entity_name, keys in sample:
            matched = {k for k in keys if _key_matches(k, substrings)}
            label_counter.update(matched)
            if not matched:
                no_match.append(entity_name)

        report[concept] = {
            "label_counts": label_counter,
            "no_match":     no_match,
        }
    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _bar(fraction: float, width: int = 10) -> str:
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)


def render(
    report: dict[str, dict],
    sample: list[tuple[str, set[str]]],
    seed: int,
    source_label: str = "random EDGAR companyfacts JSONs",
) -> str:
    n = len(sample)
    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        "# XBRL Label Audit",
        "",
        f"**Sample:** {n} {source_label}  ",
        f"**Seed:** {seed}  ",
        f"**Companies sampled:** {', '.join(name for name, _ in sample)}",
        "",
        "---",
        "",
    ]

    # ── Summary table ────────────────────────────────────────────────────────
    lines += [
        "## Summary",
        "",
        "| Concept | Distinct labels | No-match companies |",
        "|---------|----------------|-------------------|",
    ]
    for concept, data in report.items():
        n_labels   = len(data["label_counts"])
        n_no_match = len(data["no_match"])
        flag = f"⚠ {n_no_match}/{n}" if n_no_match else f"✓ 0/{n}"
        lines.append(f"| {concept} | {n_labels} | {flag} |")
    lines += ["", "---", ""]

    # ── Per-concept detail ────────────────────────────────────────────────────
    for concept, data in report.items():
        counts: Counter = data["label_counts"]
        no_match: list[str] = data["no_match"]

        lines.append(f"## {concept}")
        lines.append("")

        if counts:
            lines += [
                f"| Label | n/{n} | Coverage |",
                "|-------|------|---------|",
            ]
            for label, count in counts.most_common():
                frac = count / n
                lines.append(
                    f"| `{label}` | {count} | {frac:.0%} {_bar(frac)} |"
                )
        else:
            lines.append("*No matching us-gaap keys found in any sampled company.*")

        lines.append("")

        if no_match:
            lines.append(
                f"**⚠ No match in {len(no_match)}/{n} companies:** "
                + ", ".join(f"*{e}*" for e in no_match)
            )
        else:
            lines.append(f"**✓ All {n} companies have at least one matching label.**")

        lines += ["", "---", ""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n",    type=int, default=DEFAULT_SAMPLE,
                        help="Number of companies to sample (default: 20)")
    parser.add_argument(
        "--universe",
        choices=["sp500"],
        default=None,
        help="Sample from a specific universe instead of random JSONs",
    )
    args = parser.parse_args()

    if not COMPANYFACTS_DIR.exists():
        print(
            f"ERROR: {COMPANYFACTS_DIR} not found. "
            "Run scripts/download_edgar_bulk.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.universe == "sp500":
        sample = load_sp500_sample(COMPANYFACTS_DIR, args.n, args.seed)
        source_label = "S&P 500 companies (sampled from Wikipedia constituent list)"
    else:
        sample = load_sample(COMPANYFACTS_DIR, args.n, args.seed)
        source_label = "random EDGAR companyfacts JSONs"

    print(
        f"Loaded {len(sample)} company JSONs  (universe={args.universe or 'random'}, seed={args.seed})",
        file=sys.stderr,
    )

    report = analyse(sample, CONCEPTS)
    print(render(report, sample, args.seed, source_label))


if __name__ == "__main__":
    main()
