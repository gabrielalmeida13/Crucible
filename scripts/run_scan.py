"""Entry point for the monthly scan pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from crucible.cleaner import clean  # noqa: E402
from crucible.config import load_config  # noqa: E402
from crucible.fetcher import fetch_universe  # noqa: E402
from crucible.filters import apply_filters  # noqa: E402
from crucible.logging_setup import configure_logging  # noqa: E402
from crucible.scorer import score  # noqa: E402


def main() -> int:
    """Run the full monthly scan pipeline."""
    parser = argparse.ArgumentParser(description="Crucible monthly stock scan")
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Run with a specific set of tickers (test mode)",
    )
    args = parser.parse_args()

    config = load_config()
    configure_logging(config.log_level)
    logger = logging.getLogger(__name__)

    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    logger.info(
        "Crucible scan starting — universe=%s  tickers=%s",
        config.universe,
        args.tickers or "full universe",
    )

    run_ts, *_ = fetch_universe(
        config.universe, raw_dir, tickers=args.tickers or None
    )
    processed = clean(raw_dir, run_ts, processed_dir)
    filtered = apply_filters(processed, config.filters)
    shortlist = score(filtered, config)

    logger.info(
        "Pipeline complete — %d processed → %d filtered → %d scored",
        len(processed),
        len(filtered),
        len(shortlist),
    )

    if not shortlist.empty:
        cols = ["sector", "roic_proxy_avg", "gross_margin_avg",
                "net_debt_ebitda", "quality_score", "valuation_score",
                "fx_penalty", "composite_score"]
        logger.info("Shortlist (top %d):\n%s", len(shortlist),
                    shortlist[cols].to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
