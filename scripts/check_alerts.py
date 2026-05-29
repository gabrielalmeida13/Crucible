#!/usr/bin/env python3
"""Lightweight alert checker — no EDGAR snapshot or price data needed.

Checks:
  - Portfolio recommendation changes vs last logged run in SQLite
  - Monthly reminder on the 1st of each month

Suitable for a daily cron job:
  0 8 * * * cd /path/to/Crucible && uv run python scripts/check_alerts.py

Does NOT re-run the screener.  Reads only:
  - data/portfolio.csv       (held positions)
  - data/crucible_picks.db   (historical recommendations)

Alert channels are configured via .env:
  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID   (preferred)
  ALERT_EMAIL_FROM + ALERT_EMAIL_TO + ALERT_SMTP_HOST + ALERT_SMTP_PORT + ALERT_SMTP_PASSWORD
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from crucible.alerts import (
    check_alerts_from_history,
    check_monthly_reminder,
    dispatch_alerts,
)
from crucible.portfolio import load_portfolio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PICKS_DB = ROOT / "data" / "crucible_picks.db"
PORTFOLIO_CSV = ROOT / "data" / "portfolio.csv"


def main() -> None:
    positions = load_portfolio(PORTFOLIO_CSV)
    held_tickers = list(positions["ticker"]) if not positions.empty else None

    alerts = check_alerts_from_history(db_path=PICKS_DB, tickers=held_tickers)

    reminder = check_monthly_reminder(db_path=PICKS_DB)
    if reminder:
        alerts.append(reminder)

    if not alerts:
        log.info("No alerts.")
        return

    dispatch_alerts(alerts)


if __name__ == "__main__":
    main()
