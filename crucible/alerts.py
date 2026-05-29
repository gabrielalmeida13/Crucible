"""Portfolio and system alerts for the Crucible screener.

Public API
----------
check_portfolio_alerts   — fresh evaluation + history comparison; use in run_monthly
check_alerts_from_history — SQLite-only; no snapshot needed; use in daily cron
check_monthly_reminder   — first-of-month reminder if screener not yet run
send_telegram_alert      — push via Telegram Bot API
send_email_alert         — push via SMTP
dispatch_alerts          — print to stdout + send via configured channel
"""
from __future__ import annotations

import logging
import smtplib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from crucible.config import CrucibleConfig
from crucible.portfolio import evaluate_portfolio, load_portfolio

log = logging.getLogger(__name__)

_PICKS_DB = Path(__file__).resolve().parent.parent / "data" / "crucible_picks.db"


@dataclass
class Alert:
    """A single actionable alert raised by the screener."""

    kind: str
    # EXIT_SIGNAL | MOMENTUM_NEGATIVE_2M | HOLD_TO_REVIEW | MONTHLY_REMINDER
    ticker: Optional[str]  # None for system-level alerts like MONTHLY_REMINDER
    message: str


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


def _query_last_n_recs(
    db_path: Path,
    tickers: list[str],
    n: int = 2,
) -> dict[str, list[dict]]:
    """Return the last n logged records per ticker from monthly_picks.

    Returns {ticker: [newest, ..., oldest]} limited to n entries.
    Each dict: run_date, recommendation, momentum_3m.
    """
    if not db_path.exists() or not tickers:
        return {t: [] for t in tickers}

    placeholders = ", ".join("?" for _ in tickers)
    sql = (
        f"SELECT ticker, run_date, recommendation, momentum_3m "
        f"FROM monthly_picks "
        f"WHERE ticker IN ({placeholders}) "
        f"ORDER BY ticker, run_date DESC"
    )
    result: dict[str, list[dict]] = {t: [] for t in tickers}
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute(sql, tickers):
                t = row["ticker"]
                if t in result and len(result[t]) < n:
                    result[t].append(
                        {
                            "run_date": row["run_date"],
                            "recommendation": row["recommendation"],
                            "momentum_3m": row["momentum_3m"],
                        }
                    )
    except sqlite3.Error:
        log.warning("alerts: failed to query monthly_picks", exc_info=True)
    return result


def _has_run_this_month(db_path: Path, year_month: str) -> bool:
    """Return True if monthly_picks has any row whose run_date starts with YYYY-MM."""
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM monthly_picks WHERE run_date LIKE ? LIMIT 1",
                (f"{year_month}%",),
            ).fetchone()
        return row is not None
    except sqlite3.Error:
        log.warning("alerts: failed to check run history", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Core alert logic
# ---------------------------------------------------------------------------


def check_portfolio_alerts(
    portfolio_csv: Path,
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    config: CrucibleConfig,
    db_path: Path = _PICKS_DB,
) -> list[Alert]:
    """Evaluate the portfolio and return actionable alerts.

    Call this BEFORE log_monthly_picks so that SQLite holds only prior-run data.

    Alert types produced:
    - EXIT_SIGNAL       — position moved to EXIT_SIGNAL vs last logged state
    - MOMENTUM_NEGATIVE_2M — momentum_3m negative this month AND in last logged run
    - HOLD_TO_REVIEW    — position moved from HOLD/REINFORCE to REVIEW
    """
    positions = load_portfolio(portfolio_csv)
    if positions.empty:
        return []

    eval_df = evaluate_portfolio(positions, fund_by_date, prices, config)
    if eval_df.empty:
        return []

    tickers = list(eval_df.index)
    history = _query_last_n_recs(db_path, tickers, n=1)

    latest_snap = fund_by_date[max(fund_by_date.keys())] if fund_by_date else pd.DataFrame()
    alerts: list[Alert] = []

    for ticker in tickers:
        curr_rec = eval_df.at[ticker, "recommendation"].lower()
        prev_runs = history.get(ticker, [])
        prev_rec = prev_runs[0]["recommendation"].lower() if prev_runs else None

        # (a) EXIT_SIGNAL transition
        if curr_rec == "exit_signal" and prev_rec not in ("exit_signal", None):
            alerts.append(
                Alert(
                    kind="EXIT_SIGNAL",
                    ticker=ticker,
                    message=(
                        f"EXIT SIGNAL — {ticker}: failed track filters with negative 3m momentum. "
                        f"Previous recommendation was {prev_rec.upper()}. Consider selling."
                    ),
                )
            )

        # (b) Consecutive negative momentum: current snapshot + last logged run
        curr_mom: float | None = None
        if (
            not latest_snap.empty
            and "momentum_3m" in latest_snap.columns
            and ticker in latest_snap.index
        ):
            v = latest_snap.at[ticker, "momentum_3m"]
            if pd.notna(v):
                curr_mom = float(v)

        prev_mom_raw = prev_runs[0].get("momentum_3m") if prev_runs else None
        prev_mom = float(prev_mom_raw) if prev_mom_raw is not None else None

        if curr_mom is not None and curr_mom < 0 and prev_mom is not None and prev_mom < 0:
            alerts.append(
                Alert(
                    kind="MOMENTUM_NEGATIVE_2M",
                    ticker=ticker,
                    message=(
                        f"MOMENTUM WARNING — {ticker}: 3-month momentum negative for at least "
                        f"two consecutive months (current {curr_mom:+.1%}, prior {prev_mom:+.1%}). "
                        "Review position."
                    ),
                )
            )

        # (c) HOLD/REINFORCE → REVIEW
        if curr_rec == "review" and prev_rec in ("hold", "reinforce"):
            alerts.append(
                Alert(
                    kind="HOLD_TO_REVIEW",
                    ticker=ticker,
                    message=(
                        f"DOWNGRADE — {ticker}: recommendation changed from "
                        f"{prev_rec.upper()} to REVIEW. "
                        "Passes filters but dropped out of top-20 ranking. Monitor closely."
                    ),
                )
            )

    return alerts


def check_alerts_from_history(
    db_path: Path = _PICKS_DB,
    tickers: list[str] | None = None,
) -> list[Alert]:
    """SQLite-only alert check — no snapshot or price data needed.

    Compares the two most recent logged runs per ticker.  Intended for the
    daily cron job in scripts/check_alerts.py.

    tickers — restrict to these tickers (pass held positions from portfolio.csv).
              None = all tickers that appear in the last two distinct run_dates.
    """
    if not db_path.exists():
        return []

    if tickers is None:
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT ticker FROM monthly_picks "
                    "WHERE run_date IN ("
                    "  SELECT DISTINCT run_date FROM monthly_picks "
                    "  ORDER BY run_date DESC LIMIT 2"
                    ")"
                ).fetchall()
            tickers = [r[0] for r in rows]
        except sqlite3.Error:
            log.warning("alerts: failed to discover tickers", exc_info=True)
            return []

    if not tickers:
        return []

    history = _query_last_n_recs(db_path, tickers, n=2)
    alerts: list[Alert] = []

    for ticker, runs in history.items():
        if len(runs) < 2:
            continue  # need two data points

        curr = runs[0]
        prev = runs[1]
        curr_rec = curr["recommendation"].lower() if curr["recommendation"] else ""
        prev_rec = prev["recommendation"].lower() if prev["recommendation"] else ""

        # (a) EXIT_SIGNAL transition
        if curr_rec == "exit_signal" and prev_rec not in ("exit_signal", ""):
            alerts.append(
                Alert(
                    kind="EXIT_SIGNAL",
                    ticker=ticker,
                    message=(
                        f"EXIT SIGNAL — {ticker}: last run shows EXIT_SIGNAL "
                        f"(was {prev_rec.upper()}). Consider selling."
                    ),
                )
            )

        # (b) Consecutive negative momentum
        curr_mom_raw = curr.get("momentum_3m")
        prev_mom_raw = prev.get("momentum_3m")
        if curr_mom_raw is not None and prev_mom_raw is not None:
            curr_mom = float(curr_mom_raw)
            prev_mom = float(prev_mom_raw)
            if curr_mom < 0 and prev_mom < 0:
                alerts.append(
                    Alert(
                        kind="MOMENTUM_NEGATIVE_2M",
                        ticker=ticker,
                        message=(
                            f"MOMENTUM WARNING — {ticker}: 3-month momentum negative in "
                            f"two consecutive runs ({prev_mom:+.1%}, {curr_mom:+.1%}). "
                            "Review position."
                        ),
                    )
                )

        # (c) HOLD/REINFORCE → REVIEW
        if curr_rec == "review" and prev_rec in ("hold", "reinforce"):
            alerts.append(
                Alert(
                    kind="HOLD_TO_REVIEW",
                    ticker=ticker,
                    message=(
                        f"DOWNGRADE — {ticker}: last run shows REVIEW "
                        f"(was {prev_rec.upper()}). Dropped from top-20. Monitor closely."
                    ),
                )
            )

    return alerts


def check_monthly_reminder(
    db_path: Path = _PICKS_DB,
    today: datetime | None = None,
) -> Alert | None:
    """Return a MONTHLY_REMINDER alert if today is the 1st and no run exists this month.

    Returns None when the condition is not met (not the 1st, or already run).
    """
    if today is None:
        today = datetime.now(timezone.utc)

    if today.day != 1:
        return None

    year_month = today.strftime("%Y-%m")
    if _has_run_this_month(db_path, year_month):
        return None

    return Alert(
        kind="MONTHLY_REMINDER",
        ticker=None,
        message=(
            f"Crucible: monthly screener not yet run for {year_month}. "
            f"Run scripts/run_monthly.py --track 2 --budget 100"
        ),
    )


# ---------------------------------------------------------------------------
# Notification channels
# ---------------------------------------------------------------------------


def send_telegram_alert(message: str, token: str, chat_id: str) -> bool:
    """Send message via Telegram Bot API. Returns True on success."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": message},
            timeout=10,
        )
        if not resp.ok:
            log.warning(
                "Telegram alert failed: %s — %s", resp.status_code, resp.text[:200]
            )
            return False
        return True
    except Exception:
        log.warning("Telegram alert failed", exc_info=True)
        return False


def send_email_alert(message: str, smtp_config: dict) -> bool:
    """Send message via SMTP (STARTTLS on port 587 by default).

    smtp_config keys: from_addr, to_addr, host, port (int), password
    Returns True on success, False if config is incomplete or send fails.
    """
    from_addr = smtp_config.get("from_addr", "")
    to_addr = smtp_config.get("to_addr", "")
    host = smtp_config.get("host", "")
    port = int(smtp_config.get("port", 587))
    password = smtp_config.get("password", "")

    if not all([from_addr, to_addr, host, password]):
        log.warning("Email alert skipped — incomplete SMTP config")
        return False

    msg = MIMEText(message, "plain")
    msg["Subject"] = "Crucible Alert"
    msg["From"] = from_addr
    msg["To"] = to_addr

    try:
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            server.login(from_addr, password)
            server.sendmail(from_addr, [to_addr], msg.as_string())
        return True
    except Exception:
        log.warning("Email alert failed", exc_info=True)
        return False


def dispatch_alerts(alerts: list[Alert]) -> None:
    """Print alerts to stdout and send via configured channel.

    Prefers Telegram (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID).
    Falls back to email (ALERT_EMAIL_* + ALERT_SMTP_*).
    Gracefully skips if neither channel is configured.
    """
    import os

    for a in alerts:
        print(f"\n[ALERT] {a.message}")

    if not alerts:
        return

    combined = "\n\n".join(a.message for a in alerts)

    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    tg_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if tg_token and tg_chat_id:
        if send_telegram_alert(combined, tg_token, tg_chat_id):
            log.info("Alerts dispatched via Telegram (%d alert(s))", len(alerts))
            return

    smtp_cfg = {
        "from_addr": os.getenv("ALERT_EMAIL_FROM", ""),
        "to_addr": os.getenv("ALERT_EMAIL_TO", ""),
        "host": os.getenv("ALERT_SMTP_HOST", ""),
        "port": os.getenv("ALERT_SMTP_PORT", "587"),
        "password": os.getenv("ALERT_SMTP_PASSWORD", ""),
    }
    if smtp_cfg["from_addr"] and smtp_cfg["to_addr"] and smtp_cfg["host"]:
        if send_email_alert(combined, smtp_cfg):
            log.info("Alerts dispatched via email (%d alert(s))", len(alerts))
            return

    log.info("No notification channel configured — alerts printed to stdout only")
