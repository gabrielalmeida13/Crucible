"""Portfolio position tracker for the Crucible screener.

Reads open positions from data/portfolio.csv, evaluates each against the
current fundamental snapshot using its track's filter and scorer, and
produces hold/review/exit recommendations.

CSV schema  (data/portfolio.csv)
  ticker      — stock ticker as used in EDGAR / yfinance (e.g. AAPL)
  entry_price — price paid per share, in account currency
  entry_date  — ISO date of purchase (e.g. 2025-03-31)
  track       — integer 1, 2, or 3 (the Crucible track that selected this pick)
  shares      — number of shares held
  entry_pfcf  — P/FCF at purchase date (optional; copy from monthly picks output)
  entry_ps    — P/S  at purchase date (optional; copy from monthly picks output)

entry_pfcf and entry_ps are used for the REINFORCE valuation check.  If omitted,
evaluate_portfolio attempts a lookup in fund_by_date at entry_date; if that also
fails the valuation condition is skipped and REINFORCE will not trigger.

Note on Track 3 in live monthly runs
  The p_fcf_vs_history filter requires 12–60 months of prior P/FCF data.
  In a live monthly run (single-date snapshot) that history is not attached,
  so attach_p_fcf_history has not been called.  The filter degrades gracefully
  to a no-op in that case (see track3_value.filter_p_fcf_vs_history).
  Positions on Track 3 will be evaluated against the remaining filters only.
"""
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from crucible.config import CrucibleConfig
from crucible.tracks import track1_quality, track2_growth, track3_value

log = logging.getLogger(__name__)

PORTFOLIO_CSV = Path(__file__).resolve().parent.parent / "data" / "portfolio.csv"

# Rank threshold: positions ranked 1–TOP_N in their track get HOLD; beyond that REVIEW.
_TOP_N = 20

_CSV_DTYPES: dict[str, type] = {
    "ticker":      str,
    "entry_price": float,
    "track":       int,
    "shares":      float,
    "entry_pfcf":  float,   # optional — P/FCF at entry date
    "entry_ps":    float,   # optional — P/S  at entry date
}


class PositionRecommendation(Enum):
    REINFORCE    = "REINFORCE"    # passes filters; rank ≤ 5; current P/FCF or P/S ≥ 10% cheaper
                                  # than at entry; AND market_value < 3× equal-weight allocation —
                                  # all three required; consider adding to the position
    HOLD         = "HOLD"         # passes track filters; composite score ranks in top-N this month
    REVIEW       = "REVIEW"       # passes filters but score slipped below top-N, OR fails filters
                                  # without negative momentum confirmation — watch, do not act
    EXIT_SIGNAL  = "EXIT_SIGNAL"  # fails track filters AND 3-month momentum is negative:
                                  # both conditions required; consider selling
    DATA_MISSING = "DATA_MISSING" # ticker absent from snapshot or insufficient EDGAR data
                                  # this month — hold without action, not an exit signal


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_portfolio(csv_path: Path = PORTFOLIO_CSV) -> pd.DataFrame:
    """Load positions from CSV.

    Returns an empty DataFrame (with correct columns) if the file does not
    exist, is empty, or contains only the header row.  Optional columns
    entry_pfcf and entry_ps are filled with NaN when absent from the file.
    """
    _empty = pd.DataFrame(
        columns=["ticker", "entry_price", "entry_date", "track", "shares",
                 "entry_pfcf", "entry_ps"]
    )
    if not csv_path.exists():
        return _empty
    # Only coerce columns that are actually present; avoid KeyError on optional ones.
    present_dtypes = {k: v for k, v in _CSV_DTYPES.items()}
    df = pd.read_csv(csv_path, dtype=present_dtypes, parse_dates=["entry_date"])
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df = df.dropna(subset=["ticker", "entry_price", "track"])
    if df.empty:
        return _empty
    # Ensure optional valuation columns exist even if not in the file
    for col in ("entry_pfcf", "entry_ps"):
        if col not in df.columns:
            df[col] = float("nan")
    log.info("Loaded %d position(s) from %s", len(df), csv_path)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _score_universe_by_track(
    snapshot: pd.DataFrame,
    config: CrucibleConfig,
) -> dict[int, pd.DataFrame]:
    """Run filter + score for all three tracks on the full snapshot.

    Returns {track_num: scored_df} where scored_df is sorted by
    composite_score descending and contains only tickers that passed
    the track's Layer 1 filters.
    """
    results: dict[int, pd.DataFrame] = {}

    try:
        t1 = track1_quality.apply_filters(snapshot, config.filters)
        if not t1.empty:
            results[1] = track1_quality.score(t1, config)
    except Exception:
        log.warning("Track 1 scoring failed during portfolio evaluation", exc_info=True)

    try:
        t2 = track2_growth.apply_filters(snapshot, config.track2_filters)
        if not t2.empty:
            mom_mask = t2["momentum_raw"].notna() & (t2["momentum_raw"] > 0)
            t2 = t2[mom_mask]
        if not t2.empty:
            results[2] = track2_growth.score(t2, config, config.track2_score_weights)
    except Exception:
        log.warning("Track 2 scoring failed during portfolio evaluation", exc_info=True)

    try:
        t3 = track3_value.apply_filters(snapshot, config.track3_filters)
        if not t3.empty:
            results[3] = track3_value.score(t3, config, config.track3_score_weights)
    except Exception:
        log.warning("Track 3 scoring failed during portfolio evaluation", exc_info=True)

    return results


def _latest_price(ticker: str, prices: pd.DataFrame) -> float | None:
    """Return the most recent non-null month-end close for ticker."""
    if ticker not in prices.columns:
        return None
    series = prices[ticker].dropna()
    return float(series.iloc[-1]) if not series.empty else None


def _entry_valuations(
    ticker: str,
    pos: pd.Series,
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
) -> tuple[float, float]:
    """Return (entry_pfcf, entry_ps) for a position row.

    Lookup priority:
      1. Latest fund_by_date snapshot whose date ≤ entry_date (backtest mode).
      2. entry_pfcf / entry_ps columns in the CSV row (live mode fallback).
    Both values are NaN when unavailable.
    """
    pfcf: float = float("nan")
    ps:   float = float("nan")

    entry_date = pos.get("entry_date", pd.NaT)
    if not pd.isna(entry_date) and fund_by_date:
        try:
            ed = pd.Timestamp(entry_date)
            if ed.tzinfo is None:
                ed = ed.tz_localize("UTC")
            prior = [d for d in fund_by_date if d <= ed]
            if prior:
                snap = fund_by_date[max(prior)]
                if ticker in snap.index:
                    if "p_fcf" in snap.columns:
                        pfcf = float(snap.at[ticker, "p_fcf"])
                    if "p_s" in snap.columns:
                        ps = float(snap.at[ticker, "p_s"])
        except Exception:
            pass  # fall through to CSV columns

    if np.isnan(pfcf):
        v = pos.get("entry_pfcf", float("nan"))
        if not pd.isna(v):
            pfcf = float(v)
    if np.isnan(ps):
        v = pos.get("entry_ps", float("nan"))
        if not pd.isna(v):
            ps = float(v)

    return pfcf, ps


def _check_reinforce(
    ticker: str,
    rank: float,
    snapshot: pd.DataFrame,
    pos: pd.Series,
    market_val: float,
    monthly_budget: float,
    n_positions: int,
    latest_date: pd.Timestamp,
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
) -> bool:
    """Return True when all three REINFORCE conditions are satisfied.

    1. rank_in_track ≤ 5
    2. Current P/FCF or P/S is ≥ 10% below the entry-date valuation
       (at least one of the two multiples must satisfy this; skipped if
       neither entry valuation is available)
    3. market_value < 3 × equal_weight_allocation
       where equal_weight = monthly_budget × months_invested / n_positions
    """
    # Condition 1
    if np.isnan(rank) or rank > 5:
        return False

    # Condition 2 — valuation cheaper than at entry
    entry_pfcf, entry_ps = _entry_valuations(ticker, pos, fund_by_date)
    curr_pfcf = (
        float(snapshot.at[ticker, "p_fcf"])
        if "p_fcf" in snapshot.columns and ticker in snapshot.index
        else float("nan")
    )
    curr_ps = (
        float(snapshot.at[ticker, "p_s"])
        if "p_s" in snapshot.columns and ticker in snapshot.index
        else float("nan")
    )

    pfcf_cheaper = (
        not np.isnan(entry_pfcf) and not np.isnan(curr_pfcf)
        and entry_pfcf > 0 and curr_pfcf <= entry_pfcf * 0.90
    )
    ps_cheaper = (
        not np.isnan(entry_ps) and not np.isnan(curr_ps)
        and entry_ps > 0 and curr_ps <= entry_ps * 0.90
    )
    if not pfcf_cheaper and not ps_cheaper:
        # Valuation data absent or neither multiple has dropped ≥ 10%
        return False

    # Condition 3 — position is not already 3× overweight vs equal allocation
    if np.isnan(market_val) or monthly_budget <= 0 or n_positions <= 0:
        return False

    entry_date = pos.get("entry_date", pd.NaT)
    if pd.isna(entry_date):
        return False

    try:
        ed = pd.Timestamp(entry_date)
        if ed.tzinfo is None:
            ed = ed.tz_localize("UTC")
        months_invested = max(1, round((latest_date - ed).days / 30.44))
    except Exception:
        return False

    equal_weight = monthly_budget * months_invested / n_positions
    return float(market_val) < 3.0 * equal_weight


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_portfolio(
    positions: pd.DataFrame,
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    config: CrucibleConfig,
    monthly_budget: float = 0.0,
) -> pd.DataFrame:
    """Evaluate each open position against the latest snapshot.

    Re-runs the track-specific filter + scorer on the full universe snapshot
    and determines whether each position warrants holding.

    Recommendation logic (evaluated in this order)
      REINFORCE    — passes filters; rank ≤ 5; current P/FCF or P/S ≥ 10% below
                     entry-date valuation; market_value < 3× equal-weight allocation
                     (all three required; monthly_budget must be > 0)
      HOLD         — passes filters AND rank ≤ TOP_N
      REVIEW       — passes filters but rank > TOP_N, OR fails filters without
                     negative 3m momentum confirmation
      EXIT_SIGNAL  — fails filters AND momentum_3m < 0 (both conditions required)
      DATA_MISSING — ticker absent from snapshot or insufficient_data=True;
                     hold without action

    monthly_budget is required for the REINFORCE equal-weight calculation.
    Pass 0.0 (default) to disable REINFORCE evaluation.

    Returns a DataFrame indexed by ticker with columns:
      track, shares, entry_price, entry_date, cost_basis,
      current_price, market_value, return_pct, gain_loss,
      passes_filters, current_score, rank_in_track, recommendation
    """
    if positions.empty:
        return pd.DataFrame()

    latest_date     = max(fund_by_date.keys())
    snapshot        = fund_by_date[latest_date]
    scored_by_track = _score_universe_by_track(snapshot, config)

    n_positions = len(positions)
    log.info(
        "evaluate_portfolio: snapshot %s, %d positions, tracks present: %s, budget=%.0f",
        latest_date.date(),
        n_positions,
        sorted(scored_by_track.keys()),
        monthly_budget,
    )

    rows: list[dict] = []

    for _, pos in positions.iterrows():
        ticker      = str(pos["ticker"])
        entry_price = float(pos["entry_price"])
        track_num   = int(pos["track"])
        shares      = float(pos["shares"])
        entry_date  = pos.get("entry_date", pd.NaT)

        curr_price = _latest_price(ticker, prices)
        cost_basis = entry_price * shares
        market_val = curr_price * shares if curr_price is not None else float("nan")
        ret_pct    = (curr_price / entry_price - 1.0) if curr_price is not None else float("nan")
        gain_loss  = market_val - cost_basis if curr_price is not None else float("nan")

        scored = scored_by_track.get(track_num)

        if ticker not in snapshot.index:
            # Truly absent this month (delisted, dropped from universe, or ticker mismatch)
            passes_filters = False
            current_score  = float("nan")
            rank_in_track  = float("nan")
            rec            = PositionRecommendation.DATA_MISSING

        elif bool(snapshot.at[ticker, "insufficient_data"]) if "insufficient_data" in snapshot.columns else False:
            # Present in snapshot but EDGAR data too sparse to evaluate fundamentals
            passes_filters = False
            current_score  = float("nan")
            rank_in_track  = float("nan")
            rec            = PositionRecommendation.DATA_MISSING

        elif scored is None or ticker not in scored.index:
            # Ticker has data but did not survive the track's Layer 1 filters.
            # Use 3-month momentum to distinguish a confirmed exit from a borderline case.
            passes_filters = False
            current_score  = float("nan")
            rank_in_track  = float("nan")
            mom_3m = (
                snapshot.at[ticker, "momentum_3m"]
                if "momentum_3m" in snapshot.columns
                else float("nan")
            )
            if pd.notna(mom_3m) and float(mom_3m) < 0:
                rec = PositionRecommendation.EXIT_SIGNAL
            else:
                # Fundamentals flagged but market not yet confirming — watch, don't act
                rec = PositionRecommendation.REVIEW

        else:
            passes_filters = True
            current_score  = float(scored.at[ticker, "composite_score"])
            # scored is sorted descending; position in index = 0-based rank
            rank_in_track  = float(scored.index.get_loc(ticker) + 1)
            if rank_in_track <= _TOP_N:
                if _check_reinforce(
                    ticker, rank_in_track, snapshot, pos,
                    market_val, monthly_budget, n_positions,
                    latest_date, fund_by_date,
                ):
                    rec = PositionRecommendation.REINFORCE
                else:
                    rec = PositionRecommendation.HOLD
            else:
                rec = PositionRecommendation.REVIEW

        rows.append({
            "ticker":         ticker,
            "track":          track_num,
            "shares":         shares,
            "entry_price":    entry_price,
            "entry_date":     entry_date,
            "cost_basis":     round(cost_basis, 2),
            "current_price":  round(curr_price, 4) if curr_price is not None else float("nan"),
            "market_value":   round(market_val, 2) if not np.isnan(market_val) else float("nan"),
            "return_pct":     ret_pct,
            "gain_loss":      round(gain_loss, 2) if not np.isnan(gain_loss) else float("nan"),
            "passes_filters": passes_filters,
            "current_score":  current_score,
            "rank_in_track":  rank_in_track,
            "recommendation": rec.value,
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).set_index("ticker")
    log.info(
        "evaluate_portfolio: REINFORCE=%d  HOLD=%d  REVIEW=%d  EXIT_SIGNAL=%d  DATA_MISSING=%d",
        (result["recommendation"] == "REINFORCE").sum(),
        (result["recommendation"] == "HOLD").sum(),
        (result["recommendation"] == "REVIEW").sum(),
        (result["recommendation"] == "EXIT_SIGNAL").sum(),
        (result["recommendation"] == "DATA_MISSING").sum(),
    )
    return result


def allocation_advice(
    portfolio: pd.DataFrame,
    monthly_budget: float,
    shortlists: dict[int, pd.DataFrame],
) -> str:
    """Return a Markdown string advising how to allocate the monthly budget.

    portfolio      — output of evaluate_portfolio (may be empty)
    monthly_budget — cash available for new purchases (account currency)
    shortlists     — {track_num: scored_df} for this month's top-N candidates

    Logic
      1. Flag REINFORCE positions — top-5, cheaper than entry, room to add capital.
      2. Flag EXIT_SIGNAL positions (failed filters + negative 3m momentum) — act first.
      3. Flag REVIEW positions that may be worth replacing.
      4. Show DATA_MISSING positions as informational — do not treat as exit signals.
      5. From the shortlists, suggest up to 3 new candidates not currently held,
         and split the budget equally across them.
    """
    lines: list[str] = ["## Allocation Advice", ""]

    if portfolio.empty and not shortlists:
        lines.append("*No open positions and no shortlists available.*")
        return "\n".join(lines)

    reinforces   = portfolio[portfolio["recommendation"] == "REINFORCE"]    if not portfolio.empty else pd.DataFrame()
    exit_sigs    = portfolio[portfolio["recommendation"] == "EXIT_SIGNAL"]  if not portfolio.empty else pd.DataFrame()
    reviews      = portfolio[portfolio["recommendation"] == "REVIEW"]       if not portfolio.empty else pd.DataFrame()
    holds        = portfolio[portfolio["recommendation"] == "HOLD"]         if not portfolio.empty else pd.DataFrame()
    data_missing = portfolio[portfolio["recommendation"] == "DATA_MISSING"] if not portfolio.empty else pd.DataFrame()
    held         = set(portfolio.index) if not portfolio.empty else set()

    # --- Current positions ---------------------------------------------------

    if not reinforces.empty:
        lines += ["### 🔼 Reinforce — top-5 rank, cheaper than entry, room to add", ""]
        for tkr, row in reinforces.iterrows():
            ret_str  = f"{row['return_pct']:+.1%}" if not np.isnan(row["return_pct"]) else "—"
            rank_str = f"rank {int(row['rank_in_track'])}"
            mv_str   = f"{row['market_value']:,.0f}" if not np.isnan(row["market_value"]) else "—"
            lines.append(
                f"- **{tkr}** (T{int(row['track'])}) &nbsp; "
                f"return {ret_str} &nbsp; {rank_str} &nbsp; value {mv_str} — "
                "valuation has dipped ≥ 10% below entry and position is underweight. "
                "Consider adding to this position before deploying into a new name."
            )
        lines.append("")

    if not holds.empty:
        lines += ["### ✓ Hold", ""]
        for tkr, row in holds.iterrows():
            ret_str  = f"{row['return_pct']:+.1%}" if not np.isnan(row["return_pct"]) else "—"
            rank_str = f"rank {int(row['rank_in_track'])}" if not np.isnan(row["rank_in_track"]) else "—"
            mv_str   = f"{row['market_value']:,.0f}" if not np.isnan(row["market_value"]) else "—"
            lines.append(
                f"- **{tkr}** (T{int(row['track'])}) &nbsp; "
                f"return {ret_str} &nbsp; {rank_str} this month &nbsp; value {mv_str}"
            )
        lines.append("")

    if not reviews.empty:
        lines += ["### ↘ Review — monitor, no action required yet", ""]
        for tkr, row in reviews.iterrows():
            ret_str  = f"{row['return_pct']:+.1%}" if not np.isnan(row["return_pct"]) else "—"
            rank_str = f"rank {int(row['rank_in_track'])}" if not np.isnan(row["rank_in_track"]) else "—"
            filter_note = (
                "still passes filters" if row["passes_filters"]
                else "failed filters but 3m momentum not negative"
            )
            lines.append(
                f"- **{tkr}** (T{int(row['track'])}) &nbsp; "
                f"return {ret_str} &nbsp; {rank_str} — {filter_note}; "
                "hold unless a clearly better candidate is available."
            )
        lines.append("")

    if not exit_sigs.empty:
        lines += ["### ⚠ Exit signal — failed filters AND negative 3m momentum", ""]
        for tkr, row in exit_sigs.iterrows():
            ret_str = f"{row['return_pct']:+.1%}" if not np.isnan(row["return_pct"]) else "—"
            gl_str  = f"{row['gain_loss']:+,.0f}" if not np.isnan(row["gain_loss"]) else "—"
            lines.append(
                f"- **{tkr}** (T{int(row['track'])}) &nbsp; "
                f"return {ret_str} (P&L {gl_str}) — "
                "fundamentals deteriorated and price momentum confirming. Consider selling."
            )
        lines.append("")

    if not data_missing.empty:
        lines += ["### ℹ No data this month — hold without action", ""]
        for tkr, row in data_missing.iterrows():
            ret_str = f"{row['return_pct']:+.1%}" if not np.isnan(row["return_pct"]) else "—"
            lines.append(
                f"- **{tkr}** (T{int(row['track'])}) &nbsp; "
                f"return {ret_str} — not in snapshot or insufficient EDGAR data this month. "
                "Not an exit signal; re-evaluate next month."
            )
        lines.append("")

    # --- New allocation ------------------------------------------------------

    lines.append(f"### New allocation  (budget: {monthly_budget:,.0f})")
    lines.append("")

    if not shortlists:
        lines.append("*No shortlist provided for this month — run the screener first.*")
        return "\n".join(lines)

    # Collect candidates from shortlists, excluding already-held tickers.
    # Preserve per-track ordering (shortlist is already sorted by composite_score).
    candidates: list[tuple[int, str, float]] = []
    for track_num in sorted(shortlists):
        scored_df = shortlists[track_num]
        for tkr in scored_df.index:
            if tkr in held:
                continue
            score = (
                float(scored_df.at[tkr, "composite_score"])
                if "composite_score" in scored_df.columns
                else float("nan")
            )
            candidates.append((track_num, tkr, score))
            if len([c for c in candidates if c[0] == track_num]) >= 5:
                break  # cap contribution per track

    if not candidates:
        lines.append("*All shortlist candidates are already held — no new positions suggested.*")
        return "\n".join(lines)

    # Top 3 across tracks, sorted by score descending
    candidates.sort(key=lambda x: x[2] if not np.isnan(x[2]) else -1.0, reverse=True)
    top3       = candidates[:3]
    n_new      = len(top3)
    alloc_each = monthly_budget / n_new if n_new > 0 else 0.0

    lines.append(f"Top {n_new} candidate(s) not currently held:")
    lines.append("")
    for track_num, tkr, score in top3:
        score_str = f"{score:.3f}" if not np.isnan(score) else "—"
        lines.append(
            f"- **{tkr}** (Track {track_num}, score {score_str}) — "
            f"suggested allocation ~{alloc_each:,.0f}"
        )

    lines += [
        "",
        f"*Equal-weight split of {monthly_budget:,.0f} across {n_new} position(s). "
        "Final sizing is your call — adjust for conviction and existing concentration.*",
    ]

    return "\n".join(lines)
