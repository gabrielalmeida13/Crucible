"""Phase 5.1 — Options strategy analysis for XTB American-style options.

Context
-------
XTB Portugal offers buy-only American-style options on ~110 US stocks.
Max expiry ~231 days. Standard contracts of 100 shares. Useful for:
  1. Protective puts  — lock in gains on large winners without selling
  2. Leveraged calls  — gain 100-share exposure for less capital than buying outright
  3. Index puts       — hedge portfolio direction via SPY/QQQ

Important constraint: with a €100/month budget, a single ATM contract on most
SP500 stocks costs €800–2000+. The practical floor for using options is
~€500 accumulated capital.

Public API
----------
check_iv_rank(ticker, lookback_days=252) -> IVRankResult
suggest_options_strategy(ticker, action, current_price, budget_eur,
                         expiry_days=180) -> OptionsAdvice
"""
from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTRACT_SIZE        = 100          # shares per option contract
MAX_EXPIRY_DAYS      = 231          # XTB Portugal limit
IV_RANK_WARN         = 60.0         # flag if approximate IV rank > this
LIQUIDITY_SPREAD_MAX = 0.25         # flag if (ask-bid)/ask > 25%
PAYOFF_MOVES         = [-0.25, -0.10, 0.00, +0.10, +0.25, +0.50]

_FALLBACK_EUR_USD    = 1.08         # used if FX fetch fails


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class IVRankResult:
    """Current implied volatility context for a ticker."""

    ticker: str
    current_iv: float        # ATM option IV (annualized, decimal, e.g. 0.45 = 45%)
    iv_rank: float           # percentile 0–100 vs 1Y rolling 30d realised vol
    hv_30d: float            # current 30-day realised vol (annualized)
    hv_252d: float           # current 252-day realised vol (annualized)
    warning: bool            # iv_rank > IV_RANK_WARN
    expiry_used: str         # which option expiry was used for current_iv
    note: str                # methodology explanation for the user


@dataclass
class PayoffRow:
    """Single scenario in the payoff comparison table."""

    move_label: str          # e.g. "+25%"
    move_pct: float          # e.g. 0.25
    stock_price_usd: float
    option_pnl_eur: float    # total P&L on N contracts at expiry (EUR)
    option_return_pct: float # option_pnl_eur / cost_total_eur × 100
    shares_pnl_eur: float    # same EUR invested in shares → P&L
    shares_return_pct: float # move_pct × 100 (always equal to the move)


@dataclass
class OptionsAdvice:
    """Full options analysis for a ticker/action/budget combination."""

    ticker: str
    action: str              # "new_position" | "protect_gains" | "hedge_portfolio"
    current_price_usd: float
    eur_usd_rate: float      # EUR/USD (e.g. 1.08 means 1 EUR = $1.08)
    current_price_eur: float

    # Contract details
    expiry_date: str
    days_to_expiry: int
    option_type: str         # "call" | "put"
    strike: float
    bid_usd: float
    ask_usd: float
    premium_usd: float       # mid-price per share
    premium_eur: float       # per share, converted
    iv_at_strike: float      # option's own implied volatility

    # Position sizing
    contracts_affordable: int
    cost_per_contract_eur: float   # = CONTRACT_SIZE × premium_eur
    cost_total_eur: float          # = contracts_affordable × cost_per_contract_eur

    # Risk metrics
    breakeven_usd: float
    max_loss_eur: float      # worst case (expires worthless) = cost_total_eur

    # IV context
    iv_rank_result: IVRankResult

    # Payoff table
    payoff_table: list[PayoffRow] = field(default_factory=list)

    # Recommendation
    recommendation: str = ""
    rationale: list[str] = field(default_factory=list)
    warning: str | None = None
    liquidity_warning: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_eur_usd_rate() -> float:
    """Fetch current EUR/USD rate from yfinance; fall back to hardcoded value."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rate = float(yf.Ticker("EURUSD=X").fast_info.last_price)
        if 0.8 < rate < 1.6:
            return rate
    except Exception:
        pass
    log.debug("FX fetch failed — using fallback EUR/USD=%.2f", _FALLBACK_EUR_USD)
    return _FALLBACK_EUR_USD


def _to_eur(usd: float, eur_usd: float) -> float:
    """Convert USD → EUR given EUR/USD rate (e.g. 1.08 → $1.08 per €1)."""
    return usd / eur_usd if eur_usd > 0 else usd


def _annualised_hv(returns: pd.Series, window: int) -> pd.Series:
    """Rolling annualised historical volatility (std of log returns × √252)."""
    log_ret = np.log(1 + returns)
    return log_ret.rolling(window).std() * math.sqrt(252)


def _nearest_expiry(
    expirations: tuple[str, ...],
    target_days: int,
    max_days: int = MAX_EXPIRY_DAYS,
) -> str | None:
    """Return the expiry string closest to target_days and within max_days."""
    now = datetime.now()
    valid: list[tuple[int, str]] = []
    for exp in expirations:
        try:
            days = (datetime.strptime(exp, "%Y-%m-%d") - now).days
        except ValueError:
            continue
        if 5 <= days <= max_days:
            valid.append((abs(days - target_days), exp))
    if not valid:
        return None
    return min(valid, key=lambda x: x[0])[1]


def _mid_premium(bid: float, ask: float) -> float:
    """Compute mid-price; fall back to ask if bid is zero (no market)."""
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return ask if ask > 0 else 0.0


def _select_call(
    calls: pd.DataFrame,
    current_price: float,
) -> pd.Series | None:
    """Select the ATM or nearest ITM call with bid > 0 and meaningful OI.

    Prefers the call whose strike is closest to (and ≤) current_price.
    Falls back to nearest OTM if no ITM with liquidity.
    """
    liquid = calls[(calls["bid"] > 0) & (calls["openInterest"] > 0)].copy()
    if liquid.empty:
        liquid = calls[calls["ask"] > 0].copy()
    if liquid.empty:
        return None

    # Prefer slightly ITM (strike ≤ current_price) for delta ~0.6–0.8
    itm = liquid[liquid["strike"] <= current_price]
    if not itm.empty:
        return itm.iloc[(itm["strike"] - current_price).abs().argsort()[:1]].iloc[0]

    # Fallback: nearest OTM
    return liquid.iloc[(liquid["strike"] - current_price).abs().argsort()[:1]].iloc[0]


def _select_put(
    puts: pd.DataFrame,
    current_price: float,
    otm_pct: float = 0.05,
) -> pd.Series | None:
    """Select a put near (1 - otm_pct) × current_price with bid > 0.

    Default 5% OTM: provides protection below 95% of current price at
    lower premium than ATM while still covering most large adverse moves.
    """
    liquid = puts[(puts["bid"] > 0) & (puts["openInterest"] > 0)].copy()
    if liquid.empty:
        liquid = puts[puts["ask"] > 0].copy()
    if liquid.empty:
        return None

    target_strike = current_price * (1 - otm_pct)
    return liquid.iloc[(liquid["strike"] - target_strike).abs().argsort()[:1]].iloc[0]


def _build_payoff_table(
    option_type: str,
    strike: float,
    premium_usd: float,
    current_price_usd: float,
    contracts: int,
    eur_usd: float,
) -> list[PayoffRow]:
    """Build payoff comparison: option P&L vs equivalent investment in shares."""
    if contracts <= 0:
        return []

    total_cost_usd = contracts * CONTRACT_SIZE * premium_usd
    total_cost_eur = _to_eur(total_cost_usd, eur_usd)
    shares_buyable  = total_cost_usd / current_price_usd if current_price_usd > 0 else 0

    rows: list[PayoffRow] = []
    for move in PAYOFF_MOVES:
        new_price = current_price_usd * (1 + move)

        if option_type == "call":
            intrinsic = max(0.0, new_price - strike)
        else:  # put
            intrinsic = max(0.0, strike - new_price)

        option_pnl_usd   = (intrinsic - premium_usd) * CONTRACT_SIZE * contracts
        option_pnl_eur   = _to_eur(option_pnl_usd, eur_usd)
        option_ret_pct   = (option_pnl_eur / total_cost_eur * 100) if total_cost_eur > 0 else 0.0

        shares_pnl_usd   = shares_buyable * (new_price - current_price_usd)
        shares_pnl_eur   = _to_eur(shares_pnl_usd, eur_usd)
        shares_ret_pct   = move * 100

        label = f"{move:+.0%}".replace("+0%", " 0%")
        rows.append(PayoffRow(
            move_label=label,
            move_pct=move,
            stock_price_usd=round(new_price, 2),
            option_pnl_eur=round(option_pnl_eur, 0),
            option_return_pct=round(option_ret_pct, 1),
            shares_pnl_eur=round(shares_pnl_eur, 0),
            shares_return_pct=round(shares_ret_pct, 1),
        ))
    return rows


def _build_recommendation(
    action: str,
    ticker: str,
    option_type: str,
    strike: float,
    contracts: int,
    cost_total_eur: float,
    breakeven_usd: float,
    current_price_usd: float,
    iv_rank_result: IVRankResult,
    budget_eur: float,
) -> tuple[str, list[str], str | None]:
    """Return (recommendation, rationale_bullets, warning_or_None)."""
    warning = None
    rationale: list[str] = []

    # IV warning
    if iv_rank_result.warning:
        warning = (
            f"IV is elevated — current option IV ({iv_rank_result.current_iv:.0%}) "
            f"is in the {iv_rank_result.iv_rank:.0f}th percentile vs 1-year "
            f"historical realised vol. You are paying above-average for optionality. "
            f"Consider buying shares instead and waiting for IV to normalise."
        )

    # Budget insufficient for even 1 contract
    if contracts == 0:
        min_cost = cost_total_eur  # already computed as 0-contract cost placeholder
        # Actually when contracts=0 cost_total_eur=0; compute 1-contract cost
        single_cost = cost_total_eur  # will be overridden by caller
        recommendation = f"Insufficient budget for 1 contract."
        rationale = [
            f"Minimum required: 1 contract = 100 shares × premium.",
            f"Available budget: €{budget_eur:,.0f}.",
            f"Accumulate more capital or consider a lower-priced ticker.",
        ]
        return recommendation, rationale, warning

    if action == "new_position":
        breakeven_pct = (breakeven_usd / current_price_usd - 1) * 100
        recommendation = (
            f"Buy {contracts} {ticker} {option_type.upper()} contract(s) at "
            f"strike ${strike:.0f} — total cost €{cost_total_eur:,.0f}."
        )
        rationale = [
            f"Gains exposure equivalent to {contracts * CONTRACT_SIZE} shares "
            f"for €{cost_total_eur:,.0f} vs €{current_price_usd * contracts * CONTRACT_SIZE / iv_rank_result.current_iv * iv_rank_result.current_iv:,.0f} to buy shares outright.",
            f"Breakeven at ${breakeven_usd:.2f} ({breakeven_pct:+.1f}% from current ${current_price_usd:.2f}).",
            f"Max loss: €{cost_total_eur:,.0f} (premium paid) if {ticker} is below "
            f"${strike:.0f} at expiry.",
            f"Max gain: unlimited if {ticker} rises strongly above ${breakeven_usd:.2f}.",
        ]
        if iv_rank_result.warning:
            rationale.insert(0, f"⚠ High IV: consider shares instead (see warning above).")

    elif action == "protect_gains":
        protection_pct = (1 - strike / current_price_usd) * 100
        recommendation = (
            f"Buy {contracts} {ticker} PUT contract(s) at strike ${strike:.0f} — "
            f"cost €{cost_total_eur:,.0f} (insurance)."
        )
        rationale = [
            f"Protects {contracts * CONTRACT_SIZE} shares against falls below "
            f"${strike:.0f} ({protection_pct:.1f}% OTM from ${current_price_usd:.2f}).",
            f"Breakeven: stock must fall below ${breakeven_usd:.2f} for the put to "
            f"profit (compensates insurance cost).",
            f"Upside fully preserved — gains above ${current_price_usd:.2f} are unaffected.",
            f"Consider this insurance: useful if position is large relative to portfolio.",
        ]

    else:  # hedge_portfolio
        protection_pct = (1 - strike / current_price_usd) * 100
        recommendation = (
            f"Buy {contracts} {ticker} PUT contract(s) at strike ${strike:.0f} — "
            f"cost €{cost_total_eur:,.0f} (portfolio hedge)."
        )
        rationale = [
            f"Provides directional hedge: profits if {ticker} falls below "
            f"${breakeven_usd:.2f} at expiry.",
            f"Current protection level: {protection_pct:.1f}% below market.",
            f"Works as a portfolio hedge if {ticker} is SPY/QQQ (index ETF).",
            f"Alternative: use a larger put budget for deeper protection or more contracts.",
        ]

    return recommendation, rationale, warning


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_iv_rank(
    ticker: str,
    lookback_days: int = 252,
) -> IVRankResult:
    """Compute approximate IV rank for ticker.

    Methodology (yfinance limitation: no historical IV data available)
    -------------------------------------------------------------------
    1. Fetch current ATM option IV from the nearest-expiry option chain.
    2. Compute rolling 30-day annualised historical volatility (HV) over
       the past `lookback_days` trading days of daily price returns.
    3. IV rank = percentile of current_iv vs the 1Y HV distribution.
       Higher rank = options more expensive relative to historical norms.

    This is an approximation: true IV rank compares current IV to
    historical IV values, not realised vol. However, IV and HV are
    correlated, making this a useful directional indicator.

    Raises ValueError if no option chain or price history is available.
    """
    t = yf.Ticker(ticker)

    # 1. Current ATM option IV
    expirations = t.options
    if not expirations:
        raise ValueError(f"{ticker}: no options available (not on XTB options list?)")

    # Use the nearest expiry that's at least 14 days out for meaningful IV
    expiry_used = _nearest_expiry(expirations, target_days=30, max_days=90) or expirations[0]
    try:
        chain = t.option_chain(expiry_used)
    except Exception as exc:
        raise ValueError(f"{ticker}: could not fetch option chain: {exc}") from exc

    # Find ATM call IV (most representative of overall vol level)
    try:
        current_price = float(t.fast_info.last_price)
    except Exception:
        current_price = float(chain.calls["strike"].median())

    atm_calls = chain.calls[
        (chain.calls["impliedVolatility"] > 0.01)  # filter out invalid IV
        & (chain.calls["strike"] >= current_price * 0.90)
        & (chain.calls["strike"] <= current_price * 1.10)
    ]
    if atm_calls.empty:
        atm_calls = chain.calls[chain.calls["impliedVolatility"] > 0.01]
    if atm_calls.empty:
        raise ValueError(f"{ticker}: no valid ATM IV found in option chain")

    # Use the call closest to ATM
    atm_row = atm_calls.iloc[
        (atm_calls["strike"] - current_price).abs().argsort()[:1]
    ].iloc[0]
    current_iv = float(atm_row["impliedVolatility"])

    # 2. Historical realised volatility (rolling 30d, annualised)
    hist = t.history(period=f"{lookback_days + 60}d")
    if hist.empty or len(hist) < 30:
        raise ValueError(f"{ticker}: insufficient price history for IV rank computation")

    returns = hist["Close"].pct_change().dropna()
    hv30_series = _annualised_hv(returns, 30).dropna()
    hv252_series = _annualised_hv(returns, min(252, len(returns))).dropna()

    if hv30_series.empty:
        raise ValueError(f"{ticker}: could not compute rolling HV")

    hv_30d   = float(hv30_series.iloc[-1])
    hv_252d  = float(hv252_series.iloc[-1]) if not hv252_series.empty else hv_30d

    # 3. IV rank: percentile of current_iv vs the HV distribution
    hv_vals = hv30_series.values
    iv_rank = float((hv_vals < current_iv).mean() * 100)

    return IVRankResult(
        ticker=ticker,
        current_iv=current_iv,
        iv_rank=iv_rank,
        hv_30d=hv_30d,
        hv_252d=hv_252d,
        warning=iv_rank > IV_RANK_WARN,
        expiry_used=expiry_used,
        note=(
            "IV rank = percentile of current ATM IV vs 1-year rolling 30-day "
            "realised vol. Approximate: true IV rank requires historical IV data "
            "(unavailable from yfinance). Values > 60 suggest options are expensive."
        ),
    )


def suggest_options_strategy(
    ticker: str,
    action: Literal["new_position", "protect_gains", "hedge_portfolio"],
    current_price: float,
    budget_eur: float,
    expiry_days: int = 180,
) -> OptionsAdvice:
    """Analyse an options strategy and return a full OptionsAdvice.

    Parameters
    ----------
    ticker       : Stock ticker (e.g. "APH") or index ETF ("SPY", "QQQ")
    action       : "new_position"   — long call to gain leveraged upside
                   "protect_gains"  — protective put to hedge existing position
                   "hedge_portfolio"— index put to hedge portfolio direction
    current_price: Current stock price in USD
    budget_eur   : Available capital in EUR (used for position sizing)
    expiry_days  : Target days to expiry; capped at MAX_EXPIRY_DAYS (231)

    Returns OptionsAdvice with contract details, payoff table, and
    a recommendation that accounts for IV level and budget constraints.

    Note: all P&L figures assume the option is held to expiry (no early
    exercise modelled). American-style options may be exercised early
    but the payoff analysis shows intrinsic value at expiry as the
    primary reference.
    """
    expiry_days = min(expiry_days, MAX_EXPIRY_DAYS)
    eur_usd = _get_eur_usd_rate()
    current_price_eur = _to_eur(current_price, eur_usd)

    # 1. IV rank context
    try:
        iv_rank_result = check_iv_rank(ticker)
    except Exception as exc:
        log.info("check_iv_rank failed for %s: %s", ticker, exc)
        iv_rank_result = IVRankResult(
            ticker=ticker, current_iv=float("nan"), iv_rank=float("nan"),
            hv_30d=float("nan"), hv_252d=float("nan"), warning=False,
            expiry_used="—",
            note="IV rank unavailable (option chain or price history could not be fetched).",
        )

    # 2. Find target expiry
    t = yf.Ticker(ticker)
    expirations = t.options
    if not expirations:
        raise ValueError(
            f"{ticker} has no listed options. "
            "Check that it is available on XTB's options list."
        )

    expiry_date = _nearest_expiry(expirations, expiry_days, MAX_EXPIRY_DAYS)
    if expiry_date is None:
        # Fallback: use whatever is available within 231 days
        expiry_date = _nearest_expiry(expirations, expiry_days, 999)
    if expiry_date is None:
        raise ValueError(
            f"{ticker}: no expiry found within {MAX_EXPIRY_DAYS} days "
            f"(available: {expirations})"
        )

    days_to_expiry = (
        datetime.strptime(expiry_date, "%Y-%m-%d") - datetime.now()
    ).days

    # 3. Fetch option chain and select contract
    chain = t.option_chain(expiry_date)

    if action == "new_position":
        option_type = "call"
        selected = _select_call(chain.calls, current_price)
    else:
        option_type = "put"
        otm_pct = 0.05 if action == "protect_gains" else 0.05
        selected = _select_put(chain.puts, current_price, otm_pct=otm_pct)

    if selected is None:
        raise ValueError(
            f"{ticker}: no liquid {option_type} found for expiry {expiry_date}. "
            "Check that this expiry has open interest."
        )

    strike       = float(selected["strike"])
    bid_usd      = float(selected.get("bid", 0) or 0)
    ask_usd      = float(selected.get("ask", 0) or 0)
    premium_usd  = _mid_premium(bid_usd, ask_usd)
    iv_at_strike = float(selected.get("impliedVolatility", float("nan")) or float("nan"))
    premium_eur  = _to_eur(premium_usd, eur_usd)

    # Liquidity warning
    liquidity_warning: str | None = None
    if ask_usd > 0 and bid_usd > 0:
        spread_pct = (ask_usd - bid_usd) / ask_usd
        if spread_pct > LIQUIDITY_SPREAD_MAX:
            liquidity_warning = (
                f"Wide bid-ask spread: ${bid_usd:.2f}–${ask_usd:.2f} "
                f"({spread_pct:.0%}). Real execution cost may be higher than "
                f"the mid-price estimate of ${premium_usd:.2f}."
            )

    # 4. Position sizing
    cost_per_contract_eur = CONTRACT_SIZE * premium_eur
    if cost_per_contract_eur > 0:
        contracts_affordable = int(budget_eur // cost_per_contract_eur)
    else:
        contracts_affordable = 0
    cost_total_eur = contracts_affordable * cost_per_contract_eur

    # 5. Breakeven and max loss
    if option_type == "call":
        breakeven_usd = strike + premium_usd
    else:
        breakeven_usd = strike - premium_usd
    max_loss_eur = cost_total_eur  # full premium if expires worthless

    # 6. Payoff table
    payoff_table = _build_payoff_table(
        option_type, strike, premium_usd,
        current_price, contracts_affordable, eur_usd,
    )

    # 7. Recommendation
    recommendation, rationale, warning = _build_recommendation(
        action, ticker, option_type, strike,
        contracts_affordable, cost_total_eur,
        breakeven_usd, current_price, iv_rank_result, budget_eur,
    )

    # Fix rationale for 0-contract case (include actual min cost)
    if contracts_affordable == 0 and cost_per_contract_eur > 0:
        recommendation = (
            f"Insufficient budget for 1 contract "
            f"(minimum €{cost_per_contract_eur:,.0f} required)."
        )
        rationale = [
            f"1 contract = {CONTRACT_SIZE} shares × €{premium_eur:.2f}/share "
            f"= €{cost_per_contract_eur:,.0f}.",
            f"Your budget: €{budget_eur:,.0f}.",
            f"Accumulate at least €{math.ceil(cost_per_contract_eur / 100) * 100:,.0f} "
            f"before using this strategy.",
        ]
        # Recompute cost_total_eur for display (show 1-contract cost even if unaffordable)
        cost_total_eur = cost_per_contract_eur  # for reference

    return OptionsAdvice(
        ticker=ticker,
        action=action,
        current_price_usd=current_price,
        eur_usd_rate=eur_usd,
        current_price_eur=current_price_eur,
        expiry_date=expiry_date,
        days_to_expiry=days_to_expiry,
        option_type=option_type,
        strike=strike,
        bid_usd=bid_usd,
        ask_usd=ask_usd,
        premium_usd=premium_usd,
        premium_eur=premium_eur,
        iv_at_strike=iv_at_strike,
        contracts_affordable=contracts_affordable,
        cost_per_contract_eur=cost_per_contract_eur,
        cost_total_eur=cost_total_eur,
        breakeven_usd=breakeven_usd,
        max_loss_eur=max_loss_eur,
        iv_rank_result=iv_rank_result,
        payoff_table=payoff_table,
        recommendation=recommendation,
        rationale=rationale,
        warning=warning,
        liquidity_warning=liquidity_warning,
    )
