"""Single authoritative snapshot builder for the Crucible screener.

Migrated from scripts/run_backtest.py (canonical version) with the following
key differences:
- All functions are public (no leading underscore)
- compute_snapshot_row accepts edgar_dir as an explicit parameter
- enable_insider=False by default (Form 4 calls are too slow for bulk work)
- attach_momentum adds both momentum_raw (12-1m) and momentum_3m (3-1m)
- Six additional Track 2 columns: revenue_growth_yr1, revenue_growth_yr2,
  fcf_positive_last2yr, fcf_trajectory, gross_margin_yr1_change, p_s

Usage
-----
  from crucible.snapshot import build_snapshot, build_snapshots_parallel, attach_momentum
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

from crucible.fetcher import _load_dei_shares_cached, _to_float, get_shares_outstanding

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Raw XBRL tags
# ---------------------------------------------------------------------------

_RAW_XBRL_TAGS: frozenset[str] = frozenset({
    # Revenue
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    # Net Income
    "NetIncomeLoss",
    "ProfitLoss",
    # Operating CF
    "NetCashProvidedByUsedInOperatingActivities",
    # CapEx
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "CapitalExpenditures",
    # Total Debt
    "LongTermDebt",
    "DebtAndCapitalLeaseObligations",
    # Cash
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsAndShortTermInvestments",
    # Equity
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    # Gross Profit
    "GrossProfit",
    # EBITDA components (for leverage filter)
    "OperatingIncomeLoss",
    "DepreciationDepletionAndAmortization",
    "DepreciationAndAmortization",
    "Depreciation",
    # Interest expense (for interest coverage)
    "InterestExpense",
    "InterestAndDebtExpense",
    # Total Assets (for asset_growth_yoy — Fama-French CMA proxy)
    "Assets",
    # Deferred Revenue (book-to-bill proxy for Track 2)
    "DeferredRevenueCurrent",
    "DeferredRevenue",
})

# ---------------------------------------------------------------------------
# Insider / buyback caches (module-level)
# ---------------------------------------------------------------------------

_SUBMISSIONS_CACHE: dict[str, dict] = {}   # CIK10 → SEC submissions JSON
_FORM4_SIGNAL_CACHE: dict[tuple[str, str], float | None] = {}  # (CIK10, as_of) → signal

_EDGAR_UA: str = os.environ.get("EDGAR_USER_AGENT", "Crucible gabrielserens@gmail.com")


# ---------------------------------------------------------------------------
# Public: Raw XBRL loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=600)
def load_raw_annual_facts(
    padded_cik: str,
    edgar_dir_str: str,
) -> dict[str, list[dict]]:
    """Load all 10-K/10-K/A FY records for a CIK, keyed by raw XBRL tag.

    Returns {tag: [{end, val, filed}, ...]} with NO date filter applied.
    LRU-cached so the same JSON is not re-read across snapshot dates.
    """
    json_path = Path(edgar_dir_str) / f"CIK{padded_cik}.json"
    if not json_path.exists():
        return {}
    raw = json.loads(json_path.read_bytes())
    us_gaap: dict = raw.get("facts", {}).get("us-gaap", {})
    result: dict[str, list[dict]] = {}
    for tag in _RAW_XBRL_TAGS:
        concept = us_gaap.get(tag)
        if not concept:
            continue
        usd_recs: list[dict] = concept.get("units", {}).get("USD", [])
        collected = []
        for rec in usd_recs:
            if rec.get("form") not in {"10-K", "10-K/A"}:
                continue
            if rec.get("fp") != "FY":
                continue
            filed = rec.get("filed", "")
            end = rec.get("end", "")
            val = rec.get("val")
            if not filed or val is None:
                continue
            collected.append({"end": end, "val": float(val), "filed": filed})
        if collected:
            result[tag] = collected
    return result


def get_raw_pivoted(
    cik: str,
    as_of_date: pd.Timestamp,
    edgar_dir: Path,
) -> dict[str, pd.Series]:
    """Build {xbrl_tag: Series(fiscal_year_end → value)} for a CIK as of as_of_date.

    Each tag's records are filtered to filed <= as_of_date and deduped by fiscal year
    (latest amendment wins). Winner selection happens AFTER the date filter so a tag
    with only post-cutoff data doesn't block an older tag from being found.
    """
    all_facts = load_raw_annual_facts(cik.zfill(10), str(edgar_dir))
    as_of_str = as_of_date.strftime("%Y-%m-%d")
    result: dict[str, pd.Series] = {}
    for tag, records in all_facts.items():
        filtered = [r for r in records if r["filed"] <= as_of_str]
        if not filtered:
            continue
        by_fy: dict[str, dict] = {}
        for rec in filtered:
            fy = rec.get("end", "")[:4]
            existing = by_fy.get(fy)
            if existing is None or rec["filed"] > existing["filed"]:
                by_fy[fy] = rec
        if by_fy:
            idx = pd.to_datetime([v["end"] for v in by_fy.values()], utc=True)
            vals = [v["val"] for v in by_fy.values()]
            result[tag] = pd.Series(vals, index=idx, dtype=float).sort_index()
    return result


def get_metric(pivoted: dict[str, pd.Series], keys: list[str]) -> pd.Series:
    """Return the first non-empty Series from pivoted whose key appears in keys."""
    for key in keys:
        s = pivoted.get(key)
        if s is not None and not s.empty:
            return s
    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Public: Math helpers
# ---------------------------------------------------------------------------


def linear_slope(values: list[float]) -> float | None:
    """OLS slope for a list of evenly-spaced values."""
    n = len(values)
    if n < 2:
        return None
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((i - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


def prices_at(prices: pd.DataFrame, date: pd.Timestamp) -> dict[str, float]:
    """Extract {ticker: price} for the closest month-end on or before date."""
    idx = prices.index
    pos = int(idx.searchsorted(date, side="right")) - 1
    if pos < 0:
        return {}
    row = prices.iloc[pos]
    return {
        col: float(row[col])
        for col in prices.columns
        if col != "SP500" and pd.notna(row[col]) and float(row[col]) > 0
    }


# ---------------------------------------------------------------------------
# Private: buyback signal
# ---------------------------------------------------------------------------


def _compute_buyback_signal(
    cik: str,
    as_of_date: pd.Timestamp,
    edgar_dir: Path,
) -> float | None:
    """Year-over-year change in shares outstanding from EDGAR DEI 10-K filings.

    Returns (prior_shares − current_shares) / prior_shares.
    Positive = net buyback; negative = net dilution.
    """
    records = _load_dei_shares_cached(cik.zfill(10), str(edgar_dir))
    as_of_str = as_of_date.strftime("%Y-%m-%d")
    annual = sorted(
        [r for r in records if r["form"] in {"10-K", "10-K/A"} and r["filed"] <= as_of_str],
        key=lambda r: r["filed"],
        reverse=True,
    )
    if len(annual) < 2:
        return None
    current = annual[0]["val"]
    latest_dt = pd.Timestamp(annual[0]["filed"])
    window_lo = (latest_dt - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    window_hi = (latest_dt - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    prior_candidates = [r for r in annual[1:] if window_lo <= r["filed"] <= window_hi]
    if not prior_candidates:
        return None
    prior = prior_candidates[0]["val"]
    return (prior - current) / prior if prior > 0 else None


# ---------------------------------------------------------------------------
# Public: Snapshot row computation
# ---------------------------------------------------------------------------


def compute_snapshot_row(
    ticker: str,
    pivoted: dict[str, pd.Series],
    price: float | None = None,
    shares: float | None = None,
    cik: str | None = None,
    as_of_date: pd.Timestamp | None = None,
    edgar_dir: Path | None = None,
    enable_insider: bool = False,
) -> dict:
    """Derive the processed row that apply_filters() and score() expect.

    edgar_dir is required for the share_buyback_signal computation.
    enable_insider=False by default (Form 4 calls are too slow for bulk work).
    """
    rev    = get_metric(pivoted, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"])
    gp     = get_metric(pivoted, ["GrossProfit"])
    ni     = get_metric(pivoted, ["NetIncomeLoss", "ProfitLoss"])
    ocf    = get_metric(pivoted, ["NetCashProvidedByUsedInOperatingActivities"])
    capex  = get_metric(pivoted, ["PaymentsToAcquirePropertyPlantAndEquipment", "CapitalExpenditures"])
    td     = get_metric(pivoted, ["LongTermDebt", "DebtAndCapitalLeaseObligations"])
    cash   = get_metric(pivoted, ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments"])
    equity = get_metric(pivoted, ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"])
    op_inc   = get_metric(pivoted, ["OperatingIncomeLoss"])
    da       = get_metric(pivoted, ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation"])
    interest = get_metric(pivoted, ["InterestExpense", "InterestAndDebtExpense"])

    data_years = len(rev)

    roic_vals: list[float] = []
    for fy in ni.index:
        ni_v = _to_float(ni.get(fy))
        eq_v = _to_float(equity.get(fy))
        td_v = _to_float(td.get(fy))
        if ni_v is not None and eq_v is not None and td_v is not None:
            denom = eq_v + td_v
            if denom > 0:
                roic_vals.append(ni_v / denom)
    roic_avg = sum(roic_vals) / len(roic_vals) if roic_vals else None

    # FCF = OCF − |CapEx|, aligned by fiscal year
    fcf_by_fy: dict = {}
    for fy in ocf.index:
        o = _to_float(ocf.get(fy))
        c = _to_float(capex.get(fy))
        if o is not None and c is not None:
            fcf_by_fy[fy] = o - abs(c)
    fcf = pd.Series(fcf_by_fy, dtype=float).sort_index() if fcf_by_fy else pd.Series(dtype=float)

    fcf_vals   = [_to_float(v) for v in fcf.values]
    fcf_pos    = float(sum(1 for v in fcf_vals if v is not None and v > 0))
    fcf_latest = _to_float(fcf.iloc[-1]) if not fcf.empty else None

    td_last = _to_float(td.iloc[-1])   if not td.empty   else None
    ca_last = _to_float(cash.iloc[-1]) if not cash.empty else None

    # EBITDA = Operating Income + D&A (most recent fiscal year with both)
    eb_last: float | None = None
    if not op_inc.empty and not da.empty:
        common = op_inc.index.intersection(da.index)
        if len(common):
            oi = _to_float(op_inc.get(common[-1]))
            d  = _to_float(da.get(common[-1]))
            if oi is not None and d is not None:
                eb_last = oi + d

    nd_eb: float | None = None
    if td_last is not None and ca_last is not None and eb_last and eb_last > 0:
        nd_eb = (td_last - ca_last) / eb_last

    rev_vals = [_to_float(v) for v in rev.values if _to_float(v) is not None]
    rev_growth = (
        float(sum(1 for i in range(1, len(rev_vals)) if rev_vals[i] > rev_vals[i - 1]))
        if len(rev_vals) >= 2 else None
    )

    gm_vals: list[float] = []
    for fy in rev.index:
        r = _to_float(rev.get(fy))
        g = _to_float(gp.get(fy))
        if r and r > 0 and g is not None:
            gm_vals.append(g / r)

    # 5-year average FCF (Shiller-style: normalises against own earnings history)
    _LOOKBACK = 5
    fcf_recent = list(fcf.tail(_LOOKBACK).dropna().values) if not fcf.empty else []
    fcf_5yr_avg: float | None = (sum(fcf_recent) / len(fcf_recent)) if fcf_recent else None
    fcf_positive_years_last5: int = sum(1 for v in fcf_recent if v > 0)

    # 5-year average EBITDA
    ebitda_5yr_avg: float | None = None
    if not op_inc.empty and not da.empty:
        common_fys = op_inc.index.intersection(da.index)
        ebitda_vals: list[float] = []
        for fy in common_fys[-_LOOKBACK:]:
            oi = _to_float(op_inc.get(fy))
            d = _to_float(da.get(fy))
            if oi is not None and d is not None:
                ebitda_vals.append(oi + d)
        if ebitda_vals:
            ebitda_5yr_avg = sum(ebitda_vals) / len(ebitda_vals)

    # 5-year average net income (all years, including negative; None if avg is negative)
    ni_5yr_avg: float | None = None
    if not ni.empty:
        ni_recent = [_to_float(v) for v in ni.tail(_LOOKBACK).values]
        ni_valid = [v for v in ni_recent if v is not None]
        if ni_valid:
            avg = sum(ni_valid) / len(ni_valid)
            ni_5yr_avg = avg if avg > 0 else None

    # ── Feature 1: interest_coverage = EBIT / interest expense (latest year) ──
    interest_coverage: float | None = None
    if not op_inc.empty and not interest.empty:
        common_int = op_inc.index.intersection(interest.index)
        for fy in reversed(common_int):
            ebit_v = _to_float(op_inc.get(fy))
            int_v  = _to_float(interest.get(fy))
            if ebit_v is not None and int_v is not None and int_v != 0:
                raw = ebit_v / int_v
                interest_coverage = max(-10.0, min(50.0, raw))
                break

    # ── Feature 2: cfo_to_ni = OCF / Net Income (latest year) ──
    cfo_to_ni: float | None = None
    if not ocf.empty and not ni.empty:
        common_cfo = ocf.index.intersection(ni.index)
        for fy in reversed(common_cfo):
            ocf_v = _to_float(ocf.get(fy))
            ni_v  = _to_float(ni.get(fy))
            if ocf_v is not None and ni_v is not None and ni_v != 0:
                raw = ocf_v / ni_v
                cfo_to_ni = max(-2.0, min(5.0, raw))
                break

    # ── Feature 3: capex_intensity = |CapEx| / Revenue (latest year) ──
    capex_intensity: float | None = None
    if not capex.empty and not rev.empty:
        common_cap = capex.index.intersection(rev.index)
        for fy in reversed(common_cap):
            cap_v = _to_float(capex.get(fy))
            rev_v = _to_float(rev.get(fy))
            if cap_v is not None and rev_v is not None and rev_v > 0:
                capex_intensity = abs(cap_v) / rev_v
                break

    # ── Feature 4: operating_margin_trend = OLS slope of (OpInc/Rev) ──
    om_vals: list[float] = []
    for fy in op_inc.index:
        oi_v = _to_float(op_inc.get(fy))
        rv_v = _to_float(rev.get(fy))
        if oi_v is not None and rv_v is not None and rv_v > 0:
            om_vals.append(oi_v / rv_v)
    operating_margin_trend = linear_slope(om_vals)

    # ── Feature 5: revenue_acceleration ──
    # recent YoY growth (rev[-1]/rev[-2]−1) minus prior YoY growth (rev[-3]/rev[-4]−1).
    # Requires at least 4 revenue data points.
    revenue_acceleration: float | None = None
    rv_list = [v for v in ((_to_float(rev.iloc[i]) for i in range(len(rev))) if not rev.empty else [])
               if v is not None and v > 0]
    if len(rv_list) >= 4:
        recent_yoy = rv_list[-1] / rv_list[-2] - 1.0
        prior_yoy  = rv_list[-3] / rv_list[-4] - 1.0
        revenue_acceleration = recent_yoy - prior_yoy

    # ── NEW Track 2: revenue_growth_yr1 and revenue_growth_yr2 ──
    revenue_growth_yr1: float | None = None
    revenue_growth_yr2: float | None = None
    rv_pos = [v for v in rev_vals if v > 0]  # rev_vals already computed above
    if len(rv_pos) >= 2:
        revenue_growth_yr1 = rv_pos[-1] / rv_pos[-2] - 1.0
    if len(rv_pos) >= 3:
        revenue_growth_yr2 = rv_pos[-2] / rv_pos[-3] - 1.0

    # ── NEW Track 2: fcf_positive_last2yr ──
    fcf_positive_last2yr: int | None = None
    if not fcf.empty:
        last2 = [_to_float(v) for v in fcf.tail(2).values]
        fcf_positive_last2yr = sum(1 for v in last2 if v is not None and v > 0)

    # ── NEW Track 2: fcf_trajectory (normalised OLS slope of FCF) ──
    # Slope divided by mean(|FCF|) → dimensionless rate of change, comparable across sizes.
    fcf_nonnull = [v for v in fcf_vals if v is not None]
    fcf_trajectory: float | None = None
    if len(fcf_nonnull) >= 2:
        slope = linear_slope(fcf_nonnull)
        mean_abs = sum(abs(v) for v in fcf_nonnull) / len(fcf_nonnull)
        fcf_trajectory = slope / mean_abs if mean_abs != 0.0 else None

    # ── NEW Track 2: gross_margin_yr1_change ──
    gross_margin_yr1_change: float | None = None
    if len(gm_vals) >= 2:
        gross_margin_yr1_change = gm_vals[-1] - gm_vals[-2]

    # ── Feature 6: share_buyback_signal ──
    share_buyback_signal: float | None = (
        _compute_buyback_signal(cik, as_of_date, edgar_dir)
        if cik is not None and as_of_date is not None and edgar_dir is not None
        else None
    )

    # ── Feature 7: insider_buy_ratio ──
    # Disabled by default — Form 4 calls are too slow for bulk backtest work.
    insider_buy_ratio: float | None = None

    # ── Phase 4.7 — Track 2 enrichment features ──

    # Feature 8: asset_growth_yoy — Fama-French CMA proxy
    # High asset growth predicts underperformance (over-investment signal).
    assets = get_metric(pivoted, ["Assets"])
    asset_growth_yoy: float | None = None
    if len(assets) >= 2:
        a_n  = _to_float(assets.iloc[-1])
        a_n1 = _to_float(assets.iloc[-2])
        if a_n is not None and a_n1 is not None and a_n1 != 0:
            raw = (a_n - a_n1) / abs(a_n1)
            asset_growth_yoy = max(-1.0, min(2.0, raw))

    # Feature 9: deferred_revenue_growth — book-to-bill proxy (Track 2)
    # Rising deferred revenue = collecting cash before delivering = order backlog signal.
    deferred = get_metric(pivoted, ["DeferredRevenueCurrent", "DeferredRevenue"])
    deferred_revenue_growth: float | None = None
    if len(deferred) >= 2:
        d_n  = _to_float(deferred.iloc[-1])
        d_n1 = _to_float(deferred.iloc[-2])
        if d_n is not None and d_n1 is not None and d_n1 != 0:
            raw = (d_n - d_n1) / abs(d_n1)
            deferred_revenue_growth = max(-1.0, min(3.0, raw))

    # Feature 10: eps_surprise_last_q — annual Net Income as EPS proxy
    # Positive YoY growth = company beating its own prior-year baseline.
    eps_surprise_last_q: float | None = None
    if len(ni) >= 2:
        ni_n  = _to_float(ni.iloc[-1])
        ni_n1 = _to_float(ni.iloc[-2])
        if ni_n is not None and ni_n1 is not None and ni_n1 != 0:
            raw = (ni_n - ni_n1) / abs(ni_n1)
            eps_surprise_last_q = max(-2.0, min(5.0, raw))

    # Valuation multiples: price × EDGAR shares → market cap → multiples
    _MAX_MULTIPLE = 200.0
    p_fcf_val: float | None = None
    ev_ebitda_val: float | None = None
    p_e_val: float | None = None
    p_s_val: float | None = None

    if price is not None and price > 0 and shares is not None and shares > 0:
        mkt_cap = price * shares
        td_v = td_last or 0.0
        ca_v = ca_last or 0.0

        if fcf_5yr_avg is not None and fcf_5yr_avg > 0:
            p_fcf_val = min(mkt_cap / fcf_5yr_avg, _MAX_MULTIPLE)

        if ebitda_5yr_avg is not None and ebitda_5yr_avg > 0:
            ev = mkt_cap + td_v - ca_v
            if ev > 0:
                ev_ebitda_val = min(ev / ebitda_5yr_avg, _MAX_MULTIPLE)

        if ni_5yr_avg is not None and ni_5yr_avg > 0:
            p_e_val = min(mkt_cap / ni_5yr_avg, _MAX_MULTIPLE)

        # ── NEW Track 2: p_s = market cap / most recent annual revenue ──
        if rev_vals and rev_vals[-1] is not None and rev_vals[-1] > 0:
            p_s_val = min(mkt_cap / rev_vals[-1], _MAX_MULTIPLE)

    return {
        "ticker":                        ticker,
        "sector":                        None,
        "sub_industry":                  None,
        "currency":                      "USD",
        "p_e":                           p_e_val,
        "p_fcf":                         p_fcf_val,
        "ev_ebitda":                     ev_ebitda_val,
        "data_years":                    data_years,
        "insufficient_data":             data_years < 3,
        "roic_proxy_avg":                roic_avg,
        "fcf_latest":                    fcf_latest,
        "fcf_positive_years":            fcf_pos if not fcf.empty else None,
        "net_debt_ebitda":               nd_eb,
        "revenue_growth_positive_years": rev_growth,
        "gross_margin_latest":           gm_vals[-1] if gm_vals else None,
        "gross_margin_avg":              sum(gm_vals) / len(gm_vals) if gm_vals else None,
        "gross_margin_trend_slope":      linear_slope(gm_vals),
        "interest_coverage":             interest_coverage,
        "cfo_to_ni":                     cfo_to_ni,
        "capex_intensity":               capex_intensity,
        "operating_margin_trend":        operating_margin_trend,
        "revenue_acceleration":          revenue_acceleration,
        "share_buyback_signal":          share_buyback_signal,
        "insider_buy_ratio":             insider_buy_ratio,
        # ── Track 2 new columns ──
        "revenue_growth_yr1":            revenue_growth_yr1,
        "revenue_growth_yr2":            revenue_growth_yr2,
        "fcf_positive_last2yr":          fcf_positive_last2yr,
        "fcf_trajectory":                fcf_trajectory,
        "gross_margin_yr1_change":       gross_margin_yr1_change,
        "p_s":                           p_s_val,
        # ── Track 3 new columns ──
        "fcf_positive_years_last5":      fcf_positive_years_last5,
        # ── Phase 4.7 new columns ──
        "asset_growth_yoy":              asset_growth_yoy,
        "deferred_revenue_growth":       deferred_revenue_growth,
        "eps_surprise_last_q":           eps_surprise_last_q,
    }


# ---------------------------------------------------------------------------
# Public: Snapshot building
# ---------------------------------------------------------------------------


def build_snapshot(
    date: pd.Timestamp,
    tickers: list[str],
    cik_map: dict[str, str],
    edgar_dir: Path,
    prices_at_date: dict[str, float] | None = None,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Build a point-in-time fundamentals snapshot for one monthly date.

    prices_at_date, if provided, maps ticker → closing price as of date and is
    used to compute valuation multiples (P/FCF, EV/EBITDA, P/E, P/S) via EDGAR shares.
    """
    rows = []
    for ticker in tickers:
        cik = cik_map.get(ticker.upper())
        pivoted = get_raw_pivoted(cik, date, edgar_dir) if cik else {}
        price = prices_at_date.get(ticker) if prices_at_date else None
        shares = get_shares_outstanding(cik, date, edgar_dir) if cik else None
        rows.append(compute_snapshot_row(
            ticker, pivoted, price, shares, cik=cik, as_of_date=date, edgar_dir=edgar_dir,
        ))

    df = pd.DataFrame(rows).set_index("ticker")
    n_ok = int((~df["insufficient_data"]).sum())
    n_val = int(df["p_fcf"].notna().sum())
    log.debug(
        "%s  %d/%d sufficient data  %d/%d have p_fcf",
        date.date(), n_ok, len(tickers), n_val, len(tickers),
    )
    return date, df


def build_snapshots_parallel(
    tickers: list[str],
    dates: pd.DatetimeIndex,
    cik_map: dict[str, str],
    edgar_dir: Path,
    prices: pd.DataFrame | None = None,
    workers: int = 4,
    universe: str = "unknown",
    use_cache: bool = True,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Build monthly fundamentals snapshots in parallel (memory-safe).

    prices, if provided, is used to populate valuation multiples at each snapshot
    date via EDGAR shares outstanding. workers=4 keeps peak concurrent memory safe.

    Results are cached to data/cache/ by joblib. Pass use_cache=False to force rebuild.
    """
    start_tag = dates[0].strftime("%Y%m")
    end_tag   = dates[-1].strftime("%Y%m")
    cache_path = _CACHE_DIR / f"snapshots_{universe}_{start_tag}_{end_tag}.pkl"

    if use_cache and cache_path.exists():
        log.info("Cache HIT — loading snapshots from %s", cache_path)
        return joblib.load(cache_path)

    log.info("Cache MISS — building %d snapshots from EDGAR (universe=%s)", len(dates), universe)

    result: dict[pd.Timestamp, pd.DataFrame] = {}
    total = len(dates)
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                build_snapshot,
                d,
                tickers,
                cik_map,
                edgar_dir,
                prices_at(prices, d) if prices is not None else None,
            ): d
            for d in dates
        }
        for future in as_completed(futures):
            date, df = future.result()
            result[date] = df
            done += 1
            if done % 12 == 0 or done == total:
                log.info("Snapshots: %d / %d built", done, total)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, cache_path, compress=3)
    log.info("Snapshots cached → %s", cache_path)

    return result


# ---------------------------------------------------------------------------
# Public: Momentum attachment
# ---------------------------------------------------------------------------


def attach_momentum(
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
) -> None:
    """Add momentum_raw (12-1m) and momentum_3m (3-1m) columns in-place."""
    price_idx = prices.index

    for date, df in fund_by_date.items():
        pos = int(price_idx.searchsorted(date, side="right")) - 1
        pos_1m  = pos - 1    # D-1 month close
        pos_12m = pos - 12   # D-12 months close
        pos_3m  = pos - 4    # D-4 months close → 3-month window

        mom_map:  dict[str, float] = {}
        mom3_map: dict[str, float] = {}

        if pos_1m >= 0:
            p_1m = prices.iloc[pos_1m]

            if pos_12m >= 0:
                p_12m = prices.iloc[pos_12m]
                for ticker in df.index:
                    if ticker not in prices.columns:
                        continue
                    v1  = p_1m.get(ticker)
                    v12 = p_12m.get(ticker)
                    if pd.notna(v1) and pd.notna(v12) and float(v12) > 0:
                        mom_map[ticker] = float(v1) / float(v12) - 1.0

            if pos_3m >= 0:
                p_3m = prices.iloc[pos_3m]
                for ticker in df.index:
                    if ticker not in prices.columns:
                        continue
                    v1 = p_1m.get(ticker)
                    v3 = p_3m.get(ticker)
                    if pd.notna(v1) and pd.notna(v3) and float(v3) > 0:
                        mom3_map[ticker] = float(v1) / float(v3) - 1.0

        df["momentum_raw"] = pd.Series(mom_map,  dtype=float)
        df["momentum_3m"]  = pd.Series(mom3_map, dtype=float)


# ---------------------------------------------------------------------------
# Public: P/FCF history attachment (Track 3 price-attach layer)
# ---------------------------------------------------------------------------


def attach_p_fcf_history(
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    lookback: int = 60,
    min_history: int = 12,
) -> None:
    """Add p_fcf_5yr_avg, p_fcf_5yr_std, p_fcf_vs_history columns in-place.

    For each snapshot date, looks back at up to `lookback` prior monthly snapshots
    to compute rolling per-ticker P/FCF statistics. Requires prices to have been
    attached (p_fcf column must be populated by build_snapshots_parallel).

    p_fcf_vs_history = (p_fcf_5yr_avg - p_fcf_current) / p_fcf_5yr_std
    Positive value means the current P/FCF is below the historical mean (statistically cheap).
    A value > 1.0 means current P/FCF is more than one std below its own history.
    """
    dates = sorted(fund_by_date.keys())

    # Build a p_fcf panel: each column is one snapshot date's p_fcf Series
    pfcf_by_date = {
        date: fund_by_date[date]["p_fcf"]
        for date in dates
        if "p_fcf" in fund_by_date[date].columns
    }
    if not pfcf_by_date:
        log.warning("attach_p_fcf_history: no p_fcf column found — skipping")
        return

    # rows = dates, cols = tickers; NaN where ticker had no price/data that month
    pfcf_panel: pd.DataFrame = pd.DataFrame(pfcf_by_date).T.reindex(dates)

    for i, date in enumerate(dates):
        df = fund_by_date[date]
        hist = pfcf_panel.iloc[max(0, i - lookback): i]   # prior months, exclusive

        if len(hist) < min_history:
            df["p_fcf_5yr_avg"]    = float("nan")
            df["p_fcf_5yr_std"]    = float("nan")
            df["p_fcf_vs_history"] = float("nan")
            continue

        # Require each ticker to have at least min_history valid observations
        valid_counts = hist.notna().sum(axis=0)
        sufficient   = valid_counts >= min_history

        avg = hist.mean(axis=0).where(sufficient)
        std = hist.std(axis=0, ddof=1).where(sufficient)

        current = df["p_fcf"] if "p_fcf" in df.columns else pd.Series(dtype=float)
        vsh = ((avg - current) / std).where(std > 0)

        df["p_fcf_5yr_avg"]    = avg.reindex(df.index)
        df["p_fcf_5yr_std"]    = std.reindex(df.index)
        df["p_fcf_vs_history"] = vsh.reindex(df.index)

    log.info("attach_p_fcf_history: processed %d snapshot dates", len(dates))
