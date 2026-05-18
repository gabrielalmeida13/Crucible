"""Comprehensive unit tests for backtest.py — synthetic data only, no API calls."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crucible.backtest import (
    BacktestConfig,
    BacktestResult,
    MonthlyResult,
    _advance,
    _benchmark_return,
    _portfolio_return,
    _single_return,
    cumulative_return_series,
    generate_report,
    generate_ticker_contribution,
    hit_rate,
    max_drawdown,
    run_backtest,
    run_sensitivity,
    sharpe_ratio,
    ticker_contribution_analysis,
    total_return,
)
from crucible.config import CrucibleConfig, FilterThresholds

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# 60 monthly end-of-month dates: Jan-2018 → Dec-2022
_DATES = pd.date_range("2018-01-31", periods=60, freq="ME")


def _cfg(**kwargs) -> CrucibleConfig:
    """CrucibleConfig with USD account so synthetic USD tickers get no FX penalty."""
    return CrucibleConfig(
        universe="SP500",
        db_path=":memory:",
        log_level="WARNING",
        fmp_api_key="",
        account_currency="USD",
        filters=kwargs.pop("filters", FilterThresholds()),
    )


def _good_row(**kw) -> dict:
    """Minimal row that passes all default FilterThresholds."""
    base = dict(
        sector="Technology",
        sub_industry="Software",
        currency="USD",
        p_e=20.0,
        p_fcf=15.0,
        ev_ebitda=10.0,
        data_years=5,
        insufficient_data=False,
        roic_proxy_avg=0.25,
        fcf_latest=1e9,
        fcf_positive_years=4.0,
        net_debt_ebitda=1.0,
        revenue_growth_positive_years=4.0,
        gross_margin_latest=0.45,
        gross_margin_avg=0.44,
        gross_margin_trend_slope=0.01,
    )
    base.update(kw)
    return base


def _bad_row(**kw) -> dict:
    """Row that fails the ROIC filter (roic_proxy_avg=0.01 << threshold 0.15)."""
    return _good_row(roic_proxy_avg=0.01, **kw)


def _make_fund_df(good_tickers: list[str], bad_tickers: list[str]) -> pd.DataFrame:
    """One fundamentals snapshot with good and bad tickers."""
    rows = [_good_row() for _ in good_tickers] + [_bad_row() for _ in bad_tickers]
    tickers = good_tickers + bad_tickers
    return pd.DataFrame(rows, index=pd.Index(tickers, name="ticker"))


def _make_fundamentals(
    dates: pd.DatetimeIndex,
    good_tickers: list[str],
    bad_tickers: list[str],
) -> dict[pd.Timestamp, pd.DataFrame]:
    return {d: _make_fund_df(good_tickers, bad_tickers) for d in dates}


def _make_prices(
    dates: pd.DatetimeIndex,
    monthly_returns: dict[str, float],
) -> pd.DataFrame:
    """Price DataFrame where each ticker grows at a fixed monthly compound rate."""
    data = {
        ticker: [100.0 * (1.0 + r) ** i for i in range(len(dates))]
        for ticker, r in monthly_returns.items()
    }
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# total_return
# ---------------------------------------------------------------------------


def test_total_return_empty() -> None:
    assert total_return([]) == 0.0


def test_total_return_single_positive() -> None:
    assert abs(total_return([0.10]) - 0.10) < 1e-9


def test_total_return_single_negative() -> None:
    assert abs(total_return([-0.20]) - (-0.20)) < 1e-9


def test_total_return_all_zeros() -> None:
    assert total_return([0.0, 0.0, 0.0]) == 0.0


def test_total_return_compounds_correctly() -> None:
    result = total_return([0.10, 0.20, 0.30])
    expected = 1.10 * 1.20 * 1.30 - 1
    assert abs(result - expected) < 1e-9


def test_total_return_negative_sequence() -> None:
    result = total_return([-0.10, -0.10, -0.10])
    expected = 0.90 ** 3 - 1
    assert abs(result - expected) < 1e-9


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------


def test_sharpe_ratio_single_period_is_nan() -> None:
    assert math.isnan(sharpe_ratio([0.05]))


def test_sharpe_ratio_zero_variance_is_nan() -> None:
    # Constant returns → zero std → NaN
    assert math.isnan(sharpe_ratio([0.01, 0.01, 0.01]))


def test_sharpe_ratio_known_value() -> None:
    # returns=[0.0, 0.02], rf=0 → annualised Sharpe ≈ 2.449
    result = sharpe_ratio([0.0, 0.02], risk_free_annual=0.0)
    assert abs(result - 2.449) < 0.001


def test_sharpe_ratio_positive_for_excess_positive() -> None:
    # All returns well above zero → positive Sharpe
    result = sharpe_ratio([0.05, 0.04, 0.06, 0.05], risk_free_annual=0.0)
    assert result > 0


def test_sharpe_ratio_negative_for_excess_negative() -> None:
    # All returns below risk-free
    result = sharpe_ratio([-0.01, -0.02, -0.01, -0.02], risk_free_annual=0.04)
    assert result < 0


def test_sharpe_ratio_annualises_by_sqrt12() -> None:
    # With rf=0, Sharpe = mean/std * sqrt(12)
    rets = [0.01, 0.02, 0.01, 0.02]
    excess = np.array(rets)
    expected = (excess.mean() / excess.std(ddof=1)) * np.sqrt(12)
    result = sharpe_ratio(rets, risk_free_annual=0.0)
    assert abs(result - expected) < 1e-9


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_empty() -> None:
    assert max_drawdown([]) == 0.0


def test_max_drawdown_monotone_rise() -> None:
    # No drawdown on a consistently rising path
    assert max_drawdown([0.05, 0.05, 0.05]) == 0.0


def test_max_drawdown_known_sequence() -> None:
    # [+10%, -20%, +10%] → wealth [1.1, 0.88, 0.968]
    # Peak always 1.1 after first period → MDD = (0.88 - 1.1) / 1.1 = -0.2
    result = max_drawdown([0.10, -0.20, 0.10])
    assert abs(result - (-0.20)) < 1e-9


def test_max_drawdown_always_non_positive() -> None:
    # Drawdown can never be positive
    for returns in [[0.1, -0.3, 0.2], [0.05, 0.05], [-0.1, -0.2]]:
        assert max_drawdown(returns) <= 0.0


def test_max_drawdown_picks_worst_trough() -> None:
    # [+50%, -60%] → wealth [1.5, 0.6] → peak always 1.5 → MDD = (0.6-1.5)/1.5 = -0.6
    result = max_drawdown([0.50, -0.60])
    assert abs(result - (-0.60)) < 1e-9


# ---------------------------------------------------------------------------
# hit_rate
# ---------------------------------------------------------------------------


def test_hit_rate_empty_is_nan() -> None:
    assert math.isnan(hit_rate([]))


def test_hit_rate_all_positive() -> None:
    assert hit_rate([0.1, 0.2, 0.3]) == 1.0


def test_hit_rate_all_negative() -> None:
    assert hit_rate([-0.1, -0.2]) == 0.0


def test_hit_rate_exactly_half() -> None:
    assert hit_rate([0.1, 0.1, -0.1, -0.1]) == 0.5


def test_hit_rate_excludes_zero_return() -> None:
    # Zero returns are not "positive", so should not be counted
    assert hit_rate([0.0, 0.0, 0.1]) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# cumulative_return_series
# ---------------------------------------------------------------------------


def test_cumulative_return_series_empty() -> None:
    result = cumulative_return_series([])
    assert list(result) == [1.0]


def test_cumulative_return_series_single() -> None:
    result = cumulative_return_series([0.10])
    assert abs(result.iloc[0] - 1.10) < 1e-9


def test_cumulative_return_series_compounds() -> None:
    result = cumulative_return_series([0.10, 0.10])
    assert abs(result.iloc[-1] - 1.21) < 1e-9


def test_cumulative_return_series_monotone_for_positive_returns() -> None:
    result = cumulative_return_series([0.01, 0.02, 0.01])
    assert all(result.iloc[i] < result.iloc[i + 1] for i in range(len(result) - 1))


# ---------------------------------------------------------------------------
# _advance
# ---------------------------------------------------------------------------


def test_advance_normal_case() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29"),
                            pd.Timestamp("2020-03-31")])
    result = _advance(pd.Timestamp("2020-01-31"), idx, 1)
    assert result == pd.Timestamp("2020-02-29")


def test_advance_multi_step() -> None:
    idx = _DATES[:10]
    result = _advance(_DATES[3], idx, 3)
    assert result == _DATES[6]


def test_advance_past_end_returns_none() -> None:
    idx = _DATES[:5]
    result = _advance(_DATES[4], idx, 1)
    assert result is None


def test_advance_from_last_date_returns_none() -> None:
    idx = _DATES[:3]
    result = _advance(_DATES[2], idx, 2)
    assert result is None


# ---------------------------------------------------------------------------
# _single_return
# ---------------------------------------------------------------------------


def test_single_return_normal() -> None:
    dates = _DATES[:3]
    prices = _make_prices(dates, {"AAPL": 0.10})
    r = _single_return("AAPL", dates[0], dates[1], prices)
    assert abs(r - 0.10) < 1e-9


def test_single_return_missing_ticker_returns_none() -> None:
    dates = _DATES[:3]
    prices = _make_prices(dates, {"AAPL": 0.10})
    assert _single_return("MSFT", dates[0], dates[1], prices) is None


def test_single_return_zero_price_returns_none() -> None:
    dates = _DATES[:2]
    prices = pd.DataFrame({"AAPL": [0.0, 100.0]}, index=dates)
    assert _single_return("AAPL", dates[0], dates[1], prices) is None


def test_single_return_nan_price_returns_none() -> None:
    dates = _DATES[:2]
    prices = pd.DataFrame({"AAPL": [float("nan"), 110.0]}, index=dates)
    assert _single_return("AAPL", dates[0], dates[1], prices) is None


# ---------------------------------------------------------------------------
# _portfolio_return and _benchmark_return
# ---------------------------------------------------------------------------


def test_portfolio_return_equal_weighted() -> None:
    dates = _DATES[:3]
    prices = _make_prices(dates, {"A": 0.10, "B": 0.20})
    r = _portfolio_return(["A", "B"], dates[0], dates[1], prices)
    assert abs(r - 0.15) < 1e-9  # equal-weighted mean of 10% and 20%


def test_portfolio_return_missing_tickers_excluded() -> None:
    dates = _DATES[:3]
    prices = _make_prices(dates, {"A": 0.10})
    # MISSING is not in prices; only A contributes
    r = _portfolio_return(["A", "MISSING"], dates[0], dates[1], prices)
    assert abs(r - 0.10) < 1e-9


def test_portfolio_return_all_missing_returns_zero() -> None:
    dates = _DATES[:3]
    prices = _make_prices(dates, {"A": 0.10})
    r = _portfolio_return(["GONE1", "GONE2"], dates[0], dates[1], prices)
    assert r == 0.0


def test_benchmark_return_uses_named_column() -> None:
    dates = _DATES[:3]
    prices = _make_prices(dates, {"SP500": 0.08, "OTHER": 0.50})
    r = _benchmark_return(dates[0], dates[1], prices, "SP500")
    assert abs(r - 0.08) < 1e-9


def test_benchmark_return_falls_back_to_equal_weight() -> None:
    dates = _DATES[:3]
    prices = _make_prices(dates, {"A": 0.10, "B": 0.20})
    # "BENCH" column does not exist → fallback
    r = _benchmark_return(dates[0], dates[1], prices, "BENCH")
    assert abs(r - 0.15) < 1e-9


# ---------------------------------------------------------------------------
# BacktestResult and MonthlyResult
# ---------------------------------------------------------------------------


def test_backtest_result_empty_to_dataframe() -> None:
    result = BacktestResult(monthly_results=[], hit_rate_returns=[], bt_config=BacktestConfig())
    df = result.to_dataframe()
    assert df.empty
    assert list(df.columns) == ["date", "portfolio_return", "benchmark_return", "n_picks"]


def test_backtest_result_to_dataframe_has_correct_columns() -> None:
    mr = MonthlyResult(
        date=_DATES[0], portfolio_return=0.05,
        benchmark_return=0.02, n_picks=3,
    )
    result = BacktestResult(monthly_results=[mr], hit_rate_returns=[0.10], bt_config=BacktestConfig())
    df = result.to_dataframe()
    assert len(df) == 1
    assert list(df.columns) == ["date", "portfolio_return", "benchmark_return", "n_picks"]


def test_backtest_result_portfolio_returns() -> None:
    results = [
        MonthlyResult(date=_DATES[i], portfolio_return=float(i),
                      benchmark_return=0.0, n_picks=1)
        for i in range(3)
    ]
    bt = BacktestResult(monthly_results=results, hit_rate_returns=[], bt_config=BacktestConfig())
    assert bt.portfolio_returns() == [0.0, 1.0, 2.0]


def test_backtest_result_benchmark_returns() -> None:
    results = [
        MonthlyResult(date=_DATES[i], portfolio_return=0.0,
                      benchmark_return=float(i) * 0.1, n_picks=1)
        for i in range(3)
    ]
    bt = BacktestResult(monthly_results=results, hit_rate_returns=[], bt_config=BacktestConfig())
    assert bt.benchmark_returns() == pytest.approx([0.0, 0.1, 0.2])


# ---------------------------------------------------------------------------
# run_backtest — structure and edge cases
# ---------------------------------------------------------------------------


def test_run_backtest_raises_when_too_few_months() -> None:
    dates = _DATES[:5]
    bt_cfg = BacktestConfig(train_months=10)
    funds = _make_fundamentals(dates, ["G1"], ["B1"])
    prices = _make_prices(dates, {"G1": 0.01, "B1": 0.0, "SP500": 0.005})
    with pytest.raises(ValueError, match="more than"):
        run_backtest(funds, prices, _cfg(), bt_cfg)


def test_run_backtest_no_results_if_prices_dont_cover_test_dates() -> None:
    # 6 fundamental dates (3 train + 3 test), prices only cover training window
    fund_dates = _DATES[:6]
    price_dates = _DATES[:4]  # only goes up to training window + 1
    bt_cfg = BacktestConfig(train_months=3, top_n=5, hit_rate_months=2)
    funds = _make_fundamentals(fund_dates, ["G1", "G2"], [])
    prices = _make_prices(price_dates, {"G1": 0.02, "G2": 0.02, "SP500": 0.005})
    result = run_backtest(funds, prices, _cfg(), bt_cfg)
    # test dates = fund_dates[3:6]; none of them have a valid next price date
    # because prices only covers to fund_dates[3]
    assert len(result.monthly_results) <= 1  # at most 1 result (date[3] → date[4])


def test_run_backtest_correct_number_of_test_months() -> None:
    # 10 fundamentals dates, 3 train → 7 test; prices cover all + 1 extra
    fund_dates = _DATES[:10]
    price_dates = _DATES[:11]
    bt_cfg = BacktestConfig(train_months=3, top_n=5, hit_rate_months=2)
    funds = _make_fundamentals(fund_dates, ["G1", "G2"], [])
    prices = _make_prices(price_dates, {"G1": 0.02, "G2": 0.02, "SP500": 0.005})
    result = run_backtest(funds, prices, _cfg(), bt_cfg)
    # Last test date = fund_dates[9]; next date = price_dates[10] ✓
    assert len(result.monthly_results) == 7


def test_run_backtest_all_fail_filters_produces_no_monthly_results() -> None:
    fund_dates = _DATES[:8]
    price_dates = _DATES[:12]
    bt_cfg = BacktestConfig(train_months=3, top_n=5, hit_rate_months=2)
    # Only bad tickers in the fundamentals — all fail ROIC filter
    funds = _make_fundamentals(fund_dates, [], ["B1", "B2", "B3"])
    prices = _make_prices(price_dates, {"B1": 0.0, "B2": 0.0, "B3": 0.0, "SP500": 0.005})
    result = run_backtest(funds, prices, _cfg(), bt_cfg)
    assert len(result.monthly_results) == 0
    assert result.hit_rate_returns == []


def test_run_backtest_n_picks_never_exceeds_top_n() -> None:
    fund_dates = _DATES[:10]
    price_dates = _DATES[:14]
    bt_cfg = BacktestConfig(train_months=3, top_n=2, hit_rate_months=2)
    # 5 good tickers but top_n=2
    funds = _make_fundamentals(fund_dates, ["G1", "G2", "G3", "G4", "G5"], [])
    returns_map = {t: 0.01 for t in ["G1", "G2", "G3", "G4", "G5"]}
    returns_map["SP500"] = 0.005
    prices = _make_prices(price_dates, returns_map)
    result = run_backtest(funds, prices, _cfg(), bt_cfg)
    assert all(m.n_picks <= 2 for m in result.monthly_results)


def test_run_backtest_tickers_stored_in_monthly_results() -> None:
    fund_dates = _DATES[:6]
    price_dates = _DATES[:8]
    bt_cfg = BacktestConfig(train_months=3, top_n=5, hit_rate_months=2)
    funds = _make_fundamentals(fund_dates, ["G1", "G2"], ["B1"])
    prices = _make_prices(price_dates, {"G1": 0.02, "G2": 0.02, "B1": 0.0, "SP500": 0.005})
    result = run_backtest(funds, prices, _cfg(), bt_cfg)
    for m in result.monthly_results:
        assert isinstance(m.tickers, list)
        # Bad ticker should never be picked
        assert "B1" not in m.tickers
        # Good tickers should be present
        assert "G1" in m.tickers or "G2" in m.tickers


def test_run_backtest_hit_rate_returns_populated() -> None:
    # 10 fund months, 3 train, hit_rate_months=3, prices extend 10+3=13 months
    fund_dates = _DATES[:10]
    price_dates = _DATES[:13]
    bt_cfg = BacktestConfig(train_months=3, top_n=5, hit_rate_months=3)
    funds = _make_fundamentals(fund_dates, ["G1", "G2"], [])
    prices = _make_prices(price_dates, {"G1": 0.02, "G2": 0.02, "SP500": 0.005})
    result = run_backtest(funds, prices, _cfg(), bt_cfg)
    assert len(result.hit_rate_returns) > 0


def test_run_backtest_hit_rate_empty_when_prices_too_short() -> None:
    # Prices only cover to last test date — no 12-month forward prices available
    fund_dates = _DATES[:8]
    price_dates = _DATES[:9]  # one extra, enough for 1-month but not 12-month
    bt_cfg = BacktestConfig(train_months=3, top_n=5, hit_rate_months=12)
    funds = _make_fundamentals(fund_dates, ["G1", "G2"], [])
    prices = _make_prices(price_dates, {"G1": 0.02, "G2": 0.02, "SP500": 0.005})
    result = run_backtest(funds, prices, _cfg(), bt_cfg)
    assert result.hit_rate_returns == []


def test_run_backtest_high_quality_outperforms_benchmark() -> None:
    """Good tickers (2% monthly) should clearly beat the benchmark (0.5% monthly)."""
    fund_dates = _DATES[:36]   # 24 train + 12 test
    price_dates = _DATES[:48]  # test dates + 12 extra for hit rate
    bt_cfg = BacktestConfig(train_months=24, top_n=4, hit_rate_months=12)

    good_tickers = ["G1", "G2", "G3", "G4"]
    bad_tickers = ["B1", "B2", "B3", "B4"]
    funds = _make_fundamentals(fund_dates, good_tickers, bad_tickers)

    returns_map = {t: 0.02 for t in good_tickers}
    returns_map.update({t: -0.005 for t in bad_tickers})
    returns_map["SP500"] = 0.005
    prices = _make_prices(price_dates, returns_map)

    result = run_backtest(funds, prices, _cfg(), bt_cfg)
    assert len(result.monthly_results) > 0

    port_total = total_return(result.portfolio_returns())
    bench_total = total_return(result.benchmark_returns())
    assert port_total > bench_total


def test_run_backtest_uses_default_bt_config_when_none() -> None:
    """Passing bt_config=None must use BacktestConfig() defaults without error."""
    fund_dates = _DATES[:26]   # 24 train + 2 test
    price_dates = _DATES[:38]  # covers hit_rate_months=12
    funds = _make_fundamentals(fund_dates, ["G1"], [])
    prices = _make_prices(price_dates, {"G1": 0.01, "SP500": 0.005})
    result = run_backtest(funds, prices, _cfg(), bt_config=None)
    assert isinstance(result, BacktestResult)


# ---------------------------------------------------------------------------
# run_sensitivity
# ---------------------------------------------------------------------------


def test_sensitivity_returns_one_row_per_threshold() -> None:
    fund_dates = _DATES[:8]
    price_dates = _DATES[:12]
    bt_cfg = BacktestConfig(train_months=3, top_n=5, hit_rate_months=3)
    good_tickers = ["G1", "G2", "G3"]
    funds = _make_fundamentals(fund_dates, good_tickers, [])
    prices = _make_prices(price_dates, {t: 0.01 for t in good_tickers} | {"SP500": 0.005})
    thresholds = (0.10, 0.15, 0.20)
    sens = run_sensitivity(funds, prices, _cfg(), bt_cfg, roic_thresholds=thresholds)
    assert len(sens) == len(thresholds)


def test_sensitivity_has_required_columns() -> None:
    fund_dates = _DATES[:8]
    price_dates = _DATES[:12]
    bt_cfg = BacktestConfig(train_months=3, top_n=5, hit_rate_months=3)
    good_tickers = ["G1", "G2"]
    funds = _make_fundamentals(fund_dates, good_tickers, [])
    prices = _make_prices(price_dates, {t: 0.01 for t in good_tickers} | {"SP500": 0.005})
    sens = run_sensitivity(funds, prices, _cfg(), bt_cfg, roic_thresholds=(0.15, 0.20))
    for col in ("roic_min", "portfolio_total_return", "benchmark_total_return",
                "sharpe_ratio", "max_drawdown", "hit_rate", "n_test_months", "avg_picks"):
        assert col in sens.columns, f"Missing column: {col}"


def test_sensitivity_stricter_roic_threshold_fewer_picks() -> None:
    """Raising ROIC threshold must reduce or maintain avg_picks."""
    fund_dates = _DATES[:8]
    price_dates = _DATES[:12]
    bt_cfg = BacktestConfig(train_months=3, top_n=10, hit_rate_months=3)

    # High-ROIC tickers (pass 0.20 threshold): roic=0.30
    # Mid-ROIC tickers (pass 0.15 but fail 0.20): roic=0.16
    high_rows = {d: pd.DataFrame(
        [_good_row(roic_proxy_avg=0.30), _good_row(roic_proxy_avg=0.16)],
        index=pd.Index(["H1", "M1"], name="ticker"),
    ) for d in fund_dates}

    prices = _make_prices(price_dates, {"H1": 0.02, "M1": 0.01, "SP500": 0.005})

    sens = run_sensitivity(
        high_rows, prices, _cfg(), bt_cfg,
        roic_thresholds=(0.15, 0.20),
    )
    picks_at_015 = sens.loc[sens["roic_min"] == 0.15, "avg_picks"].iloc[0]
    picks_at_020 = sens.loc[sens["roic_min"] == 0.20, "avg_picks"].iloc[0]
    # With 0.15 both H1 and M1 pass; with 0.20 only H1 passes → fewer picks
    assert picks_at_020 <= picks_at_015


def test_sensitivity_benchmark_returns_equal_across_thresholds() -> None:
    """Benchmark return is independent of the ROIC threshold used."""
    fund_dates = _DATES[:8]
    price_dates = _DATES[:12]
    bt_cfg = BacktestConfig(train_months=3, top_n=5, hit_rate_months=3)
    good_tickers = ["G1", "G2"]
    funds = _make_fundamentals(fund_dates, good_tickers, [])
    prices = _make_prices(price_dates, {t: 0.02 for t in good_tickers} | {"SP500": 0.005})
    sens = run_sensitivity(funds, prices, _cfg(), bt_cfg, roic_thresholds=(0.15, 0.18))
    bench_vals = sens["benchmark_total_return"].dropna().tolist()
    if len(bench_vals) == 2:
        assert abs(bench_vals[0] - bench_vals[1]) < 1e-9


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


def _make_result(n: int = 12, port_monthly: float = 0.02,
                 bench_monthly: float = 0.008) -> BacktestResult:
    """Build a synthetic BacktestResult with n monthly observations."""
    monthly = [
        MonthlyResult(
            date=_DATES[i],
            portfolio_return=port_monthly,
            benchmark_return=bench_monthly,
            n_picks=3,
            tickers=["G1", "G2", "G3"],
        )
        for i in range(n)
    ]
    hr_returns = [0.15] * (n * 2) + [-0.05] * (n // 2)  # ~80% hit rate
    return BacktestResult(monthly_results=monthly, hit_rate_returns=hr_returns,
                          bt_config=BacktestConfig())


def test_generate_report_creates_file(tmp_path: Path) -> None:
    result = _make_result()
    sensitivity = pd.DataFrame([{"roic_min": 0.15, "portfolio_total_return": 0.20,
                                  "benchmark_total_return": 0.10, "sharpe_ratio": 1.2,
                                  "max_drawdown": -0.05, "hit_rate": 0.70,
                                  "n_test_months": 12, "avg_picks": 3.0}])
    out = tmp_path / "report.md"
    generate_report(result, sensitivity, out)
    assert out.exists()


def test_generate_report_creates_parent_directory(tmp_path: Path) -> None:
    result = _make_result()
    sensitivity = pd.DataFrame()
    out = tmp_path / "nested" / "deep" / "report.md"
    generate_report(result, sensitivity, out)
    assert out.exists()


def test_generate_report_contains_performance_summary(tmp_path: Path) -> None:
    result = _make_result()
    out = tmp_path / "r.md"
    generate_report(result, pd.DataFrame(), out)
    content = out.read_text()
    assert "Performance Summary" in content
    assert "Total return" in content
    assert "Sharpe" in content


def test_generate_report_contains_walkforward_params(tmp_path: Path) -> None:
    result = _make_result()
    out = tmp_path / "r.md"
    generate_report(result, pd.DataFrame(), out)
    content = out.read_text()
    assert "Walk-forward" in content
    assert "Training window" in content


def test_generate_report_outperformance_verdict(tmp_path: Path) -> None:
    result = _make_result(port_monthly=0.03, bench_monthly=0.005)
    out = tmp_path / "r.md"
    generate_report(result, pd.DataFrame(), out)
    content = out.read_text()
    assert "outperformed" in content.lower()


def test_generate_report_underperformance_verdict(tmp_path: Path) -> None:
    result = _make_result(port_monthly=0.001, bench_monthly=0.03)
    out = tmp_path / "r.md"
    generate_report(result, pd.DataFrame(), out)
    content = out.read_text()
    assert "underperformed" in content.lower()


def test_generate_report_with_empty_sensitivity(tmp_path: Path) -> None:
    result = _make_result()
    out = tmp_path / "r.md"
    generate_report(result, pd.DataFrame(), out)
    content = out.read_text()
    assert "No sensitivity data" in content


def test_generate_report_contains_data_integrity_note(tmp_path: Path) -> None:
    result = _make_result()
    out = tmp_path / "r.md"
    generate_report(result, pd.DataFrame(), out)
    assert "FMP" in out.read_text()


def test_generate_report_empty_monthly_results_does_not_crash(tmp_path: Path) -> None:
    result = BacktestResult(monthly_results=[], hit_rate_returns=[], bt_config=BacktestConfig())
    out = tmp_path / "r.md"
    generate_report(result, pd.DataFrame(), out)
    assert out.exists()


# ---------------------------------------------------------------------------
# ticker_contribution_analysis
# ---------------------------------------------------------------------------


def _make_result_with_returns(
    tickers_by_month: list[dict[str, float]],
) -> BacktestResult:
    """Build BacktestResult where each monthly result has ticker_returns pre-set."""
    monthly = [
        MonthlyResult(
            date=_DATES[i],
            portfolio_return=float(sum(v for v in tr.values()) / max(len(tr), 1)),
            benchmark_return=0.0,
            n_picks=len(tr),
            tickers=list(tr.keys()),
            ticker_returns=tr,
        )
        for i, tr in enumerate(tickers_by_month)
    ]
    return BacktestResult(monthly_results=monthly, hit_rate_returns=[], bt_config=BacktestConfig())


def test_ticker_contribution_analysis_empty_result() -> None:
    result = BacktestResult(monthly_results=[], hit_rate_returns=[], bt_config=BacktestConfig())
    df = ticker_contribution_analysis(result)
    assert df.empty


def test_ticker_contribution_analysis_sums_correctly() -> None:
    result = _make_result_with_returns([
        {"AAPL": 0.10, "MSFT": 0.05},
        {"AAPL": 0.08, "GOOG": 0.03},
    ])
    df = ticker_contribution_analysis(result)
    aapl = df[df["ticker"] == "AAPL"].iloc[0]
    assert aapl["pick_count"] == 2
    assert aapl["total_contribution"] == pytest.approx(0.18)
    assert aapl["avg_return_pct"] == pytest.approx(9.0)


def test_ticker_contribution_analysis_sorted_descending() -> None:
    result = _make_result_with_returns([
        {"LOW": 0.01, "HIGH": 0.20, "MID": 0.10},
    ])
    df = ticker_contribution_analysis(result)
    assert df.iloc[0]["ticker"] == "HIGH"
    assert df.iloc[1]["ticker"] == "MID"
    assert df.iloc[2]["ticker"] == "LOW"


def test_ticker_contribution_analysis_pick_counts() -> None:
    result = _make_result_with_returns([
        {"A": 0.05},
        {"A": 0.03, "B": 0.02},
        {"B": 0.01},
    ])
    df = ticker_contribution_analysis(result)
    counts = dict(zip(df["ticker"], df["pick_count"]))
    assert counts["A"] == 2
    assert counts["B"] == 2


def test_run_backtest_populates_ticker_returns() -> None:
    fund_dates = _DATES[:6]
    price_dates = _DATES[:8]
    bt_cfg = BacktestConfig(train_months=3, top_n=5, hit_rate_months=2)
    funds = _make_fundamentals(fund_dates, ["G1", "G2"], ["B1"])
    prices = _make_prices(price_dates, {"G1": 0.02, "G2": 0.02, "B1": 0.0, "SP500": 0.005})
    result = run_backtest(funds, prices, _cfg(), bt_cfg)
    for m in result.monthly_results:
        assert isinstance(m.ticker_returns, dict)
        for t, r in m.ticker_returns.items():
            assert t in m.tickers
            assert isinstance(r, float)


# ---------------------------------------------------------------------------
# generate_ticker_contribution
# ---------------------------------------------------------------------------


def test_generate_ticker_contribution_creates_file(tmp_path: Path) -> None:
    result = _make_result_with_returns([
        {"AAPL": 0.10, "MSFT": 0.05},
        {"AAPL": 0.08, "GOOG": -0.02},
    ])
    out = tmp_path / "contrib.md"
    generate_ticker_contribution(result, out, roic_threshold=0.15)
    assert out.exists()


def test_generate_ticker_contribution_empty_result(tmp_path: Path) -> None:
    result = BacktestResult(monthly_results=[], hit_rate_returns=[], bt_config=BacktestConfig())
    out = tmp_path / "contrib.md"
    generate_ticker_contribution(result, out)
    assert out.exists()


def test_generate_ticker_contribution_contains_key_sections(tmp_path: Path) -> None:
    tickers = {f"T{i:02d}": 0.01 * i for i in range(1, 25)}
    result = _make_result_with_returns([tickers])
    out = tmp_path / "contrib.md"
    generate_ticker_contribution(result, out)
    content = out.read_text()
    assert "Top 5 concentration" in content
    assert "Top 20 contributors" in content
    assert "Full contribution table" in content


def test_generate_ticker_contribution_top5_pct_in_output(tmp_path: Path) -> None:
    result = _make_result_with_returns([
        {"BIG": 0.50, "MED": 0.10, "SML": 0.05, "XS1": 0.01, "XS2": 0.01, "XS3": 0.01},
    ])
    out = tmp_path / "contrib.md"
    generate_ticker_contribution(result, out)
    content = out.read_text()
    assert "%" in content
    assert "BIG" in content
