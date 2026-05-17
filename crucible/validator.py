"""Pandera schemas for all major DataFrames. Enforced at cleaner.py output."""

from __future__ import annotations

import pandera as pa
from pandera.typing import Index, Series

PROCESSED_COLUMNS: list[str] = [
    "sector",
    "sub_industry",
    "currency",
    "p_e",
    "p_fcf",
    "ev_ebitda",
    "data_years",
    "insufficient_data",
    "roic_proxy_avg",
    "fcf_latest",
    "fcf_positive_years",
    "net_debt_ebitda",
    "revenue_growth_positive_years",
    "gross_margin_latest",
    "gross_margin_avg",
    "gross_margin_trend_slope",
]


class ProcessedFundamentalsSchema(pa.DataFrameModel):
    """Schema for the cleaned processed fundamentals DataFrame (one row per ticker)."""

    ticker: Index[str] = pa.Field(check_name=True)

    # From info snapshot
    sector: Series[str] = pa.Field(nullable=True)
    sub_industry: Series[str] = pa.Field(nullable=True)
    currency: Series[str] = pa.Field(nullable=True)
    p_e: Series[float] = pa.Field(nullable=True, ge=0)
    p_fcf: Series[float] = pa.Field(nullable=True)
    ev_ebitda: Series[float] = pa.Field(nullable=True)

    # Data availability
    data_years: Series[int] = pa.Field(nullable=False, ge=0, le=20)
    insufficient_data: Series[bool] = pa.Field(nullable=False)

    # ROIC proxy: Net Income / (Total Assets − Current Liabilities)
    roic_proxy_avg: Series[float] = pa.Field(nullable=True, ge=-10.0, le=100.0)

    # Free cash flow
    fcf_latest: Series[float] = pa.Field(nullable=True)
    fcf_positive_years: Series[float] = pa.Field(nullable=True, ge=0, le=10)

    # Leverage
    net_debt_ebitda: Series[float] = pa.Field(nullable=True, ge=-100.0, le=1000.0)

    # Revenue growth
    revenue_growth_positive_years: Series[float] = pa.Field(nullable=True, ge=0, le=10)

    # Gross margin
    gross_margin_latest: Series[float] = pa.Field(nullable=True, ge=-1.0, le=1.0)
    gross_margin_avg: Series[float] = pa.Field(nullable=True, ge=-1.0, le=1.0)
    gross_margin_trend_slope: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = True
        # coerce=True lets pandera normalise pandas dtype encodings (object vs
        # string[pyarrow], NoneType vs float NaN) without relaxing range or
        # nullability checks defined in pa.Field.
        coerce = True
