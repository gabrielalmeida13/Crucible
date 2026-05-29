"""Central configuration: thresholds, universe definitions, and score weights."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

UniverseID = Literal["SP500", "EUROPE_LARGE", "JAPAN_ADR", "XTB_FULL"]


@dataclass(frozen=True)
class FilterThresholds:
    """Layer 1 hard-rule thresholds. Change only with explicit developer instruction."""

    roic_min: float = 0.15
    fcf_positive_min_years: int = 4
    fcf_lookback_years: int = 5
    net_debt_ebitda_max: float = 3.0
    revenue_growth_positive_min_years: int = 3
    revenue_growth_lookback_years: int = 5
    gross_margin_min_slope: float = -0.005


@dataclass(frozen=True)
class ScoreWeights:
    """Layer 2 composite score weights. Must sum to 1.0."""

    quality: float = 0.60
    valuation: float = 0.30
    momentum: float = 0.10
    ml_score: float = 0.0  # non-zero only when ML model artifact is wired in

    def __post_init__(self) -> None:
        total = self.quality + self.valuation + self.momentum + self.ml_score
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"ScoreWeights must sum to 1.0 (got {total})")


@dataclass(frozen=True)
class Track2FilterThresholds:
    """Layer 1 hard-rule thresholds for Track 2 (Growth Inflection)."""

    revenue_growth_min_pct: float = 0.08
    gross_margin_min: float = 0.30
    fcf_positive_last2yr_min: int = 1
    net_debt_ebitda_soft_max: float = 8.0


@dataclass(frozen=True)
class Track2ScoreWeights:
    """Layer 2 composite score weights for Track 2. Must sum to 1.0."""

    growth_quality: float = 0.50
    momentum: float = 0.30
    valuation: float = 0.20

    def __post_init__(self) -> None:
        total = self.growth_quality + self.momentum + self.valuation
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Track2ScoreWeights must sum to 1.0 (got {total})")


@dataclass(frozen=True)
class Track3FilterThresholds:
    """Layer 1 hard-rule thresholds for Track 3 (Value Recovery)."""

    roic_proxy_min: float = 0.08
    p_fcf_vs_history_min: float = 1.0    # current P/FCF > 1 std below own 5yr avg
    fcf_positive_min_years: int = 2      # of last 5 years
    buyback_signal_min: float = 0.03     # > 3% net reduction in shares outstanding
    gm_recovery_change_min: float = 0.02  # gross_margin_yr1_change > 2pp


@dataclass(frozen=True)
class Track3ScoreWeights:
    """Layer 2 composite score weights for Track 3. Must sum to 1.0."""

    value: float = 0.50
    recovery_signal: float = 0.30
    balance_sheet: float = 0.20

    def __post_init__(self) -> None:
        total = self.value + self.recovery_signal + self.balance_sheet
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Track3ScoreWeights must sum to 1.0 (got {total})")


@dataclass(frozen=True)
class FXConfig:
    """FX conversion cost parameters."""

    conversion_penalty: float = -0.5


@dataclass(frozen=True)
class CrucibleConfig:
    """Root configuration object assembled from environment and defaults."""

    universe: UniverseID = field(
        default_factory=lambda: os.getenv("CRUCIBLE_UNIVERSE", "SP500")  # type: ignore[return-value]
    )
    db_path: str = field(
        default_factory=lambda: os.getenv("CRUCIBLE_DB_PATH", "data/crucible.db")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("CRUCIBLE_LOG_LEVEL", "INFO")
    )
    fmp_api_key: str = field(
        default_factory=lambda: os.getenv("CRUCIBLE_FMP_API_KEY", "")
    )
    account_currency: str = field(
        default_factory=lambda: os.getenv("CRUCIBLE_ACCOUNT_CURRENCY", "EUR")
    )
    filters: FilterThresholds = field(default_factory=FilterThresholds)
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)
    fx: FXConfig = field(default_factory=FXConfig)
    track2_filters: Track2FilterThresholds = field(default_factory=Track2FilterThresholds)
    track2_score_weights: Track2ScoreWeights = field(default_factory=Track2ScoreWeights)
    track3_filters: Track3FilterThresholds = field(default_factory=Track3FilterThresholds)
    track3_score_weights: Track3ScoreWeights = field(default_factory=Track3ScoreWeights)


def load_config() -> CrucibleConfig:
    """Load configuration from environment variables, falling back to defaults."""
    return CrucibleConfig()
