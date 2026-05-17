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


@dataclass(frozen=True)
class ScoreWeights:
    """Layer 2 composite score weights. Must sum to 1.0."""

    quality: float = 0.60
    valuation: float = 0.40

    def __post_init__(self) -> None:
        if abs(self.quality + self.valuation - 1.0) > 1e-9:
            raise ValueError("ScoreWeights must sum to 1.0")


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


def load_config() -> CrucibleConfig:
    """Load configuration from environment variables, falling back to defaults."""
    return CrucibleConfig()
