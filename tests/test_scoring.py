"""Tests for cross-sectional scoring."""

import numpy as np
import pandas as pd
import pytest

from core.factors import FACTOR_SCORE_COLUMNS
from core.scoring import _composite_and_coverage, score_universe_df


def _minimal_config() -> dict:
    weights = {family: 1.0 / len(FACTOR_SCORE_COLUMNS) for family in FACTOR_SCORE_COLUMNS}
    return {"factor_weights": weights, "universe": {"sector_scoring": False}}


def test_composite_excludes_missing_factors():
    weights = {"value": 0.5, "momentum": 0.5}
    row = pd.Series({"pct_value": 80.0, "pct_momentum": np.nan})
    composite, coverage = _composite_and_coverage(row, weights)
    assert composite == 80.0
    assert coverage == 50.0


def test_composite_coverage_zero_when_all_missing():
    weights = {"value": 0.5, "momentum": 0.5}
    row = pd.Series({"pct_value": np.nan, "pct_momentum": np.nan})
    composite, coverage = _composite_and_coverage(row, weights)
    assert composite is None
    assert coverage == 0.0


def test_score_universe_df_adds_factor_coverage():
    df = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "sector": "Tech",
                "value_composite": 0.1,
                "momentum_12_1": 0.2,
                "quality_composite": 0.3,
                "low_volatility": 0.4,
                "investment": -0.1,
                "earnings_revisions": 1.0,
                "financial_strength": 7.0,
                "garp": 1.5,
                "balance_sheet_strength": 0.2,
                "graham_value": 1.1,
                "downside_protection": 0.3,
                "earnings_quality": 0.01,
                "shareholder_yield": 0.04,
                "roic": 0.15,
                "altman_z": 3.0,
            },
            {
                "ticker": "BBB",
                "sector": "Tech",
                "value_composite": 0.2,
                "momentum_12_1": 0.1,
                "quality_composite": 0.2,
                "low_volatility": 0.3,
                "investment": -0.2,
                "earnings_revisions": 0.5,
                "financial_strength": 6.0,
                "garp": 1.0,
                "balance_sheet_strength": 0.1,
                "graham_value": 0.9,
                "downside_protection": 0.2,
                "earnings_quality": 0.02,
                "shareholder_yield": 0.03,
                "roic": 0.10,
                "altman_z": 2.5,
            },
            {
                "ticker": "CCC",
                "sector": "Tech",
                "value_composite": 0.15,
                "momentum_12_1": 0.15,
                "quality_composite": 0.25,
                "low_volatility": 0.35,
                "investment": -0.15,
                "earnings_revisions": 0.75,
                "financial_strength": 6.5,
                "garp": 1.25,
                "balance_sheet_strength": 0.15,
                "graham_value": 1.0,
                "downside_protection": 0.25,
                "earnings_quality": 0.015,
                "shareholder_yield": 0.035,
                "roic": 0.12,
                "altman_z": 2.75,
            },
        ]
    )
    scored = score_universe_df(df, _minimal_config(), group_col=None)
    assert "factor_coverage_pct" in scored.columns
    assert scored["factor_coverage_pct"].iloc[0] == pytest.approx(100.0)
