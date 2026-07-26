"""Tests for fund factor computation and fund cross-sectional scoring."""

import numpy as np
import pandas as pd

from core.config import get_fund_factor_weights
from core.data import normalize_expense_ratio, normalize_fund_yield
from core.fund_factors import FUND_FACTOR_SCORE_COLUMNS, compute_fund_factors
from core.scoring import score_universe_df


def _raw_fund(**overrides):
    base = {
        "ticker": "TEST",
        "expense_ratio": 0.002,
        "return_1y": 0.12,
        "return_3y": 0.10,
        "return_5y": 0.09,
        "momentum_12_1": 0.08,
        "volatility_12m": 0.15,
        "max_drawdown": -0.20,
        "distribution_yield": 0.015,
    }
    base.update(overrides)
    return base


def test_low_cost_inverts_expense_ratio():
    factors = compute_fund_factors(_raw_fund(expense_ratio=0.005))
    assert factors["low_cost"] == -0.005


def test_low_cost_none_when_expense_missing():
    factors = compute_fund_factors(_raw_fund(expense_ratio=None))
    assert factors["low_cost"] is None


def test_sharpe_is_return_over_volatility():
    factors = compute_fund_factors(_raw_fund(return_1y=0.12, volatility_12m=0.15))
    assert factors["sharpe_1y"] == 0.12 / 0.15


def test_sharpe_none_without_volatility():
    factors = compute_fund_factors(_raw_fund(volatility_12m=None))
    assert factors["sharpe_1y"] is None


def test_drawdown_protection_negates_magnitude():
    factors = compute_fund_factors(_raw_fund(max_drawdown=-0.30))
    assert factors["drawdown_protection"] == -0.30
    # Positive-sign drawdowns are treated as magnitudes too.
    factors = compute_fund_factors(_raw_fund(max_drawdown=0.30))
    assert factors["drawdown_protection"] == -0.30


def test_all_score_columns_are_computed():
    factors = compute_fund_factors(_raw_fund())
    for cols in FUND_FACTOR_SCORE_COLUMNS.values():
        for col in cols:
            assert col in factors
            assert factors[col] is not None


def test_fund_weights_default_and_config_override():
    defaults = get_fund_factor_weights({})
    assert set(defaults) == set(FUND_FACTOR_SCORE_COLUMNS)
    assert defaults["cost"] == 0.25
    custom = get_fund_factor_weights({"fund_factor_weights": {"cost": 0.5}})
    assert custom["cost"] == 0.5


def test_score_universe_df_with_fund_columns():
    rng = np.random.default_rng(7)
    n = 30
    df = pd.DataFrame(
        {
            "ticker": [f"F{i}" for i in range(n)],
            "category": ["Large Blend"] * 15 + ["Bond"] * 15,
            "low_cost": -rng.uniform(0.0003, 0.02, n),
            "return_3y": rng.normal(0.08, 0.05, n),
            "return_5y": rng.normal(0.08, 0.05, n),
            "sharpe_1y": rng.normal(0.6, 0.3, n),
            "low_volatility": rng.uniform(3, 12, n),
            "drawdown_protection": -rng.uniform(0.05, 0.4, n),
            "momentum_12_1": rng.normal(0.06, 0.1, n),
            "distribution_yield": rng.uniform(0.0, 0.05, n),
        }
    )
    weights = get_fund_factor_weights({})
    scored = score_universe_df(
        df,
        config={},
        group_col="category",
        factor_columns=FUND_FACTOR_SCORE_COLUMNS,
        weights=weights,
    )
    assert "composite" in scored.columns
    assert scored["composite"].notna().all()
    assert scored["composite"].between(0, 100).all()
    assert (scored["factor_coverage_pct"] == 100.0).all()
    for family in FUND_FACTOR_SCORE_COLUMNS:
        assert f"pct_{family}" in scored.columns


def test_normalize_expense_ratio_units():
    # annualReportExpenseRatio is a fraction; netExpenseRatio is percent points.
    assert normalize_expense_ratio({"annualReportExpenseRatio": 0.0004}) == 0.0004
    assert normalize_expense_ratio({"netExpenseRatio": 0.03}) == 0.0003
    assert normalize_expense_ratio({"annualReportExpenseRatio": 0.0004, "netExpenseRatio": 0.04}) == 0.0004
    # Zero is "unreported", not free.
    assert normalize_expense_ratio({"annualReportExpenseRatio": 0.0}) is None
    assert normalize_expense_ratio({}) is None


def test_normalize_fund_yield_units():
    import pytest

    assert normalize_fund_yield({"yield": 0.0107}) == 0.0107
    assert normalize_fund_yield({"dividendYield": 1.07}) == pytest.approx(0.0107)
    assert normalize_fund_yield({"dividendYield": 0.012}) == 0.012
    assert normalize_fund_yield({}) is None
