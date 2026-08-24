"""Tests for factor computations."""

import pandas as pd

from core.analysts import recommendation_period_shift
from core.factors import (
    compute_balance_sheet_strength,
    compute_piotroski_f_score,
    compute_shareholder_yield,
    compute_graham_value,
    FACTOR_SCORE_COLUMNS,
)


def test_piotroski_financial_strength_normalized():
    raw = {
        "net_income": 100,
        "total_assets": 1000,
        "operating_cashflow": 120,
        "net_income_prior": 80,
        "total_assets_prior": 900,
    }
    result = compute_piotroski_f_score(raw)
    assert result["piotroski_f_score"] == 4.0
    assert result["financial_strength"] == 9.0


def test_balance_sheet_strength_uses_ratio_debt_to_equity():
    raw = {
        "total_cash": 100,
        "total_debt": 200,
        "market_cap": 1000,
        "debt_to_equity": 2.86236,
    }
    result = compute_balance_sheet_strength(raw)
    assert result["low_leverage"] == 1.0 / (1.0 + 2.86236)
    # balance_sheet_strength composite no longer returned (moved to scoring layer)
    assert "net_cash_to_mcap" in result
    assert "low_leverage" in result


def test_shareholder_yield_ignores_acquisition_cashflow():
    raw = {
        "market_cap": 1_000_000,
        "dividends_paid": -50_000,
        "repurchase_of_stock": None,
    }
    result = compute_shareholder_yield(raw)
    assert result["shareholder_yield"] == 0.05
    assert result["net_buybacks"] is None


def test_estimate_revisions_group_is_zacks_style():
    """Live revision factor uses estimate-table sub-signals, not keyword scraping."""
    assert "estimate_revisions" in FACTOR_SCORE_COLUMNS
    assert FACTOR_SCORE_COLUMNS["estimate_revisions"] == [
        "revision_agreement",
        "revision_magnitude",
        "earnings_surprise",
    ]
    all_sub_cols = [col for cols in FACTOR_SCORE_COLUMNS.values() for col in cols]
    assert "earnings_revisions" not in all_sub_cols


def test_graham_ratio_not_in_composite_value_group():
    """graham_ratio drives the bargain score; keeping it out of the value group
    prevents the same signal from being double-counted across both Buy gates."""
    assert "graham_ratio" not in FACTOR_SCORE_COLUMNS["value"]
    all_sub_cols = [col for cols in FACTOR_SCORE_COLUMNS.values() for col in cols]
    assert "graham_ratio" not in all_sub_cols


def test_recommendation_period_shift_detects_upgrades():
    recs = pd.DataFrame(
        [
            {"period": "0m", "strongBuy": 6, "buy": 10, "hold": 4, "sell": 0, "strongSell": 0},
            {"period": "-1m", "strongBuy": 3, "buy": 8, "hold": 6, "sell": 1, "strongSell": 0},
        ]
    )
    revision_score, upgrades, downgrades = recommendation_period_shift(recs)
    assert revision_score is not None
    assert revision_score > 0
    assert upgrades > 0
    assert downgrades == 0


def test_factor_score_columns_are_lists():
    """All groups in FACTOR_SCORE_COLUMNS should map to non-empty lists of column names."""
    for family, cols in FACTOR_SCORE_COLUMNS.items():
        assert isinstance(cols, list), f"{family} should map to a list"
        assert len(cols) >= 1, f"{family} list is empty"
        for col in cols:
            assert isinstance(col, str), f"{family}: {col!r} should be a string"


def test_factor_score_columns_no_duplicates():
    """No sub-signal column should appear in more than one factor group."""
    all_cols: list[str] = []
    for cols in FACTOR_SCORE_COLUMNS.values():
        all_cols.extend(cols)
    assert len(all_cols) == len(set(all_cols)), "Duplicate sub-signal columns detected"


def test_graham_value_excludes_current_ratio_from_composite():
    """current_ratio should be returned for display but graham_value composite is gone."""
    raw = {
        "price": 50.0,
        "trailing_eps": 3.0,
        "book_value": 20.0,
        "current_ratio_info": 2.5,
        "current_assets": None,
        "current_liabilities": None,
    }
    result = compute_graham_value(raw)
    assert result["graham_ratio"] is not None
    assert result["current_ratio"] == 2.5
    # The old graham_value composite (avg of ratio + current_ratio) is gone
    assert "graham_value" not in result


def test_downside_protection_not_in_factor_score_columns():
    """downside_protection was removed from composite scoring (0.75 corr with momentum)."""
    all_sub_cols = [col for cols in FACTOR_SCORE_COLUMNS.values() for col in cols]
    assert "downside_protection" not in all_sub_cols
    assert "downside_protection" not in FACTOR_SCORE_COLUMNS
