"""Tests for factor computations."""

import pandas as pd

from core.analysts import recommendation_period_shift
from core.factors import (
    compute_balance_sheet_strength,
    compute_earnings_revisions,
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


def test_earnings_revisions_period_format():
    recs = pd.DataFrame(
        [
            {"period": "0m", "strongBuy": 5, "buy": 10, "hold": 4, "sell": 0, "strongSell": 0},
            {"period": "-1m", "strongBuy": 3, "buy": 8, "hold": 6, "sell": 1, "strongSell": 0},
        ]
    )
    raw = {
        "recommendations": recs,
        "price": 100,
        "target_mean": 110,
    }
    result = compute_earnings_revisions(raw)
    assert result["earnings_revisions"] is not None


def test_earnings_revisions_no_target_upside_blend():
    """earnings_revisions must not include the analyst target price upside blend."""
    # Stock with no rec history but a high target mean — score should be None
    # (no double-counting with the analyst_upside good-buy gate).
    raw = {
        "recommendations": pd.DataFrame(),
        "price": 100,
        "target_mean": 150,  # 50% upside — must NOT flow into earnings_revisions
    }
    result = compute_earnings_revisions(raw)
    assert result["earnings_revisions"] is None


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
