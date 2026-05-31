"""Tests for factor computations."""

import pandas as pd

from core.analysts import recommendation_period_shift
from core.factors import (
    compute_balance_sheet_strength,
    compute_earnings_revisions,
    compute_piotroski_f_score,
    compute_shareholder_yield,
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
