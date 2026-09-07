"""Tests for analyst aggregation."""

import pandas as pd
import pytest

from core.analysts import aggregate_analyst_data, recommendation_period_shift


def test_aggregate_analyst_period_format_weighted_mean():
    recs = pd.DataFrame(
        [
            {"period": "0m", "strongBuy": 2, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0},
        ]
    )
    raw = {
        "recommendations": recs,
        "price": 100,
        "target_mean": 120,
        "target_low": 90,
        "target_high": 140,
        "recommendation_key": "buy",
        "num_analysts": 2,
    }
    result = aggregate_analyst_data(raw)
    assert result["mean_rating_score"] == 5.0
    assert result["consensus_label"] == "Strong Buy"
    assert result["buy_count"] == 2
    assert result["implied_upside_pct"] == pytest.approx(20.0)


def test_aggregate_analyst_period_shift_counts():
    recs = pd.DataFrame(
        [
            {"period": "0m", "strongBuy": 6, "buy": 10, "hold": 4, "sell": 0, "strongSell": 0},
            {"period": "-1m", "strongBuy": 3, "buy": 8, "hold": 6, "sell": 1, "strongSell": 0},
        ]
    )
    raw = {"recommendations": recs, "price": 50, "target_mean": 55}
    result = aggregate_analyst_data(raw)
    assert result["recent_upgrades"] > 0
    assert result["recent_downgrades"] == 0


def test_recommendation_period_shift_downgrade():
    recs = pd.DataFrame(
        [
            {"period": "0m", "strongBuy": 1, "buy": 2, "hold": 10, "sell": 3, "strongSell": 1},
            {"period": "-1m", "strongBuy": 4, "buy": 8, "hold": 5, "sell": 0, "strongSell": 0},
        ]
    )
    _, upgrades, downgrades = recommendation_period_shift(recs)
    assert downgrades > 0
    assert upgrades == 0


def test_recommendation_key_only_fallback():
    result = aggregate_analyst_data(
        {"recommendation_key": "buy", "price": 50, "target_mean": 60, "num_analysts": 8}
    )
    assert result["consensus_label"] == "Buy"
    assert result["buy_count"] == 1
    assert result["implied_upside_pct"] == pytest.approx(20.0)


def test_firm_grade_table():
    recs = pd.DataFrame(
        [
            {"firm": "GS", "toGrade": "Overweight", "action": "up", "date": "2026-01-01"},
            {"firm": "MS", "toGrade": "Hold", "action": "init", "date": "2026-01-02"},
            {"firm": "JPM", "toGrade": "Sell", "action": "down", "date": "2026-01-03"},
        ]
    )
    result = aggregate_analyst_data({"recommendations": recs, "price": 10, "target_mean": 12})
    assert result["buy_count"] == 1
    assert result["hold_count"] == 1
    assert result["sell_count"] == 1
    assert result["recent_upgrades"] == 1
    assert result["recent_downgrades"] == 1
    assert any(a["firm"] == "GS" for a in result["recent_actions"])
