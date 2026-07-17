"""Config helpers."""

import pytest

from core.config import get_bargain_weights, get_factor_weights, get_thresholds, load_config
from core.factors import FACTOR_SCORE_COLUMNS


def test_get_thresholds_includes_bargain_min():
    thresholds = get_thresholds(load_config())
    assert "bargain_min" in thresholds
    assert thresholds["bargain_min"] == pytest.approx(50.0, abs=30.0)
    assert thresholds.get("require_implied_upside") is False


def test_get_bargain_weights_three_components():
    weights = get_bargain_weights(load_config())
    assert set(weights.keys()) == {
        "margin_of_safety",
        "valuation_vs_history",
        "discount_52w",
    }
    assert weights["margin_of_safety"] > 0
    assert weights["valuation_vs_history"] > 0
    assert pytest.approx(sum(weights.values()), rel=1e-3) == 1.0


def test_get_bargain_weights_no_removed_components():
    weights = get_bargain_weights(load_config())
    assert "discount_ath" not in weights
    assert "analyst_upside" not in weights
    assert "rsi_oversold" not in weights


def test_get_factor_weights_eight_groups():
    weights = get_factor_weights(load_config())
    expected_groups = {
        "value", "garp", "quality", "balance_sheet",
        "momentum", "low_volatility", "capital_discipline", "earnings_revisions",
    }
    assert set(weights.keys()) == expected_groups


def test_get_factor_weights_match_factor_score_columns():
    """Weights keys must align with FACTOR_SCORE_COLUMNS group names."""
    weights = get_factor_weights(load_config())
    assert set(weights.keys()) == set(FACTOR_SCORE_COLUMNS.keys())


def test_get_factor_weights_sum_to_approximately_one():
    weights = get_factor_weights(load_config())
    assert pytest.approx(sum(weights.values()), rel=1e-3) == 1.0


def test_evidence_based_priors_favor_quality_and_value():
    weights = get_factor_weights(load_config())
    assert weights["quality"] >= 0.20
    assert weights["value"] >= 0.20
    assert weights["garp"] < weights["quality"]
