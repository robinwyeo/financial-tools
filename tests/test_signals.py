"""Tests for short-interest flags and the uncertainty badge."""

import pytest

from core.signals import compute_short_interest, compute_uncertainty


def test_short_interest_flag_on_days_to_cover():
    out = compute_short_interest({"short_ratio": 6.2, "short_percent_of_float": 0.04})
    assert out["high_short_interest"] is True
    assert out["short_ratio"] == pytest.approx(6.2)


def test_short_interest_flag_on_float_percent():
    out = compute_short_interest({"short_ratio": 2.0, "short_percent_of_float": 12.0})
    assert out["short_percent_of_float"] == pytest.approx(0.12)
    assert out["high_short_interest"] is True


def test_short_interest_quiet_when_low():
    out = compute_short_interest({"short_ratio": 1.5, "short_percent_of_float": 0.03})
    assert out["high_short_interest"] is False


def test_uncertainty_low_when_clean():
    out = compute_uncertainty(
        factor_coverage_pct=100.0,
        volatility_12m=0.18,
        target_high=110,
        target_low=90,
        target_mean=100,
    )
    assert out["label"] == "Low"
    assert out["threshold_bump"] == 0
    assert out["estimate_dispersion"] == pytest.approx(0.20)


def test_uncertainty_high_widens_hurdle():
    out = compute_uncertainty(
        factor_coverage_pct=60.0,
        volatility_12m=0.50,
        target_high=160,
        target_low=80,
        target_mean=100,
    )
    assert out["label"] == "High"
    assert out["points"] == 3
    assert out["threshold_bump"] == pytest.approx(6.0)


def test_uncertainty_medium_label():
    out = compute_uncertainty(
        factor_coverage_pct=100.0,
        volatility_12m=0.18,
        target_high=160,
        target_low=80,
        target_mean=100,
    )
    assert out["label"] == "Medium"
    assert out["points"] == 1
    assert out["threshold_bump"] == pytest.approx(3.0)
