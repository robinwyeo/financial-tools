"""Config helpers."""

import pytest

from core.config import get_bargain_weights, get_thresholds, load_config


def test_get_thresholds_includes_bargain_min():
    thresholds = get_thresholds(load_config())
    assert "bargain_min" in thresholds
    assert thresholds["bargain_min"] == 43.2


def test_get_bargain_weights_from_config():
    weights = get_bargain_weights(load_config())
    assert weights["margin_of_safety"] > 0
    assert pytest.approx(sum(weights.values()), rel=1e-3) == 1.0
