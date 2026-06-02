"""Config helpers."""

from core.config import get_thresholds, load_config


def test_get_thresholds_includes_bargain_min():
    thresholds = get_thresholds(load_config())
    assert "bargain_min" in thresholds
    assert thresholds["bargain_min"] == 50.0
