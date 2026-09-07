"""Tests for ui.formatting helpers."""

from ui.formatting import fmt_large_number, gauge_score_label, ordinal


def test_fmt_large_number():
    assert "B" in fmt_large_number(1.2e9)
    assert fmt_large_number(None) == "N/A"


def test_gauge_label():
    assert gauge_score_label(80) == "Good"
    assert gauge_score_label(20) == "Weak"
    assert ordinal(21) == "21st"
