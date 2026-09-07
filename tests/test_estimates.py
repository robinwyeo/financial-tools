"""Tests for Zacks-style revision factors."""

import pandas as pd
import pytest

from core.estimates import compute_revision_factors


def test_revision_factors_from_tables():
    trend = pd.DataFrame(
        {
            "current": [2.20, 2.40],
            "7daysAgo": [2.15, 2.35],
            "30daysAgo": [2.10, 2.30],
            "60daysAgo": [2.05, 2.25],
            "90daysAgo": [2.00, 2.20],
        },
        index=["0y", "+1y"],
    )
    revisions = pd.DataFrame(
        {
            "upLast30days": [4, 3],
            "downLast30days": [1, 1],
        },
        index=["0y", "+1y"],
    )
    history = pd.DataFrame(
        {"surprisePercent": [2.0, 5.0, 8.0]},
        index=["-2q", "-1q", "0q"],
    )
    out = compute_revision_factors(
        {
            "estimate_tables": {
                "eps_trend": trend,
                "eps_revisions": revisions,
                "earnings_history": history,
            }
        }
    )
    assert out["revision_agreement"] == pytest.approx((4 / 5 + 3 / 4) / 2)
    assert out["revision_magnitude"] == pytest.approx(((2.20 - 2.00) / 2.00 + (2.40 - 2.20) / 2.20) / 2)
    assert out["earnings_surprise"] == pytest.approx(0.08)
    assert out["eps_trend_sparkline"] == pytest.approx([2.00, 2.05, 2.10, 2.15, 2.20])


def test_revision_factors_empty_without_tables():
    out = compute_revision_factors({"ticker": "X"})
    assert out["revision_agreement"] is None
    assert out["revision_magnitude"] is None
    assert out["earnings_surprise"] is None
    assert out["eps_trend_sparkline"] == []


def test_surprise_already_a_fraction():
    history = pd.DataFrame({"surprisePercent": [0.05]}, index=["0q"])
    out = compute_revision_factors({"estimate_tables": {"earnings_history": history}})
    assert out["earnings_surprise"] == pytest.approx(0.05)


def test_revision_missing_columns_returns_none():
    trend = pd.DataFrame({"foo": [1]}, index=["0y"])
    revisions = pd.DataFrame({"bar": [1]}, index=["0y"])
    history = pd.DataFrame({"baz": [1]}, index=["0q"])
    out = compute_revision_factors(
        {"estimate_tables": {"eps_trend": trend, "eps_revisions": revisions, "earnings_history": history}}
    )
    assert out["revision_agreement"] is None
    assert out["revision_magnitude"] is None
    assert out["earnings_surprise"] is None
