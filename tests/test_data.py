"""Tests for financial statement extraction and normalization."""

import numpy as np
import pandas as pd
import pytest

import core.data as core_data
from core.data import (
    _compute_rsi,
    _info_is_usable,
    build_earnings_yield_history,
    extract_financial_values,
    normalize_debt_to_equity,
    percentile_rank_in_history,
)


def test_normalize_debt_to_equity_percentage():
    assert normalize_debt_to_equity(286.236) == 2.86236


def test_normalize_debt_to_equity_ratio_unchanged():
    assert normalize_debt_to_equity(1.5) == 1.5
    assert normalize_debt_to_equity(0.8) == 0.8


def test_normalize_debt_to_equity_none():
    assert normalize_debt_to_equity(None) is None


def test_extract_financial_values_uses_newest_non_null():
    df = pd.DataFrame(
        {
            "2025-12-31": [100.0],
            "2024-12-31": [80.0],
        },
        index=["Total Assets"],
    )
    latest, prior, warnings = extract_financial_values(["Total Assets"], df)
    assert latest == 100.0
    assert prior == 80.0
    assert warnings == []


def test_extract_financial_values_warns_when_newest_column_empty():
    df = pd.DataFrame(
        {
            "2025-12-31": [None],
            "2024-12-31": [80.0],
            "2023-12-31": [70.0],
        },
        index=["Total Assets"],
    )
    latest, prior, warnings = extract_financial_values(["Total Assets"], df)
    assert latest == 80.0
    assert prior == 70.0
    assert len(warnings) == 1
    assert "most recent period" in warnings[0]


def test_extract_financial_values_sorts_unordered_columns():
    df = pd.DataFrame(
        {
            "2023-12-31": [70.0],
            "2025-12-31": [100.0],
            "2024-12-31": [80.0],
        },
        index=["Net Income"],
    )
    latest, prior, _ = extract_financial_values(["Net Income"], df)
    assert latest == 100.0
    assert prior == 80.0


def test_repurchase_does_not_use_purchase_of_business():
    df = pd.DataFrame(
        {
            "2025-12-31": [-5000000000.0],
            "2024-12-31": [-1000000000.0],
        },
        index=["Purchase Of Business"],
    )
    latest, _, _ = extract_financial_values(
        [
            "Repurchase Of Capital Stock",
            "Common Stock Payments",
            "Repurchase Of Common Stock",
            "Repurchase Of Stock",
        ],
        df,
    )
    assert latest is None


def test_compute_rsi_oversold():
    """Declining prices should yield RSI below 50."""
    n = 30
    closes = 100.0 * np.exp(-np.linspace(0, 0.15, n))
    hist = pd.DataFrame({"Close": closes})
    rsi = _compute_rsi(hist, period=14)
    assert rsi is not None
    assert rsi < 50


def test_compute_rsi_insufficient_data():
    hist = pd.DataFrame({"Close": [100.0, 101.0, 99.0]})
    assert _compute_rsi(hist, period=14) is None


def test_info_is_usable_requires_price_and_name():
    assert not _info_is_usable({})
    assert not _info_is_usable({"longName": "Amazon.com, Inc."})
    assert not _info_is_usable({"currentPrice": 100.0})
    assert _info_is_usable({"longName": "Amazon.com, Inc.", "currentPrice": 100.0})


def _fake_ey_statements(now: pd.Timestamp):
    """Annual FY EBIT=100; quarterly EBIT=25 (TTM=100); shares=10; no debt/cash."""
    annual_cols = [str((now - pd.DateOffset(years=k)).date()) for k in (1, 2, 3)]
    a_income = pd.DataFrame({c: [100.0] for c in annual_cols}, index=["EBIT"])
    a_balance = pd.DataFrame(
        {c: [10.0, 0.0, 0.0] for c in annual_cols},
        index=["Ordinary Shares Number", "Total Debt", "Cash And Cash Equivalents"],
    )

    q_ends = pd.date_range(end=now, periods=6, freq="QE")
    q_cols = [str(d.date()) for d in q_ends]
    q_income = pd.DataFrame({c: [25.0] for c in q_cols}, index=["EBIT"])
    q_balance = pd.DataFrame(
        {c: [10.0, 0.0, 0.0] for c in q_cols},
        index=["Ordinary Shares Number", "Total Debt", "Cash And Cash Equivalents"],
    )
    return (
        {"income": a_income, "balance": a_balance, "cashflow": pd.DataFrame()},
        {"income": q_income, "balance": q_balance, "cashflow": pd.DataFrame()},
    )


def test_earnings_yield_history_is_annualized(monkeypatch):
    """
    Regression: history points must be in annualized EBIT/EV units, matching the
    current EY. The old builder used single-quarter EBIT, so the current annual
    EY sat above ~all of its own history (percentile pinned at ~100).
    """
    now = pd.Timestamp.now().normalize()
    annual, quarterly = _fake_ey_statements(now)
    dates = pd.date_range(end=now, periods=4 * 365, freq="D")
    prices = pd.DataFrame({"Close": np.full(len(dates), 100.0)}, index=dates)

    monkeypatch.setattr(core_data, "fetch_financials", lambda t: annual)
    monkeypatch.setattr(core_data, "fetch_quarterly_financials", lambda t: quarterly)
    monkeypatch.setattr(core_data, "fetch_price_history", lambda t, **kw: prices)
    monkeypatch.setattr(core_data, "_read_cache", lambda *a, **kw: None)
    monkeypatch.setattr(core_data, "_write_cache", lambda *a, **kw: None)

    history = build_earnings_yield_history("FAKE")

    # EV = 100 * 10 = 1000; annualized EBIT = 100 → every point is 0.10.
    assert len(history) >= 4
    for point in history:
        assert point == pytest.approx(0.10, rel=1e-6)

    # Current annual EY equals the history level → percentile must not saturate
    # relative to a like-for-like history. With the old quarterly-unit bug the
    # history was ~0.025 and 0.10 ranked at the 100th percentile by a 4x margin.
    current_ey = 100.0 / 1000.0
    pct = percentile_rank_in_history(current_ey, history)
    assert pct is not None
    # A slightly cheaper-than-history stock ranks low, not at 100.
    assert percentile_rank_in_history(current_ey * 0.9, history) == 0.0
