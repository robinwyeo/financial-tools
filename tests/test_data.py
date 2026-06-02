"""Tests for financial statement extraction and normalization."""

import numpy as np
import pandas as pd

from core.data import (
    _compute_rsi,
    _info_is_usable,
    extract_financial_values,
    normalize_debt_to_equity,
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
