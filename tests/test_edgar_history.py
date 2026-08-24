"""Tests for EDGAR-backed 10y valuation history."""

from __future__ import annotations

import pandas as pd
import pytest

from core.edgar_history import (
    history_series_from_table,
    pick_best_metric,
)
from core.edgar_history import compute_valuation_vs_history_detail


def _closes(prices: dict[str, float]) -> pd.Series:
    idx = pd.to_datetime(list(prices.keys()))
    return pd.Series(list(prices.values()), index=idx).sort_index()


def _table() -> pd.DataFrame:
    rows = []
    for year, ebit, ocf, book, shares in [
        (2015, 80, 90, 400, 10),
        (2016, 90, 100, 420, 10),
        (2017, 100, 110, 440, 10),
        (2018, 110, 120, 460, 10),
        (2019, 120, 130, 480, 10),
        (2020, 70, 80, 500, 10),
        (2021, 130, 140, 520, 10),
        (2022, 140, 150, 540, 10),
        (2023, 150, 160, 560, 10),
        (2024, 160, 170, 580, 10),
    ]:
        rows.append(
            {
                "period": pd.Timestamp(f"{year}-12-31"),
                "ebit": ebit,
                "operating_cashflow": ocf,
                "book_equity": book,
                "shares_outstanding": shares,
                "total_debt": 0.0,
                "total_cash": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_history_series_from_table_builds_three_yields():
    table = _table()
    # Price tracks EBIT (so EY is roughly stable around 0.10).
    prices = {f"{year}-12-31": ebit * 10 / 10 for year, ebit in zip(range(2015, 2025), [
        80, 90, 100, 110, 120, 70, 130, 140, 150, 160
    ])}
    histories = history_series_from_table(table, _closes(prices), years=12)
    assert len(histories["earnings_yield"]) >= 8
    assert len(histories["ocf_yield"]) >= 8
    assert len(histories["book_to_market"]) >= 8
    assert all(0.05 < ey < 0.15 for ey in histories["earnings_yield"])


def test_pick_best_metric_prefers_ebit_when_it_tracks_price():
    table = _table()
    prices = {
        f"{year}-12-31": ebit * 1.0
        for year, ebit in zip(
            range(2015, 2025), [80, 90, 100, 110, 120, 70, 130, 140, 150, 160]
        )
    }
    closes = _closes(prices)
    histories = history_series_from_table(table, closes, years=12)
    metric, corr = pick_best_metric(table, closes, histories, years=12)
    assert metric == "earnings_yield"
    assert corr is not None and corr > 0.8


def test_valuation_detail_yahoo_fallback(monkeypatch):
    monkeypatch.setattr("core.edgar_history.load_period_table", lambda t: (pd.DataFrame(), "none"))
    monkeypatch.setattr(
        "core.edgar_history.fetch_price_history",
        lambda t, **kw: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "core.data.build_earnings_yield_history",
        lambda t, years=10: [0.05, 0.06, 0.07, 0.08, 0.09],
    )
    detail = compute_valuation_vs_history_detail("FAKE", 0.09)
    assert detail["source"] == "yahoo"
    assert detail["score"] == pytest.approx(100.0)
    assert detail["n_points"] == 5
