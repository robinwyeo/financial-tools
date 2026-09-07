"""Tests for EDGAR-backed 10y valuation history."""

from __future__ import annotations

import pandas as pd
import pytest

from core.edgar_history import (
    ev_ebit_history,
    history_series_from_table,
    p_oe_history,
    compute_valuation_vs_history_detail,
)
from core.fundamentals import Fundamentals


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
    prices = {f"{year}-12-31": ebit * 10 / 10 for year, ebit in zip(range(2015, 2025), [
        80, 90, 100, 110, 120, 70, 130, 140, 150, 160
    ])}
    histories = history_series_from_table(table, _closes(prices), years=12)
    assert len(histories["earnings_yield"]) >= 8
    assert len(histories["ocf_yield"]) >= 8
    assert len(histories["book_to_market"]) >= 8
    assert all(0.05 < ey < 0.15 for ey in histories["earnings_yield"])


def test_ev_ebit_and_p_oe_histories_are_deterministic():
    idx = pd.to_datetime([f"{y}-12-31" for y in range(2015, 2025)])
    annual = pd.DataFrame(
        {
            "ebit": [80, 90, 100, 110, 120, 70, 130, 140, 150, 160],
            "operating_cashflow": [90, 100, 110, 120, 130, 80, 140, 150, 160, 170],
            "capex": [0] * 10,
            "sbc": [0] * 10,
            "debt": [0] * 10,
            "cash": [0] * 10,
            "shares_diluted": [10] * 10,
            "revenue": [200] * 10,
        },
        index=idx,
    )
    fund = Fundamentals("X", annual, annual.iloc[-1], annual.iloc[-1], "edgar", idx[-1].date())
    prices = {f"{y}-12-31": float(ebit) for y, ebit in zip(range(2015, 2025), annual["ebit"])}
    closes = _closes(prices)
    ey = ev_ebit_history(fund, closes, years=12)
    oe = p_oe_history(fund, closes, years=12)
    assert len(ey) == 10
    # price = ebit, shares = 10 → mcap = 10*ebit, EV = mcap, EBIT/EV = 0.1
    assert all(v == pytest.approx(0.1) for v in ey)
    assert len(oe) == 10


def test_valuation_detail_yahoo_fallback(monkeypatch):
    monkeypatch.setattr("core.edgar_history.fetch_price_history", lambda t, **kw: pd.DataFrame())
    monkeypatch.setattr("core.fundamentals.get_fundamentals", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no fund")))
    monkeypatch.setattr("core.data.build_earnings_yield_history", lambda t, **kw: [0.05, 0.06, 0.07, 0.08, 0.09])
    detail = compute_valuation_vs_history_detail("FAKE", 0.09)
    assert "correlation" not in detail
    assert detail["score"] is not None
    assert detail["source"] == "yahoo"
