"""Tests for the unified fundamentals API."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from core.edgar_facts import facts_to_rows
from core.fundamentals import (
    cagr,
    get_fundamentals,
    owner_earnings,
    owner_earnings_per_share,
)
from tests.test_edgar_facts import _companyfacts_fixture


def test_owner_earnings_subtracts_capex_and_sbc():
    assert owner_earnings({"operating_cashflow": 100, "capex": -20, "sbc": 10}) == 70
    assert owner_earnings({"operating_cashflow": 100, "capex": 20, "sbc": 10}, subtract_sbc=False) == 80
    assert owner_earnings({"operating_cashflow": None, "capex": 20}) is None


def test_owner_earnings_per_share():
    assert owner_earnings_per_share(
        {"operating_cashflow": 100, "capex": 20, "sbc": 10, "shares_diluted": 10}
    ) == pytest.approx(7.0)


def test_cagr_five_year():
    idx = pd.date_range("2018-12-31", periods=7, freq="YE")
    s = pd.Series([100, 110, 121, 133, 146, 161, 177], index=idx)
    # 177/100 ** (1/6) because 6 year span with 7 points; function uses years=5 → start is 6th from end
    g = cagr(s, years=5)
    assert g == pytest.approx((177 / 110) ** (1 / 5) - 1, rel=1e-6)


def test_get_fundamentals_from_edgar_fixture(monkeypatch):
    facts = facts_to_rows(_companyfacts_fixture(), cik=123)
    facts["ticker"] = "FAKE"

    monkeypatch.setattr("core.fundamentals.fetch_companyfacts_for_ticker", lambda t, **k: facts)
    monkeypatch.setattr(
        "core.fundamentals.fetch_financials",
        lambda t: {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()},
    )
    monkeypatch.setattr(
        "core.fundamentals.fetch_ttm_financials",
        lambda t: {"income": pd.DataFrame(), "cashflow": pd.DataFrame(), "balance": pd.DataFrame()},
    )
    monkeypatch.setattr("core.fundamentals._read_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.fundamentals._write_cache", lambda *a, **k: None)

    fund = get_fundamentals("FAKE", as_of=date(2025, 12, 31), force=True)
    assert fund.source in {"edgar", "mixed"}
    assert len(fund.annual.dropna(how="all")) >= 2
    assert fund.annual["revenue"].dropna().iloc[-1] == pytest.approx(50.0)


def test_get_fundamentals_yahoo_fallback(monkeypatch):
    income = pd.DataFrame(
        {
            "2021-12-31": [80.0, 10.0],
            "2022-12-31": [90.0, 12.0],
            "2023-12-31": [100.0, 14.0],
            "2024-12-31": [110.0, 16.0],
        },
        index=["Total Revenue", "Operating Income"],
    )
    balance = pd.DataFrame(
        {
            "2021-12-31": [200.0, 80.0, 10.0],
            "2022-12-31": [220.0, 90.0, 10.0],
            "2023-12-31": [240.0, 95.0, 10.0],
            "2024-12-31": [260.0, 100.0, 10.0],
        },
        index=["Total Assets", "Stockholders Equity", "Ordinary Shares Number"],
    )
    cashflow = pd.DataFrame(
        {
            "2021-12-31": [20.0, -5.0],
            "2022-12-31": [22.0, -5.0],
            "2023-12-31": [24.0, -6.0],
            "2024-12-31": [26.0, -6.0],
        },
        index=["Operating Cash Flow", "Capital Expenditure"],
    )
    monkeypatch.setattr("core.fundamentals.fetch_companyfacts_for_ticker", lambda t, **k: pd.DataFrame())
    monkeypatch.setattr(
        "core.fundamentals.fetch_financials",
        lambda t: {"income": income, "balance": balance, "cashflow": cashflow},
    )
    monkeypatch.setattr(
        "core.fundamentals.fetch_ttm_financials",
        lambda t: {"income": income.iloc[:, [-1]], "cashflow": cashflow.iloc[:, [-1]], "balance": balance.iloc[:, [-1]]},
    )
    monkeypatch.setattr("core.fundamentals._read_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.fundamentals._write_cache", lambda *a, **k: None)

    fund = get_fundamentals("SHOP.TO", force=True)
    assert fund.source == "yahoo"
    assert len(fund.annual.dropna(how="all")) == 4
    assert fund.annual["revenue"].iloc[-1] == pytest.approx(110.0)
    assert fund.ttm.get("operating_cashflow") == pytest.approx(26.0)
