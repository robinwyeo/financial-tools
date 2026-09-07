"""Tests for owner-earnings DCF, EPV, reverse DCF, expected return."""

from __future__ import annotations

import pandas as pd
import pytest

from core.fundamentals import Fundamentals, owner_earnings
from core.valuation import (
    dcf_scenarios,
    dcf_value,
    epv,
    expected_return,
    historical_growth,
    normalized_owner_earnings,
    reverse_dcf,
    valuation_summary,
)

_CFG = {
    "valuation": {
        "terminal_growth": 0.025,
        "explicit_years": 10,
        "fade_start_year": 5,
        "base_growth_cap": 0.15,
        "base_growth_floor": 0.0,
        "bear_growth_multiplier": 0.5,
        "bear_growth_cap": 0.04,
        "bear_rate_bump": 0.01,
        "subtract_sbc": True,
        "normalization_years": 3,
        "cyclical_normalization_years": 7,
        "cyclical_sectors": ["Energy"],
        "min_tax_rate": 0.15,
        "max_tax_rate": 0.35,
        "hurdle": {"floor": 0.08, "erp": 0.045, "fallback_rf": 0.042},
    }
}


def test_dcf_matches_hand_fixture():
    out = dcf_value(100, 10, 50, 30, 0.09, 0.08, _CFG)
    # Hand-computed: years 1-5 grow at 8%, fade to 2.5% by year 10, Gordon terminal.
    assert out["per_share"] == pytest.approx(217.00460754392793, rel=0.005)
    assert out["equity_value"] == pytest.approx(2170.0460754392793, rel=0.005)


def test_reverse_dcf_recovers_g1():
    produced = dcf_value(100, 10, 50, 30, 0.09, 0.08, _CFG)
    price = produced["per_share"]
    recovered = reverse_dcf(price, 100, 10, 50, 30, 0.09, _CFG)
    assert recovered["implied_g1"] == pytest.approx(0.08, abs=0.002)


def test_dcf_bear_is_below_base():
    sc = dcf_scenarios(100, 10, 50, 30, 0.09, 0.08, _CFG)
    assert sc["base"]["per_share"] > sc["bear"]["per_share"]
    assert sc["g1_bear"] == pytest.approx(0.04)


def test_epv_no_growth():
    out = epv(100, 0.20, 0.09, 50, 30, 10, _CFG)
    # NOPAT = 80; 80/0.09 + 50 - 30 = 908.89; /10 = 90.889
    assert out["per_share"] == pytest.approx(90.888888, rel=1e-4)
    assert out["tax_rate"] == pytest.approx(0.20)


def test_expected_return_caps_growth():
    # yield 10% + g 8% capped at 6% → 0.16
    assert expected_return(10, 100, 0.08, _CFG) == pytest.approx(0.16)
    assert expected_return(10, 100, None, _CFG) == pytest.approx(0.10)
    assert expected_return(None, 100, 0.05, _CFG) is None


def test_normalized_oe_non_cyclical_mean():
    idx = pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31"])
    annual = pd.DataFrame(
        {
            "operating_cashflow": [90.0, 100.0, 110.0],
            "capex": [-10.0, -10.0, -10.0],
            "sbc": [5.0, 5.0, 5.0],
            "revenue": [200.0, 220.0, 240.0],
        },
        index=idx,
    )
    ttm = pd.Series({"operating_cashflow": 120.0, "capex": -10.0, "sbc": 5.0, "revenue": 250.0})
    fund = Fundamentals("X", annual, ttm, ttm, "edgar", idx[-1].date())
    out = normalized_owner_earnings(fund, _CFG, sector="Technology")
    # TTM OE=105, FY=95, FY-1=85, FY-2=75 → mean 90
    assert out["value"] == pytest.approx(90.0)
    assert out["negative"] is False


def test_normalized_oe_cyclical_uses_margin_path():
    idx = pd.to_datetime([f"{y}-12-31" for y in range(2018, 2025)])
    annual = pd.DataFrame(
        {
            "operating_cashflow": [20.0] * 7,
            "capex": [-5.0] * 7,
            "sbc": [0.0] * 7,
            "revenue": [100.0] * 7,
        },
        index=idx,
    )
    ttm = pd.Series({"operating_cashflow": 30.0, "capex": -5.0, "sbc": 0.0, "revenue": 200.0})
    fund = Fundamentals("X", annual, ttm, ttm, "edgar", idx[-1].date())
    out = normalized_owner_earnings(fund, _CFG, sector="Energy")
    # 7y OE margin = 15/100 = 0.15; * TTM revenue 200 = 30
    assert out["method"] == "cyclical_7y_margin"
    assert out["value"] == pytest.approx(30.0)


def test_historical_growth_needs_six_years():
    idx = pd.to_datetime([f"{y}-12-31" for y in range(2020, 2025)])
    annual = pd.DataFrame(
        {"operating_cashflow": [10] * 5, "capex": [0] * 5, "sbc": [0] * 5, "shares_diluted": [1] * 5, "revenue": [10] * 5},
        index=idx,
    )
    fund = Fundamentals("X", annual, pd.Series(dtype=float), pd.Series(dtype=float), "edgar", idx[-1].date())
    assert historical_growth(fund, _CFG)["g_hist"] is None


def test_negative_oe_flagged():
    ttm = pd.Series({"operating_cashflow": 10.0, "capex": -40.0, "sbc": 0.0})
    fund = Fundamentals("X", pd.DataFrame(), ttm, ttm, "edgar", pd.Timestamp("2024-12-31").date())
    out = normalized_owner_earnings(fund, _CFG, sector="Technology")
    assert out["value"] == pytest.approx(-30.0)
    assert out["negative"] is True


def test_financials_relative_only():
    ttm = pd.Series({"net_income": 10.0, "ebit": 12.0, "shares_diluted": 5.0, "equity": 50.0})
    mrq = pd.Series({"equity": 50.0, "cash": 5.0, "debt": 8.0, "shares_diluted": 5.0})
    fund = Fundamentals("JPM", pd.DataFrame(), ttm, mrq, "edgar", pd.Timestamp("2024-12-31").date())
    summary = valuation_summary(
        fund,
        price=20.0,
        shares=5.0,
        cash=5.0,
        debt=8.0,
        market_cap=100.0,
        ev=103.0,
        sector="Financial Services",
        industry=None,
        uncertainty_label="Low",
        config=_CFG,
        rf=0.042,
    )
    assert summary["relative_only"] is True
    assert summary["dcf"] is None
    assert summary["price_to_book"] == pytest.approx(2.0)
