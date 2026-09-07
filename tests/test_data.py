"""Tests for financial statement extraction and normalization."""

import numpy as np
import pandas as pd
import pytest

import core.data as core_data
from core.data import (
    _compute_rsi,
    _ey_point,
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


def test_earnings_yield_history_omits_date_missing_cash(monkeypatch):
    """Missing cash skips that EY history point instead of treating cash as 0."""
    now = pd.Timestamp.now().normalize()
    complete_ts = (now - pd.DateOffset(years=1)).normalize()
    missing_ts = (now - pd.DateOffset(years=2)).normalize()
    complete = str(complete_ts.date())
    missing = str(missing_ts.date())

    a_income = pd.DataFrame({complete: [100.0], missing: [100.0]}, index=["EBIT"])
    a_balance = pd.DataFrame(
        {
            complete: [10.0, 0.0, 0.0],
            missing: [10.0, 0.0, None],
        },
        index=["Ordinary Shares Number", "Total Debt", "Cash And Cash Equivalents"],
    )
    annual = {"income": a_income, "balance": a_balance, "cashflow": pd.DataFrame()}
    quarterly = {
        "income": pd.DataFrame(),
        "balance": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
    }
    dates = pd.date_range(end=now, periods=4 * 365, freq="D")
    prices = pd.DataFrame({"Close": np.full(len(dates), 100.0)}, index=dates)

    monkeypatch.setattr(core_data, "fetch_financials", lambda t: annual)
    monkeypatch.setattr(core_data, "fetch_quarterly_financials", lambda t: quarterly)
    monkeypatch.setattr(core_data, "fetch_price_history", lambda t, **kw: prices)
    monkeypatch.setattr(core_data, "_read_cache", lambda *a, **kw: None)
    monkeypatch.setattr(core_data, "_write_cache", lambda *a, **kw: None)

    closes = prices["Close"]
    assert _ey_point(100.0, a_balance, missing, closes, missing_ts) is None
    no_cash_balance = pd.DataFrame(
        {complete: [10.0, 0.0]},
        index=["Ordinary Shares Number", "Total Debt"],
    )
    assert _ey_point(100.0, no_cash_balance, complete, closes, complete_ts) is None
    complete_ey = _ey_point(100.0, a_balance, complete, closes, complete_ts)
    assert complete_ey == pytest.approx(0.10, rel=1e-6)

    history = build_earnings_yield_history("FAKE")
    assert history == [complete_ey]


def _stmt(index: list[str], values: list[float], col: str = "2025-12-31") -> pd.DataFrame:
    return pd.DataFrame({col: values}, index=index)


def _patch_build_raw_metrics_deps(monkeypatch, *, info, ttm, annual, hist=None, edgar=None):
    import core.estimates as estimates
    import core.insiders as insiders

    hist = hist if hist is not None else pd.DataFrame({"Close": [100.0, 101.0]})
    monkeypatch.setattr(core_data, "fetch_ticker_info", lambda t: info)
    monkeypatch.setattr(core_data, "fetch_financials", lambda t: annual)
    monkeypatch.setattr(core_data, "fetch_ttm_financials", lambda t: ttm)
    monkeypatch.setattr(core_data, "fetch_price_history", lambda t, **kw: hist)
    monkeypatch.setattr(core_data, "fetch_analyst_recommendations", lambda t: pd.DataFrame())
    monkeypatch.setattr(core_data, "fetch_all_time_high", lambda t: 120.0)
    monkeypatch.setattr(estimates, "fetch_estimate_tables", lambda t: {})
    monkeypatch.setattr(insiders, "fetch_form4_transactions", lambda t: [])
    monkeypatch.setattr("core.data_quality.load_edgar_snapshot", lambda t: edgar)
    monkeypatch.setattr(
        "core.fundamentals.get_fundamentals",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skipped")),
    )


def test_build_raw_metrics_uses_ttm_flows_mrq_equity(monkeypatch):
    """TTM revenue + MRQ equity; Yahoo BVPS × shares is not used as book equity."""
    info = {
        "currentPrice": 450.0,
        "marketCap": 900e9,
        "enterpriseValue": 950e9,
        "bookValue": 400_000.0,  # Class-A BVPS (BRK-like mismatch)
        "sharesOutstanding": 2e9,
        "longName": "Fake Dual Class",
        "sector": "Financial Services",
        "trailingEps": 5.0,
        "totalDebt": 50e9,
        "totalCash": 80e9,
    }
    ttm = {
        "income": _stmt(
            ["Total Revenue", "Gross Profit", "EBIT", "Net Income", "Operating Income"],
            [400e9, 180e9, 120e9, 90e9, 120e9],
            "2026-06-30",
        ),
        "cashflow": _stmt(
            ["Operating Cash Flow", "Free Cash Flow", "Capital Expenditure"],
            [110e9, 90e9, -20e9],
            "2026-06-30",
        ),
        "balance": _stmt(
            [
                "Total Assets",
                "Total Liabilities Net Minority Interest",
                "Current Assets",
                "Current Liabilities",
                "Total Debt",
                "Cash Cash Equivalents And Short Term Investments",
                "Stockholders Equity",
                "Retained Earnings",
                "Ordinary Shares Number",
                "Net PPE",
                "Long Term Debt",
            ],
            [1_200e9, 700e9, 200e9, 100e9, 50e9, 80e9, 500e9, 400e9, 2e9, 150e9, 40e9],
            "2026-06-30",
        ),
    }
    annual = {
        "income": _stmt(
            ["Total Revenue", "Gross Profit", "Net Income"],
            [350e9, 160e9, 80e9],
            "2025-12-31",
        ),
        "balance": _stmt(["Total Assets", "Long Term Debt", "Ordinary Shares Number"], [1_100e9, 45e9, 1.9e9]),
        "cashflow": pd.DataFrame(),
    }
    edgar = {
        "revenue": 400e9,
        "net_income": 90e9,
        "operating_cashflow": 110e9,
        "total_debt": 50e9,
        "total_cash": 80e9,
        "book_equity": 500e9,
        "shares_outstanding": 941481,  # Class A; listing is B-equivalent 2e9
    }
    _patch_build_raw_metrics_deps(monkeypatch, info=info, ttm=ttm, annual=annual, edgar=edgar)

    de_calls = []
    orig_normalize_de = core_data.normalize_debt_to_equity

    def _spy_normalize_de(val):
        de_calls.append(val)
        return orig_normalize_de(val)

    monkeypatch.setattr(core_data, "normalize_debt_to_equity", _spy_normalize_de)

    raw = core_data.build_raw_metrics("BRK-B")
    assert raw["revenue"] == pytest.approx(400e9)
    assert raw["book_equity"] == pytest.approx(500e9)
    # Live D/E is total_debt / book_equity as a ratio, not percent / heuristic.
    assert raw["debt_to_equity"] == pytest.approx(50e9 / 500e9)
    assert de_calls == []
    assert raw["book_value"] == pytest.approx(400_000.0)
    assert raw["shares_outstanding"] == pytest.approx(2e9)
    assert raw["statement_basis"]["flows"] == "ttm"
    assert raw["statement_basis"]["balance"] == "mrq"
    from core.data_quality import warning_message
    from core.factors import compute_value_factors, compute_graham_value

    value = compute_value_factors(raw)
    assert value["book_to_market"] == pytest.approx(500e9 / 900e9)
    graham = compute_graham_value(raw)
    assert graham["graham_ratio"] is not None
    assert graham["graham_ratio"] < 2.0
    assert any(
        "keeping listing shares" in warning_message(w) for w in raw["data_warnings"]
    )


def test_build_raw_metrics_negative_equity_and_missing_ev(monkeypatch):
    info = {
        "currentPrice": 10.0,
        "marketCap": 5e9,
        "bookValue": 1.0,
        "sharesOutstanding": 500e6,
        "longName": "Neg Equity Co",
        "sector": "Consumer Cyclical",
    }
    ttm = {
        "income": _stmt(["Total Revenue", "EBIT", "Net Income"], [2e9, 0.2e9, 0.1e9]),
        "cashflow": _stmt(["Operating Cash Flow"], [0.15e9]),
        "balance": _stmt(
            ["Total Assets", "Total Liabilities Net Minority Interest", "Stockholders Equity"],
            [8e9, 9e9, -1e9],
        ),
    }
    annual = {
        "income": pd.DataFrame(),
        "balance": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
    }
    _patch_build_raw_metrics_deps(monkeypatch, info=info, ttm=ttm, annual=annual)
    raw = core_data.build_raw_metrics("NEG")
    assert raw["book_equity"] == pytest.approx(-1e9)
    assert raw["enterprise_value"] is None
    from core.data_quality import warning_message

    assert any("EV unavailable" in warning_message(w) for w in raw["data_warnings"])
    from core.factors import compute_balance_sheet_strength

    bs = compute_balance_sheet_strength(raw)
    assert bs["negative_equity"] is True
    assert bs["low_leverage"] is None


def test_build_raw_metrics_rejects_nonpositive_yahoo_ev(monkeypatch):
    """Negative Yahoo EV is discarded; use market_cap + debt - cash, never a negative EV."""
    info = {
        "currentPrice": 450.0,
        "marketCap": 900e9,
        "enterpriseValue": -1e9,
        "bookValue": 50.0,
        "sharesOutstanding": 2e9,
        "longName": "Neg EV Co",
        "sector": "Financial Services",
        "totalDebt": 50e9,
        "totalCash": 80e9,
    }
    ttm = {
        "income": _stmt(["Total Revenue", "EBIT", "Net Income"], [400e9, 120e9, 90e9]),
        "cashflow": _stmt(["Operating Cash Flow"], [110e9]),
        "balance": _stmt(
            [
                "Total Assets",
                "Total Liabilities Net Minority Interest",
                "Stockholders Equity",
                "Total Debt",
                "Cash Cash Equivalents And Short Term Investments",
            ],
            [1_200e9, 700e9, 500e9, 50e9, 80e9],
        ),
    }
    annual = {
        "income": pd.DataFrame(),
        "balance": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
    }
    _patch_build_raw_metrics_deps(monkeypatch, info=info, ttm=ttm, annual=annual)
    raw = core_data.build_raw_metrics("NEG-EV")
    constructed = 900e9 + 50e9 - 80e9
    assert raw["enterprise_value"] == pytest.approx(constructed)
    assert raw["enterprise_value"] > 0
    assert raw["earnings_yield_current"] == pytest.approx(120e9 / constructed)
    from core.data_quality import warning_message

    assert any("not positive" in warning_message(w) for w in raw["data_warnings"])


def test_build_raw_metrics_fx_converts_market_fields(monkeypatch):
    info = {
        "currentPrice": 100.0,
        "marketCap": 100e9,
        "currency": "CAD",
        "financialCurrency": "USD",
        "longName": "Shop CAD",
        "sector": "Technology",
        "trailingEps": 1.0,
    }
    ttm = {
        "income": _stmt(["Total Revenue", "EBIT", "Net Income"], [8e9, 1e9, 0.6e9]),
        "cashflow": _stmt(["Operating Cash Flow"], [0.8e9]),
        "balance": _stmt(
            [
                "Total Assets",
                "Total Liabilities Net Minority Interest",
                "Stockholders Equity",
                "Total Debt",
                "Cash Cash Equivalents And Short Term Investments",
            ],
            [20e9, 8e9, 12e9, 2e9, 1e9],
        ),
    }
    annual = {
        "income": pd.DataFrame(),
        "balance": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
    }
    _patch_build_raw_metrics_deps(monkeypatch, info=info, ttm=ttm, annual=annual)
    monkeypatch.setattr(core_data, "fetch_fx_rate", lambda pair: 0.73)
    raw = core_data.build_raw_metrics("SHOP.TO")
    assert raw["currency"] == "CAD"
    assert raw["financial_currency"] == "USD"
    assert raw["fx_to_financial"] == pytest.approx(0.73)
    assert raw["price_listing"] == pytest.approx(100.0)
    assert raw["price"] == pytest.approx(73.0)
    assert raw["market_cap"] == pytest.approx(73e9)
    assert raw["earnings_yield_current"] == pytest.approx(1e9 / (73e9 + 2e9 - 1e9))
    from core.data_quality import warning_message

    assert any("FX_CONVERTED" in str(w) or "converted CAD" in warning_message(w) for w in raw["data_warnings"])


def test_currency_symbol_cad():
    assert core_data.currency_symbol("CAD") == "C$"
    assert core_data.currency_symbol("USD") == "$"
