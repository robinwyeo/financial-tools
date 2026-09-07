"""Coverage backfill for HTTP/IO helpers with fakes (no live network)."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from core.fundamentals import Fundamentals, history_quality_inputs
from core.rates import fetch_fred_history, pit_risk_free
from core.sec import cik_padded, fetch_cik_ticker_map, sec_get, ticker_to_cik
from core.universe import _wiki_symbol_list, load_universe_snapshot
from core.fund_universe import (
    SnapshotIncompleteError,
    build_fund_universe_snapshot,
    default_fund_universe,
    fund_snapshot_path,
    load_fund_universe_snapshot,
)
from core.providers.edgar import EdgarProvider
from core.providers.finnhub import FinnhubProvider
from core.providers.sharadar import SharadarProvider
from core.providers.yahoo import YahooProvider


def test_sec_get_rate_limit_and_headers(monkeypatch):
    calls = []

    class _Resp:
        def raise_for_status(self):
            return None

    def fake_get(url, headers=None, timeout=30):
        calls.append((url, headers, timeout))
        return _Resp()

    monkeypatch.setattr("core.sec.requests.get", fake_get)
    monkeypatch.setattr("core.sec._last_request_at", 0.0)
    monkeypatch.setattr("core.sec.time.sleep", lambda *_: None)
    monkeypatch.setattr("core.sec.time.time", lambda: 1.0)
    out = sec_get("https://example.test", timeout=5)
    assert isinstance(out, _Resp)
    assert calls[0][1]["User-Agent"].startswith("financial-tools")


def test_cik_map_from_cache(monkeypatch):
    monkeypatch.setattr(
        "core.sec._read_cache",
        lambda *a, **k: {"rows": [{"cik": 320193, "ticker": "AAPL", "name": "Apple"}]},
    )
    df = fetch_cik_ticker_map()
    assert int(df.iloc[0]["cik"]) == 320193
    assert ticker_to_cik("AAPL") == 320193
    assert ticker_to_cik("NOPE") is None
    assert cik_padded(320193) == "0000320193"


def test_ticker_to_cik_handles_fetch_error(monkeypatch):
    monkeypatch.setattr("core.sec.fetch_cik_ticker_map", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert ticker_to_cik("AAPL") is None


def test_wiki_fallback_on_error(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("net")))
    out = _wiki_symbol_list("https://example.test/x", ["AAA", "BBB"])
    assert out == ["AAA", "BBB"]
    tsx = _wiki_symbol_list("https://example.test/x", ["RY"], suffix=".TO")
    assert tsx == ["RY.TO"]


def test_load_universe_snapshot_missing(monkeypatch, tmp_path):
    missing = tmp_path / "nope.parquet"
    monkeypatch.setattr("core.universe.SNAPSHOT_PATH", missing)
    assert load_universe_snapshot() is None


def test_fund_universe_snapshot_roundtrip(tmp_path, monkeypatch):
    dest = tmp_path / "funds.parquet"
    meta = tmp_path / "funds.meta.json"
    monkeypatch.setattr("core.fund_universe.FUND_SNAPSHOT_PATH", dest)
    monkeypatch.setattr("core.fund_universe.FUND_SNAPSHOT_META_PATH", meta)
    monkeypatch.setattr("core.fund_universe.DATA_DIR", tmp_path)
    monkeypatch.setattr("core.fund_universe.throttle", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "core.fund_universe.build_fund_raw_metrics",
        lambda t: {"ticker": t, "name": t, "price": 10.0, "quote_type": "ETF", "category": "x", "fund_family": "y", "currency": "USD"},
    )
    monkeypatch.setattr("core.fund_universe.compute_fund_factors", lambda raw: {"low_cost": -0.001})
    df = build_fund_universe_snapshot(tickers=["AAA", "BBB"], min_success_ratio=0.9)
    assert len(df) == 2
    assert dest.exists()
    loaded = load_fund_universe_snapshot()
    assert loaded is not None
    assert fund_snapshot_path() == dest
    assert len(default_fund_universe()) > 50


def test_fund_universe_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr("core.fund_universe.FUND_SNAPSHOT_PATH", tmp_path / "funds.parquet")
    monkeypatch.setattr("core.fund_universe.FUND_SNAPSHOT_META_PATH", tmp_path / "funds.meta.json")
    monkeypatch.setattr("core.fund_universe.DATA_DIR", tmp_path)
    monkeypatch.setattr("core.fund_universe.throttle", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "core.fund_universe.build_fund_raw_metrics",
        lambda t: {"ticker": t, "price": None},
    )
    with pytest.raises(SnapshotIncompleteError):
        build_fund_universe_snapshot(tickers=["A", "B", "C"], min_success_ratio=0.9)


def test_fred_history_and_pit(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "observations": [
                    {"date": "2024-01-02", "value": "."},
                    {"date": "2024-01-03", "value": "4.20"},
                    {"date": "2024-06-01", "value": "4.40"},
                ]
            }

    monkeypatch.setenv("FRED_API_KEY", "k")
    monkeypatch.setattr("core.rates._read_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.rates._write_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.rates.requests.get", lambda *a, **k: _Resp())
    hist = fetch_fred_history("DGS10", api_key="k", force=True)
    assert len(hist) == 2
    assert pit_risk_free(date(2024, 3, 1), hist) == pytest.approx(0.042)
    assert pit_risk_free(date(2020, 1, 1), hist) is None


def test_fred_history_from_cache(monkeypatch):
    monkeypatch.setattr(
        "core.rates._read_cache",
        lambda *a, **k: {"rows": [{"date": "2024-01-03", "value": 0.042}]},
    )
    hist = fetch_fred_history("DGS10")
    assert float(hist.iloc[0]) == pytest.approx(0.042)


def test_providers_wrap_existing(monkeypatch):
    monkeypatch.setattr("core.data.fetch_ttm_financials", lambda t: {"revenue": 1})
    monkeypatch.setattr(
        "core.data.fetch_financials",
        lambda t: {"income": pd.DataFrame({"2024": [1]}, index=["Total Revenue"])},
    )
    monkeypatch.setattr("core.estimates.fetch_estimate_tables", lambda t: {"eps_trend": pd.DataFrame()})
    y = YahooProvider()
    assert y.ttm("AAPL")["revenue"] == 1
    assert not y.annual("AAPL").empty
    assert "eps_trend" in y.estimates("AAPL")

    fund = Fundamentals(
        ticker="AAPL",
        annual=pd.DataFrame({"revenue": [1]}, index=pd.to_datetime(["2024-12-31"])),
        ttm=pd.Series({"revenue": 1.0}),
        mrq=pd.Series({"equity": 2.0}),
        source="edgar",
        as_of=date(2025, 1, 1),
        column_sources={},
    )
    monkeypatch.setattr("core.fundamentals.get_fundamentals", lambda t: fund)
    e = EdgarProvider()
    assert float(e.ttm("AAPL")["revenue"]) == 1.0
    assert "revenue" in e.annual("AAPL").columns
    assert e.estimates("AAPL") == {}

    fh = FinnhubProvider(api_key=None)
    assert fh.ttm("X") == {}
    assert fh.annual("X").empty
    assert fh.estimates("X") == {}
    fh2 = FinnhubProvider(api_key="k")

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"metric": {"pe": 10}, "data": [{"year": 2024}], "recommendation": []}

    monkeypatch.setattr("core.providers.finnhub.requests.get", lambda *a, **k: _Resp())
    assert fh2.ttm("AAPL")["pe"] == 10
    assert not fh2.annual("AAPL").empty
    recs = fh2.estimates("AAPL")
    assert "recommendation" in recs

    with pytest.raises(NotImplementedError):
        SharadarProvider().annual("AAPL")
    with pytest.raises(NotImplementedError):
        SharadarProvider().estimates("AAPL")


def test_history_quality_inputs_computes_ratios():
    idx = pd.date_range("2018-12-31", periods=7, freq="YE")
    annual = pd.DataFrame(
        {
            "revenue": [100, 110, 120, 130, 140, 150, 160],
            "gross_profit": [40, 44, 48, 50, 52, 54, 56],
            "ebit": [20] * 7,
            "net_income": [10] * 7,
            "operating_cashflow": [15] * 7,
            "capex": [-5] * 7,
            "sbc": [1] * 7,
            "shares_diluted": [10, 10, 10, 10.2, 10.3, 10.4, 10.5],
            "current_assets": [50] * 7,
            "current_liabilities": [20] * 7,
            "cash": [10] * 7,
            "ppe_net": [30] * 7,
            "equity": [80] * 7,
            "debt": [20] * 7,
        },
        index=idx,
    )
    fund = Fundamentals(
        ticker="X",
        annual=annual,
        ttm=pd.Series(
            {"revenue": 160, "operating_cashflow": 15, "capex": -5, "net_income": 10, "shares_diluted": 10.5}
        ),
        mrq=pd.Series({"equity": 80, "goodwill": 8, "total_assets": 200}),
        source="edgar",
        as_of=date(2025, 1, 1),
        column_sources={},
    )
    out = history_quality_inputs(fund, {"price": 20, "enterprise_value": 200})
    assert out["goodwill_to_equity"] == pytest.approx(0.1)
    assert out["equity_to_assets"] == pytest.approx(0.4)
    assert out["fcf_margin"] is not None
    assert out["revenue_5y_cagr"] is not None


def test_refresh_reference_cli(tmp_path, monkeypatch):
    from jobs import refresh_reference as job

    dest = tmp_path / "erp.csv"
    monkeypatch.setattr(job, "ERP_PATH", dest)
    monkeypatch.setattr(
        "sys.argv",
        ["refresh_reference.py", "--from-file", "data/reference/damodaran_erp.csv", "--kind", "erp"],
    )
    assert job.main() == 0
    assert dest.exists()


def test_fetch_ticker_info_uses_cache_and_yf(monkeypatch):
    from core.data import fetch_analyst_recommendations, fetch_financials, fetch_ticker_info, fetch_ttm_financials

    monkeypatch.setattr("core.data._read_cache", lambda *a, **k: None)
    written = []
    monkeypatch.setattr("core.data._write_cache", lambda path, data: written.append(data))

    class FakeTicker:
        def __init__(self, ticker):
            self.info = {"currentPrice": 12.0, "longName": "Fake Inc", "currency": "USD"}
            self.financials = pd.DataFrame({"2024-12-31": [100]}, index=["Total Revenue"])
            self.balance_sheet = pd.DataFrame({"2024-12-31": [200]}, index=["Total Assets"])
            self.cashflow = pd.DataFrame({"2024-12-31": [30]}, index=["Operating Cash Flow"])
            self.ttm_income_stmt = pd.DataFrame({"ttm": [110]}, index=["Total Revenue"])
            self.ttm_cashflow = pd.DataFrame({"ttm": [32]}, index=["Operating Cash Flow"])
            self.quarterly_balance_sheet = pd.DataFrame({"2025-03-31": [210]}, index=["Total Assets"])
            self.recommendations = pd.DataFrame({"period": ["0m"], "strongBuy": [1], "buy": [2], "hold": [0], "sell": [0], "strongSell": [0]})

    monkeypatch.setattr("core.data.yf.Ticker", FakeTicker)
    info = fetch_ticker_info("FAKE")
    assert info["currentPrice"] == 12.0
    fin = fetch_financials("FAKE")
    assert not fin["income"].empty
    ttm = fetch_ttm_financials("FAKE")
    assert not ttm["income"].empty
    recs = fetch_analyst_recommendations("FAKE")
    assert not recs.empty


def test_fetch_uses_cache_hit(monkeypatch):
    from core.data import fetch_financials, fetch_ticker_info, fetch_ttm_financials

    monkeypatch.setattr(
        "core.data._read_cache",
        lambda path, max_age_hours=24: (
            {"currentPrice": 1.0, "longName": "Z"}
            if "info" in str(path)
            else {"income": [], "balance": [], "cashflow": []}
        ),
    )
    assert fetch_ticker_info("X")["longName"] == "Z"
    assert fetch_financials("X")["income"].empty
    # ttm cache shape
    monkeypatch.setattr(
        "core.data._read_cache",
        lambda path, max_age_hours=48: {"income": [], "cashflow": [], "balance": []},
    )
    assert fetch_ttm_financials("X")["income"].empty


def _rich_raw() -> dict:
    return {
        "enterprise_value": 1000,
        "ebit": 80,
        "market_cap": 900,
        "book_value": 20,
        "shares_outstanding": 10,
        "book_equity": 200,
        "free_cashflow": 50,
        "gross_profit": 120,
        "total_liabilities": 220,
        "net_income": 40,
        "revenue": 300,
        "volatility_12m": 0.2,
        "momentum_12_1": 0.1,
        "total_assets_prior": 450,
        "net_income_prior": 30,
        "operating_cashflow": 60,
        "long_term_debt": 80,
        "long_term_debt_prior": 90,
        "current_assets": 150,
        "current_liabilities": 70,
        "current_assets_prior": 140,
        "current_liabilities_prior": 75,
        "shares_prior": 10.2,
        "gross_profit_prior": 110,
        "revenue_prior": 280,
        "total_cash": 40,
        "total_debt": 100,
        "debt_to_equity": 0.5,
        "dividends_paid": -10,
        "repurchase_of_stock": -20,
        "interest_expense": 5,
        "retained_earnings": 100,
        "inventory": 20,
        "inventory_prior": 18,
        "receivables": 25,
        "receivables_prior": 24,
        "trailing_pe": 18,
        "earnings_growth": 0.12,
        "dividend_yield": 0.02,
        "fcf_conversion_3y": 0.9,
        "share_cagr_3y": -0.01,
        "revenue_5y_cagr": 0.08,
        "gross_margin_5y_delta": 0.01,
        "roic_5y_mean": 0.15,
        "roic_5y_std": 0.02,
        "net_debt_to_ebitda": 1.2,
        "interest_coverage": 8.0,
        "price": 90,
        "eps": 5,
        "bvps": 20,
        "roic": 0.12,
        "capex": -15,
        "ppe_net": 80,
        "cash": 40,
        "debt": 100,
        "form4_transactions": [{"insider": "A", "code": "P", "value": 1000, "is_officer": True}],
        "estimate_tables": {},
    }


def test_compute_all_factors_kitchen_sink():
    from core.factors import compute_all_factors

    out = compute_all_factors(_rich_raw())
    assert out["earnings_yield"] == pytest.approx(80 / 1000)
    assert out["financial_strength"] is not None
    assert "altman_z_pp" in out
    assert "insider_buying" in out


def test_valuation_summary_operating_company():
    from core.valuation import valuation_summary

    idx = pd.date_range("2016-12-31", periods=9, freq="YE")
    annual = pd.DataFrame(
        {
            "revenue": [80 + i * 5 for i in range(9)],
            "gross_profit": [40 + i * 2 for i in range(9)],
            "ebit": [20 + i for i in range(9)],
            "net_income": [12 + i for i in range(9)],
            "operating_cashflow": [18 + i for i in range(9)],
            "capex": [-6] * 9,
            "sbc": [1] * 9,
            "shares_diluted": [10] * 9,
            "cash": [50] * 9,
            "debt": [30] * 9,
            "tax_expense": [3] * 9,
            "pretax_income": [15] * 9,
        },
        index=idx,
    )
    ttm = pd.Series(
        {
            "revenue": 120,
            "ebit": 28,
            "operating_cashflow": 26,
            "capex": -6,
            "sbc": 1,
            "net_income": 20,
            "shares_diluted": 10,
            "tax_expense": 4,
            "pretax_income": 24,
        }
    )
    mrq = pd.Series({"cash": 50, "debt": 30, "equity": 80, "shares_diluted": 10})
    fund = Fundamentals("AAPL", annual, ttm, mrq, "edgar", idx[-1].date(), {})
    peers = pd.DataFrame({"ev_to_ebit": [12.0, 14.0], "p_to_oe": [18.0, 20.0]})
    out = valuation_summary(
        fund,
        price=80.0,
        shares=10.0,
        cash=50.0,
        debt=30.0,
        market_cap=800.0,
        ev=780.0,
        sector="Technology",
        industry=None,
        uncertainty_label="Low",
        analyst_growth=0.10,
        sector_peers_df=peers,
        rf=0.042,
    )
    assert out["relative_only"] is False
    assert out["dcf"]["base"]["per_share"] is not None
    assert out["reverse_dcf"]["implied_g1"] is not None
    assert out["hurdle"]["rate"] == pytest.approx(0.087)


def test_score_fund_with_and_without_universe(monkeypatch):
    from core.scoring import score_fund

    raw = {
        "ticker": "VTI",
        "name": "Vanguard",
        "quote_type": "ETF",
        "category": "Large Blend",
        "fund_family": "Vanguard",
        "currency": "USD",
        "price": 100.0,
        "nav_price": 99.0,
        "expense_ratio": 0.003,
        "return_1y": 0.1,
        "return_3y": 0.09,
        "return_5y": 0.08,
        "momentum_12_1": 0.07,
        "volatility_12m": 0.15,
        "max_drawdown": -0.2,
        "distribution_yield": 0.015,
        "fifty_two_week_high": 110.0,
        "rsi_14": 40.0,
    }
    monkeypatch.setattr("core.scoring.build_fund_raw_metrics", lambda t: raw)
    empty = score_fund("VTI", {}, fund_universe_df=pd.DataFrame())
    assert empty.get("warning")

    uni = pd.DataFrame(
        [
            {**raw, "ticker": "VOO", "low_cost": -0.004, "return_3y": 0.08, "return_5y": 0.07, "sharpe_1y": 0.6, "low_volatility": 6.0, "drawdown_protection": -0.2, "momentum_12_1": 0.05, "distribution_yield": 0.01},
            {**raw, "ticker": "SPY", "low_cost": -0.009, "return_3y": 0.07, "return_5y": 0.06, "sharpe_1y": 0.5, "low_volatility": 5.0, "drawdown_protection": -0.25, "momentum_12_1": 0.04, "distribution_yield": 0.012},
            {**raw, "ticker": "ITOT", "low_cost": -0.003, "return_3y": 0.085, "return_5y": 0.075, "sharpe_1y": 0.55, "low_volatility": 5.5, "drawdown_protection": -0.22, "momentum_12_1": 0.06, "distribution_yield": 0.014},
        ]
    )
    scored = score_fund("VTI", {"universe": {"sector_scoring": False}}, fund_universe_df=uni)
    assert scored["composite"] is not None
    assert scored["nav_premium"] == pytest.approx(100 / 99 - 1)


def test_score_ticker_etf_short_circuit(monkeypatch):
    from core.scoring import score_ticker

    monkeypatch.setattr("core.scoring.is_fund", lambda t: True)
    monkeypatch.setattr("core.scoring.get_security_type", lambda t: "ETF")
    out = score_ticker("SPY", {})
    assert out["is_etf"] is True


def test_fetch_fx_rate(monkeypatch):
    from core.data import fetch_fx_rate

    assert fetch_fx_rate("USDUSD") == 1.0
    assert fetch_fx_rate("xx") is None
    monkeypatch.setattr("core.data._read_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.data._write_cache", lambda *a, **k: None)

    class FakeFX:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period="5d"):
            if "CADUSD" in self.symbol:
                return pd.DataFrame({"Close": [0.73]})
            return pd.DataFrame()

    monkeypatch.setattr("core.data.yf.Ticker", FakeFX)
    assert fetch_fx_rate("CADUSD") == pytest.approx(0.73)


def test_fetch_estimate_tables(monkeypatch):
    from core.estimates import fetch_estimate_tables

    monkeypatch.setattr("core.estimates._read_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.estimates._write_cache", lambda *a, **k: None)

    class Fake:
        def __init__(self, ticker):
            self.eps_trend = pd.DataFrame({"current": [1.0]})
            self.eps_revisions = pd.DataFrame({"upLast30days": [2]})
            self.earnings_history = pd.DataFrame({"surprisePercent": [3.0]})

    monkeypatch.setattr("core.estimates.yf.Ticker", Fake)
    tables = fetch_estimate_tables("AAPL")
    assert not tables["eps_trend"].empty


def test_wiki_symbol_list_parses_table(monkeypatch):
    html = """
    <html><body><table>
    <tr><th>Symbol</th><th>Security</th></tr>
    <tr><td>AAA</td><td>Alpha</td></tr>
    <tr><td>BRK.B</td><td>Berkshire</td></tr>
    </table></body></html>
    """

    class _Resp:
        def read(self):
            return html.encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    tickers = _wiki_symbol_list("https://example.test", ["ZZZ"])
    assert "AAA" in tickers
    assert "BRK-B" in tickers


def test_score_universe_empty_snapshot(monkeypatch):
    from core.scoring import score_fund_universe, score_universe

    monkeypatch.setattr("core.scoring.load_universe_snapshot", lambda: None)
    monkeypatch.setattr("core.scoring.load_fund_universe_snapshot", lambda: None)
    assert score_universe({}).empty
    assert score_fund_universe({}).empty

