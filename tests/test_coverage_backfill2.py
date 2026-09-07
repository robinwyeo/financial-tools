"""Additional fakes for remaining core/ coverage gaps (no live network)."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from core.fundamentals import Fundamentals
from core.universe import _wiki_symbol_list, fetch_universe_tickers, load_universe_snapshot, prefer_us_listings


def test_read_write_cache_and_safe_float(tmp_path, monkeypatch):
    from core.data import _read_cache, _safe_float, _write_cache

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert _read_cache(bad, max_age_hours=24) is None

    good = tmp_path / "good.json"
    _write_cache(good, {"a": 1, "when": pd.Timestamp("2024-01-01")})
    assert _read_cache(good, max_age_hours=24)["a"] == 1

    assert _safe_float(None) is None
    assert _safe_float(float("nan")) is None
    assert _safe_float("nope") is None
    assert _safe_float("3.5") == 3.5


def test_get_security_type_and_fund_flags(monkeypatch):
    from core.data import get_security_type, is_etf, is_fund

    monkeypatch.setattr("core.data._read_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.data._write_cache", lambda *a, **k: None)

    class Fake:
        def __init__(self, ticker):
            self.info = {"quoteType": "ETF"}

    monkeypatch.setattr("core.data.yf.Ticker", Fake)
    assert get_security_type("SPY") == "ETF"
    assert is_etf("SPY") is True
    assert is_fund("SPY") is True

    class Boom:
        def __init__(self, ticker):
            raise RuntimeError("net")

    monkeypatch.setattr("core.data.yf.Ticker", Boom)
    assert get_security_type("ZZZ") == "EQUITY"


def test_fetch_price_history_and_ath(monkeypatch):
    from core.data import fetch_all_time_high, fetch_price_history

    monkeypatch.setattr("core.data._read_cache", lambda *a, **k: None)
    written = {}
    monkeypatch.setattr("core.data._write_cache", lambda path, data: written.setdefault("ok", data))

    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    hist = pd.DataFrame({"Close": [10, 11, 12, 13, 14]}, index=idx)

    class Fake:
        def __init__(self, ticker):
            pass

        def history(self, period="2y", interval="1d", auto_adjust=True):
            return hist

    monkeypatch.setattr("core.data.yf.Ticker", Fake)
    out = fetch_price_history("AAA")
    assert not out.empty
    assert out["Close"].iloc[-1] == 14

    monkeypatch.setattr("core.data.fetch_price_history", lambda t, period="max": hist)
    assert fetch_all_time_high("AAA") == 14.0


def test_fetch_price_history_empty_and_error(monkeypatch):
    from core.data import fetch_price_history

    monkeypatch.setattr("core.data._read_cache", lambda *a, **k: None)

    class Empty:
        def __init__(self, ticker):
            pass

        def history(self, **kw):
            return pd.DataFrame()

    monkeypatch.setattr("core.data.yf.Ticker", Empty)
    assert fetch_price_history("AAA").empty

    class Boom:
        def __init__(self, ticker):
            pass

        def history(self, **kw):
            raise RuntimeError("x")

    monkeypatch.setattr("core.data.yf.Ticker", Boom)
    assert fetch_price_history("AAA").empty


def test_fetch_ticker_info_incomplete_and_error(monkeypatch):
    from core.data import fetch_ticker_info

    monkeypatch.setattr("core.data._read_cache", lambda *a, **k: {"shortName": "X"})
    monkeypatch.setattr("core.data._write_cache", lambda *a, **k: None)

    class Incomplete:
        info = {"shortName": "OnlyName"}

        def __init__(self, ticker):
            pass

    monkeypatch.setattr("core.data.yf.Ticker", Incomplete)
    info = fetch_ticker_info("AAA")
    assert info["shortName"] == "OnlyName"

    class Boom:
        def __init__(self, ticker):
            raise RuntimeError("x")

    monkeypatch.setattr("core.data._read_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.data.yf.Ticker", Boom)
    assert fetch_ticker_info("AAA") == {}


def test_fetch_fx_inverse_and_cache(monkeypatch):
    from core.data import fetch_fx_rate

    monkeypatch.setattr("core.data._read_cache", lambda *a, **k: {"rate": 0.8})
    assert fetch_fx_rate("CADUSD") == pytest.approx(0.8)

    monkeypatch.setattr("core.data._read_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.data._write_cache", lambda *a, **k: None)

    class Fake:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period="5d"):
            if "USDCAD" in self.symbol:
                return pd.DataFrame({"Close": [1.25]})
            raise RuntimeError("no pair")

    monkeypatch.setattr("core.data.yf.Ticker", Fake)
    assert fetch_fx_rate("CADUSD") == pytest.approx(0.8)


def test_fetch_statements_live_paths(monkeypatch):
    from core.data import (
        fetch_analyst_recommendations,
        fetch_financials,
        fetch_quarterly_financials,
        fetch_ttm_financials,
    )

    monkeypatch.setattr("core.data._read_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.data._write_cache", lambda *a, **k: None)
    cols = [pd.Timestamp("2023-12-31"), pd.Timestamp("2024-12-31")]
    income = pd.DataFrame({cols[0]: [100], cols[1]: [110]}, index=["Total Revenue"])
    recs = pd.DataFrame({"strongBuy": [5], "buy": [4], "hold": [3], "sell": [0], "strongSell": [0]})

    class T:
        financials = income
        balance_sheet = income
        cashflow = income
        ttm_income_stmt = income
        ttm_cashflow = income
        quarterly_balance_sheet = income
        quarterly_financials = income
        quarterly_cashflow = income
        recommendations = recs

        def __init__(self, ticker):
            pass

    monkeypatch.setattr("core.data.yf.Ticker", T)
    assert not fetch_financials("AAA")["income"].empty
    assert not fetch_ttm_financials("AAA")["income"].empty
    assert not fetch_quarterly_financials("AAA")["income"].empty
    assert not fetch_analyst_recommendations("AAA").empty

    class Boom:
        def __init__(self, ticker):
            raise RuntimeError("x")

    monkeypatch.setattr("core.data.yf.Ticker", Boom)
    assert fetch_financials("AAA")["income"].empty
    assert fetch_ttm_financials("AAA")["income"].empty
    assert fetch_quarterly_financials("AAA")["income"].empty
    assert fetch_analyst_recommendations("AAA").empty


def test_fetch_fund_info_holdings_and_trailing_return(monkeypatch):
    from core.data import (
        _compute_drawdown_metrics,
        _compute_momentum_12_1,
        _compute_rsi,
        _compute_trailing_return,
        _compute_volatility_12m,
        fetch_etf_holdings,
        fetch_fund_info,
    )

    monkeypatch.setattr(
        "core.data.fetch_ticker_info",
        lambda t: {
            "currentPrice": 50,
            "longName": "Fund",
            "quoteType": "ETF",
            "annualReportExpenseRatio": 0.0003,
            "yield": 0.012,
            "navPrice": 49.5,
            "ytdReturn": 8.0,
            "threeYearAverageReturn": 9.0,
            "fiveYearAverageReturn": 200.0,
            "currency": "USD",
        },
    )
    info = fetch_fund_info("VTI")
    assert info["expense_ratio"] == 0.0003
    assert info["five_year_avg_return"] == pytest.approx(2.0)

    class Hold:
        fund_holding_info = {"holdings": [{"symbol": "AAPL", "holdingPercent": 0.07}]}

        def __init__(self, ticker):
            pass

    monkeypatch.setattr("core.data.yf.Ticker", Hold)
    holdings = fetch_etf_holdings("VTI", top_n=1)
    assert holdings.iloc[0]["symbol"] == "AAPL"

    idx = pd.date_range("2020-01-01", periods=800, freq="B")
    hist = pd.DataFrame({"Close": np.linspace(100, 140, 800)}, index=idx)
    assert _compute_trailing_return(hist, 1.0) is not None
    assert _compute_trailing_return(hist, 3.0) is not None
    assert _compute_momentum_12_1(hist) is not None
    assert _compute_volatility_12m(hist) is not None
    dd = _compute_drawdown_metrics(hist)
    assert dd["max_drawdown"] is not None
    assert _compute_rsi(hist) is not None
    rsi_up = _compute_rsi(pd.DataFrame({"Close": list(range(20))}))
    assert rsi_up is not None


def test_build_earnings_yield_history_and_wrapper(monkeypatch):
    from core.data import build_earnings_yield_history, compute_valuation_vs_history

    dates = pd.date_range("2020-01-01", periods=800, freq="B")
    hist = pd.DataFrame({"Close": np.linspace(50, 80, 800)}, index=dates)
    fy = [pd.Timestamp("2022-12-31"), pd.Timestamp("2023-12-31"), pd.Timestamp("2024-12-31")]
    income = pd.DataFrame(
        {c: [20 + i * 2] for i, c in enumerate(fy)},
        index=["EBIT"],
    )
    balance = pd.DataFrame(
        {
            c: [10, 30, 5]
            for c in fy
        },
        index=["Ordinary Shares Number", "Total Debt", "Cash And Cash Equivalents"],
    )
    q_cols = [pd.Timestamp(f"2024-{m:02d}-30") for m in (3, 6, 9, 12)]
    q_income = pd.DataFrame({c: [5] for c in q_cols}, index=["EBIT"])
    q_balance = pd.DataFrame(
        {c: [10, 30, 5] for c in q_cols},
        index=["Ordinary Shares Number", "Total Debt", "Cash And Cash Equivalents"],
    )

    monkeypatch.setattr("core.data._read_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.data._write_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.data.fetch_price_history", lambda *a, **k: hist)
    monkeypatch.setattr("core.data.fetch_financials", lambda t: {"income": income, "balance": balance})
    monkeypatch.setattr(
        "core.data.fetch_quarterly_financials",
        lambda t: {"income": q_income, "balance": q_balance},
    )
    series = build_earnings_yield_history("AAA", years=10)
    assert series
    monkeypatch.setattr(
        "core.edgar_history.compute_valuation_vs_history_detail",
        lambda *a, **k: {"score": 72.0},
    )
    assert compute_valuation_vs_history("AAA", 0.08) == pytest.approx(72.0)


def test_edgar_history_companyfacts(monkeypatch):
    from core.edgar_history import (
        _annual_entries,
        _companyfacts_entries,
        _period_table_from_companyfacts,
        _price_on_or_before,
        load_period_table,
    )

    facts = {
        "facts": {
            "us-gaap": {
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            {"form": "10-K", "fp": "FY", "end": "2023-12-31", "val": 80, "filed": "2024-02-01"},
                            {"form": "10-Q", "fp": "Q1", "end": "2023-03-31", "val": 20, "filed": "2023-05-01"},
                            {"form": "10-K", "fp": "FY", "end": None, "val": 1, "filed": "2024-02-01"},
                        ]
                    }
                },
                "CommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"form": "10-K", "fp": "FY", "end": "2023-12-31", "val": 10, "filed": "2024-02-01"},
                        ]
                    }
                },
                "StockholdersEquity": {
                    "units": {
                        "USD": [
                            {"form": "10-K", "fp": "FY", "end": "2023-12-31", "val": 200, "filed": "2024-02-01"},
                        ]
                    }
                },
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": {
                    "units": {
                        "USD": [
                            {"form": "10-K", "fp": "FY", "end": "2023-12-31", "val": 210, "filed": "2024-02-01"},
                        ]
                    }
                },
            }
        }
    }
    entries = _companyfacts_entries(facts, "OperatingIncomeLoss")
    annual = _annual_entries(entries)
    assert len(annual) == 1
    monkeypatch.setattr("core.edgar_history.ticker_to_cik", lambda t: 123)
    monkeypatch.setattr("core.edgar_history._read_cache", lambda *a, **k: facts)
    cf_table = _period_table_from_companyfacts("AAA")
    assert "ebit" in cf_table.columns

    monkeypatch.setattr("core.edgar_history.ticker_to_cik", lambda t: None)
    assert _period_table_from_companyfacts("ZZZ").empty

    closes = pd.Series([10.0], index=pd.to_datetime(["2023-12-31"]))
    assert _price_on_or_before(closes, pd.Timestamp("2023-12-31")) == 10.0
    assert _price_on_or_before(closes, pd.Timestamp("2020-01-01")) is None

    monkeypatch.setattr(
        "core.edgar_facts.fetch_companyfacts_for_ticker",
        lambda t: pd.DataFrame({"x": [1]}),
    )
    monkeypatch.setattr(
        "core.edgar_facts.period_table_from_facts",
        lambda df, t: pd.DataFrame({"period": [pd.Timestamp("2023-12-31")], "ebit": [1]}),
    )
    loaded, src = load_period_table("AAA")
    assert src == "companyfacts"
    assert not loaded.empty


def test_valuation_detail_from_fundamentals(monkeypatch):
    from core.edgar_history import compute_valuation_vs_history_detail

    idx = pd.to_datetime([f"{y}-12-31" for y in range(2016, 2026)])
    annual = pd.DataFrame(
        {
            "ebit": [80 + i for i in range(10)],
            "operating_cashflow": [90 + i for i in range(10)],
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
    prices = pd.DataFrame(
        {"Close": [float(v) for v in annual["ebit"]]},
        index=idx,
    )
    monkeypatch.setattr("core.edgar_history.fetch_price_history", lambda *a, **k: prices)
    monkeypatch.setattr("core.fundamentals.get_fundamentals", lambda *a, **k: fund)
    detail = compute_valuation_vs_history_detail("X", 0.10, current_oe_yield=0.09, years=12)
    assert detail["metric"] == "ev_ebit_p_oe"
    assert detail["score"] is not None
    assert detail["source"] == "edgar"


def test_score_without_universe_and_watchlist(monkeypatch):
    from core.scoring import (
        _linear_score,
        _meaningful_mask,
        _score_without_universe,
        apply_fund_snapshot_scoring,
        evaluate_watchlist,
    )

    assert _linear_score(1, 2, 2) == 0.0
    mask = _meaningful_mask(pd.Series(["", "ok", "unknown", None]), "name")
    assert bool(mask.iloc[1])
    zero = _meaningful_mask(pd.Series([0, 1, np.nan]), "x")
    assert list(zero.fillna(False)) == [True, True, False]

    raw = {
        "name": "Acme",
        "sector": "Technology",
        "price": 50,
        "market_cap": 500,
        "dividend_yield": 0.01,
        "fifty_two_week_high": 80,
        "currency": "USD",
        "data_warnings": [],
        "data_quality": {"grade": "A"},
        "ticker": "AAA",
        "earnings_yield": 0.08,
        "graham_ratio": 0.9,
    }
    factors = {"altman_z": 3.0, "altman_z_pp": 4.0, "graham_ratio": 0.9, "earnings_yield": 0.08}
    out = _score_without_universe("AAA", raw, factors, {"consensus_label": "Buy"}, {"decision": {"mode": "legacy"}})
    assert out["composite"] is None
    assert out["warning"]
    assert "decision" in out

    fund_uni = pd.DataFrame(
        {
            "ticker": ["VTI", "VOO"],
            "composite": [70.0, 60.0],
            "factor_coverage_pct": [90.0, 80.0],
            "pct_cost": [80.0, 70.0],
        }
    )
    analysis = apply_fund_snapshot_scoring({"ticker": "VTI", "composite": 1}, fund_uni, "VTI")
    assert analysis["composite"] == 70.0
    assert apply_fund_snapshot_scoring({"ticker": "X"}, pd.DataFrame(), "X")["ticker"] == "X"
    assert apply_fund_snapshot_scoring({"ticker": "ZZZ"}, fund_uni, "ZZZ")["ticker"] == "ZZZ"

    monkeypatch.setattr("core.scoring.load_watchlist", lambda: ["AAA", "BBB"])
    monkeypatch.setattr("core.scoring.load_universe_snapshot", lambda: None)
    monkeypatch.setattr(
        "core.scoring.score_ticker",
        lambda t, cfg, uni: {"is_good_buy": t == "AAA", "ticker": t},
    )
    hits = evaluate_watchlist({"decision": {"mode": "legacy"}})
    assert hits[0]["ticker"] == "AAA"


def test_wiki_ticker_column_and_unknown_member(monkeypatch):
    html = """
    <html><body><table>
    <tr><th>Ticker</th></tr>
    <tr><td>XYZ</td></tr>
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
    assert "XYZ" in _wiki_symbol_list("https://example.test", ["ZZZ"])

    html2 = """
    <html><body><table>
    <tr><th>Code</th></tr>
    <tr><td>QQQ</td></tr>
    </table></body></html>
    """

    class _Resp2:
        def read(self):
            return html2.encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp2())
    assert "QQQ" in _wiki_symbol_list("https://example.test", ["ZZZ"])

    monkeypatch.setattr("core.universe.fetch_sp500_tickers", lambda: ["AAPL"])
    tagged = fetch_universe_tickers(["sp500", "unknown-index"])
    assert tagged == [("AAPL", "sp500")]
    assert prefer_us_listings(["AAPL", "AAPL", "SHOP.TO", "SHOP"]) == ["AAPL", "SHOP"]


def test_load_universe_snapshot_success_and_corrupt(tmp_path, monkeypatch):
    path = tmp_path / "uni.parquet"
    pd.DataFrame({"ticker": ["AAPL"]}).to_parquet(path, index=False)
    monkeypatch.setattr("core.universe.SNAPSHOT_PATH", path)
    df = load_universe_snapshot()
    assert df.iloc[0]["ticker"] == "AAPL"

    monkeypatch.setattr("pandas.read_parquet", lambda *a, **k: (_ for _ in ()).throw(OSError("bad")))
    assert load_universe_snapshot() is None


def test_cik_map_empty_and_bad_cik(monkeypatch):
    from core.sec import ticker_to_cik

    monkeypatch.setattr("core.sec.fetch_cik_ticker_map", lambda: pd.DataFrame())
    assert ticker_to_cik("AAPL") is None
    monkeypatch.setattr(
        "core.sec.fetch_cik_ticker_map",
        lambda: pd.DataFrame({"ticker": ["AAPL"], "cik": ["bad"]}),
    )
    assert ticker_to_cik("AAPL") is None


def test_cik_map_sec_http_path(monkeypatch):
    from core.sec import fetch_cik_ticker_map

    monkeypatch.setattr("core.sec._read_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.sec._write_cache", lambda *a, **k: None)

    def boom_import(*args, **kwargs):
        raise RuntimeError("no backtest store")

    import builtins

    real_import = builtins.__import__

    def guarded(name, *a, **k):
        if name == "backtest.data.edgar":
            raise RuntimeError("skip")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guarded)

    class Resp:
        def json(self):
            return {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}}

    monkeypatch.setattr("core.sec.sec_get", lambda *a, **k: Resp())
    df = fetch_cik_ticker_map(force=True)
    assert int(df.iloc[0]["cik"]) == 320193


def test_edgar_facts_fetch_and_bulk_zip(tmp_path, monkeypatch):
    from core.edgar_facts import (
        download_bulk_companyfacts,
        fetch_companyfacts,
        fetch_companyfacts_for_ticker,
        rows_from_bulk_zip,
    )
    from tests.test_edgar_facts import _companyfacts_fixture

    payload = _companyfacts_fixture()
    monkeypatch.setattr("core.edgar_facts._read_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.edgar_facts._write_cache", lambda *a, **k: None)

    class Resp:
        def json(self):
            return payload

        content = b"zip-bytes"

    monkeypatch.setattr("core.edgar_facts.sec_get", lambda *a, **k: Resp())
    assert fetch_companyfacts(123)["entityName"] == "FAKE INC"
    monkeypatch.setattr("core.edgar_facts.ticker_to_cik", lambda t: 123)
    rows = fetch_companyfacts_for_ticker("FAKE")
    assert not rows.empty
    monkeypatch.setattr("core.edgar_facts.ticker_to_cik", lambda t: None)
    assert fetch_companyfacts_for_ticker("ZZZ").empty

    monkeypatch.setattr("core.edgar_facts.sec_get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert fetch_companyfacts(1, force=True) is None
    assert download_bulk_companyfacts(tmp_path / "bulk.zip") is None

    zip_path = tmp_path / "cf.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("CIK0000000123.json", json.dumps(payload))
        zf.writestr("readme.txt", "skip")
        zf.writestr("CIK0000000999.json", "{not json")
        zf.writestr("CIK0000000456.json", json.dumps({"not": "facts"}))
    df = rows_from_bulk_zip(zip_path, ciks={123})
    assert not df.empty
    empty = rows_from_bulk_zip(zip_path, ciks={999999})
    assert empty.empty


def test_rates_fred_with_key_and_errors(monkeypatch):
    from core.rates import fetch_fred_history, fetch_fred_series, hurdle_rate

    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "observations": [
                    {"date": "2024-01-02", "value": "."},
                    {"date": "2024-01-01", "value": "4.20"},
                ]
            }

    monkeypatch.setattr("core.rates._read_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.rates._write_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.rates.requests.get", lambda *a, **k: Resp())
    assert fetch_fred_series("DGS10", api_key="abc", force=True) == pytest.approx(0.042)
    hist = fetch_fred_history("DGS10", api_key="abc", force=True)
    assert not hist.empty

    class Empty:
        def raise_for_status(self):
            return None

        def json(self):
            return {"observations": [{"value": "."}]}

    monkeypatch.setattr("core.rates.requests.get", lambda *a, **k: Empty())
    assert fetch_fred_series("DGS10", api_key="abc", force=True) is None

    class Boom:
        def raise_for_status(self):
            raise RuntimeError("x")

    monkeypatch.setattr("core.rates.requests.get", lambda *a, **k: Boom())
    assert fetch_fred_series("DGS10", api_key="abc", force=True) is None
    assert fetch_fred_history("DGS10", api_key="abc", force=True).empty

    monkeypatch.setattr(
        "core.rates.fred_observation",
        lambda sid: {"value": 0.05, "as_of": "2024-01-01"} if sid == "DGS10" else {"value": 0.06, "as_of": "2024-01-01"},
    )
    monkeypatch.setattr(
        "core.rates.market_rate_context",
        lambda: {"baa10y": {"value": 0.06}, "t10yie": {"value": 0.02}, "baa_spread": 0.01},
    )
    out = hurdle_rate(rf=None)
    assert out["rf_source"] == "FRED:DGS10"
    assert out["rf"] == pytest.approx(0.05)


def test_get_fundamentals_yahoo_and_mixed(monkeypatch):
    from core.fundamentals import (
        _annual_from_yahoo,
        attach_history_metrics,
        cagr,
        get_fundamentals,
        owner_earnings_per_share,
    )

    cols = [pd.Timestamp("2022-12-31"), pd.Timestamp("2023-12-31")]
    income = pd.DataFrame(
        {cols[0]: [100, 40, 20, 12], cols[1]: [110, 44, 22, 13]},
        index=["Total Revenue", "Gross Profit", "EBIT", "Net Income"],
    )
    balance = pd.DataFrame(
        {cols[0]: [500, 200, 50, 80, 10, 90], cols[1]: [520, 210, 55, 90, 10, 95]},
        index=[
            "Total Assets",
            "Stockholders Equity",
            "Cash And Cash Equivalents",
            "Total Debt",
            "Ordinary Shares Number",
            "Net PPE",
        ],
    )
    cashflow = pd.DataFrame(
        {cols[0]: [30, -10, 2], cols[1]: [32, -11, 2]},
        index=["Operating Cash Flow", "Capital Expenditure", "Stock Based Compensation"],
    )
    fin = {"income": income, "balance": balance, "cashflow": cashflow}
    annual = _annual_from_yahoo(fin)
    assert annual["revenue"].iloc[-1] == 110

    monkeypatch.setattr("core.fundamentals._read_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.fundamentals._write_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.fundamentals.fetch_companyfacts_for_ticker", lambda t: pd.DataFrame())
    monkeypatch.setattr("core.fundamentals.fetch_financials", lambda t: fin)
    ttm_income = pd.DataFrame({pd.Timestamp("2024-12-31"): [120, 50, 25]}, index=["Total Revenue", "Gross Profit", "EBIT"])
    ttm_cf = pd.DataFrame({pd.Timestamp("2024-12-31"): [35, -12]}, index=["Operating Cash Flow", "Capital Expenditure"])
    mrq = pd.DataFrame({pd.Timestamp("2024-12-31"): [540, 60, 12]}, index=["Total Assets", "Cash And Cash Equivalents", "Ordinary Shares Number"])
    monkeypatch.setattr(
        "core.fundamentals.fetch_ttm_financials",
        lambda t: {"income": ttm_income, "cashflow": ttm_cf, "balance": mrq},
    )
    fund = get_fundamentals("AAA", force=True)
    assert fund.source == "yahoo"
    assert fund.annual["revenue"].notna().any()

    s = pd.Series([10.0, 12.0], index=pd.to_datetime(["2022-12-31", "2023-12-31"]))
    assert cagr(s, years=5) is not None
    assert owner_earnings_per_share({"operating_cashflow": 50, "capex": -10, "sbc": 5, "shares_diluted": 10}) == pytest.approx(3.5)
    ser = pd.Series({"operating_cashflow": 50, "capex": -10, "sbc": 5, "shares_diluted": 10})
    assert owner_earnings_per_share(ser) == pytest.approx(3.5)

    monkeypatch.setattr("core.fundamentals.get_fundamentals", lambda t, **k: fund)
    raw = {"ticker": "AAA"}
    attach_history_metrics(raw)
    assert raw.get("_fundamentals") is fund


def test_quality_financials_and_value_trap_from_fund():
    from core.quality import compute_quality_score, flags_to_dicts, value_trap_flags

    df = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "sector": ["Financial Services", "Technology"],
            "gross_profitability": [0.2, 0.3],
            "roic": [0.1, 0.12],
            "fcf_margin": [0.1, 0.15],
            "earnings_quality": [0.8, 0.9],
            "fcf_conversion_3y": [0.9, 1.0],
            "financial_strength": [7, 8],
            "leverage_quality": [0.5, 0.6],
            "interest_coverage": [4, 8],
            "roic_5y_mean": [0.1, 0.12],
            "stability_roic": [0.02, 0.01],
            "gross_margin_5y_delta": [0.0, 0.01],
            "revenue_5y_cagr": [0.05, 0.08],
            "shareholder_yield": [0.03, 0.04],
            "investment": [-0.02, -0.01],
            "anti_dilution": [0.01, 0.0],
            "roe": [0.12, 0.15],
            "roa": [0.04, 0.06],
            "equity_to_assets": [0.08, 0.4],
        }
    )
    scored = compute_quality_score(df)
    assert "quality_score" in scored.columns

    idx = pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31"])
    annual = pd.DataFrame(
        {"ebit": [30, 20, 10], "total_assets": [100, 100, 100]},
        index=idx,
    )
    fund = Fundamentals("X", annual, annual.iloc[-1], annual.iloc[-1], "edgar", idx[-1].date())
    flags = value_trap_flags({"roic": 0.04, "sector": "Technology"}, fund)
    codes = {f.code for f in flags}
    assert "ROIC_DECLINING" in codes
    assert any(d["code"] == "ROIC_DECLINING" for d in flags_to_dicts(flags))


def test_providers_factory_and_finnhub_live(monkeypatch):
    from core.providers.base import get_fundamentals_providers
    from core.providers.finnhub import FinnhubProvider

    providers = get_fundamentals_providers(
        {"providers": {"fundamentals": ["edgar", "yahoo", "unknown", "sharadar"]}}
    )
    names = [p.name for p in providers]
    assert "edgar" in names
    assert "yahoo" in names

    monkeypatch.setenv("FINNHUB_API_KEY", "tok")
    p = FinnhubProvider(api_key="tok")

    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"metric": {"peTTM": 20}, "data": [{"year": 2023, "revenue": 1}]}

    monkeypatch.setattr("core.providers.finnhub.requests.get", lambda *a, **k: Resp())
    assert p.ttm("AAPL")["peTTM"] == 20
    assert not p.annual("AAPL").empty
    assert isinstance(p.estimates("AAPL"), dict)

    monkeypatch.setattr(
        "core.providers.finnhub.requests.get",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")),
    )
    assert p.ttm("AAPL") == {}
    assert p.annual("AAPL").empty
    assert p.estimates("AAPL") == {}


def test_analysts_rating_helpers_and_sell_fallback():
    from core.analysts import _rating_to_score, _score_to_label, aggregate_analyst_data

    assert _rating_to_score("overweight-buy") == 4.0
    assert _rating_to_score("zzz") is None
    assert _score_to_label(1.0) == "Sell"
    assert _score_to_label(2.0) == "Underperform"
    out = aggregate_analyst_data({"recommendation_key": "sell", "recommendations": pd.DataFrame()})
    assert out["consensus_label"] in {"Sell", "Underperform"}


def test_fund_universe_load_corrupt_and_skip_no_price(tmp_path, monkeypatch):
    from core.fund_universe import build_fund_universe_snapshot, load_fund_universe_snapshot

    path = tmp_path / "funds.parquet"
    monkeypatch.setattr("core.fund_universe.FUND_SNAPSHOT_PATH", path)
    monkeypatch.setattr("core.fund_universe.FUND_SNAPSHOT_META_PATH", tmp_path / "meta.json")
    monkeypatch.setattr("pandas.read_parquet", lambda *a, **k: (_ for _ in ()).throw(OSError("bad")))
    path.write_bytes(b"x")
    assert load_fund_universe_snapshot() is None

    monkeypatch.setattr("core.fund_universe.build_fund_raw_metrics", lambda t: {"price": None, "name": t})
    monkeypatch.setattr("core.fund_universe.throttle", lambda s: None)
    with pytest.raises(Exception):
        build_fund_universe_snapshot(tickers=["AAA"], min_success_ratio=0.9)


def test_data_quality_helpers():
    from core.data_quality import _as_warning, _pct_diff, _safe_float, presence_pct, warning_message

    assert warning_message({"message": "hi"}) == "hi"
    assert warning_message("plain") == "plain"
    assert _as_warning("x")["message"] == "x"
    assert _safe_float("bad") is None
    assert _safe_float(float("nan")) is None
    assert _pct_diff(10, 10) == 0.0
    assert presence_pct({}, required=()) == 1.0


def test_config_loader_helpers(tmp_path):
    from core.config import (
        get_provider_names,
        get_quality_weights,
        get_universe_members,
        load_config,
    )

    p = tmp_path / "c.yaml"
    p.write_text("universe:\n  members: [sp500, tsx60]\n", encoding="utf-8")
    cfg = load_config(p)
    assert get_universe_members(cfg) == ["sp500", "tsx60"]
    assert get_provider_names({}) == ["edgar", "yahoo"]
    qw = get_quality_weights({})
    assert sum(qw.values()) == pytest.approx(1.0)
