"""Data fetching with yfinance primary and optional OpenBB fallback."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from core.config import ROOT

logger = logging.getLogger(__name__)

CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Optional OpenBB
try:
    from openbb import obb

    HAS_OPENBB = True
except ImportError:
    HAS_OPENBB = False
    obb = None


def _cache_key(prefix: str, *parts: str) -> Path:
    raw = "|".join([prefix, *parts])
    digest = hashlib.md5(raw.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{prefix}_{digest}.json"


def _read_cache(path: Path, max_age_hours: float = 6) -> Any | None:
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > max_age_hours * 3600:
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, default=str)


def _safe_float(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def normalize_debt_to_equity(val: float | None) -> float | None:
    """Yahoo reports debt/equity as a percentage (e.g. 286 = 2.86x); convert to ratio."""
    if val is None:
        return None
    if abs(val) > 10:
        # 10 < |val| <= 40 is ambiguous: could be a percent (0.1x-0.4x) or a
        # genuinely extreme leverage ratio that this heuristic would misread.
        if abs(val) <= 40:
            logger.warning(
                "debt_to_equity=%.2f in ambiguous unit range; treating as percent (%.2fx)",
                val,
                val / 100.0,
            )
        return val / 100.0
    return val


def _financial_columns_newest_first(df: pd.DataFrame) -> list:
    """Return financial statement column labels sorted newest-first."""
    if df.empty:
        return []

    def _col_date(col: Any) -> pd.Timestamp:
        try:
            return pd.to_datetime(col)
        except (TypeError, ValueError):
            return pd.NaT

    dated = [(col, _col_date(col)) for col in df.columns]
    dated.sort(key=lambda item: (pd.isna(item[1]), item[1]), reverse=True)
    dated_cols = [col for col, dt in dated if pd.notna(dt)]
    return dated_cols if dated_cols else list(df.columns)


def extract_financial_values(
    col_names: list[str],
    df: pd.DataFrame,
) -> tuple[float | None, float | None, list[str]]:
    """
    Return (latest, prior, warnings) for the first matching financial statement row.
    Columns are read in date order (newest first).
    """
    warnings: list[str] = []
    if df.empty:
        return None, None, warnings

    sorted_cols = _financial_columns_newest_first(df)
    for name in col_names:
        if name not in df.index:
            continue
        row = df.loc[name]
        ordered: list[tuple[str, float]] = []
        for col in sorted_cols:
            if col not in row.index:
                continue
            fv = _safe_float(row[col])
            if fv is not None:
                ordered.append((str(col), fv))
        if not ordered:
            continue

        latest_val = ordered[0][1]
        prior_val = ordered[1][1] if len(ordered) >= 2 else None
        if sorted_cols and sorted_cols[0] in row.index and _safe_float(row[sorted_cols[0]]) is None:
            warnings.append(
                f"{name}: most recent period ({sorted_cols[0]}) empty; using {ordered[0][0]}"
            )
        return latest_val, prior_val, warnings

    return None, None, warnings


def latest_financial(col_names: list[str], df: pd.DataFrame) -> float | None:
    latest, _, _ = extract_financial_values(col_names, df)
    return latest


def prior_financial(col_names: list[str], df: pd.DataFrame) -> float | None:
    _, prior, _ = extract_financial_values(col_names, df)
    return prior


FUND_QUOTE_TYPES = frozenset({"ETF", "MUTUALFUND"})


def get_security_type(ticker: str) -> str:
    """Return the Yahoo quote type, e.g. EQUITY, ETF, MUTUALFUND (cached)."""
    cache_path = _cache_key("qtype", ticker.upper())
    cached = _read_cache(cache_path, max_age_hours=168)
    if cached is not None:
        return str(cached.get("quote_type") or "EQUITY")

    try:
        info = yf.Ticker(ticker).info or {}
        quote_type = (info.get("quoteType") or "EQUITY").upper()
        _write_cache(cache_path, {"quote_type": quote_type})
        return quote_type
    except Exception as exc:
        logger.warning("Quote type check failed for %s: %s", ticker, exc)
        return "EQUITY"


def is_etf(ticker: str) -> bool:
    """Return True if ticker appears to be an ETF."""
    return get_security_type(ticker) == "ETF"


def is_fund(ticker: str) -> bool:
    """Return True for pooled funds (ETFs and mutual funds)."""
    return get_security_type(ticker) in FUND_QUOTE_TYPES


def fetch_price_history(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV history; returns empty DataFrame on failure."""
    cache_path = _cache_key("hist", ticker.upper(), period, interval)
    cached = _read_cache(cache_path, max_age_hours=12)
    if cached is not None:
        df = pd.DataFrame(cached)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
            df = df.set_index("Date")
        return df

    try:
        hist = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()
        out = hist.reset_index()
        if "Date" in out.columns:
            out["Date"] = pd.to_datetime(out["Date"], utc=True).dt.tz_localize(None).astype(str)
        _write_cache(cache_path, out.to_dict(orient="records"))
        hist.index = pd.to_datetime(hist.index, utc=True).tz_localize(None)
        return hist
    except Exception as exc:
        logger.warning("Price history failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def fetch_all_time_high(ticker: str) -> float | None:
    """Max adjusted close from full price history (cached)."""
    cache_path = _cache_key("ath", ticker.upper())
    cached = _read_cache(cache_path, max_age_hours=24)
    if cached is not None:
        return _safe_float(cached.get("all_time_high"))

    hist = fetch_price_history(ticker, period="max")
    if hist.empty or "Close" not in hist.columns:
        return None
    ath = _safe_float(hist["Close"].max())
    if ath is not None:
        _write_cache(cache_path, {"all_time_high": ath})
    return ath


def _info_is_usable(info: dict[str, Any]) -> bool:
    """Quote summary must include price and a display name to be cached or trusted."""
    if not info:
        return False
    has_price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice")) is not None
    has_name = bool(info.get("longName") or info.get("shortName"))
    return has_price and has_name


def fetch_ticker_info(ticker: str) -> dict[str, Any]:
    """Fetch yfinance info dict with caching."""
    cache_path = _cache_key("info", ticker.upper())
    cached = _read_cache(cache_path, max_age_hours=24)
    if cached is not None and _info_is_usable(cached):
        return cached
    if cached is not None:
        logger.warning("Ignoring stale/empty info cache for %s", ticker)

    try:
        info = yf.Ticker(ticker).info or {}
        serializable = {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool, type(None)))}
        if _info_is_usable(serializable):
            _write_cache(cache_path, serializable)
            return serializable
        logger.warning("Quote info incomplete for %s (keys=%d)", ticker, len(serializable))
        return serializable
    except Exception as exc:
        logger.warning("Info fetch failed for %s: %s", ticker, exc)
        return {}


def currency_symbol(currency: str | None) -> str:
    """Display symbol for a listing currency (C$ distinguishes CAD from USD)."""
    cur = (currency or "USD").upper()
    if cur == "CAD":
        return "C$"
    if cur == "USD":
        return "$"
    return f"{cur} "


def listing_amount(raw: dict[str, Any], key: str) -> Any:
    """Prefer the listing-currency copy of a market field, else the working value."""
    listing_key = f"{key}_listing"
    if listing_key in raw and raw[listing_key] is not None:
        return raw[listing_key]
    return raw.get(key)


def fetch_fx_rate(pair: str) -> float | None:
    """
    Spot FX for a 6-letter pair like CADUSD (units of quote per 1 unit of base).

    Cached 24h. Tries Yahoo ``{pair}=X`` then the inverse pair.
    """
    raw = (pair or "").upper().replace("=", "").replace("X", "")
    if len(raw) != 6 or not raw.isalpha():
        return None
    base, quote = raw[:3], raw[3:]
    if base == quote:
        return 1.0

    cache_path = _cache_key("fx", base, quote)
    cached = _read_cache(cache_path, max_age_hours=24)
    if cached is not None:
        rate = _safe_float(cached.get("rate"))
        if rate is not None and rate > 0:
            return rate

    def _last_close(symbol: str) -> float | None:
        try:
            hist = yf.Ticker(symbol).history(period="5d")
        except Exception as exc:
            logger.debug("FX history failed for %s: %s", symbol, exc)
            return None
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        return _safe_float(hist["Close"].dropna().iloc[-1] if not hist["Close"].dropna().empty else None)

    rate = _last_close(f"{base}{quote}=X")
    if rate is None or rate <= 0:
        inv = _last_close(f"{quote}{base}=X")
        if inv is not None and inv > 0:
            rate = 1.0 / inv
    if rate is None or rate <= 0:
        logger.warning("FX rate unavailable for %s%s", base, quote)
        return None
    _write_cache(cache_path, {"rate": rate, "pair": f"{base}{quote}"})
    return rate


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    """Restore a financial DataFrame from cached records, preserving the metric-name index."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "index" in df.columns:
        df = df.set_index("index")
        df.index.name = None
    return df


def fetch_financials(ticker: str) -> dict[str, pd.DataFrame]:
    """Fetch annual income statement, balance sheet, cash flow."""
    cache_path = _cache_key("fin", ticker.upper())
    cached = _read_cache(cache_path, max_age_hours=48)
    if cached is not None:
        return {k: _records_to_df(v) for k, v in cached.items()}

    try:
        t = yf.Ticker(ticker)
        result = {
            "income": _df_to_records(t.financials),
            "balance": _df_to_records(t.balance_sheet),
            "cashflow": _df_to_records(t.cashflow),
        }
        _write_cache(cache_path, result)
        return {
            "income": t.financials if t.financials is not None else pd.DataFrame(),
            "balance": t.balance_sheet if t.balance_sheet is not None else pd.DataFrame(),
            "cashflow": t.cashflow if t.cashflow is not None else pd.DataFrame(),
        }
    except Exception as exc:
        logger.warning("Financials failed for %s: %s", ticker, exc)
        return {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}


def fetch_ttm_financials(ticker: str) -> dict[str, pd.DataFrame]:
    """TTM income/cashflow plus most-recent quarterly balance sheet (cached 48h)."""
    cache_path = _cache_key("ttmfin", ticker.upper())
    cached = _read_cache(cache_path, max_age_hours=48)
    if cached is not None:
        return {k: _records_to_df(v) for k, v in cached.items()}

    empty = {"income": pd.DataFrame(), "cashflow": pd.DataFrame(), "balance": pd.DataFrame()}
    try:
        t = yf.Ticker(ticker)
        income = getattr(t, "ttm_income_stmt", None)
        cashflow = getattr(t, "ttm_cashflow", None)
        balance = getattr(t, "quarterly_balance_sheet", None)
        if not isinstance(income, pd.DataFrame):
            income = pd.DataFrame()
        if not isinstance(cashflow, pd.DataFrame):
            cashflow = pd.DataFrame()
        if not isinstance(balance, pd.DataFrame):
            balance = pd.DataFrame()
        _write_cache(
            cache_path,
            {
                "income": _df_to_records(income),
                "cashflow": _df_to_records(cashflow),
                "balance": _df_to_records(balance),
            },
        )
        return {"income": income, "cashflow": cashflow, "balance": balance}
    except Exception as exc:
        logger.warning("TTM financials failed for %s: %s", ticker, exc)
        return empty


def _statement_period_end(df: pd.DataFrame) -> str | None:
    cols = _financial_columns_newest_first(df)
    if not cols:
        return None
    return str(cols[0])


def _df_to_records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    out = df.copy()
    out.index = out.index.astype(str)
    out.columns = [str(c) for c in out.columns]
    return out.reset_index().to_dict(orient="records")


def fetch_analyst_recommendations(ticker: str) -> pd.DataFrame:
    """Fetch analyst recommendation history from yfinance."""
    cache_path = _cache_key("recs", ticker.upper())
    cached = _read_cache(cache_path, max_age_hours=24)
    if cached is not None:
        return pd.DataFrame(cached)

    try:
        recs = yf.Ticker(ticker).recommendations
        if recs is None or recs.empty:
            return pd.DataFrame()
        out = recs.reset_index()
        out.columns = [str(c) for c in out.columns]
        if "Date" in out.columns:
            out["Date"] = out["Date"].astype(str)
        _write_cache(cache_path, out.to_dict(orient="records"))
        return recs
    except Exception as exc:
        logger.warning("Recommendations failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def fetch_analyst_price_targets_openbb(ticker: str) -> pd.DataFrame:
    """Try OpenBB price targets; returns empty DataFrame if unavailable."""
    if not HAS_OPENBB:
        return pd.DataFrame()
    try:
        result = obb.equity.estimates.price_target(symbol=ticker, provider="yfinance")
        if result and hasattr(result, "results") and result.results is not None:
            df = result.to_df()
            return df if df is not None else pd.DataFrame()
    except Exception as exc:
        logger.debug("OpenBB price targets failed for %s: %s", ticker, exc)
    return pd.DataFrame()


def normalize_expense_ratio(info: dict[str, Any]) -> float | None:
    """
    Expense ratio as a decimal fraction (0.0009 = 0.09%).

    Yahoo exposes two fields with different units: ``annualReportExpenseRatio``
    is a fraction, while the newer ``netExpenseRatio`` is in percentage points.
    """
    frac = _safe_float(info.get("annualReportExpenseRatio"))
    if frac is not None and frac > 0:
        return frac
    pct = _safe_float(info.get("netExpenseRatio"))
    if pct is not None and pct > 0:
        return pct / 100.0
    return None


def normalize_fund_yield(info: dict[str, Any]) -> float | None:
    """
    Distribution/dividend yield as a decimal fraction (0.013 = 1.3%).

    ``yield`` is a fraction; newer ``dividendYield`` is in percentage points.
    Values above 0.15 are assumed to be percentages (15%+ fund yields are
    implausible as fractions in this universe).
    """
    def _percent_to_fraction(v: float, field: str) -> float:
        # 0.15 < v < 1.0 is ambiguous: a sub-1% "percent" value or a 15%+
        # fraction. Flag it so bad units don't silently skew the income factor.
        if v < 1.0:
            logger.warning(
                "%s=%.4f in ambiguous unit range; treating as percent (%.4f)",
                field,
                v,
                v / 100.0,
            )
        return v / 100.0

    val = _safe_float(info.get("yield"))
    if val is None:
        val = _safe_float(info.get("dividendYield"))
        if val is not None and val > 0.15:
            val = _percent_to_fraction(val, "dividendYield")
    elif val > 0.15:
        val = _percent_to_fraction(val, "yield")
    return val


def _normalize_avg_return(val: float | None) -> float | None:
    """Annualized return as fraction; values beyond ±1.5 are percentage points."""
    if val is None:
        return None
    if abs(val) > 1.5:
        return val / 100.0
    return val


def fetch_fund_info(ticker: str) -> dict[str, Any]:
    """Metadata view for pooled funds (ETFs and mutual funds)."""
    info = fetch_ticker_info(ticker)
    price = _safe_float(
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("navPrice")
        or info.get("previousClose")
    )
    return {
        "symbol": ticker.upper(),
        "name": info.get("longName") or info.get("shortName"),
        "quote_type": (info.get("quoteType") or "").upper(),
        "category": info.get("category"),
        "fund_family": info.get("fundFamily"),
        "currency": (info.get("currency") or "USD").upper(),
        "exchange": info.get("fullExchangeName") or info.get("exchange"),
        "expense_ratio": normalize_expense_ratio(info),
        "total_assets": _safe_float(info.get("totalAssets")),
        "yield": normalize_fund_yield(info),
        "nav_price": _safe_float(info.get("navPrice")),
        "current_price": price,
        "ytd_return": _normalize_avg_return(_safe_float(info.get("ytdReturn"))),
        "three_year_avg_return": _normalize_avg_return(
            _safe_float(info.get("threeYearAverageReturn"))
        ),
        "five_year_avg_return": _normalize_avg_return(
            _safe_float(info.get("fiveYearAverageReturn"))
        ),
        "beta_3y": _safe_float(info.get("beta3Year")),
        "fund_inception": info.get("fundInceptionDate"),
        "fifty_two_week_high": _safe_float(info.get("fiftyTwoWeekHigh")),
        "fifty_two_week_low": _safe_float(info.get("fiftyTwoWeekLow")),
        "description": info.get("longBusinessSummary"),
    }


def fetch_etf_info(ticker: str) -> dict[str, Any]:
    """Backward-compatible alias for :func:`fetch_fund_info`."""
    return fetch_fund_info(ticker)


def fetch_etf_holdings(ticker: str, top_n: int = 10) -> pd.DataFrame:
    """Top ETF holdings when available via yfinance."""
    try:
        t = yf.Ticker(ticker)
        if hasattr(t, "fund_holding_info"):
            holdings = t.fund_holding_info
            if isinstance(holdings, dict) and "holdings" in holdings:
                df = pd.DataFrame(holdings["holdings"])
                return df.head(top_n)
    except Exception:
        pass
    return pd.DataFrame()


def _compute_trailing_return(hist: pd.DataFrame, years: float) -> float | None:
    """Annualized (CAGR) trailing return over ``years`` from daily closes."""
    if hist.empty or "Close" not in hist.columns:
        return None
    closes = hist["Close"].dropna()
    span = int(round(252 * years))
    # Require at least ~90% of the window so young funds don't get inflated CAGRs.
    if len(closes) < int(span * 0.9):
        return None
    start = _safe_float(closes.iloc[max(-len(closes), -span - 1)])
    end = _safe_float(closes.iloc[-1])
    if start is None or end is None or start <= 0 or end <= 0:
        return None
    total = end / start
    if years <= 1:
        return total - 1.0
    return float(total ** (1.0 / years) - 1.0)


def build_fund_raw_metrics(ticker: str) -> dict[str, Any]:
    """
    Assemble raw inputs for fund (ETF / mutual fund) factor computation.

    Uses fund metadata plus price/NAV history; company financial statements do
    not exist for pooled funds, so no balance-sheet style metrics are attempted.
    """
    fund = fetch_fund_info(ticker)
    hist = fetch_price_history(ticker, period="max")

    price = fund.get("current_price")
    if price is None and not hist.empty and "Close" in hist.columns:
        price = _safe_float(hist["Close"].iloc[-1])

    return_1y = _compute_trailing_return(hist, 1.0)
    return_3y = _compute_trailing_return(hist, 3.0)
    return_5y = _compute_trailing_return(hist, 5.0)
    if return_3y is None:
        return_3y = fund.get("three_year_avg_return")
    if return_5y is None:
        return_5y = fund.get("five_year_avg_return")

    momentum_12_1 = _compute_momentum_12_1(hist)
    volatility_12m = _compute_volatility_12m(hist)
    drawdown_metrics = _compute_drawdown_metrics(hist)
    rsi_14 = _compute_rsi(hist)

    all_time_high = None
    if not hist.empty and "Close" in hist.columns:
        all_time_high = _safe_float(hist["Close"].max())

    fifty_two_week_high = fund.get("fifty_two_week_high")
    fifty_two_week_low = fund.get("fifty_two_week_low")
    if not hist.empty and "Close" in hist.columns:
        window = hist["Close"].dropna().tail(252)
        if fifty_two_week_high is None and len(window) > 0:
            fifty_two_week_high = _safe_float(window.max())
        if fifty_two_week_low is None and len(window) > 0:
            fifty_two_week_low = _safe_float(window.min())

    return {
        "ticker": ticker.upper(),
        "name": fund.get("name") or ticker.upper(),
        "quote_type": fund.get("quote_type"),
        "category": fund.get("category"),
        "fund_family": fund.get("fund_family"),
        "currency": fund.get("currency"),
        "exchange": fund.get("exchange"),
        "price": price,
        "nav_price": fund.get("nav_price"),
        "expense_ratio": fund.get("expense_ratio"),
        "total_assets": fund.get("total_assets"),
        "distribution_yield": fund.get("yield"),
        "beta_3y": fund.get("beta_3y"),
        "fund_inception": fund.get("fund_inception"),
        "ytd_return": fund.get("ytd_return"),
        "return_1y": return_1y,
        "return_3y": return_3y,
        "return_5y": return_5y,
        "momentum_12_1": momentum_12_1,
        "volatility_12m": volatility_12m,
        "max_drawdown": drawdown_metrics.get("max_drawdown"),
        "downside_deviation": drawdown_metrics.get("downside_deviation"),
        "rsi_14": rsi_14,
        "fifty_two_week_high": fifty_two_week_high,
        "fifty_two_week_low": fifty_two_week_low,
        "all_time_high": all_time_high,
        "description": fund.get("description"),
        "price_history": hist,
    }


_FLOW_REVENUE = ["Total Revenue", "Operating Revenue"]
_FLOW_GROSS = ["Gross Profit"]
_FLOW_NI = ["Net Income", "Net Income Common Stockholders"]
_FLOW_EBIT = ["EBIT", "Operating Income"]
_FLOW_OI = ["Operating Income"]
_FLOW_OCF = ["Operating Cash Flow"]
_FLOW_FCF = ["Free Cash Flow"]
_FLOW_CAPEX = ["Capital Expenditure", "Capital Expenditures"]
_FLOW_SBC = ["Stock Based Compensation", "Share Based Compensation"]
_FLOW_INTEREST = ["Interest Expense", "Interest Expense Non Operating"]
_FLOW_TAX = ["Tax Provision", "Income Tax Expense"]
_FLOW_PRETAX = ["Pretax Income"]
_FLOW_DA = ["Reconciled Depreciation", "Depreciation And Amortization"]
_FLOW_DIV = [
    "Cash Dividends Paid",
    "Common Stock Dividend Paid",
    "Payment Of Dividends",
    "Dividends Paid",
]
_FLOW_BUYBACK = [
    "Repurchase Of Capital Stock",
    "Common Stock Payments",
    "Repurchase Of Common Stock",
    "Repurchase Of Stock",
]
_BS_ASSETS = ["Total Assets"]
_BS_LIAB = ["Total Liabilities Net Minority Interest", "Total Liabilities"]
_BS_CA = ["Current Assets"]
_BS_CL = ["Current Liabilities"]
_BS_LTD = ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]
_BS_STD = [
    "Current Debt",
    "Current Debt And Capital Lease Obligation",
    "Short Long Term Debt",
]
_BS_DEBT = ["Total Debt"]
_BS_CASH = [
    "Cash Cash Equivalents And Short Term Investments",
    "Cash And Cash Equivalents",
    "Cash",
]
_BS_EQUITY = ["Stockholders Equity", "Common Stock Equity"]
_BS_RE = [
    "Retained Earnings",
    "Retained Earnings Total Equity",
    "Retained Earnings Accumulated Deficit",
]
_BS_SHARES = ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"]
_BS_PPE = ["Net PPE", "Net Property Plant And Equipment"]
_BS_GW = ["Goodwill"]


def build_raw_metrics(ticker: str) -> dict[str, Any]:
    """
    Assemble raw inputs needed for factor computation for a single ticker.

    Flows (income/cashflow) use TTM statements with annual fallback.
    Balance-sheet items use the most recent quarterly sheet with annual fallback.
    Book equity is the statement equity line, never Yahoo BVPS × shares.
    """
    info = fetch_ticker_info(ticker)
    fin = fetch_financials(ticker)
    ttm = fetch_ttm_financials(ticker)
    hist = fetch_price_history(ticker, period="2y")
    recs = fetch_analyst_recommendations(ticker)
    from core.estimates import fetch_estimate_tables
    from core.insiders import fetch_form4_transactions

    estimate_tables = fetch_estimate_tables(ticker)
    try:
        form4_transactions = fetch_form4_transactions(ticker)
    except Exception as exc:
        logger.warning("Form 4 fetch failed for %s: %s", ticker, exc)
        form4_transactions = []

    price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    if price is None and not hist.empty and "Close" in hist.columns:
        price = _safe_float(hist["Close"].iloc[-1])
    market_cap = _safe_float(info.get("marketCap"))
    yahoo_ev = _safe_float(info.get("enterpriseValue"))
    yahoo_book_value = _safe_float(info.get("bookValue"))
    yahoo_shares = _safe_float(info.get("sharesOutstanding"))

    annual_income = fin.get("income", pd.DataFrame())
    annual_balance = fin.get("balance", pd.DataFrame())
    annual_cashflow = fin.get("cashflow", pd.DataFrame())
    ttm_income = ttm.get("income", pd.DataFrame())
    ttm_cashflow = ttm.get("cashflow", pd.DataFrame())
    mrq_balance = ttm.get("balance", pd.DataFrame())
    data_warnings: list[str] = []

    def _latest(names: list[str], df: pd.DataFrame) -> float | None:
        latest, _, warns = extract_financial_values(names, df)
        data_warnings.extend(warns)
        return latest

    def _annual_pair(names: list[str], df: pd.DataFrame) -> tuple[float | None, float | None]:
        latest, prior, warns = extract_financial_values(names, df)
        data_warnings.extend(warns)
        return latest, prior

    used_ttm_flow = False

    def _flow(names: list[str], ttm_df: pd.DataFrame, ann_df: pd.DataFrame) -> float | None:
        nonlocal used_ttm_flow
        val = _latest(names, ttm_df)
        if val is not None:
            used_ttm_flow = True
            return val
        return _latest(names, ann_df)

    used_mrq = False

    def _bs(names: list[str]) -> float | None:
        nonlocal used_mrq
        val = _latest(names, mrq_balance)
        if val is not None:
            used_mrq = True
            return val
        return _latest(names, annual_balance)

    revenue = _flow(_FLOW_REVENUE, ttm_income, annual_income)
    gross_profit = _flow(_FLOW_GROSS, ttm_income, annual_income)
    net_income = _flow(_FLOW_NI, ttm_income, annual_income)
    ebit = _flow(_FLOW_EBIT, ttm_income, annual_income)
    operating_income = _flow(_FLOW_OI, ttm_income, annual_income)
    interest_expense = _flow(_FLOW_INTEREST, ttm_income, annual_income)
    tax_expense = _flow(_FLOW_TAX, ttm_income, annual_income)
    pretax_income = _flow(_FLOW_PRETAX, ttm_income, annual_income)
    depreciation = _flow(_FLOW_DA, ttm_income, annual_income)
    sbc = _flow(_FLOW_SBC, ttm_income, annual_income)
    operating_cashflow = _flow(_FLOW_OCF, ttm_cashflow, annual_cashflow)
    free_cashflow = _flow(_FLOW_FCF, ttm_cashflow, annual_cashflow)
    capex = _flow(_FLOW_CAPEX, ttm_cashflow, annual_cashflow)
    dividends_paid = _flow(_FLOW_DIV, ttm_cashflow, annual_cashflow)
    repurchase_of_stock = _flow(_FLOW_BUYBACK, ttm_cashflow, annual_cashflow)

    if interest_expense is not None and interest_expense < 0:
        interest_expense = abs(interest_expense)
    if free_cashflow is None and operating_cashflow is not None and capex is not None:
        free_cashflow = operating_cashflow - abs(capex)

    _, revenue_prior = _annual_pair(_FLOW_REVENUE, annual_income)
    _, gross_profit_prior = _annual_pair(_FLOW_GROSS, annual_income)
    _, net_income_prior = _annual_pair(_FLOW_NI, annual_income)
    total_assets_fy, total_assets_prior = _annual_pair(_BS_ASSETS, annual_balance)
    _, total_liabilities_prior = _annual_pair(_BS_LIAB, annual_balance)
    _, current_assets_prior = _annual_pair(_BS_CA, annual_balance)
    _, current_liabilities_prior = _annual_pair(_BS_CL, annual_balance)
    long_term_debt_fy, long_term_debt_prior = _annual_pair(_BS_LTD, annual_balance)
    _, shares_prior = _annual_pair(_BS_SHARES, annual_balance)

    total_assets = _bs(_BS_ASSETS)
    if total_assets is None:
        total_assets = total_assets_fy
    total_liabilities = _bs(_BS_LIAB)
    current_assets = _bs(_BS_CA)
    current_liabilities = _bs(_BS_CL)
    long_term_debt = _bs(_BS_LTD)
    if long_term_debt is None:
        long_term_debt = long_term_debt_fy
    short_term_debt = _bs(_BS_STD)
    retained_earnings = _bs(_BS_RE)
    book_equity = _bs(_BS_EQUITY)
    ppe_net = _bs(_BS_PPE)
    goodwill = _bs(_BS_GW)
    shares = _bs(_BS_SHARES)
    if shares is None:
        shares = yahoo_shares
    # Dual-class share-basis reconciliation. Statement/EDGAR share rows sometimes
    # report a single class (e.g. Berkshire Class A) whose count is inconsistent
    # with the listing price and per-share earnings. market_cap / price is the
    # share count on the same basis as `price` and `trailing_eps`, so per-share
    # metrics (Graham, BVPS) must use it when the statement figure is materially
    # smaller; otherwise BVPS is inflated by orders of magnitude.
    implied_shares = None
    if market_cap is not None and price is not None and price > 0:
        implied_shares = market_cap / price
    if (
        implied_shares is not None
        and implied_shares > 0
        and (shares is None or shares <= 0 or implied_shares / shares > 1.5)
    ):
        if shares is not None and shares > 0:
            data_warnings.append(
                {
                    "code": "SHARE_BASIS",
                    "message": (
                        f"shares_outstanding: statement {shares:.4g} inconsistent "
                        f"with market cap / price {implied_shares:.4g}; using "
                        f"market-basis shares for per-share metrics (dual-class)"
                    ),
                    "severity": "warning",
                }
            )
        shares = implied_shares

    total_debt = _bs(_BS_DEBT)
    if total_debt is None and long_term_debt is not None and short_term_debt is not None:
        total_debt = long_term_debt + short_term_debt
    if total_debt is None:
        total_debt = _safe_float(info.get("totalDebt"))

    total_cash = _bs(_BS_CASH)
    if total_cash is None:
        total_cash = _safe_float(info.get("totalCash"))

    # D/E from the balance sheet only — do not guess Yahoo percent vs ratio units.
    debt_to_equity = None
    if book_equity is not None and book_equity > 0 and total_debt is not None:
        debt_to_equity = total_debt / book_equity

    currency = (info.get("currency") or "USD").upper()
    financial_currency = (info.get("financialCurrency") or currency).upper()
    fx_to_financial = 1.0
    price_listing = price
    market_cap_listing = market_cap
    if currency != financial_currency:
        fx = fetch_fx_rate(f"{currency}{financial_currency}")
        if fx is None or fx <= 0:
            fx_to_financial = None
            data_warnings.append(
                {
                    "code": "FX_MISSING",
                    "message": (
                        f"listing {currency} vs financials {financial_currency}; "
                        "FX unavailable, ratios not converted"
                    ),
                    "severity": "error",
                }
            )
        else:
            fx_to_financial = fx

            def _fx(val: float | None) -> float | None:
                return None if val is None else val * fx

            price = _fx(price)
            market_cap = _fx(market_cap)
            yahoo_ev = _fx(yahoo_ev)
            data_warnings.append(
                {
                    "code": "FX_CONVERTED",
                    "message": (
                        f"converted {currency} price/market fields to {financial_currency} "
                        f"at {fx:.4f}"
                    ),
                    "severity": "info",
                }
            )

    enterprise_value = yahoo_ev
    if enterprise_value is not None and enterprise_value <= 0:
        data_warnings.append(
            "EV unavailable: Yahoo enterpriseValue is not positive; discarding"
        )
        enterprise_value = None
    if enterprise_value is None and market_cap is not None:
        if total_debt is not None and total_cash is not None:
            enterprise_value = market_cap + total_debt - total_cash
        else:
            data_warnings.append(
                "EV unavailable: missing debt or cash; not zero-filling"
            )
    if enterprise_value is not None and enterprise_value <= 0:
        data_warnings.append(
            "EV unavailable: constructed enterprise value is not positive"
        )
        enterprise_value = None

    momentum_12_1 = _compute_momentum_12_1(hist)
    volatility_12m = _compute_volatility_12m(hist)
    drawdown_metrics = _compute_drawdown_metrics(hist)
    rsi_14 = _compute_rsi(hist)
    all_time_high = fetch_all_time_high(ticker)

    trailing_pe = _safe_float(info.get("trailingPE"))
    trailing_eps = _safe_float(info.get("trailingEps"))
    earnings_growth = _safe_float(info.get("earningsGrowth"))
    dividend_yield = _safe_float(info.get("dividendYield"))
    trailing_peg_ratio = _safe_float(info.get("trailingPegRatio"))
    current_ratio_info = _safe_float(info.get("currentRatio"))

    current_ey = None
    if ebit is not None and enterprise_value is not None and enterprise_value > 0:
        current_ey = ebit / enterprise_value

    target_mean = _safe_float(info.get("targetMeanPrice"))
    target_low = _safe_float(info.get("targetLowPrice"))
    target_high = _safe_float(info.get("targetHighPrice"))
    recommendation_key = info.get("recommendationKey")
    num_analysts = _safe_float(info.get("numberOfAnalystOpinions"))

    fifty_two_week_high = _safe_float(info.get("fiftyTwoWeekHigh"))
    fifty_two_week_low = _safe_float(info.get("fiftyTwoWeekLow"))
    if not hist.empty and "Close" in hist.columns:
        closes = hist["Close"].dropna()
        if fifty_two_week_high is None and len(closes) > 0:
            window = closes.tail(min(252, len(closes)))
            fifty_two_week_high = _safe_float(window.max())
        if fifty_two_week_low is None and len(closes) > 0:
            window = closes.tail(min(252, len(closes)))
            fifty_two_week_low = _safe_float(window.min())
    fifty_two_week_high_listing = fifty_two_week_high
    fifty_two_week_low_listing = fifty_two_week_low
    all_time_high_listing = all_time_high
    target_mean_listing = target_mean
    target_low_listing = target_low
    target_high_listing = target_high
    if fx_to_financial is not None and fx_to_financial != 1.0:
        def _to_fin(val: float | None) -> float | None:
            return None if val is None else val * fx_to_financial

        fifty_two_week_high = _to_fin(fifty_two_week_high)
        fifty_two_week_low = _to_fin(fifty_two_week_low)
        all_time_high = _to_fin(all_time_high)
        target_mean = _to_fin(target_mean)
        target_low = _to_fin(target_low)
        target_high = _to_fin(target_high)
    exchange = info.get("fullExchangeName") or info.get("exchange")
    sector = info.get("sector")
    industry = info.get("industry")
    name = info.get("longName") or info.get("shortName") or ticker.upper()

    statement_basis = {
        "flows": "ttm" if used_ttm_flow else "annual",
        "balance": "mrq" if used_mrq else "annual",
        "flow_period_end": _statement_period_end(ttm_income if used_ttm_flow else annual_income),
        "balance_period_end": _statement_period_end(mrq_balance if used_mrq else annual_balance),
    }

    out = {
        "ticker": ticker.upper(),
        "name": name,
        "sector": sector,
        "industry": industry,
        "price": price,
        "price_listing": price_listing,
        "price_fin": price,
        "market_cap": market_cap,
        "market_cap_listing": market_cap_listing,
        "market_cap_fin": market_cap,
        "enterprise_value": enterprise_value,
        "book_value": yahoo_book_value,
        "book_equity": book_equity,
        "shares_outstanding": shares,
        "total_assets": total_assets,
        "total_assets_prior": total_assets_prior,
        "total_liabilities": total_liabilities,
        "total_liabilities_prior": total_liabilities_prior,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "current_assets_prior": current_assets_prior,
        "current_liabilities_prior": current_liabilities_prior,
        "long_term_debt": long_term_debt,
        "long_term_debt_prior": long_term_debt_prior,
        "short_term_debt": short_term_debt,
        "debt_st": short_term_debt,
        "shares_prior": shares_prior,
        "gross_profit": gross_profit,
        "gross_profit_prior": gross_profit_prior,
        "net_income": net_income,
        "net_income_prior": net_income_prior,
        "ebit": ebit,
        "revenue": revenue,
        "revenue_prior": revenue_prior,
        "operating_income": operating_income,
        "operating_cashflow": operating_cashflow,
        "free_cashflow": free_cashflow,
        "capex": capex,
        "sbc": sbc,
        "interest_expense": interest_expense,
        "tax_expense": tax_expense,
        "pretax_income": pretax_income,
        "depreciation": depreciation,
        "ppe_net": ppe_net,
        "goodwill": goodwill,
        "dividends_paid": dividends_paid,
        "repurchase_of_stock": repurchase_of_stock,
        "retained_earnings": retained_earnings,
        "momentum_12_1": momentum_12_1,
        "volatility_12m": volatility_12m,
        "max_drawdown": drawdown_metrics.get("max_drawdown"),
        "downside_deviation": drawdown_metrics.get("downside_deviation"),
        "trailing_pe": trailing_pe,
        "trailing_eps": trailing_eps,
        "earnings_growth": earnings_growth,
        "dividend_yield": dividend_yield,
        "trailing_peg_ratio": trailing_peg_ratio,
        "total_cash": total_cash,
        "total_debt": total_debt,
        "debt_to_equity": debt_to_equity,
        "current_ratio_info": current_ratio_info,
        "target_mean": target_mean,
        "target_low": target_low,
        "target_high": target_high,
        "target_mean_listing": target_mean_listing,
        "target_low_listing": target_low_listing,
        "target_high_listing": target_high_listing,
        "recommendation_key": recommendation_key,
        "num_analysts": num_analysts,
        "recommendations": recs,
        "price_history": hist,
        "fifty_two_week_high": fifty_two_week_high,
        "fifty_two_week_low": fifty_two_week_low,
        "fifty_two_week_high_listing": fifty_two_week_high_listing,
        "fifty_two_week_low_listing": fifty_two_week_low_listing,
        "all_time_high": all_time_high,
        "all_time_high_listing": all_time_high_listing,
        "rsi_14": rsi_14,
        "earnings_yield_current": current_ey,
        "exchange": exchange,
        "currency": currency,
        "financial_currency": financial_currency,
        "fx_to_financial": fx_to_financial,
        "estimate_tables": estimate_tables,
        "form4_transactions": form4_transactions,
        "short_ratio": _safe_float(info.get("shortRatio")),
        "short_percent_of_float": _safe_float(info.get("shortPercentOfFloat")),
        "shares_short": _safe_float(info.get("sharesShort")),
        "statement_basis": statement_basis,
        "data_warnings": data_warnings,
        "ticker": ticker.upper().strip(),
    }
    from core.data_quality import attach_data_quality

    attach_data_quality(out)
    try:
        from core.fundamentals import attach_history_metrics

        attach_history_metrics(out)
    except Exception as exc:
        logger.debug("Fundamentals history skipped for %s: %s", ticker, exc)
    return out


def _compute_rsi(hist: pd.DataFrame, period: int = 14) -> float | None:
    """Wilder RSI (14-day default) from close prices."""
    if hist.empty or "Close" not in hist.columns:
        return None
    closes = hist["Close"].dropna()
    if len(closes) < period + 1:
        return None

    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    last_gain = avg_gain.iloc[-1]
    last_loss = avg_loss.iloc[-1]
    if last_loss == 0:
        return 100.0 if last_gain > 0 else 50.0
    rs = last_gain / last_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _compute_momentum_12_1(hist: pd.DataFrame) -> float | None:
    """12-1 month momentum: return from 13 months ago to 1 month ago."""
    if hist.empty or "Close" not in hist.columns:
        return None
    closes = hist["Close"].dropna()
    if len(closes) < 252:
        return None
    try:
        end_idx = -22  # ~1 month ago
        start_idx = -252  # ~12 months before that window
        p_end = closes.iloc[end_idx]
        p_start = closes.iloc[start_idx]
        if p_start and p_start > 0:
            return (p_end / p_start) - 1.0
    except (IndexError, KeyError):
        pass
    return None


def _compute_volatility_12m(hist: pd.DataFrame) -> float | None:
    if hist.empty or "Close" not in hist.columns:
        return None
    closes = hist["Close"].dropna().tail(252)
    if len(closes) < 60:
        return None
    returns = closes.pct_change().dropna()
    if returns.empty:
        return None
    return float(returns.std() * np.sqrt(252))


def _compute_drawdown_metrics(hist: pd.DataFrame) -> dict[str, float | None]:
    """Max drawdown and downside (semi) deviation from price history."""
    if hist.empty or "Close" not in hist.columns:
        return {"max_drawdown": None, "downside_deviation": None}

    closes = hist["Close"].dropna().tail(252)
    if len(closes) < 60:
        return {"max_drawdown": None, "downside_deviation": None}

    returns = closes.pct_change().dropna()
    if returns.empty:
        return {"max_drawdown": None, "downside_deviation": None}

    # Max drawdown (negative number, e.g. -0.25 for 25% drawdown)
    running_max = closes.cummax()
    drawdowns = (closes / running_max) - 1.0
    max_drawdown = float(drawdowns.min())

    # Downside deviation: std of negative returns only, annualized
    negative_returns = returns[returns < 0]
    if negative_returns.empty:
        downside_dev = 0.0
    else:
        downside_dev = float(negative_returns.std() * np.sqrt(252))

    return {"max_drawdown": max_drawdown, "downside_deviation": downside_dev}


def percentile_rank_in_history(
    current: float | None,
    history: list[float] | tuple[float, ...],
) -> float | None:
    """
    Percentile rank of ``current`` within ``history`` (inclusive), scaled 0-100.

    Higher current relative to history → higher score. For earnings yield this
    means "cheap vs own history". Requires at least 4 historical observations.
    """
    if current is None or (isinstance(current, float) and np.isnan(current)):
        return None
    vals = [float(v) for v in history if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(vals) < 4:
        return None
    # Inclusive rank: fraction of history <= current.
    n_le = sum(1 for v in vals if v <= current)
    return float(n_le / len(vals) * 100.0)


def fetch_quarterly_financials(ticker: str) -> dict[str, pd.DataFrame]:
    """Fetch quarterly income / balance / cashflow (cached)."""
    cache_path = _cache_key("qfin", ticker.upper())
    cached = _read_cache(cache_path, max_age_hours=48)
    if cached is not None:
        return {k: _records_to_df(v) for k, v in cached.items()}

    try:
        t = yf.Ticker(ticker)
        income = t.quarterly_financials if t.quarterly_financials is not None else pd.DataFrame()
        balance = t.quarterly_balance_sheet if t.quarterly_balance_sheet is not None else pd.DataFrame()
        cashflow = t.quarterly_cashflow if t.quarterly_cashflow is not None else pd.DataFrame()
        result = {
            "income": _df_to_records(income),
            "balance": _df_to_records(balance),
            "cashflow": _df_to_records(cashflow),
        }
        _write_cache(cache_path, result)
        return {"income": income, "balance": balance, "cashflow": cashflow}
    except Exception as exc:
        logger.warning("Quarterly financials failed for %s: %s", ticker, exc)
        return {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}


def _row_value_at(df: pd.DataFrame, names: list[str], col: Any) -> float | None:
    if df is None or df.empty:
        return None
    for name in names:
        if name not in df.index:
            continue
        if col not in df.columns:
            continue
        return _safe_float(df.loc[name, col])
    return None


_BALANCE_SHARES_ROWS = ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"]
_BALANCE_DEBT_ROWS = ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"]
_BALANCE_CASH_ROWS = ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"]


def _ey_point(
    annualized_ebit: float | None,
    balance: pd.DataFrame,
    balance_col: Any,
    closes: pd.Series,
    as_of: pd.Timestamp,
) -> float | None:
    """Annualized EBIT / EV as of a statement date, or None if inputs are missing."""
    if annualized_ebit is None:
        return None
    shares = _row_value_at(balance, _BALANCE_SHARES_ROWS, balance_col)
    if shares is None or shares <= 0:
        return None
    eligible = closes[closes.index <= as_of]
    if eligible.empty:
        return None
    price = _safe_float(eligible.iloc[-1])
    if price is None or price <= 0:
        return None
    total_debt = _row_value_at(balance, _BALANCE_DEBT_ROWS, balance_col)
    total_cash = _row_value_at(balance, _BALANCE_CASH_ROWS, balance_col)
    if total_debt is None or total_cash is None:
        return None
    enterprise_value = price * shares + total_debt - total_cash
    if enterprise_value <= 0:
        return None
    return float(annualized_ebit / enterprise_value)


def _statement_col_dates(df: pd.DataFrame) -> list[tuple[Any, pd.Timestamp]]:
    out: list[tuple[Any, pd.Timestamp]] = []
    for col in _financial_columns_newest_first(df):
        try:
            ts = pd.to_datetime(col)
        except (TypeError, ValueError):
            continue
        if pd.notna(ts):
            out.append((col, ts))
    return out


def build_earnings_yield_history(
    ticker: str,
    *,
    years: int = 5,
) -> list[float]:
    """
    Build an *annualized* earnings-yield (EBIT/EV) history in the same units as
    the current EY (annual EBIT / EV), so own-history percentiles compare
    like-for-like.

    Points come from two sources, all annualized:
      - annual statements: fiscal-year EBIT / EV at each fiscal year end
      - quarterly statements: trailing-4-quarter EBIT sums / EV at quarter end
        (only where 4 consecutive quarters exist)

    Yahoo typically exposes ~4 annual and ~5-6 quarterly periods, so the series
    spans roughly 4-5 years with a handful of observations. EV ≈ price × shares
    + debt − cash from the balance sheet at the same date.
    """
    # v2: annualized units (the old quarterly-EBIT series was ~4x too small,
    # which pinned the valuation-vs-history percentile at ~100 for most stocks).
    cache_path = _cache_key("eyhist2", ticker.upper(), str(years))
    cached = _read_cache(cache_path, max_age_hours=48)
    if cached is not None and isinstance(cached.get("history"), list):
        return [float(v) for v in cached["history"] if v is not None]

    hist = fetch_price_history(ticker, period="max")
    if hist.empty or "Close" not in hist.columns:
        return []
    closes = hist["Close"].dropna().sort_index()

    points: dict[pd.Timestamp, float] = {}
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)

    # Annual fiscal-year points.
    afin = fetch_financials(ticker)
    a_income = afin.get("income", pd.DataFrame())
    a_balance = afin.get("balance", pd.DataFrame())
    if not a_income.empty and not a_balance.empty:
        for col, ts in _statement_col_dates(a_income):
            if ts < cutoff:
                continue
            ebit = _row_value_at(a_income, ["EBIT", "Operating Income"], col)
            ey = _ey_point(ebit, a_balance, col, closes, ts)
            if ey is not None:
                points[ts.normalize()] = ey

    # Trailing-twelve-month points from quarterly statements.
    qfin = fetch_quarterly_financials(ticker)
    q_income = qfin.get("income", pd.DataFrame())
    q_balance = qfin.get("balance", pd.DataFrame())
    if not q_income.empty and not q_balance.empty:
        q_cols = _statement_col_dates(q_income)  # newest first
        for i, (col, ts) in enumerate(q_cols):
            if ts < cutoff:
                continue
            window = q_cols[i : i + 4]
            if len(window) < 4:
                continue
            # Require 4 consecutive quarters (~a year's span) for a valid TTM sum.
            span_days = (window[0][1] - window[-1][1]).days
            if not 240 <= span_days <= 320:
                continue
            quarter_ebits = [
                _row_value_at(q_income, ["EBIT", "Operating Income"], wcol)
                for wcol, _ in window
            ]
            if any(v is None for v in quarter_ebits):
                continue
            ttm_ebit = float(sum(quarter_ebits))  # type: ignore[arg-type]
            ey = _ey_point(ttm_ebit, q_balance, col, closes, ts)
            if ey is not None:
                # TTM points win over an annual point on the same date.
                points[ts.normalize()] = ey

    history = [points[ts] for ts in sorted(points)]
    _write_cache(cache_path, {"history": history})
    return history


def compute_valuation_vs_history(
    ticker: str,
    current_earnings_yield: float | None,
    *,
    years: int = 10,
    current_ocf_yield: float | None = None,
    current_book_to_market: float | None = None,
) -> float | None:
    """Current cheapness percentile vs own 10y EDGAR history (Yahoo fallback)."""
    from core.edgar_history import compute_valuation_vs_history_detail

    detail = compute_valuation_vs_history_detail(
        ticker,
        current_earnings_yield,
        current_ocf_yield=current_ocf_yield,
        current_book_to_market=current_book_to_market,
        years=years,
    )
    score = detail.get("score")
    return float(score) if score is not None else None


def throttle(seconds: float = 0.3) -> None:
    """Simple rate limit between batch requests."""
    time.sleep(seconds)
