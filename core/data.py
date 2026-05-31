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


def is_etf(ticker: str) -> bool:
    """Return True if ticker appears to be an ETF."""
    cache_path = _cache_key("etf", ticker.upper())
    cached = _read_cache(cache_path, max_age_hours=168)
    if cached is not None:
        return bool(cached.get("is_etf"))

    try:
        info = yf.Ticker(ticker).info or {}
        quote_type = (info.get("quoteType") or "").upper()
        is_etf_flag = quote_type == "ETF"
        _write_cache(cache_path, {"is_etf": is_etf_flag})
        return is_etf_flag
    except Exception as exc:
        logger.warning("ETF check failed for %s: %s", ticker, exc)
        return False


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


def fetch_ticker_info(ticker: str) -> dict[str, Any]:
    """Fetch yfinance info dict with caching."""
    cache_path = _cache_key("info", ticker.upper())
    cached = _read_cache(cache_path, max_age_hours=24)
    if cached is not None:
        return cached

    try:
        info = yf.Ticker(ticker).info or {}
        serializable = {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool, type(None)))}
        _write_cache(cache_path, serializable)
        return serializable
    except Exception as exc:
        logger.warning("Info fetch failed for %s: %s", ticker, exc)
        return {}


def fetch_financials(ticker: str) -> dict[str, pd.DataFrame]:
    """Fetch income statement, balance sheet, cash flow."""
    cache_path = _cache_key("fin", ticker.upper())
    cached = _read_cache(cache_path, max_age_hours=48)
    if cached is not None:
        return {k: pd.DataFrame(v) for k, v in cached.items()}

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


def fetch_etf_info(ticker: str) -> dict[str, Any]:
    """Lightweight ETF metadata view."""
    info = fetch_ticker_info(ticker)
    return {
        "symbol": ticker.upper(),
        "name": info.get("longName") or info.get("shortName"),
        "category": info.get("category"),
        "fund_family": info.get("fundFamily"),
        "expense_ratio": info.get("annualReportExpenseRatio"),
        "total_assets": info.get("totalAssets"),
        "yield": info.get("yield") or info.get("dividendYield"),
        "nav_price": info.get("navPrice"),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "description": info.get("longBusinessSummary"),
    }


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


def build_raw_metrics(ticker: str) -> dict[str, Any]:
    """
    Assemble raw inputs needed for factor computation for a single ticker.
    """
    info = fetch_ticker_info(ticker)
    fin = fetch_financials(ticker)
    hist = fetch_price_history(ticker, period="2y")
    recs = fetch_analyst_recommendations(ticker)

    price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    market_cap = _safe_float(info.get("marketCap"))
    enterprise_value = _safe_float(info.get("enterpriseValue"))
    book_value = _safe_float(info.get("bookValue"))
    shares = _safe_float(info.get("sharesOutstanding"))

    income = fin["income"]
    balance = fin["balance"]
    cashflow = fin["cashflow"]

    # Latest annual columns (yfinance: columns are dates, most recent first)
    def latest(col_names: list[str], df: pd.DataFrame) -> float | None:
        if df.empty:
            return None
        for name in col_names:
            if name in df.index:
                row = df.loc[name]
                for v in row:
                    fv = _safe_float(v)
                    if fv is not None:
                        return fv
        return None

    def prior(col_names: list[str], df: pd.DataFrame) -> float | None:
        if df.empty:
            return None
        for name in col_names:
            if name in df.index:
                row = df.loc[name]
                vals = [_safe_float(v) for v in row]
                vals = [v for v in vals if v is not None]
                if len(vals) >= 2:
                    return vals[1]
        return None

    total_assets = latest(["Total Assets"], balance)
    total_assets_prior = prior(["Total Assets"], balance)
    total_liabilities = latest(["Total Liabilities Net Minority Interest", "Total Liabilities"], balance)
    total_liabilities_prior = prior(["Total Liabilities Net Minority Interest", "Total Liabilities"], balance)
    current_assets = latest(["Current Assets"], balance)
    current_liabilities = latest(["Current Liabilities"], balance)
    current_assets_prior = prior(["Current Assets"], balance)
    current_liabilities_prior = prior(["Current Liabilities"], balance)
    long_term_debt = latest(["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], balance)
    long_term_debt_prior = prior(["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], balance)
    shares_prior = prior(["Ordinary Shares Number", "Share Issued"], balance)

    gross_profit = latest(["Gross Profit"], income)
    net_income = latest(["Net Income", "Net Income Common Stockholders"], income)
    net_income_prior = prior(["Net Income", "Net Income Common Stockholders"], income)
    ebit = latest(["EBIT", "Operating Income"], income)
    revenue = latest(["Total Revenue", "Operating Revenue"], income)
    revenue_prior = prior(["Total Revenue", "Operating Revenue"], income)
    operating_income = latest(["Operating Income"], income)

    operating_cashflow = latest(["Operating Cash Flow"], cashflow)
    free_cashflow = latest(["Free Cash Flow"], cashflow)

    gross_profit_prior = prior(["Gross Profit"], income)
    total_assets_for_turnover_prior = total_assets_prior

    # Price-based metrics
    momentum_12_1 = _compute_momentum_12_1(hist)
    volatility_12m = _compute_volatility_12m(hist)
    drawdown_metrics = _compute_drawdown_metrics(hist)

    # Valuation / balance-sheet fields from info (book-inspired metrics)
    trailing_pe = _safe_float(info.get("trailingPE"))
    trailing_eps = _safe_float(info.get("trailingEps"))
    earnings_growth = _safe_float(info.get("earningsGrowth"))
    dividend_yield = _safe_float(info.get("dividendYield"))
    trailing_peg_ratio = _safe_float(info.get("trailingPegRatio"))
    total_cash = _safe_float(info.get("totalCash"))
    total_debt = _safe_float(info.get("totalDebt"))
    debt_to_equity = _safe_float(info.get("debtToEquity"))
    current_ratio_info = _safe_float(info.get("currentRatio"))

    # Analyst targets from info
    target_mean = _safe_float(info.get("targetMeanPrice"))
    target_low = _safe_float(info.get("targetLowPrice"))
    target_high = _safe_float(info.get("targetHighPrice"))
    recommendation_key = info.get("recommendationKey")
    num_analysts = _safe_float(info.get("numberOfAnalystOpinions"))

    sector = info.get("sector") or "Unknown"
    industry = info.get("industry") or "Unknown"
    name = info.get("longName") or info.get("shortName") or ticker.upper()

    return {
        "ticker": ticker.upper(),
        "name": name,
        "sector": sector,
        "industry": industry,
        "price": price,
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "book_value": book_value,
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
        "recommendation_key": recommendation_key,
        "num_analysts": num_analysts,
        "recommendations": recs,
        "price_history": hist,
    }


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


def throttle(seconds: float = 0.3) -> None:
    """Simple rate limit between batch requests."""
    time.sleep(seconds)
