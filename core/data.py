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
    """Fetch income statement, balance sheet, cash flow."""
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
    if price is None and not hist.empty and "Close" in hist.columns:
        price = _safe_float(hist["Close"].iloc[-1])
    market_cap = _safe_float(info.get("marketCap"))
    enterprise_value = _safe_float(info.get("enterpriseValue"))
    book_value = _safe_float(info.get("bookValue"))
    shares = _safe_float(info.get("sharesOutstanding"))

    income = fin["income"]
    balance = fin["balance"]
    cashflow = fin["cashflow"]
    data_warnings: list[str] = []

    def _fin(col_names: list[str], df: pd.DataFrame) -> tuple[float | None, float | None]:
        latest, prior, warns = extract_financial_values(col_names, df)
        data_warnings.extend(warns)
        return latest, prior

    (total_assets, total_assets_prior) = _fin(["Total Assets"], balance)
    (total_liabilities, total_liabilities_prior) = _fin(
        ["Total Liabilities Net Minority Interest", "Total Liabilities"], balance
    )
    (current_assets, current_assets_prior) = _fin(["Current Assets"], balance)
    (current_liabilities, current_liabilities_prior) = _fin(["Current Liabilities"], balance)
    (long_term_debt, long_term_debt_prior) = _fin(
        ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], balance
    )
    (_, shares_prior) = _fin(["Ordinary Shares Number", "Share Issued"], balance)

    (gross_profit, gross_profit_prior) = _fin(["Gross Profit"], income)
    (net_income, net_income_prior) = _fin(["Net Income", "Net Income Common Stockholders"], income)
    (ebit, _) = _fin(["EBIT", "Operating Income"], income)
    (revenue, revenue_prior) = _fin(["Total Revenue", "Operating Revenue"], income)
    (operating_income, _) = _fin(["Operating Income"], income)

    (operating_cashflow, _) = _fin(["Operating Cash Flow"], cashflow)
    (free_cashflow, _) = _fin(["Free Cash Flow"], cashflow)
    (dividends_paid, _) = _fin(
        ["Cash Dividends Paid", "Common Stock Dividend Paid", "Payment Of Dividends", "Dividends Paid"],
        cashflow,
    )
    (repurchase_of_stock, _) = _fin(
        [
            "Repurchase Of Capital Stock",
            "Common Stock Payments",
            "Repurchase Of Common Stock",
            "Repurchase Of Stock",
        ],
        cashflow,
    )

    (retained_earnings, _) = _fin(
        ["Retained Earnings", "Retained Earnings Total Equity", "Retained Earnings Accumulated Deficit"],
        balance,
    )

    # Price-based metrics
    momentum_12_1 = _compute_momentum_12_1(hist)
    volatility_12m = _compute_volatility_12m(hist)
    drawdown_metrics = _compute_drawdown_metrics(hist)
    rsi_14 = _compute_rsi(hist)
    all_time_high = fetch_all_time_high(ticker)

    # Valuation / balance-sheet fields from info (book-inspired metrics)
    trailing_pe = _safe_float(info.get("trailingPE"))
    trailing_eps = _safe_float(info.get("trailingEps"))
    earnings_growth = _safe_float(info.get("earningsGrowth"))
    dividend_yield = _safe_float(info.get("dividendYield"))
    trailing_peg_ratio = _safe_float(info.get("trailingPegRatio"))
    total_cash = _safe_float(info.get("totalCash"))
    total_debt = _safe_float(info.get("totalDebt"))
    debt_to_equity = normalize_debt_to_equity(_safe_float(info.get("debtToEquity")))
    current_ratio_info = _safe_float(info.get("currentRatio"))

    # Analyst targets from info
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
    exchange = info.get("fullExchangeName") or info.get("exchange")
    sector = info.get("sector")
    industry = info.get("industry")
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
        "recommendation_key": recommendation_key,
        "num_analysts": num_analysts,
        "recommendations": recs,
        "price_history": hist,
        "fifty_two_week_high": fifty_two_week_high,
        "fifty_two_week_low": fifty_two_week_low,
        "all_time_high": all_time_high,
        "rsi_14": rsi_14,
        "exchange": exchange,
        "data_warnings": data_warnings,
    }


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


def throttle(seconds: float = 0.3) -> None:
    """Simple rate limit between batch requests."""
    time.sleep(seconds)
