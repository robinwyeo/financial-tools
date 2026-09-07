"""EDGAR-backed 10+ year own-history valuation for the live bargain score."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from core.data import (
    _cache_key,
    _read_cache,
    _write_cache,
    fetch_price_history,
    percentile_rank_in_history,
)
from core.sec import cik_padded, sec_get, ticker_to_cik

logger = logging.getLogger(__name__)

VALUATION_HISTORY_YEARS = 10
_MIN_POINTS = 4
_MIN_POINTS_FOR_CORR = 8
_MIN_ABS_CORR = 0.30

# Same us-gaap tags the backtest ingest uses (backtest/data/edgar.py TAG_MAP).
_COMPANYFACTS_TAGS: dict[str, str] = {
    "OperatingIncomeLoss": "ebit",
    "NetCashProvidedByUsedInOperatingActivities": "operating_cashflow",
    "StockholdersEquity": "book_equity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "book_equity",
    "CommonStockSharesOutstanding": "shares_outstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic": "shares_outstanding",
    "LongTermDebtAndCapitalLeaseObligations": "total_debt",
    "LongTermDebt": "total_debt",
    "CashAndCashEquivalentsAtCarryingValue": "total_cash",
    "CashCashEquivalentsAndShortTermInvestments": "total_cash",
}

_FLOW_FIELDS = frozenset({"ebit", "operating_cashflow"})


def _companyfacts_entries(facts: dict[str, Any], tag: str) -> list[dict[str, Any]]:
    node = ((facts.get("facts") or {}).get("us-gaap") or {}).get(tag) or {}
    units = node.get("units") or {}
    entries = units.get("USD") or units.get("USD/shares") or []
    if tag in ("CommonStockSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"):
        entries = units.get("shares") or units.get("USD") or entries
    return entries if isinstance(entries, list) else []


def _annual_entries(entries: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in entries:
        form = str(item.get("form") or "")
        fp = str(item.get("fp") or "")
        if "10-K" not in form and fp != "FY":
            continue
        end = pd.to_datetime(item.get("end"), errors="coerce")
        val = item.get("val")
        if pd.isna(end) or val is None:
            continue
        rows.append(
            {
                "period": end,
                "amount": float(val),
                "filed": pd.to_datetime(item.get("filed"), errors="coerce"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["period", "amount"])
    df = pd.DataFrame(rows).sort_values(["period", "filed"])
    return df.groupby("period", as_index=False).last()[["period", "amount"]]


def _period_table_from_companyfacts(ticker: str) -> pd.DataFrame:
    cik = ticker_to_cik(ticker)
    if cik is None:
        return pd.DataFrame()
    cache_path = _cache_key("cfacts", str(cik))
    cached = _read_cache(cache_path, max_age_hours=48)
    if cached is None:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded(cik)}.json"
        try:
            cached = sec_get(url, timeout=60).json()
            _write_cache(cache_path, cached)
        except Exception as exc:
            logger.warning("Company facts fetch failed for %s (CIK %s): %s", ticker, cik, exc)
            return pd.DataFrame()

    collected: dict[str, pd.DataFrame] = {}
    for tag, field in _COMPANYFACTS_TAGS.items():
        series = _annual_entries(_companyfacts_entries(cached, tag))
        if series.empty:
            continue
        series = series.rename(columns={"amount": field})
        if field in collected:
            # Prefer the first tag that produced data; don't overwrite with emptier later tags.
            merged = collected[field].merge(series, on="period", how="outer", suffixes=("", "_new"))
            if f"{field}_new" in merged.columns:
                merged[field] = merged[field].combine_first(merged[f"{field}_new"])
                merged = merged.drop(columns=[f"{field}_new"])
            collected[field] = merged
        else:
            collected[field] = series
    if not collected:
        return pd.DataFrame()
    out = None
    for frame in collected.values():
        out = frame if out is None else out.merge(frame, on="period", how="outer")
    return out.sort_values("period") if out is not None else pd.DataFrame()


def _price_on_or_before(closes: pd.Series, as_of: pd.Timestamp) -> float | None:
    eligible = closes[closes.index <= as_of]
    if eligible.empty:
        return None
    price = eligible.iloc[-1]
    if price is None or (isinstance(price, float) and np.isnan(price)) or price <= 0:
        return None
    return float(price)


def _closes_index(hist: pd.DataFrame) -> pd.Series:
    if hist.empty or "Close" not in hist.columns:
        return pd.Series(dtype=float)
    closes = hist["Close"].dropna().copy()
    closes.index = pd.to_datetime(closes.index, utc=True).tz_localize(None)
    return closes.sort_index()


def history_series_from_table(
    table: pd.DataFrame,
    closes: pd.Series,
    *,
    years: int = VALUATION_HISTORY_YEARS,
) -> dict[str, list[float]]:
    """Build annualized yield histories (EY, OCF yield, B/M) from a period table."""
    if table.empty:
        return {"earnings_yield": [], "ocf_yield": [], "book_to_market": []}
    cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(years=years)
    ey: list[float] = []
    ocf_y: list[float] = []
    btm: list[float] = []
    for _, row in table.iterrows():
        period = pd.to_datetime(row.get("period"), errors="coerce")
        if pd.isna(period) or period < cutoff:
            continue
        price = _price_on_or_before(closes, pd.Timestamp(period))
        shares = row.get("shares_outstanding")
        if price is None or shares is None or (isinstance(shares, float) and np.isnan(shares)) or shares <= 0:
            continue
        mcap = price * float(shares)
        if mcap <= 0:
            continue
        debt = float(row["total_debt"]) if pd.notna(row.get("total_debt")) else 0.0
        cash = float(row["total_cash"]) if pd.notna(row.get("total_cash")) else 0.0
        ev = mcap + debt - cash
        ebit = row.get("ebit")
        if ev > 0 and ebit is not None and pd.notna(ebit):
            ey.append(float(ebit) / ev)
        ocf = row.get("operating_cashflow")
        if ocf is not None and pd.notna(ocf):
            ocf_y.append(float(ocf) / mcap)
        book = row.get("book_equity")
        if book is not None and pd.notna(book):
            btm.append(float(book) / mcap)
    return {"earnings_yield": ey, "ocf_yield": ocf_y, "book_to_market": btm}


def ev_ebit_history(
    fund,
    closes: pd.Series,
    *,
    years: int = VALUATION_HISTORY_YEARS,
) -> list[float]:
    """Annual EBIT/EV yields (higher = cheaper)."""
    from core.fundamentals import Fundamentals

    if fund is None or getattr(fund, "annual", None) is None or fund.annual.empty:
        return []
    cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(years=years)
    out: list[float] = []
    for period, row in fund.annual.iterrows():
        ts = pd.Timestamp(period)
        if ts < cutoff:
            continue
        price = _price_on_or_before(closes, ts)
        shares = row.get("shares_diluted")
        ebit = row.get("ebit")
        if price is None or shares is None or pd.isna(shares) or shares <= 0 or ebit is None or pd.isna(ebit):
            continue
        mcap = price * float(shares)
        debt = float(row["debt"]) if pd.notna(row.get("debt")) else 0.0
        cash = float(row["cash"]) if pd.notna(row.get("cash")) else 0.0
        ev = mcap + debt - cash
        if ev > 0:
            out.append(float(ebit) / ev)
    return out


def p_oe_history(
    fund,
    closes: pd.Series,
    *,
    years: int = VALUATION_HISTORY_YEARS,
    subtract_sbc: bool = True,
) -> list[float]:
    """Annual owner-earnings yields (higher = cheaper)."""
    from core.fundamentals import owner_earnings

    if fund is None or getattr(fund, "annual", None) is None or fund.annual.empty:
        return []
    cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(years=years)
    out: list[float] = []
    for period, row in fund.annual.iterrows():
        ts = pd.Timestamp(period)
        if ts < cutoff:
            continue
        price = _price_on_or_before(closes, ts)
        shares = row.get("shares_diluted")
        oe = owner_earnings(row, subtract_sbc=subtract_sbc)
        if price is None or shares is None or pd.isna(shares) or shares <= 0 or oe is None:
            continue
        mcap = price * float(shares)
        if mcap > 0:
            out.append(float(oe) / mcap)
    return out


def load_period_table(ticker: str) -> tuple[pd.DataFrame, str]:
    """Always use consolidated companyfacts (same source live and backtest).

    The old FSDS parquet mixed segments and YTD amounts; live scoring no longer
    reads it even when the backtest store is present locally.
    """
    try:
        from core.edgar_facts import fetch_companyfacts_for_ticker, period_table_from_facts

        facts = fetch_companyfacts_for_ticker(ticker)
        table = period_table_from_facts(facts, ticker)
        if not table.empty:
            return table, "companyfacts"
    except Exception as exc:
        logger.debug("edgar_facts period table failed for %s: %s", ticker, exc)
    table = _period_table_from_companyfacts(ticker)
    if not table.empty:
        return table, "companyfacts"
    return pd.DataFrame(), "none"


def compute_valuation_vs_history_detail(
    ticker: str,
    current_earnings_yield: float | None,
    *,
    current_ocf_yield: float | None = None,
    current_book_to_market: float | None = None,
    current_oe_yield: float | None = None,
    years: int = VALUATION_HISTORY_YEARS,
) -> dict[str, Any]:
    """EV/EBIT and P/OE cheapness vs own 10y history. score = mean of the two percentiles."""
    del current_ocf_yield, current_book_to_market
    empty = {
        "score": None,
        "metric": None,
        "source": None,
        "n_points": 0,
        "years": years,
        "ev_ebit": {"current": None, "median_10y": None, "percentile": None, "n": 0},
        "p_oe": {"current": None, "median_10y": None, "percentile": None, "n": 0},
    }
    ticker = ticker.upper().strip()
    hist = fetch_price_history(ticker, period="max")
    closes = _closes_index(hist)
    fund = None
    source = "none"
    try:
        from core.fundamentals import get_fundamentals

        fund = get_fundamentals(ticker)
        source = fund.source
    except Exception as exc:
        logger.debug("fundamentals for valuation history failed: %s", exc)

    ey_hist = ev_ebit_history(fund, closes, years=years) if fund is not None else []
    oe_hist = p_oe_history(fund, closes, years=years) if fund is not None else []

    def _block(current_yield: float | None, series: list[float]) -> dict[str, Any]:
        pct = percentile_rank_in_history(current_yield, series)
        median_yield = float(np.median(series)) if series else None
        median_mult = (1.0 / median_yield) if median_yield else None
        current_mult = (1.0 / current_yield) if current_yield else None
        return {
            "current": current_mult,
            "median_10y": median_mult,
            "percentile": pct,
            "n": len(series),
        }

    ev_block = _block(current_earnings_yield, ey_hist)
    poe_block = _block(current_oe_yield, oe_hist)
    scores = [b["percentile"] for b in (ev_block, poe_block) if b["percentile"] is not None]
    score = float(np.mean(scores)) if scores else None
    n_points = max(len(ey_hist), len(oe_hist), 0)
    if score is not None:
        return {
            "score": score,
            "metric": "ev_ebit_p_oe",
            "source": source,
            "n_points": int(n_points),
            "years": years,
            "ev_ebit": ev_block,
            "p_oe": poe_block,
        }

    from core.data import build_earnings_yield_history

    yahoo = build_earnings_yield_history(ticker, years=years)
    yahoo_score = percentile_rank_in_history(current_earnings_yield, yahoo)
    if yahoo_score is None:
        return empty
    return {
        "score": float(yahoo_score),
        "metric": "ev_ebit",
        "source": "yahoo",
        "n_points": len(yahoo),
        "years": years,
        "ev_ebit": _block(current_earnings_yield, yahoo),
        "p_oe": poe_block,
    }

