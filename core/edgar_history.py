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


def _load_edgar_parquet() -> pd.DataFrame | None:
    try:
        from backtest.data.edgar import load_fundamentals

        return load_fundamentals()
    except Exception as exc:
        logger.debug("EDGAR parquet unavailable: %s", exc)
        return None


def _latest_per_period(sub: pd.DataFrame, field: str, *, annual_only: bool) -> pd.DataFrame:
    rows = sub[sub["field"] == field].copy()
    if rows.empty:
        return pd.DataFrame(columns=["period", "amount"])
    if annual_only and "qtrs" in rows.columns:
        qtrs = pd.to_numeric(rows["qtrs"], errors="coerce")
        form = rows["form"].astype(str) if "form" in rows.columns else pd.Series("", index=rows.index)
        rows = rows[(qtrs == 4) | form.str.contains("10-K", na=False)]
    if rows.empty:
        return pd.DataFrame(columns=["period", "amount"])
    rows["period"] = pd.to_datetime(rows["period"], errors="coerce")
    rows = rows.dropna(subset=["period", "amount"])
    rows = rows.sort_values(["period", "filed"] if "filed" in rows.columns else ["period"])
    latest = rows.groupby("period", as_index=False).last()
    return latest[["period", "amount"]].rename(columns={"amount": field})


def _period_table_from_parquet(fundamentals: pd.DataFrame, ticker: str) -> pd.DataFrame:
    sub = fundamentals[fundamentals["ticker"].astype(str).str.upper() == ticker.upper()].copy()
    if sub.empty:
        return pd.DataFrame()
    frames = [
        _latest_per_period(sub, "ebit", annual_only=True),
        _latest_per_period(sub, "operating_cashflow", annual_only=True),
        _latest_per_period(sub, "book_equity", annual_only=False),
        _latest_per_period(sub, "shares_outstanding", annual_only=False),
        _latest_per_period(sub, "total_debt", annual_only=False),
        _latest_per_period(sub, "total_cash", annual_only=False),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on="period", how="outer")
    return out.sort_values("period")


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


def _spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < _MIN_POINTS_FOR_CORR or len(a) != len(b):
        return None
    sa = pd.Series(a)
    sb = pd.Series(b)
    if sa.nunique() < 3 or sb.nunique() < 3:
        return None
    corr = sa.rank().corr(sb.rank())
    if corr is None or (isinstance(corr, float) and np.isnan(corr)):
        return None
    return float(corr)


def _aligned_fundamental_and_price(
    table: pd.DataFrame,
    closes: pd.Series,
    field: str,
    *,
    years: int,
) -> tuple[list[float], list[float]]:
    cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(years=years)
    fund_vals: list[float] = []
    prices: list[float] = []
    for _, row in table.iterrows():
        period = pd.to_datetime(row.get("period"), errors="coerce")
        val = row.get(field)
        if pd.isna(period) or period < cutoff or val is None or pd.isna(val):
            continue
        price = _price_on_or_before(closes, pd.Timestamp(period))
        if price is None:
            continue
        fund_vals.append(float(val))
        prices.append(price)
    return fund_vals, prices


def pick_best_metric(
    table: pd.DataFrame,
    closes: pd.Series,
    histories: dict[str, list[float]],
    *,
    years: int,
) -> tuple[str | None, float | None]:
    """
    Pick the yield whose underlying fundamental best tracks the stock's price
    (GuruFocus-style: use the historically most relevant multiple).
    """
    field_for_metric = {
        "earnings_yield": "ebit",
        "ocf_yield": "operating_cashflow",
        "book_to_market": "book_equity",
    }
    best_metric: str | None = None
    best_corr: float | None = None
    for metric, field in field_for_metric.items():
        if len(histories.get(metric) or []) < _MIN_POINTS:
            continue
        fund_vals, prices = _aligned_fundamental_and_price(table, closes, field, years=years)
        corr = _spearman(fund_vals, prices)
        if corr is None:
            continue
        if best_corr is None or abs(corr) > abs(best_corr):
            best_corr = corr
            best_metric = metric
    if best_corr is not None and abs(best_corr) < _MIN_ABS_CORR:
        return None, best_corr
    return best_metric, best_corr


def load_period_table(ticker: str) -> tuple[pd.DataFrame, str]:
    """Prefer the ingested EDGAR parquet; fall back to per-ticker company facts."""
    parquet = _load_edgar_parquet()
    if parquet is not None and not parquet.empty:
        table = _period_table_from_parquet(parquet, ticker)
        if not table.empty:
            return table, "edgar_parquet"
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
    years: int = VALUATION_HISTORY_YEARS,
) -> dict[str, Any]:
    """
    Current cheapness vs the stock's own 10y EDGAR history (0-100).

    Builds three yield histories (EBIT/EV, OCF/mcap, book/mcap) and scores the
    current value against the metric whose fundamental has the strongest
    historical rank-correlation with price. Falls back to an equal-weight
    average of available percentiles, then to Yahoo EY history.
    """
    empty = {
        "score": None,
        "metric": None,
        "source": None,
        "n_points": 0,
        "correlation": None,
        "years": years,
        "percentiles": {},
    }
    ticker = ticker.upper().strip()
    hist = fetch_price_history(ticker, period="max")
    closes = _closes_index(hist)
    table, source = load_period_table(ticker)
    histories = history_series_from_table(table, closes, years=years) if not table.empty else {
        "earnings_yield": [],
        "ocf_yield": [],
        "book_to_market": [],
    }

    current = {
        "earnings_yield": current_earnings_yield,
        "ocf_yield": current_ocf_yield,
        "book_to_market": current_book_to_market,
    }
    percentiles: dict[str, float] = {}
    for metric, series in histories.items():
        pct = percentile_rank_in_history(current.get(metric), series)
        if pct is not None:
            percentiles[metric] = pct

    best_metric, best_corr = pick_best_metric(table, closes, histories, years=years)
    score = None
    metric_used = best_metric
    if best_metric and best_metric in percentiles:
        score = percentiles[best_metric]
    elif percentiles:
        score = float(np.mean(list(percentiles.values())))
        metric_used = "average"

    n_points = max((len(v) for v in histories.values()), default=0)
    if score is not None:
        return {
            "score": float(score),
            "metric": metric_used,
            "source": source,
            "n_points": int(n_points),
            "correlation": best_corr,
            "years": years,
            "percentiles": percentiles,
        }

    # Yahoo fallback: annualized EBIT/EV over whatever history Yahoo exposes.
    from core.data import build_earnings_yield_history

    yahoo = build_earnings_yield_history(ticker, years=years)
    yahoo_score = percentile_rank_in_history(current_earnings_yield, yahoo)
    if yahoo_score is None:
        return empty
    return {
        "score": float(yahoo_score),
        "metric": "earnings_yield",
        "source": "yahoo",
        "n_points": len(yahoo),
        "correlation": None,
        "years": years,
        "percentiles": {"earnings_yield": float(yahoo_score)},
    }
