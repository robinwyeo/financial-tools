"""Unified 10y annual + TTM fundamentals from EDGAR with Yahoo fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from core.data import (
    _BS_ASSETS,
    _BS_CA,
    _BS_CASH,
    _BS_CL,
    _BS_DEBT,
    _BS_EQUITY,
    _BS_GW,
    _BS_LIAB,
    _BS_PPE,
    _BS_SHARES,
    _FLOW_BUYBACK,
    _FLOW_CAPEX,
    _FLOW_DIV,
    _FLOW_EBIT,
    _FLOW_GROSS,
    _FLOW_INTEREST,
    _FLOW_NI,
    _FLOW_OCF,
    _FLOW_PRETAX,
    _FLOW_REVENUE,
    _FLOW_SBC,
    _FLOW_TAX,
    _cache_key,
    _read_cache,
    _row_value_at,
    _write_cache,
    fetch_financials,
    fetch_ttm_financials,
)
from core.edgar_facts import (
    FLOW_FIELDS,
    INSTANT_FIELDS,
    annual_series,
    fetch_companyfacts_for_ticker,
    fundamentals_as_of_structured,
)

logger = logging.getLogger(__name__)

ANNUAL_COLUMNS: tuple[str, ...] = (
    "revenue",
    "gross_profit",
    "ebit",
    "net_income",
    "operating_cashflow",
    "capex",
    "sbc",
    "dividends",
    "buybacks",
    "interest_expense",
    "tax_expense",
    "tax_paid",
    "amortization",
    "pretax_income",
    "total_assets",
    "total_liabilities",
    "current_assets",
    "current_liabilities",
    "cash",
    "debt",
    "equity",
    "shares_diluted",
    "goodwill",
    "ppe_net",
)

_EDGAR_ANNUAL_MAP: dict[str, str] = {
    "revenue": "revenue",
    "gross_profit": "gross_profit",
    "ebit": "ebit",
    "net_income": "net_income",
    "operating_cashflow": "operating_cashflow",
    "capex": "capex",
    "sbc": "sbc",
    "dividends": "dividends_paid",
    "buybacks": "repurchase_of_stock",
    "interest_expense": "interest_expense",
    "tax_expense": "tax_expense",
    "tax_paid": "tax_paid",
    "amortization": "amortization",
    "pretax_income": "pretax_income",
    "total_assets": "total_assets",
    "total_liabilities": "total_liabilities",
    "current_assets": "current_assets",
    "current_liabilities": "current_liabilities",
    "cash": "total_cash",
    "equity": "book_equity",
    "shares_diluted": "shares_diluted",
    "goodwill": "goodwill",
    "ppe_net": "ppe_net",
}

_YAHOO_ROWS: dict[str, list[str]] = {
    "revenue": _FLOW_REVENUE,
    "gross_profit": _FLOW_GROSS,
    "ebit": _FLOW_EBIT,
    "net_income": _FLOW_NI,
    "operating_cashflow": _FLOW_OCF,
    "capex": _FLOW_CAPEX,
    "sbc": _FLOW_SBC,
    "dividends": _FLOW_DIV,
    "buybacks": _FLOW_BUYBACK,
    "interest_expense": _FLOW_INTEREST,
    "tax_expense": _FLOW_TAX,
    "pretax_income": _FLOW_PRETAX,
    "total_assets": _BS_ASSETS,
    "total_liabilities": _BS_LIAB,
    "current_assets": _BS_CA,
    "current_liabilities": _BS_CL,
    "cash": _BS_CASH,
    "debt": _BS_DEBT,
    "equity": _BS_EQUITY,
    "shares_diluted": _BS_SHARES,
    "goodwill": _BS_GW,
    "ppe_net": _BS_PPE,
}


@dataclass
class Fundamentals:
    ticker: str
    annual: pd.DataFrame
    ttm: pd.Series
    mrq: pd.Series
    source: str
    as_of: date
    column_sources: dict[str, str] = field(default_factory=dict)


def owner_earnings(
    row: pd.Series | dict[str, Any],
    subtract_sbc: bool = True,
) -> float | None:
    """OCF - |CapEx| - SBC (SBC optional)."""
    if isinstance(row, dict):
        ocf = row.get("operating_cashflow")
        capex = row.get("capex")
        sbc = row.get("sbc")
    else:
        ocf = row.get("operating_cashflow") if "operating_cashflow" in row.index else None
        capex = row.get("capex") if "capex" in row.index else None
        sbc = row.get("sbc") if "sbc" in row.index else None
    if ocf is None or (isinstance(ocf, float) and np.isnan(ocf)):
        return None
    spend = abs(float(capex)) if capex is not None and not (isinstance(capex, float) and np.isnan(capex)) else 0.0
    sbc_v = 0.0
    if subtract_sbc and sbc is not None and not (isinstance(sbc, float) and np.isnan(sbc)):
        sbc_v = abs(float(sbc))
    return float(ocf) - spend - sbc_v


def owner_earnings_per_share(
    row: pd.Series | dict[str, Any],
    subtract_sbc: bool = True,
) -> float | None:
    oe = owner_earnings(row, subtract_sbc=subtract_sbc)
    if oe is None:
        return None
    if isinstance(row, dict):
        shares = row.get("shares_diluted") or row.get("shares_outstanding")
    else:
        shares = None
        if "shares_diluted" in row.index:
            shares = row.get("shares_diluted")
        if shares is None and "shares_outstanding" in row.index:
            shares = row.get("shares_outstanding")
    if shares is None or (isinstance(shares, float) and (np.isnan(shares) or shares <= 0)):
        return None
    return float(oe) / float(shares)


def cagr(series: pd.Series, years: int = 5) -> float | None:
    """CAGR from the point ``years`` ago to the latest value. Requires positive endpoints."""
    s = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if s.empty or years <= 0:
        return None
    if len(s) < years + 1:
        if len(s) < 2:
            return None
        start, end = float(s.iloc[0]), float(s.iloc[-1])
        span = years
        if hasattr(s.index[0], "year") and hasattr(s.index[-1], "year"):
            span = max(int(s.index[-1].year - s.index[0].year), 1)
        if start <= 0 or end <= 0:
            return None
        return float((end / start) ** (1.0 / span) - 1.0)
    start, end = float(s.iloc[-(years + 1)]), float(s.iloc[-1])
    if start <= 0 or end <= 0:
        return None
    return float((end / start) ** (1.0 / years) - 1.0)


def margin_series(numer: pd.Series, denom: pd.Series) -> pd.Series:
    n = pd.to_numeric(numer, errors="coerce")
    d = pd.to_numeric(denom, errors="coerce")
    out = n / d.replace(0, np.nan)
    return out


def _empty_annual() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ANNUAL_COLUMNS))


def _series_from_mapping(mapping: dict[str, Any]) -> pd.Series:
    data = {col: mapping.get(col) for col in ANNUAL_COLUMNS}
    return pd.Series(data, dtype="float64")


def _annual_from_edgar(facts_df: pd.DataFrame, years: int = 10) -> pd.DataFrame:
    if facts_df is None or facts_df.empty:
        return _empty_annual()
    pieces: dict[str, pd.Series] = {}
    for dest, src in _EDGAR_ANNUAL_MAP.items():
        fy = annual_series(facts_df, src, years=years)
        if fy.empty:
            continue
        ser = fy.set_index(pd.to_datetime(fy["period"]))["value"]
        ser = pd.to_numeric(ser, errors="coerce")
        pieces[dest] = ser[~ser.index.duplicated(keep="last")]
    # Debt = LT + ST when both exist.
    ltd = annual_series(facts_df, "long_term_debt", years=years)
    std = annual_series(facts_df, "debt_st", years=years)
    debt = None
    if not ltd.empty:
        debt = ltd.set_index(pd.to_datetime(ltd["period"]))["value"]
    if not std.empty:
        st_s = std.set_index(pd.to_datetime(std["period"]))["value"]
        debt = st_s if debt is None else debt.add(st_s, fill_value=0.0)
    if debt is not None:
        pieces["debt"] = pd.to_numeric(debt, errors="coerce")
    shares = annual_series(facts_df, "shares_outstanding", years=years)
    if "shares_diluted" not in pieces and not shares.empty:
        pieces["shares_diluted"] = pd.to_numeric(
            shares.set_index(pd.to_datetime(shares["period"]))["value"],
            errors="coerce",
        )
    if not pieces:
        return _empty_annual()
    annual = pd.concat(pieces, axis=1).sort_index()
    for col in ANNUAL_COLUMNS:
        if col not in annual.columns:
            annual[col] = np.nan
    return annual[list(ANNUAL_COLUMNS)]


def _yahoo_col_key(df: pd.DataFrame, period: pd.Timestamp) -> Any | None:
    if df is None or df.empty:
        return None
    target = pd.Timestamp(period).normalize()
    for col in df.columns:
        ts = pd.to_datetime(col, errors="coerce")
        if pd.notna(ts) and pd.Timestamp(ts).normalize() == target:
            return col
    return None


def _yahoo_statement_columns(income: pd.DataFrame, balance: pd.DataFrame, cashflow: pd.DataFrame) -> list[pd.Timestamp]:
    cols: list[pd.Timestamp] = []
    for df in (income, balance, cashflow):
        if df is None or df.empty:
            continue
        for col in df.columns:
            ts = pd.to_datetime(col, errors="coerce")
            if pd.notna(ts):
                cols.append(pd.Timestamp(ts).normalize())
    if not cols:
        return []
    uniq = sorted(set(cols))
    return uniq[-10:]


def _yahoo_cell(df: pd.DataFrame, names: list[str], col: Any) -> float | None:
    if df is None or df.empty:
        return None
    return _row_value_at(df, names, col)


def _annual_from_yahoo(fin: dict[str, pd.DataFrame]) -> pd.DataFrame:
    income = fin.get("income", pd.DataFrame())
    balance = fin.get("balance", pd.DataFrame())
    cashflow = fin.get("cashflow", pd.DataFrame())
    periods = _yahoo_statement_columns(income, balance, cashflow)
    if not periods:
        return _empty_annual()
    rows = []
    index = []
    for period in periods:
        row: dict[str, float | None] = {}
        for field, names in _YAHOO_ROWS.items():
            if field in {"operating_cashflow", "capex", "sbc", "dividends", "buybacks"}:
                val = _yahoo_cell(cashflow, names, _yahoo_col_key(cashflow, period))
            elif field in {
                "total_assets",
                "total_liabilities",
                "current_assets",
                "current_liabilities",
                "cash",
                "debt",
                "equity",
                "shares_diluted",
                "goodwill",
                "ppe_net",
            }:
                val = _yahoo_cell(balance, names, _yahoo_col_key(balance, period))
            else:
                val = _yahoo_cell(income, names, _yahoo_col_key(income, period))
            row[field] = val
        rows.append(row)
        index.append(period)
    annual = pd.DataFrame(rows, index=pd.DatetimeIndex(index))
    for col in ANNUAL_COLUMNS:
        if col not in annual.columns:
            annual[col] = np.nan
    return annual[list(ANNUAL_COLUMNS)].sort_index()


def _ttm_mrq_from_structured(structured: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    ttm_map: dict[str, Any] = {}
    flow = structured.get("flow_ttm") or {}
    fy = structured.get("flow_fy") or {}
    mrq_raw = structured.get("instant_mrq") or {}
    mapping_flow = {
        "revenue": "revenue",
        "gross_profit": "gross_profit",
        "ebit": "ebit",
        "net_income": "net_income",
        "operating_cashflow": "operating_cashflow",
        "capex": "capex",
        "sbc": "sbc",
        "dividends": "dividends_paid",
        "buybacks": "repurchase_of_stock",
        "interest_expense": "interest_expense",
        "tax_expense": "tax_expense",
        "pretax_income": "pretax_income",
    }
    for dest, src in mapping_flow.items():
        val = flow.get(src)
        if val is None:
            val = fy.get(src)
        ttm_map[dest] = val
    mapping_inst = {
        "total_assets": "total_assets",
        "total_liabilities": "total_liabilities",
        "current_assets": "current_assets",
        "current_liabilities": "current_liabilities",
        "cash": "total_cash",
        "equity": "book_equity",
        "goodwill": "goodwill",
        "ppe_net": "ppe_net",
        "shares_diluted": "shares_diluted",
    }
    mrq_map: dict[str, Any] = {}
    for dest, src in mapping_inst.items():
        mrq_map[dest] = mrq_raw.get(src)
        ttm_map[dest] = mrq_raw.get(src)
    ltd = mrq_raw.get("long_term_debt")
    std = mrq_raw.get("debt_st")
    debt = None
    if ltd is not None or std is not None:
        debt = (ltd or 0.0) + (std or 0.0)
    mrq_map["debt"] = debt
    ttm_map["debt"] = debt
    if ttm_map.get("shares_diluted") is None:
        ttm_map["shares_diluted"] = mrq_raw.get("shares_outstanding")
        mrq_map["shares_diluted"] = mrq_raw.get("shares_outstanding")
    return _series_from_mapping(ttm_map), _series_from_mapping(mrq_map)


def _ttm_mrq_from_yahoo(ttm: dict[str, pd.DataFrame], annual: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    income = ttm.get("income", pd.DataFrame())
    cashflow = ttm.get("cashflow", pd.DataFrame())
    balance = ttm.get("balance", pd.DataFrame())
    ttm_col = None
    mrq_col = None
    if isinstance(income, pd.DataFrame) and not income.empty:
        ttm_col = income.columns[0]
    if isinstance(cashflow, pd.DataFrame) and not cashflow.empty and ttm_col is None:
        ttm_col = cashflow.columns[0]
    if isinstance(balance, pd.DataFrame) and not balance.empty:
        mrq_col = balance.columns[0]
    ttm_map: dict[str, Any] = {}
    mrq_map: dict[str, Any] = {}
    for field, names in _YAHOO_ROWS.items():
        if field in {"operating_cashflow", "capex", "sbc", "dividends", "buybacks"}:
            ttm_map[field] = _yahoo_cell(cashflow, names, ttm_col) if ttm_col is not None else None
        elif field in {
            "total_assets",
            "total_liabilities",
            "current_assets",
            "current_liabilities",
            "cash",
            "debt",
            "equity",
            "shares_diluted",
            "goodwill",
            "ppe_net",
        }:
            val = _yahoo_cell(balance, names, mrq_col) if mrq_col is not None else None
            ttm_map[field] = val
            mrq_map[field] = val
        else:
            ttm_map[field] = _yahoo_cell(income, names, ttm_col) if ttm_col is not None else None
    if annual is not None and not annual.empty:
        last = annual.iloc[-1]
        for col in ANNUAL_COLUMNS:
            if ttm_map.get(col) is None and col in last.index:
                ttm_map[col] = last[col]
    return _series_from_mapping(ttm_map), _series_from_mapping(mrq_map)


def _column_sources(annual: pd.DataFrame, source: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if annual is None or annual.empty:
        return out
    for col in annual.columns:
        if annual[col].notna().any():
            out[col] = source
    return out


def _from_cache(payload: dict[str, Any], ticker: str) -> Fundamentals | None:
    try:
        annual_rows = payload.get("annual") or []
        annual = pd.DataFrame(annual_rows)
        if "period" in annual.columns:
            annual["period"] = pd.to_datetime(annual["period"])
            annual = annual.set_index("period")
        as_of_raw = payload.get("as_of")
        as_of = pd.Timestamp(as_of_raw).date() if as_of_raw else date.today()
        ttm = pd.Series(payload.get("ttm") or {}, dtype="float64")
        mrq = pd.Series(payload.get("mrq") or {}, dtype="float64")
        return Fundamentals(
            ticker=ticker,
            annual=annual,
            ttm=ttm,
            mrq=mrq,
            source=str(payload.get("source") or "mixed"),
            as_of=as_of,
            column_sources=payload.get("column_sources") or {},
        )
    except Exception:
        return None


def _to_cache(fund: Fundamentals) -> dict[str, Any]:
    annual = fund.annual.copy()
    if annual.index.name is None:
        annual.index.name = "period"
    reset = annual.reset_index()
    reset["period"] = reset["period"].astype(str)
    return {
        "ticker": fund.ticker,
        "source": fund.source,
        "as_of": fund.as_of.isoformat(),
        "annual": reset.to_dict(orient="records"),
        "ttm": {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else v) for k, v in fund.ttm.to_dict().items()},
        "mrq": {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else v) for k, v in fund.mrq.to_dict().items()},
        "column_sources": fund.column_sources,
    }


def get_fundamentals(
    ticker: str,
    *,
    as_of: date | None = None,
    force: bool = False,
) -> Fundamentals:
    """10y annual + TTM/MRQ series. EDGAR preferred; Yahoo fills gaps / non-filers."""
    ticker = ticker.upper().strip()
    as_of = as_of or date.today()
    cache_path = _cache_key("fundamentals", ticker, as_of.isoformat())
    if not force:
        cached = _read_cache(cache_path, max_age_hours=24)
        if isinstance(cached, dict):
            loaded = _from_cache(cached, ticker)
            if loaded is not None:
                return loaded

    facts_df = pd.DataFrame()
    try:
        facts_df = fetch_companyfacts_for_ticker(ticker)
    except Exception as exc:
        logger.debug("EDGAR facts unavailable for %s: %s", ticker, exc)

    edgar_annual = _annual_from_edgar(facts_df) if facts_df is not None and not facts_df.empty else _empty_annual()
    ttm = pd.Series(dtype="float64")
    mrq = pd.Series(dtype="float64")
    if facts_df is not None and not facts_df.empty:
        structured = fundamentals_as_of_structured(as_of, facts_df, ticker=ticker)
        ttm, mrq = _ttm_mrq_from_structured(structured)

    yahoo_annual = _empty_annual()
    try:
        fin = fetch_financials(ticker)
        yahoo_annual = _annual_from_yahoo(fin)
        ttm_yahoo, mrq_yahoo = _ttm_mrq_from_yahoo(fetch_ttm_financials(ticker), yahoo_annual)
    except Exception as exc:
        logger.debug("Yahoo statements unavailable for %s: %s", ticker, exc)
        ttm_yahoo, mrq_yahoo = pd.Series(dtype="float64"), pd.Series(dtype="float64")

    if edgar_annual is not None and not edgar_annual.empty and edgar_annual.dropna(how="all").shape[0] >= 1:
        annual = edgar_annual
        source = "edgar"
        col_src = _column_sources(annual, "edgar")
        if yahoo_annual is not None and not yahoo_annual.empty:
            for col in ANNUAL_COLUMNS:
                if col in annual.columns and annual[col].notna().sum() == 0 and col in yahoo_annual.columns:
                    aligned = yahoo_annual[col].reindex(annual.index)
                    if aligned.notna().any():
                        annual[col] = aligned
                        col_src[col] = "yahoo"
                        source = "mixed"
        if ttm.dropna().empty:
            ttm = ttm_yahoo
            source = "mixed"
        if mrq.dropna().empty:
            mrq = mrq_yahoo
            source = "mixed"
    else:
        annual = yahoo_annual
        ttm = ttm_yahoo if not ttm_yahoo.dropna().empty else ttm
        mrq = mrq_yahoo if not mrq_yahoo.dropna().empty else mrq
        source = "yahoo"
        col_src = _column_sources(annual, "yahoo")

    fund = Fundamentals(
        ticker=ticker,
        annual=annual if annual is not None else _empty_annual(),
        ttm=ttm,
        mrq=mrq,
        source=source,
        as_of=as_of,
        column_sources=col_src,
    )
    try:
        _write_cache(cache_path, _to_cache(fund))
    except Exception:
        pass
    return fund


def history_quality_inputs(fund: Fundamentals, raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Derived 5y/3y history metrics stored on the universe snapshot."""
    raw = raw or {}
    annual = fund.annual if fund.annual is not None else _empty_annual()
    subtract = True
    out: dict[str, Any] = {
        "fundamentals_source": fund.source,
        "goodwill_to_equity": None,
        "fcf_margin": None,
        "fcf_conversion_3y": None,
        "share_cagr_3y": None,
        "revenue_5y_cagr": None,
        "gross_margin_5y_delta": None,
        "roic_5y_mean": None,
        "roic_5y_std": None,
        "owner_earnings_norm": None,
        "ev_to_ebit": None,
        "p_to_oe": None,
        "equity_to_assets": None,
    }

    equity = fund.mrq.get("equity") if fund.mrq is not None else None
    goodwill = fund.mrq.get("goodwill") if fund.mrq is not None else None
    assets = fund.mrq.get("total_assets") if fund.mrq is not None else None
    if goodwill is not None and equity and float(equity) > 0:
        out["goodwill_to_equity"] = float(goodwill) / float(equity)
    if equity is not None and assets and float(assets) > 0:
        out["equity_to_assets"] = float(equity) / float(assets)

    ttm = fund.ttm if fund.ttm is not None else pd.Series(dtype="float64")
    revenue = ttm.get("revenue") if "revenue" in ttm.index else raw.get("revenue")
    ocf = ttm.get("operating_cashflow") if "operating_cashflow" in ttm.index else raw.get("operating_cashflow")
    capex = ttm.get("capex") if "capex" in ttm.index else raw.get("capex")
    fcf = None
    if ocf is not None:
        fcf = float(ocf) - abs(float(capex or 0.0))
    ni = ttm.get("net_income") if "net_income" in ttm.index else raw.get("net_income")
    if fcf is not None and revenue and float(revenue) != 0:
        out["fcf_margin"] = float(fcf) / float(revenue)

    if not annual.empty and "operating_cashflow" in annual.columns:
        oe_rows = []
        ni_rows = []
        for _, row in annual.tail(3).iterrows():
            oe = owner_earnings(row, subtract_sbc=subtract)
            oe_rows.append(oe)
            ni_rows.append(row.get("net_income"))
        fcf_vals = []
        for oe, ni_v in zip(oe_rows, ni_rows):
            if oe is None or ni_v is None or (isinstance(ni_v, float) and np.isnan(ni_v)) or ni_v == 0:
                continue
            fcf_vals.append(float(oe) / float(ni_v) if oe is not None else None)
        if fcf_vals:
            out["fcf_conversion_3y"] = float(np.nanmean(fcf_vals))

    if not annual.empty and "shares_diluted" in annual.columns:
        out["share_cagr_3y"] = cagr(annual["shares_diluted"], years=3)
    if not annual.empty and "revenue" in annual.columns:
        out["revenue_5y_cagr"] = cagr(annual["revenue"], years=5)
    if not annual.empty and "gross_profit" in annual.columns and "revenue" in annual.columns:
        gm = margin_series(annual["gross_profit"], annual["revenue"]).dropna()
        if len(gm) >= 2:
            span = min(5, len(gm) - 1)
            out["gross_margin_5y_delta"] = float(gm.iloc[-1] - gm.iloc[-(span + 1)])

    if not annual.empty:
        roics = []
        for _, row in annual.iterrows():
            ebit = row.get("ebit")
            nwc = None
            ca, cl, cash = row.get("current_assets"), row.get("current_liabilities"), row.get("cash")
            debt_st = 0.0
            if ca is not None and cl is not None and cash is not None:
                nwc = (float(ca) - float(cash)) - (float(cl) - debt_st)
            ppe = row.get("ppe_net")
            ic = None
            if nwc is not None and ppe is not None:
                ic = float(nwc) + float(ppe)
            else:
                eq, debt, cash_v = row.get("equity"), row.get("debt"), row.get("cash")
                if eq is not None:
                    ic = float(eq) + float(debt or 0.0) - float(cash_v or 0.0)
            assets_v = row.get("total_assets")
            if ic is not None and assets_v and abs(ic) < 0.10 * abs(float(assets_v)):
                ic = 0.10 * abs(float(assets_v))
            if ebit is not None and ic and ic != 0:
                roics.append(float(np.clip(float(ebit) / float(ic), -1.0, 2.0)))
        if len(roics) >= 3:
            tail = roics[-5:]
            out["roic_5y_mean"] = float(np.mean(tail))
            out["roic_5y_std"] = float(np.std(tail, ddof=0)) if len(tail) >= 2 else 0.0

    oe_ttm = owner_earnings(ttm, subtract_sbc=subtract)
    out["owner_earnings_norm"] = oe_ttm
    ebit = ttm.get("ebit") if "ebit" in ttm.index else raw.get("ebit")
    ev = raw.get("enterprise_value")
    price = raw.get("price")
    shares = ttm.get("shares_diluted") if "shares_diluted" in ttm.index else raw.get("shares_outstanding")
    mcap = raw.get("market_cap")
    if ebit and ev and float(ev) > 0:
        out["ev_to_ebit"] = float(ev) / float(ebit)
    if oe_ttm and shares and price and float(oe_ttm) != 0:
        out["p_to_oe"] = float(price) * float(shares) / float(oe_ttm)
    elif oe_ttm and mcap and float(oe_ttm) != 0:
        out["p_to_oe"] = float(mcap) / float(oe_ttm)
    return out


def attach_history_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    ticker = raw.get("ticker")
    if not ticker:
        return raw
    try:
        fund = get_fundamentals(str(ticker))
        raw.update(history_quality_inputs(fund, raw))
        raw["_fundamentals"] = fund
    except Exception as exc:
        logger.debug("History metrics skipped for %s: %s", ticker, exc)
    return raw
