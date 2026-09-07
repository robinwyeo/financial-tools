"""Historical factor panel reconstruction."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from backtest.constants import (
    BACKTEST_FACTOR_FAMILIES,
    BARGAIN_BACKTEST_COMPONENTS,
    DATA_STORE,
    QUARTER_ENDS,
    VALUATION_HISTORY_QUARTERS,
)
from backtest.data.constituents import load_membership
from backtest.data.edgar import fundamentals_as_of, load_fundamentals
from backtest.data.prices import load_prices, price_history_as_of, price_on_or_before
from core.data import (
    _compute_drawdown_metrics,
    _compute_momentum_12_1,
    _compute_rsi,
    _compute_volatility_12m,
    normalize_debt_to_equity,
    percentile_rank_in_history,
)
from core.factors import FACTOR_SCORE_COLUMNS, QUALITY_SCORE_COLUMNS, compute_all_factors
from core.rates import pit_risk_free
from core.scoring import compute_bargain_score

logger = logging.getLogger(__name__)

FACTOR_PANEL_PATH = DATA_STORE / "factor_panel.parquet"

# Layer-A quality inputs that are not already in BACKTEST_FACTOR_FAMILIES.
_QUALITY_INPUT_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(col for cols in QUALITY_SCORE_COLUMNS.values() for col in cols)
)
_INTERNAL_PANEL_COLS: tuple[str, ...] = ("_valuation", "_value_trap_flags")
# annual_series cuts off at now-N years; 40y covers BACKTEST_START with room.
_PIT_ANNUAL_YEARS = 40


def _pit_fundamentals(
    ticker: str,
    as_of: date,
    fundamentals: pd.DataFrame,
) -> Any | None:
    """Point-in-time Fundamentals from the companyfacts store (filed <= as_of)."""
    from core.edgar_facts import _filed_on_or_before, fundamentals_as_of_structured
    from core.fundamentals import (
        Fundamentals,
        _annual_from_edgar,
        _empty_annual,
        _ttm_mrq_from_structured,
    )

    if fundamentals is None or fundamentals.empty:
        return None
    sub = fundamentals
    if "ticker" in sub.columns:
        sub = sub[sub["ticker"].astype(str).str.upper() == ticker.upper()]
    if sub.empty:
        return None
    pit = _filed_on_or_before(sub, as_of)
    structured = fundamentals_as_of_structured(as_of, pit, ticker=ticker)
    ttm, mrq = _ttm_mrq_from_structured(structured)
    annual = _annual_from_edgar(pit, years=_PIT_ANNUAL_YEARS) if not pit.empty else _empty_annual()
    return Fundamentals(
        ticker=ticker.upper(),
        annual=annual if annual is not None else _empty_annual(),
        ttm=ttm,
        mrq=mrq,
        source="edgar",
        as_of=as_of,
        column_sources={},
    )


def _decision_analysis(raw: dict[str, Any], *, quality_score: float | None = None) -> dict[str, Any]:
    q = quality_score if quality_score is not None else raw.get("quality_score")
    if isinstance(q, float) and np.isnan(q):
        q = None
    flags = raw.get("_value_trap_flags")
    if not isinstance(flags, list):
        flags = []
    valuation = raw.get("_valuation")
    if not isinstance(valuation, dict):
        valuation = {}
    return {
        "ticker": raw.get("ticker"),
        "price": raw.get("price"),
        "sector": raw.get("sector"),
        "quality_score": q,
        "quality_percentile": q,
        "data_quality": raw.get("data_quality") or {},
        "distress_flag": raw.get("distress_flag"),
        "altman_z": raw.get("altman_z"),
        "altman_z_pp": raw.get("altman_z_pp"),
        "value_trap_flags": flags,
        "uncertainty": {"label": "Medium"},
        "valuation": valuation,
        "momentum_12_1": raw.get("momentum_12_1"),
        "rsi_14": raw.get("rsi_14"),
    }


def _attach_history_valuation_decision(
    raw: dict[str, Any],
    fund: Any,
    as_of: date,
    *,
    rf: float | None = None,
) -> None:
    """Fill quality history inputs, PIT valuation, and a preliminary decision_label."""
    from core.decision import decide
    from core.fundamentals import history_quality_inputs
    from core.quality import flags_to_dicts, value_trap_flags
    from core.valuation import valuation_summary

    hist = history_quality_inputs(fund, raw)
    for key, val in hist.items():
        if val is not None:
            raw[key] = val
        elif raw.get(key) is None:
            raw[key] = val
    raw["_fundamentals"] = fund
    if rf is None:
        rf = pit_risk_free(as_of)
    shares = raw.get("shares_outstanding")
    if shares is None and fund.ttm is not None:
        shares = fund.ttm.get("shares_diluted")
    cash = raw.get("total_cash")
    if cash is None and fund.mrq is not None:
        cash = fund.mrq.get("cash")
    debt = raw.get("total_debt")
    if debt is None and fund.mrq is not None:
        debt = fund.mrq.get("debt")
    raw["_valuation"] = valuation_summary(
        fund,
        price=raw.get("price"),
        shares=shares,
        cash=cash,
        debt=debt,
        market_cap=raw.get("market_cap"),
        ev=raw.get("enterprise_value"),
        sector=raw.get("sector"),
        industry=raw.get("industry"),
        uncertainty_label="Medium",
        analyst_growth=None,
        rf=rf,
    )
    try:
        raw["_value_trap_flags"] = flags_to_dicts(value_trap_flags(raw, fund))
    except Exception:
        raw["_value_trap_flags"] = []
    try:
        raw["decision_label"] = decide(_decision_analysis(raw)).label
    except Exception:
        raw["decision_label"] = None


def _flags_after_factors(raw: dict[str, Any], factors: dict[str, Any]) -> list[dict[str, Any]]:
    from core.quality import flags_to_dicts, value_trap_flags

    merged = dict(raw)
    merged.update({k: v for k, v in factors.items() if v is not None})
    try:
        return flags_to_dicts(value_trap_flags(merged, raw.get("_fundamentals")))
    except Exception:
        flags = raw.get("_value_trap_flags")
        return flags if isinstance(flags, list) else []


def _attach_quality_and_decision(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional quality_score per quarter, then decide() each row."""
    from core.decision import decide
    from core.quality import compute_quality_score

    out = panel.copy()
    if out.empty:
        out["quality_score"] = []
        out["decision_label"] = []
        return out

    frames: list[pd.DataFrame] = []
    grouped = (
        [(None, out)]
        if "quarter_end" not in out.columns
        else list(out.groupby("quarter_end", sort=False))
    )
    for _, grp in grouped:
        group_col = "sector" if "sector" in grp.columns else None
        try:
            scored = compute_quality_score(grp, group_col=group_col)
        except Exception:
            scored = grp.copy()
            scored["quality_score"] = None
        labels: list[str | None] = []
        for _, row in scored.iterrows():
            q = row.get("quality_score")
            if isinstance(q, float) and np.isnan(q):
                q = None
            try:
                labels.append(decide(_decision_analysis(row.to_dict(), quality_score=q)).label)
            except Exception:
                labels.append(row.get("decision_label") if isinstance(row.get("decision_label"), str) else None)
        scored = scored.copy()
        scored["decision_label"] = labels
        frames.append(scored)
    merged = pd.concat(frames, ignore_index=True)
    drop = [c for c in _INTERNAL_PANEL_COLS if c in merged.columns]
    if drop:
        merged = merged.drop(columns=drop)
    return merged


def _realized_earnings_growth(fund: dict[str, float]) -> float | None:
    ni = fund.get("net_income")
    ni_prior = fund.get("net_income_prior")
    if ni is None or ni_prior is None or ni_prior == 0:
        return None
    return (ni / ni_prior) - 1.0


def _build_raw_row(
    ticker: str,
    as_of: date,
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    rf: float | None = None,
) -> dict[str, Any]:
    fund = fundamentals_as_of(as_of, ticker, fundamentals)
    price = price_on_or_before(prices, ticker, as_of)
    hist = price_history_as_of(prices, ticker, as_of)

    shares = fund.get("shares_outstanding")
    market_cap = price * shares if price and shares else None
    book_value = fund.get("book_value")
    book_equity = fund.get("book_equity")
    total_debt = fund.get("total_debt")
    total_cash = fund.get("total_cash")

    debt_to_equity = None
    if total_debt is not None and book_equity and book_equity != 0:
        debt_to_equity = normalize_debt_to_equity(total_debt / book_equity * 100.0)

    trailing_eps = None
    if fund.get("net_income") is not None and shares:
        trailing_eps = fund["net_income"] / shares

    trailing_pe = None
    if trailing_eps and trailing_eps > 0 and price:
        trailing_pe = price / trailing_eps

    earnings_growth = _realized_earnings_growth(fund)
    dividend_yield = None
    if fund.get("dividends_paid") is not None and market_cap and market_cap > 0:
        dividend_yield = abs(fund["dividends_paid"]) / market_cap

    ebit = fund.get("ebit")
    enterprise_value = None
    if market_cap is not None:
        enterprise_value = market_cap + (total_debt or 0.0) - (total_cash or 0.0)

    current_ratio = None
    ca = fund.get("current_assets")
    cl = fund.get("current_liabilities")
    if ca is not None and cl and cl > 0:
        current_ratio = ca / cl

    fifty_two_week_high = None
    all_time_high = None
    if not hist.empty and "Close" in hist.columns:
        closes = hist["Close"].dropna()
        if len(closes) > 0:
            window = closes.tail(min(252, len(closes)))
            fifty_two_week_high = float(window.max())
            all_time_high = float(closes.max())

    raw: dict[str, Any] = {
        "ticker": ticker.upper(),
        "price": price,
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "book_value": book_value,
        "book_equity": book_equity,
        "shares_outstanding": shares,
        "shares_prior": fund.get("shares_outstanding_prior"),
        "total_assets": fund.get("total_assets"),
        "total_assets_prior": fund.get("total_assets_prior"),
        "total_liabilities": fund.get("total_liabilities"),
        "current_assets": ca,
        "current_liabilities": cl,
        "current_assets_prior": fund.get("current_assets_prior"),
        "current_liabilities_prior": fund.get("current_liabilities_prior"),
        "long_term_debt": fund.get("long_term_debt"),
        "long_term_debt_prior": fund.get("long_term_debt_prior"),
        "gross_profit": fund.get("gross_profit"),
        "gross_profit_prior": fund.get("gross_profit_prior"),
        "net_income": fund.get("net_income"),
        "net_income_prior": fund.get("net_income_prior"),
        "ebit": ebit,
        "revenue": fund.get("revenue"),
        "revenue_prior": fund.get("revenue_prior"),
        "operating_cashflow": fund.get("operating_cashflow"),
        "free_cashflow": fund.get("free_cashflow"),
        "ppe_net": fund.get("ppe_net"),
        "interest_expense": fund.get("interest_expense"),
        "goodwill": fund.get("goodwill"),
        "debt_st": fund.get("debt_st"),
        "dividends_paid": fund.get("dividends_paid"),
        "repurchase_of_stock": fund.get("repurchase_of_stock"),
        "retained_earnings": fund.get("retained_earnings"),
        "total_cash": total_cash,
        "total_debt": total_debt,
        "debt_to_equity": debt_to_equity,
        "current_ratio_info": current_ratio,
        "trailing_pe": trailing_pe,
        "trailing_eps": trailing_eps,
        "earnings_growth": earnings_growth,
        "dividend_yield": dividend_yield,
        "trailing_peg_ratio": None,
        "momentum_12_1": _compute_momentum_12_1(hist) if not hist.empty else None,
        "volatility_12m": _compute_volatility_12m(hist) if not hist.empty else None,
        "max_drawdown": None,
        "downside_deviation": None,
        "fifty_two_week_high": fifty_two_week_high,
        "all_time_high": all_time_high,
        "rsi_14": _compute_rsi(hist) if not hist.empty else None,
        "recommendations": pd.DataFrame(),
        "target_mean": None,
        "capex": fund.get("capex"),
        "sbc": fund.get("sbc"),
        "owner_earnings_norm": None,
        "p_to_oe": None,
        "ev_to_ebit": None,
        "revenue_5y_cagr": None,
        "share_cagr_3y": None,
        "fcf_conversion_3y": None,
        "gross_margin_5y_delta": None,
        "roic_5y_mean": None,
        "roic_5y_std": None,
        "decision_label": None,
        "quality_score": None,
    }
    if ebit and enterprise_value and enterprise_value > 0:
        raw["ev_to_ebit"] = float(enterprise_value) / float(ebit)
    try:
        from core.fundamentals import owner_earnings

        oe = owner_earnings(
            {
                "operating_cashflow": fund.get("operating_cashflow"),
                "capex": fund.get("capex"),
                "sbc": fund.get("sbc"),
            }
        )
        raw["owner_earnings_norm"] = oe
        if oe and market_cap and oe != 0:
            raw["p_to_oe"] = float(market_cap) / float(oe)
    except Exception:
        pass
    if not hist.empty:
        dd = _compute_drawdown_metrics(hist)
        raw["max_drawdown"] = dd.get("max_drawdown")
        raw["downside_deviation"] = dd.get("downside_deviation")
    try:
        pit_fund = _pit_fundamentals(ticker, as_of, fundamentals)
        if pit_fund is not None:
            _attach_history_valuation_decision(raw, pit_fund, as_of, rf=rf)
    except Exception:
        logger.debug("PIT quality/valuation skipped for %s as_of %s", ticker, as_of, exc_info=True)
    return raw


def compute_historical_factors(raw: dict[str, Any]) -> dict[str, float | None]:
    """Compute factor sub-signal columns, excluding earnings_revisions."""
    all_factors = compute_all_factors(raw)
    all_factors.pop("earnings_revisions", None)
    return all_factors


def compute_historical_bargain(
    raw: dict[str, Any],
    factors: dict[str, Any],
    *,
    valuation_vs_history: float | None = None,
) -> dict[str, Any]:
    """Bargain score using long-horizon valuation components (no RSI / analyst upside)."""
    return compute_bargain_score(
        price=raw.get("price"),
        graham_ratio=factors.get("graham_ratio"),
        fifty_two_week_high=raw.get("fifty_two_week_high"),
        valuation_vs_history=valuation_vs_history,
    )


def _attach_valuation_vs_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time EV/EBIT and P/OE cheapness vs each ticker's trailing 10y history."""
    out = panel.copy()
    if out.empty:
        out["valuation_vs_history"] = []
        return out
    out["quarter_end"] = pd.to_datetime(out["quarter_end"])
    out = out.sort_values(["ticker", "quarter_end"])
    ey_col = "earnings_yield" if "earnings_yield" in out.columns else None
    poe_col = None
    if "p_to_oe" in out.columns:
        poe_col = "p_to_oe"
    elif "oe_yield" in out.columns:
        poe_col = "oe_yield"
    scores: list[float | None] = []
    for ticker, grp in out.groupby("ticker", sort=False):
        ey = grp[ey_col].tolist() if ey_col else [None] * len(grp)
        poe_raw = grp[poe_col].tolist() if poe_col else [None] * len(grp)
        # Invert P/OE multiple into a yield when values look like multiples (> 1.5).
        poe_yield = []
        for v in poe_raw:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                poe_yield.append(None)
            elif float(v) > 1.5:
                poe_yield.append(1.0 / float(v))
            else:
                poe_yield.append(float(v))
        for i, current in enumerate(ey):
            start = max(0, i - VALUATION_HISTORY_QUARTERS)
            ey_hist = [
                float(v)
                for v in ey[start:i]
                if v is not None and not (isinstance(v, float) and np.isnan(v))
            ]
            poe_hist = [
                float(v)
                for v in poe_yield[start:i]
                if v is not None and not (isinstance(v, float) and np.isnan(v))
            ]
            parts = []
            p1 = percentile_rank_in_history(
                None if current is None or (isinstance(current, float) and np.isnan(current)) else float(current),
                ey_hist,
            )
            cur_poe = poe_yield[i]
            p2 = percentile_rank_in_history(
                None if cur_poe is None or (isinstance(cur_poe, float) and np.isnan(cur_poe)) else float(cur_poe),
                poe_hist,
            )
            if p1 is not None:
                parts.append(p1)
            if p2 is not None:
                parts.append(p2)
            scores.append(float(np.mean(parts)) if parts else None)
    out["valuation_vs_history"] = scores
    return out


def _weighted_bargain_from_components(
    components: dict[str, float | None],
    weights: dict[str, float] | None = None,
) -> float | None:
    """Blend precomputed 0-100 bargain components with default weights."""
    from core.scoring import BARGAIN_COMPONENT_WEIGHTS

    wmap = weights or BARGAIN_COMPONENT_WEIGHTS
    weighted_sum = 0.0
    weight_available = 0.0
    for key, val in components.items():
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        w = float(wmap.get(key, 0.0))
        weighted_sum += float(val) * w
        weight_available += w
    if weight_available <= 0:
        return None
    return weighted_sum / weight_available


def _recompute_panel_bargain(panel: pd.DataFrame) -> pd.DataFrame:
    """Recompute bargain_score / components with the long-horizon formula."""
    from core.scoring import _linear_score

    out = panel.copy()
    if "valuation_vs_history" not in out.columns:
        out = _attach_valuation_vs_history(out)

    scores: list[float | None] = []
    mos_list: list[float | None] = []
    val_list: list[float | None] = []
    d52_list: list[float | None] = []

    for _, row in out.iterrows():
        graham = row.get("graham_ratio")
        mos = None
        if graham is not None and not (isinstance(graham, float) and np.isnan(graham)) and graham > 0:
            mos = _linear_score(float(graham), 0.30, 1.30)

        val = row.get("valuation_vs_history")
        if val is not None and isinstance(val, float) and np.isnan(val):
            val = None

        # Prefer freshly computed discount; fall back to stored component score.
        d52 = None
        price = row.get("price")
        high = row.get("fifty_two_week_high")
        if (
            price is not None
            and high is not None
            and not (isinstance(high, float) and np.isnan(high))
            and high > 0
            and price > 0
        ):
            d52 = _linear_score(1.0 - (float(price) / float(high)), 0.0, 0.30)
        else:
            stored = row.get("bargain_discount_52w")
            if stored is not None and not (isinstance(stored, float) and np.isnan(stored)):
                d52 = float(stored)

        components = {
            "margin_of_safety": mos,
            "valuation_vs_history": None if val is None else float(val),
            "discount_52w": d52,
        }
        scores.append(_weighted_bargain_from_components(components))
        mos_list.append(mos)
        val_list.append(None if val is None else float(val))
        d52_list.append(d52)

    out["bargain_score"] = scores
    out["bargain_margin_of_safety"] = mos_list
    out["bargain_valuation_vs_history"] = val_list
    out["bargain_discount_52w"] = d52_list
    if "bargain_rsi_oversold" in out.columns:
        out = out.drop(columns=["bargain_rsi_oversold"])
    return out


def enrich_factor_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach valuation-vs-history and recompute long-horizon bargain scores."""
    panel = _attach_valuation_vs_history(panel)
    return _recompute_panel_bargain(panel)


def build_factor_panel(
    quarter_ends: list[date] | None = None,
    force: bool = False,
    max_quarters: int | None = None,
) -> pd.DataFrame:
    """Build quarterly factor panel for historical backtesting.

    Stores all sub-signal columns for each factor group so the backtest engine
    can apply rank-then-average aggregation, mirroring core.scoring.score_universe_df.
    """
    if FACTOR_PANEL_PATH.exists() and not force:
        return pd.read_parquet(FACTOR_PANEL_PATH)

    DATA_STORE.mkdir(parents=True, exist_ok=True)
    membership = load_membership()
    fundamentals = load_fundamentals()
    prices = load_prices()

    qends = quarter_ends or QUARTER_ENDS
    if max_quarters is not None:
        qends = qends[:max_quarters]

    # Collect all sub-signal column names for the backtestable groups
    backtest_sub_cols: list[str] = [
        col
        for family in BACKTEST_FACTOR_FAMILIES
        for col in FACTOR_SCORE_COLUMNS.get(family, [])
    ]

    from core.rates import fetch_fred_history

    dgs10 = fetch_fred_history("DGS10")
    dgs10_hist = dgs10 if dgs10 is not None and not dgs10.empty else None

    rows: list[dict[str, Any]] = []
    for qi, qend in enumerate(qends, start=1):
        logger.info("Building factors for %s (%d/%d)", qend, qi, len(qends))
        rf = pit_risk_free(qend, history=dgs10_hist)
        tickers = membership[membership["quarter_end"] == qend]["ticker"].astype(str).tolist()
        for ticker in tickers:
            raw = _build_raw_row(ticker, qend, fundamentals, prices, rf=rf)
            if raw.get("price") is None:
                continue
            factors = compute_historical_factors(raw)
            row: dict[str, Any] = {
                "quarter_end": qend,
                "ticker": ticker.upper(),
                "price": raw.get("price"),
                "market_cap": raw.get("market_cap"),
                "fifty_two_week_high": raw.get("fifty_two_week_high"),
                "ev_to_ebit": raw.get("ev_to_ebit")
                if raw.get("ev_to_ebit") is not None
                else factors.get("ev_to_ebit"),
                "p_to_oe": raw.get("p_to_oe") if raw.get("p_to_oe") is not None else factors.get("p_to_oe"),
                "altman_z": factors.get("altman_z"),
                "altman_z_pp": factors.get("altman_z_pp"),
                "decision_label": raw.get("decision_label"),
                "_valuation": raw.get("_valuation") or {},
                "_value_trap_flags": _flags_after_factors(raw, factors),
            }
            for col in backtest_sub_cols:
                row[col] = factors.get(col)
            for col in _QUALITY_INPUT_COLUMNS:
                if col not in row:
                    val = factors.get(col)
                    row[col] = val if val is not None else raw.get(col)
            # Placeholder bargain; enrich_factor_panel fills valuation + score.
            row["bargain_score"] = None
            for comp in BARGAIN_BACKTEST_COMPONENTS:
                row[f"bargain_{comp}"] = None
            rows.append(row)

    panel = pd.DataFrame(rows)
    panel = _attach_quality_and_decision(panel)
    panel = enrich_factor_panel(panel)
    panel.to_parquet(FACTOR_PANEL_PATH, index=False)
    logger.info("Saved factor panel with %d rows", len(panel))
    return panel


def _reblend_bargain_score(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute the blended bargain_score from stored component columns using the
    current default weights (vectorized).

    The stored bargain_score was blended with whatever weights were active when
    the panel was built; components are weight-independent, so re-blending on
    load keeps calibration/simulation consistent with the live configuration.
    """
    from core.scoring import BARGAIN_COMPONENT_WEIGHTS

    comp_cols = {key: f"bargain_{key}" for key in BARGAIN_BACKTEST_COMPONENTS}
    if not all(col in panel.columns for col in comp_cols.values()):
        return panel

    out = panel.copy()
    weighted_sum = pd.Series(0.0, index=out.index)
    weight_available = pd.Series(0.0, index=out.index)
    for key, col in comp_cols.items():
        w = float(BARGAIN_COMPONENT_WEIGHTS.get(key, 0.0))
        vals = pd.to_numeric(out[col], errors="coerce")
        available = vals.notna()
        weighted_sum += vals.fillna(0.0) * w * available
        weight_available += w * available
    out["bargain_score"] = np.where(
        weight_available > 0, weighted_sum / weight_available, np.nan
    )
    return out


def load_factor_panel(*, enrich: bool = True) -> pd.DataFrame:
    """Load factor panel; enrich with valuation-vs-history bargain if needed."""
    if not FACTOR_PANEL_PATH.exists():
        return build_factor_panel()
    panel = pd.read_parquet(FACTOR_PANEL_PATH)
    if not enrich:
        return panel
    needs_enrich = (
        "valuation_vs_history" not in panel.columns
        or "bargain_valuation_vs_history" not in panel.columns
        or "bargain_rsi_oversold" in panel.columns
    )
    if needs_enrich:
        logger.info("Enriching factor panel with long-horizon bargain components")
        panel = enrich_factor_panel(panel)
        panel.to_parquet(FACTOR_PANEL_PATH, index=False)
    return _reblend_bargain_score(panel)
